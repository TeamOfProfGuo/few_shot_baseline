# encoding:utf-8
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

import models
import utils
from utils.few_shot import *
from .models import register


@register('adapt')
class AdaptClassifier(nn.Module):

    def __init__(self, encoder, encoder_args, classifier, classifier_args,
                 meta_train, meta_train_args):
        super().__init__()
        self.backbone = encoder # resnet12
        self.n_classes = classifier_args['n_classes']
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)

        if meta_train:
            self.feat_level = meta_train_args['feat_level']
            self.norm = 'norm'
            if meta_train_args['learn_temp']:
                self.temp = nn.Parameter(torch.tensor(meta_train_args['temp']))
            else:
                self.temp = meta_train_args['temp']

            if meta_train_args['learn_thresh']:
                self.thresh = nn.Parameter(torch.tensor(meta_train_args['thresh']))
            else:
                self.thresh = meta_train_args['thresh']   # 因为我后面要用 thresh.repeat()

            if meta_train_args['learn_tp']:
                self.tp = nn.Parameter(torch.tensor(1.0))
            else:
                self.tp = 1.0

            if self.feat_level == 23:
                fea_dim = 128+256
            elif self.feat_level == 34:
                fea_dim = 256+512
            elif self.feat_level == 4:
                fea_dim = 512

            reduce_dim = meta_train_args['reduce_dim']
            self.down_mid = nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False)
            if fea_dim == reduce_dim:
                self.down_mid.weight.data.fill_(1.0)
                #nn.ReLU(inplace=True),
                # nn.Dropout2d(p=0.5)

    def forward(self, x):  # 用于pretraining
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def inner_loop(self, x_shot, s_label, aug=False, init_weight=False):   # x_shot[N,3,h,w]

        # finetune linear classifier
        inner_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                          weight_decay=0.001)
        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda() if torch.cuda.is_available() else loss_function

        batch_size = 4
        support_size = x_shot.shape[0]  # 25
        for epoch in range(1, 100+1):        # finetune 100 个epoch
            if aug:
                crop = transforms.RandomResizedCrop(x_shot.shape[-2:])
                x_shot = torch.cat([crop(image).unsqueeze_(0) for image in x_shot], dim=0)
            z_support = self.encoder(x_shot) # backbone生成的feature, [25,512],[75,512]

            rand_id = np.random.permutation(support_size)
            epoch_loss, n_iter = 0.0, 0
            for i in range(0, support_size, batch_size):
                # 选择当前batch
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)])
                selected_id = selected_id.cuda() if torch.cuda.is_available() else selected_id
                z_batch = z_support[selected_id]
                y_batch = s_label[selected_id]
                scores = self.classifier(z_batch)
                loss = loss_function(scores, y_batch)
                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()

                epoch_loss += loss
                n_iter += 1
            if epoch%50==1:
                print('the loss after epoch {} is {}'.format(epoch, epoch_loss/n_iter))

    def outer_loop(self, x_s, x_q, y_s, meta_args):

        assert x_s.dim()==6, "x_shot should have 5 dim: eps, n_way, n_shot, ch, h, w"
        eps, n_way, n_shot, ch, h, w = x_s.shape
        y_s = y_s.view(eps, -1)

        q_logits0, q_logits = [], []

        for ep in range(eps):  # 针对其中一个episode predict logits for the query images
            x_shot = x_s[ep].view(n_way*n_shot, ch, h, w)
            y_shot = y_s[ep]
            x_query = x_q[ep]

            # ================== 训练classifier =====================
            self.train()
            self.reset_cls_weight()
            self.inner_loop(x_shot, y_shot, init_weight=False)  # 更新的weight在get_CAM中使用

            # ==================== 计算prototype ==================
            self.eval()

            cam_lst, cls_id, logits, mid_feat = self.get_CAM(x_shot, y_shot)  # cam_lst element: [1, 5, 5]
            feat_conv = self.down_mid(mid_feat)           # [25,256,10,10]
            cam = torch.cat( [l for l in cam_lst], dim=0) # [25, 5, 5]

            # support weighted feature
            sw_feat = weighted_feat(feat_conv, cam, norm=self.norm,  T=self.temp, thresh=self.thresh)  #[25, 256]
            sw_feat = sw_feat.view(meta_args['n_way'], meta_args['n_shot'], -1)  # [5, 5, 256]
            protos = sw_feat.mean(dim=1)  # [5, 256]

            # =============== 根据prototype对query分类 ==================
            cam_lst, cls_id, logits0, mid_feat = self.get_CAM(x_query) # cam_lst element [5, 5, 5]
            feat_conv = self.down_mid(mid_feat)  # [75, 256, 10, 10]
            cam = torch.cat( [l for l in cam_lst], dim=0 ) # [75*5, 5, 5]

            qw_feat = weighted_feat(feat_conv, cam, norm=self.norm, T=self.temp, thresh=self.thresh)  # [375, 256]
            qw_feat = qw_feat.view(75, 5, 256)  # 75 n_query, 5way, 256channel
            logits = utils.compute_logits_localize(qw_feat, protos, metric='cos', temp=self.tp)  # [75,5]

            # 结束本episode的计算，输出结果
            q_logits0.append(logits0)
            q_logits.append(logits)

        q_logits0 = torch.cat(q_logits0, dim=0)
        q_logits = torch.cat(q_logits, dim=0)
        return q_logits0, q_logits

    def get_CAM(self, x, y=None):  # accept同一episode中的多张图片 x: [B,3,h,w] y:[1]
        if self.backbone == 'resnet12':
            layer_lst =['layer2', 'layer3', 'layer4']

        # forward pass中保留中间层的输出
        feat_blobs = []
        def hook_feature(module, input, output):
            feat_blobs.append(output)

        handles = {}
        for name in layer_lst:
            handles[name] = self.encoder._modules.get(name).register_forward_hook(hook_feature)

        with torch.no_grad():
            logits = self.forward(x)
            feat2, feat3, feat4 = feat_blobs  # [100,128,20,20] #[100,256,10,10] #[100,512,5,5]
            for k, v in handles.items():
                handles[k].remove()

        # generate the class activation maps
        if y is not None:
            class_idx = y.unsqueeze(1)  # [100,1]
        else:
            class_idx = torch.arange(self.n_classes).repeat(len(x), 1)  # [100,5]
            if torch.cuda.is_available():
                class_idx = class_idx.cuda()
        weight_softmax = self.classifier.state_dict()['linear.weight']  # [5, 512] 共512个channel
        bias_softmax = self.classifier.state_dict()['linear.bias']  # [5]对应5个class 在cpu上/如果从gpu读取的话在gpu上

        bz, nc, h, w = feat4.shape   # batch_size=1 [1, 512, 5, 5]
        cam_lst = []
        for i in range(len(x)):
            weight_i = torch.cat( [weight_softmax[idx].unsqueeze(0) for idx in class_idx[i]], dim=0 )  #[len(idx), 512]
            bias_i = bias_softmax[class_idx[i]].unsqueeze(1)  # [len(idx), 1]
            cam_i = torch.matmul(weight_i, feat4[i].reshape(nc, h*w))   # [len(idx), 25]
            cam_i = cam_i + bias_i       # 单个img针对所有class的CAM, 其长度取决于：选取了多少个y_idx
            cam_lst.append(cam_i.view(cam_i.shape[0], h, w))        # list, 每个element[len(idx), h, w]

        if self.feat_level ==23:
            mid_feat = torch.cat( (F.interpolate(feat2,size=feat3.shape[-2:]), feat3), dim=1 )  # [1,ch,h,w]
        elif self.feat_level == 34:
            mid_feat = torch.cat( (F.interpolate(feat3,size=feat4.shape[-2:]), feat4), dim=1 )  # [1,ch,h,w]
        elif self.feat_level == 4:
            mid_feat = feat4
        return cam_lst, class_idx, logits, mid_feat
        # cam_lst:[100]每项[len(idx],5, 5],class_idx:[100,5way]或者[100,1], logits:[100,5way], mid_feat[100, 384, 10, 10]

    def reset_cls_weight(self):
        for name, module in self.classifier.named_children():
            if isinstance(module, nn.Linear):
                print('resetting', name)
                module.reset_parameters()



@register('linear-classifier')
class LinearClassifier(nn.Module):

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


@register('nn-classifier')
class NNClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, metric='cos', temp=None):
        super().__init__()
        self.proto = nn.Parameter(torch.empty(n_classes, in_dim))
        nn.init.kaiming_uniform_(self.proto, a=math.sqrt(5))
        if temp is None:
            if metric == 'cos':
                temp = nn.Parameter(torch.tensor(10.))
            else:
                temp = 1.0
        self.metric = metric
        self.temp = temp

    def forward(self, x):
        return utils.compute_logits(x, self.proto, self.metric, self.temp)


