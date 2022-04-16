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
                 meta_train, cam_args,  meta_train_args):
        super().__init__()
        self.backbone = encoder # resnet12
        self.n_classes = classifier_args['n_classes']
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)
        self.meta_train_args = meta_train_args
        self.cam_args = cam_args

        self.metric = meta_train_args.get('dist', 'cos')
        self.base_mean = None                                         # this is placeholder, need to update before using eu dist with centering

        if meta_train:

            if cam_args['learn_thresh']:
                self.thresh = nn.Parameter(torch.tensor(cam_args['thresh']))
            else:
                self.thresh = cam_args['thresh']   # 因为我后面要用 thresh.repeat()

            if cam_args['learn_temp']:
                self.temp = nn.Parameter(torch.tensor(cam_args['temp']))
            else:
                self.temp = cam_args['temp']

            if meta_train_args['learn_tp']:
                self.tp = nn.Parameter(torch.tensor(1.0))
            else:
                self.tp = 1.0

            self.feat_level = meta_train_args['feat_level']
            self.fea_dim = []
            dim = [64, 128, 256, 512]
            for level in str(self.feat_level):
                self.fea_dim.append(dim[int(level)-1])

            self.feat_adapt = meta_train_args['feat_adapt']
            if meta_train_args['feat_adapt'] == 'idt':
                self.down_mid = nn.Identity()
            elif meta_train_args['feat_adapt'] == 'lr':
                self.down_mid = nn.Sequential(nn.Linear(sum(self.fea_dim), sum(self.fea_dim)),
                                              nn.ReLU()                    # 没有必要有negative weight
                                              )
            elif meta_train_args['feat_adapt'] == 'wt':      # 分别求low_level和high level的similarity然后求和
                self.down_mid = nn.Parameter(torch.tensor(0.1))

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
            #if epoch%50==1:
                #print('the loss after epoch {} is {}'.format(epoch, epoch_loss/n_iter))

    def outer_loop(self, x_s, x_q, y_s, meta_args):

        assert x_s.dim()==6, "x_shot should have 5 dim: eps, n_way, n_shot, ch, h, w"
        eps, n_way, n_shot, ch, h, w = x_s.shape
        y_s = y_s.view(eps, -1)  # [ep, 25]

        q_logits0, q_logits = [], []

        if len(str(self.feat_level)) == 2:
            sub1, sub2 = [], []

        for ep in range(eps):  # 针对其中一个episode predict logits for the query images
            x_shot = x_s[ep].view(n_way*n_shot, ch, h, w)
            y_shot = y_s[ep]
            x_query = x_q[ep]    # [73,3,80,80]

            # ================== 训练classifier =====================
            self.train()
            self.reset_cls_weight()
            self.inner_loop(x_shot, y_shot, init_weight=False)  # 更新的weight在get_CAM中使用

            # ==================== 计算prototype ==================
            self.eval()

            cam_lst, cls_id, logits, mid_feat = self.get_CAM(x_shot, y_shot)  # cam_lst element: [1, 5, 5]
            cam = torch.cat( [l for l in cam_lst], dim=0)  # [25, 5, 5]

            # support weighted feature
            cam_norm = self.cam_args.get('norm', 'norm')
            sw_feat = weighted_feat(mid_feat, cam, norm=cam_norm,  T=self.temp, thresh=self.thresh)  #[25, 256]
            sw_feat = sw_feat.view(meta_args['n_way'], meta_args['n_shot'], -1)  # [5, 5, 256]
            protos = sw_feat.mean(dim=1)  # [5, 256]

            # =============== 根据prototype对query分类 ==================
            cam_lst, cls_id, logits0, mid_feat = self.get_CAM(x_query)  # cam_lst element [5, 5, 5]
            cam = torch.cat( [l for l in cam_lst], dim=0 )  # [75*5, 5, 5] 每个query img对应5(way)个cam
            qw_feat = weighted_feat(mid_feat, cam, norm=cam_norm, T=self.temp, thresh=self.thresh)  # [375, 256] 每个query对应5(way)个weighted feature

            # =========== transform protos and qw_feat ==============
            if self.feat_adapt not in ['wt']:  # 基于unified feature
                if self.feat_adapt == 'lr':
                    task_rep = protos.mean(dim=0).unsqueeze(0)   # [1, 512]
                    adapt_wt = self.down_mid(task_rep)  # [1, 512]
                    protos = torch.mul(protos, adapt_wt)  # [5, 512]
                    qw_feat = torch.mul(qw_feat, adapt_wt)  # [375, 512] 375=75*5way
                elif self.feat_adapt == 'idt':
                    pass
                qw_feat = qw_feat.view(x_query.shape[0], n_way, -1)  # 75 n_query, 5way, 256channel
                logits = utils.compute_logits_localize(qw_feat, protos, metric=self.metric, base_mean=self.base_mean)  # [75,5] 其实就是求了similarity
            else:
                dim1, dim2 = self.fea_dim
                sub_protos1, sub_protos2 = protos[:, 0:dim1], protos[:, dim1:]     # [5,256], [5,512]
                sub_qwfeat1 = qw_feat[:,0:dim1].view(x_query.shape[0], n_way, -1)  # [75,5,256]
                sub_qwfeat2 = qw_feat[:,dim1: ].view(x_query.shape[0], n_way, -1)
                sub_logits1 = utils.compute_logits_localize(sub_qwfeat1, sub_protos1, metric=self.metric, base_mean=self.base_mean)  # low level
                sub_logits2 = utils.compute_logits_localize(sub_qwfeat2, sub_protos2, metric=self.metric, base_mean=self.base_mean)   # high level
                logits = sub_logits1 * self.down_mid + sub_logits2

            logits = logits*self.tp

            # 结束本episode的计算，输出结果
            q_logits0.append(logits0)
            q_logits.append(logits)

            if len(str(self.feat_level)) == 2:
                sub1.append(sub_logits1)  # low level
                sub2.append(sub_logits2)

        q_logits0 = torch.cat(q_logits0, dim=0)
        q_logits = torch.cat(q_logits, dim=0)

        if len(str(self.feat_level))==1:
            return q_logits0, q_logits
        else:
            sub1 = torch.cat(sub1, dim=0)
            sub2 = torch.cat(sub2, dim=0)
            return q_logits0, q_logits, sub1, sub2

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
        elif self.feat_level == 3:
            mid_feat = feat3
        return cam_lst, class_idx, logits, mid_feat
        # cam_lst:[100]每项[len(idx],5, 5],class_idx:[100,5way]或者[100,1], logits:[100,5way], mid_feat[100, 384, 10, 10]

    def reset_cls_weight(self):
        for name, module in self.classifier.named_children():
            if isinstance(module, nn.Linear):
                print('resetting', name)
                module.reset_parameters()

    def get_base_mean(self, mean_list):
        out_mean, mid_mean = mean_list
        out_mean, mid_mean = torch.from_numpy(out_mean), torch.from_numpy(mid_mean)
        if torch.cuda.is_available():
                out_mean, mid_mean = out_mean.cuda(), mid_mean.cuda()
        if self.feat_level == 3:
            self.base_mean = mid_mean
        elif self.feat_level == 4:
            self.base_mean = out_mean
        elif self.feat_level == 34:
            self.base_mean = [mid_mean, out_mean]



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


