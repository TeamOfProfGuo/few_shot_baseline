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
            self.mid = 23
            self.norm = 'norm'
            if meta_train_args['learn_temp']:
                self.temp = nn.Parameter(torch.tensor(meta_train_args['temp']))
            else:
                self.temp = meta_train_args['temp']

            if meta_train_args['learn_thresh']:
                self.thresh = nn.Parameter(torch.tensor(meta_train_args['thresh']))
            else:
                self.thresh = meta_train_args['thresh']

            if meta_train_args['learn_tp']:
                self.tp = nn.Parameter(torch.tensor(1.0))
            else:
                self.tp = 1.0

            if self.mid == 23:
                fea_dim = 128+256
            else:
                fea_dim = 256+512
            reduce_dim=256
            self.down_mid =  nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False)
                #nn.ReLU(inplace=True),
                # nn.Dropout2d(p=0.5)

    def forward(self, x):  # 用于pretraining
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def inner_loop(self, x_shot, s_label, aug=False, init_weight=False):

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

    def outer_loop(self, x_shot, x_query, y_shot, y_query, meta_args):

        # ================== 训练classifier =====================
        self.train()
        self.reset_cls_weight()
        self.inner_loop(x_shot, y_shot, init_weight=False)  # 更新的weight在get_CAM中使用

        # ==================== 计算prototype ==================
        self.eval()

        sw_feat = []   # support weighted feature
        for i in range(len(x_shot)):
            x = x_shot[i:i + 1]
            y = y_shot[i:i + 1]
            cam, cls_id, _, mid_feat = self.get_CAM(x, y)
            cam = cam[0]
            feat_conv = self.down_mid(mid_feat).squeeze(0)
            feat = weighted_feat(feat_conv, cam, norm=self.norm,  T=self.temp, thresh=self.thresh)  # 当前图片的prototype/weighted feature  #dim [512]
            sw_feat.append(feat)

        sw_feat = torch.cat([feat.unsqueeze(dim=0) for feat in sw_feat], dim=0)  # [25, 512]
        sw_feat = sw_feat.view(meta_args['n_way'], meta_args['n_shot'], -1)
        protos = sw_feat.mean(dim=1)  # [5, 512]

        # =============== 根据prototype对query分类 ==================
        q_logits0, q_logits=[], []
        for i in range(len(x_query)):
            x = x_query[i: i + 1]  # [1,3,80,80]
            cam, cls_id, logits0, mid_feat = self.get_CAM(x)  # No y input, 会输出5个class所对应的cam
            q_logits0.append(logits0.squeeze(0))

            cls_feat = []
            for j in range(len(cam)):
                feat_conv = self.down_mid(mid_feat).squeeze(0)
                feat = weighted_feat(feat_conv, cam[j], norm=self.norm, T=self.temp, thresh=self.thresh)
                cls_feat.append(feat)
            q_feat = torch.cat([feat.unsqueeze(dim=0) for feat in cls_feat], dim=0)  # [5, 512]对应5个class
            logits = utils.compute_logits_localize(q_feat, protos, metric='cos', temp=self.tp) # [5]
            q_logits.append(logits)

        logits0 = torch.cat([l.unsqueeze(0) for l in q_logits0], dim=0)  # [75, 5]
        logits  = torch.cat([l.unsqueeze(0) for l in q_logits ], dim=0)  # [75, 5]
        return logits0, logits

    def get_CAM(self, x, y=None):  # 只accept一张图片 x: [1,3,h,w] y:[1]
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
            feat2, feat3, feat4 = feat_blobs  # [1,128,20,20] #[1,256,10,10] #[1,512,5,5]
            for k, v in handles.items():
                handles[k].remove()

        # generate the class activation maps
        if y is not None:
            class_idx = y
        else:
            class_idx = torch.arange(self.n_classes)
            if torch.cuda.is_available():
                class_idx = class_idx.cuda()
        weight_softmax = self.classifier.state_dict()['linear.weight']  # [5, 512] 共512个channel
        bias_softmax = self.classifier.state_dict()['linear.bias']  # [5]对应5个class 在cpu上/如果从gpu读取的话在gpu上

        bz, nc, h, w = feat4.shape   # batch_size=1 [1, 512, 5, 5]
        cam_lst = []
        for idx in class_idx:
            cam = torch.matmul(weight_softmax[idx], feat4.reshape((nc, h * w)) ) # array[hw]
            cam = cam.reshape(h, w) + bias_softmax[idx]/(h*w)
            cam_lst.append(cam)

        if self.mid ==23:
            mid_feat = torch.cat( (F.interpolate(feat2,size=feat3.shape[-2:]), feat3), dim=1 )  #[1, ch, h, w]
        return cam_lst, class_idx, logits, mid_feat

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


