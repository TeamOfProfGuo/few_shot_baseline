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


@register('classifier')
class Classifier(nn.Module):
    
    def __init__(self, encoder, encoder_args, classifier, classifier_args):
        super().__init__()
        self.backbone = encoder # resnet12
        self.n_classes = classifier_args['n_classes']
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)

    def forward(self, x):  # 用于pretraining
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def inner_loop(self, x_shot, s_label, aug=False, init_weight=False):
        z_support = self.encoder(x_shot)  # backbone生成的feature, [25,512],[75,512]

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
            z_support = self.encoder(x_shot)

            rand_id = np.random.permutation(support_size)
            epoch_loss, n_iter = 0.0, 0
            for i in range(0, support_size, batch_size):
                inner_optimizer.zero_grad()
                # 选择当前batch
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)])
                selected_id = selected_id.cuda() if torch.cuda.is_available() else selected_id
                z_batch = z_support[selected_id]
                y_batch = s_label[selected_id]
                scores = self.classifier(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                inner_optimizer.step()

                epoch_loss += loss
                n_iter += 1
            if epoch%50==1:
                print('the loss after epoch {} is {}'.format(epoch, epoch_loss/n_iter))

    def outer_loop(self, x_shot, x_query, y_shot, y_query, meta_args):

        # ================== 训练classifier =====================
        self.reset_cls_weight()
        self.inner_loop(x_shot, y_shot, init_weight=False)  # 更新的weight在get_CAM中使用

        ### ==================== 计算prototype ==================
        self.eval()

        sw_feat = []
        for i in range(len(x_shot)):
            x = x_shot[i:i + 1]
            y = y_shot[i:i + 1]
            cam, feat_conv, cls_id, _ = self.get_CAM(x, y)
            cam = cam[0]
            feat_conv = feat_conv.squeeze(dim=0)
            feat = weighted_feat(feat_conv, cam)  # 当前图片的prototype/weighted feature  #dim [ 512]
            sw_feat.append(feat)

        sw_feat = torch.cat([feat.unsqueeze(dim=0) for feat in sw_feat], dim=0)  # [25, 512]
        sw_feat = sw_feat.view(meta_args['n_way'], meta_args['n_shot'], -1)
        protos = sw_feat.mean(dim=1)  # [5, 512]

        ### =============== 根据prototype对query分类 ==================

        pred0, pred=[], []
        for i in range(len(x_query)):
            x = x_query[i: i + 1]  # [1,3,80,80]
            y = y_query[i]
            cam, feat_conv, cls_id, logits = self.get_CAM(x)  # No y input, 会输出5个class所对应的cam
            pred0.append(torch.argmax(logits))

            cls_feat = []
            for j in range(len(cam)):
                feat = weighted_feat(feat_conv.squeeze(dim=0), cam[j])
                cls_feat.append(feat)
            q_feat = torch.cat([feat.unsqueeze(dim=0) for feat in cls_feat], dim=0)  # [5, 512]对应5个class

            method = 'cos'
            protos = F.normalize(protos, dim=-1)  # [5, 512]
            q_feat = F.normalize(q_feat, dim=-1)  # [5, 512]
            sim = torch.mm(q_feat, protos.t())
            sim = torch.diagonal(sim, offset=0)
            pred.append(torch.argmax(sim))
        return torch.stack(pred0), torch.stack(pred)


    def get_CAM(self, x, y=None):  #只accept一张图片 x: [1,3,h,w] y:[1]
        if self.backbone == 'resnet12':
            finalconv_name = 'layer4'

        #hook the feature extractor
        feat_blobs = []
        def hook_feature(module, input, output):
            feat_blobs.append(output)         # feat_blobs 在cuda上 data.cpu().numpy()
        handle = self.encoder._modules.get(finalconv_name).register_forward_hook(hook_feature)

        with torch.no_grad():
            logits = self.forward(x)
            feat_conv = feat_blobs[0]
            handle.remove()

        # generate the class activation maps
        if y is not None:
            class_idx = y
        else:
            class_idx = torch.arange(self.n_classes)
            if torch.cuda.is_available():
                class_idx = class_idx.cuda()
        weight_softmax = self.classifier.state_dict()['linear.weight']  # [5, 512] 共512个channel
        bias_softmax = self.classifier.state_dict()['linear.bias']  # [5]对应5个class 在cpu上/如果从gpu读取的话在gpu上

        bz, nc, h, w = feat_conv.shape   # batch_size=1 [1, 512, 5, 5]
        cam_lst = []
        for idx in class_idx:
            cam = torch.matmul(weight_softmax[idx], feat_conv.reshape((nc, h * w)) ) # array[hw]
            cam = cam.reshape(h, w) + bias_softmax[idx]/(h*w)
            cam_lst.append(cam)
        return cam_lst, feat_conv, class_idx, logits

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


