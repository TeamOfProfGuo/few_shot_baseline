# encoding:utf-8
import math
import cv2
import numpy as np
import torch
import torch.nn as nn

import models
import utils
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

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def inner_loop(self, x_shot, s_label, init_weight=False):
        z_support = self.encoder(x_shot)  # backbone生成的feature, [25,512],[75,512]

        # finetune linear classifier
        inner_optimizer = torch.optim.SGD(self.classifier.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                        weight_decay=0.001)
        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda() if torch.cuda.is_available() else loss_function

        batch_size = 4
        support_size = x_shot.shape[0]  # 25
        for epoch in range(100):        # finetune 100 个epoch
            rand_id = np.random.permutation(support_size)
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

    def get_CAM(self, x, y=None):  #只accept一张图片 x: [1,3,h,w] y:[1]
        if self.backbone == 'resnet12':
            finalconv_name = 'layer4'

        #hook the feature extractor
        feat_blobs = []
        def hook_feature(module, input, output):
            feat_blobs.append(output.data.cpu().numpy())
            print(output.shape)
        handle = self.encoder._modules.get(finalconv_name).register_forward_hook(hook_feature)

        self.eval()  # classifier的参数也固定
        with torch.no_grad():
            logits = self.forward(x)
            feat_conv = feat_blobs[0]
            handle.remove()

        # generate the class activation maps upsample to 80x80
        weight_softmax = self.classifier.state_dict()['linear.weight'].data.numpy() # [5, 512] 共512个channel
        bias_softmax = self.classifier.state_dict()['linear.bias'].data.numpy()
        class_idx = np.array(y) if y is not None else np.arange(self.n_classes)
        bz, nc, h, w = feat_conv.shape   # batch_size=1 [1, 512, 5, 5]
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feat_conv.reshape((nc, h * w)))  # array[hw]
            cam = cam.reshape(h, w) + bias_softmax[idx]/(h*w)
            output_cam.append(cam)
            # cam = cam - np.min(cam)
            # cam_img = cam / np.max(cam)
            # cam_img = np.uint8(255 * cam_img)
            # output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam, class_idx


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


