# encoding:utf-8

import argparse
import os
import yaml
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms

import dataset
import models
import utils
import utils.few_shot as fs
from utils.few_shot import *
from dataset.samplers import CategoriesSampler


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/train_localize_mini.yaml')
parser.add_argument('--name', default=None)
parser.add_argument('--tag', default=None)
parser.add_argument('--gpu', default='0')
args = parser.parse_args([])

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu


svname = args.name
if svname is None:
    svname = 'localize_{}'.format(config['train_dataset'])
    svname += '_' + config['model_args']['encoder']
    clsfr = config['model_args']['classifier']
    if clsfr != 'linear-classifier':
        svname += '-' + clsfr
if args.tag is not None:
    svname += '_' + args.tag
save_path = os.path.join('./save', svname)
#utils.ensure_path(save_path)
#utils.set_log_path(save_path)
#writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
# yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

####============================================= Dataset ==================================================####

# train
n_way, n_shot = config['n_way'], config['n_shot']
n_query = config['n_query']

n_train_way = config['n_train_way'] if config.get('n_train_way') is not None else n_way
n_train_shot = config['n_train_shot'] if config.get('n_train_shot') is not None else n_shot
ep_per_batch = config['ep_per_batch'] if config.get('ep_per_batch') is not None else 1

config['model_args']['classifier_args'] = {'n_classes': n_train_way}

train_dataset = dataset.make(config['train_dataset'], **config['train_dataset_args'])  # 返回x:tensor[3,80,80],y:int
train_sampler = CategoriesSampler(
            train_dataset.label, config['train_batches'],
            n_train_way, n_train_shot + n_query,
            ep_per_batch=ep_per_batch)  # 生成每个batch idx: [320] = 4(ep)*5(way)*16(n_shot+n_query)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=8, pin_memory=True) #共200个batch

utils.log('train dataset: {} (x{}), {}'.format(train_dataset[0][0].shape, len(train_dataset), train_dataset.n_classes))


#### ======================================== Model and Optimizer  ========================================####

if config.get('load'):
    model_sv = torch.load(config['load'])
    model = models.load(model_sv)
else:
    model = models.make(config['model'], **config['model_args'])

if config.get('load_encoder'):
    pretrained_dict = torch.load(config['load_encoder'], map_location=lambda storage, location: storage)  # classifier模型with pretrained params
    pretrained_dict = {k:v for k, v in pretrained_dict['model_sd'].items() if 'encoder' in k}

    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

utils.log('num params: {}'.format(utils.compute_n_params(model)))
for name, param in model.named_parameters():
    if param.requires_grad:
        if 'encoder' in name:
            param.requires_grad = False


param_list = []
for name, param in model.named_parameters():
    if name in ['thresh', 'tp', 'temp']:
        param_list.append(param)

optimizer, lr_scheduler = utils.make_optimizer(param_list, config['optimizer'], **config['optimizer_args'])

# optimizer, lr_scheduler = utils.make_optimizer(model.parameters(), config['optimizer'], **config['optimizer_args'])

###==== set up

#save_epoch = config.get('save_epoch')

epoch = 1
aves_keys = ['ca', 'la']# train_loss, train_acc, val_loss, val_acc
aves = {k: utils.Averager() for k in aves_keys}  # 是否同时考虑1-shot与5—shot

###========================= train
meta_args = {'n_way': 5, 'n_shot': 5}

for i, (data, label) in enumerate(train_loader):  # data[400,3,80,80],_[400]
    if i>=1:
        break

### ========== 处理数据
if torch.cuda.is_available():
    data, label = data.cuda(), label.cuda()
x_shot, x_query = fs.split_shot_query(
    data, n_train_way, n_train_shot, n_query, ep_per_batch=ep_per_batch)  # x_shot:[4,5,5,3,80,80], x_query:[4,75,3,80,80]
x_shot, x_query = x_shot.squeeze(0), x_query.squeeze(0) # [5,5,3,80,80], way,shot x_query[75, 3, 80, 80]
x_shot = x_shot.view(n_train_way*n_train_shot, *x_shot.shape[-3:]) # [25,3,80,80]
y_shot = fs.make_nk_label(n_train_way, n_train_shot, ep_per_batch=ep_per_batch)
y_query = fs.make_nk_label(n_train_way, n_query, ep_per_batch=ep_per_batch)  # label for query:[300]

### =========== 训练模型
model.train()
logits0, logits = model.outer_loop(x_shot, x_query, y_shot, y_query, meta_args)  # [75, 5]
# print 现在的参数值
print('current thresh', model.state_dict()['thresh'])
print('current tp', model.state_dict()['tp'])
cls_weight=model.state_dict()['classifier.linear.weight']
# 更新参数
loss = F.cross_entropy(logits, y_query)
optimizer.zero_grad()
loss.backward()
optimizer.step()
#更新后的参数
print('current thresh', model.state_dict()['thresh'])
print('current tp', model.state_dict()['tp'])
new_weight=model.state_dict()['classifier.linear.weight']
#查看gradient
print(model.thresh.grad)
print(model.tp.grad)
print(model.classifier.linear.weight.grad)
print(cls_weight==new_weight)
#from torchviz import make_dot
#g = make_dot(logits, params=dict(model.named_parameters()))
#g.view()






### ======== 重新训练classifier weight

# pred0, pred = model.outer_loop(x_shot, x_query, y_shot, y_query, meta_args)
model.reset_cls_weight()
init_weight, init_bias = model.state_dict()['classifier.linear.weight'], model.state_dict()['classifier.linear.bias']
print(init_weight)
#用support image训练classifier weight
model.inner_loop(x_shot, y_shot, init_weight=False)
new_weight, new_bias = model.state_dict()['classifier.linear.weight'], model.state_dict()['classifier.linear.bias']
print(new_weight)


### ================================================== 计算prototype ==================================================
model.eval()

s_cam = []   # list of all cam
s_feat = []  # list of feat_conv
sw_feat = [] # list of weighted feat
s_logits = []

for i in range(len(x_shot)):
    x = x_shot[i:i+1]
    y = y_shot[i:i+1]
    cam, feat_conv, cls_id, logits = model.get_CAM(x, y)
    cam = cam[0]
    feat_conv = feat_conv.squeeze(dim=0)
    feat = weighted_feat(feat_conv, cam, T=model.temp, norm=model.norm, thresh=model.thresh)  # 当前图片的prototype/weighted feature  #dim [ 512]
    sw_feat.append(feat)
    s_cam.append(cam)
    s_feat.append(feat_conv)
    s_logits.append(logits)

sw_feat = torch.cat([feat.unsqueeze(dim=0) for feat in sw_feat], dim=0)   #[25, 512]
sw_feat = sw_feat.view(n_train_way, n_train_shot, -1) # [5, 5, 512]
protos = sw_feat.mean(dim=1)  # [5, 512]

### ================================================== 根据prototype分类 ================================================

pred0, pred=[], []
for i in range(len(x_query)):
    x = x_query[i: i+1]  # [1,3,80,80]
    y = y_query[i]
    cam, feat_conv, cls_id, logits = model.get_CAM(x)  # No y input, 会输出5个class所对应的cam
    pred0.append(torch.argmax(logits))

    cls_feat = []
    for j in range(len(cam)):
        feat = weighted_feat(feat_conv.squeeze(dim=0), cam[j], T=model.temp, norm=model.norm, thresh=model.thresh)
        cls_feat.append(feat)
    q_feat = torch.cat([feat.unsqueeze(dim=0) for feat in cls_feat], dim=0)   # [5, 512]对应5个class

    method = 'cos'
    protos = F.normalize(protos, dim=-1)  # [5, 512]
    q_feat = F.normalize(q_feat, dim=-1)  # [5, 512]
    sim =  torch.mm(q_feat, protos.t())
    sim = torch.diagonal(sim, offset=0)
    pred.append(torch.argmax(sim))
pred0, pred = torch.stack(pred0), torch.stack(pred)

acc0 = (pred0 == y_query).float().sum() / len(y_query)
acc = (pred == y_query).float().sum() / len(y_query)


### ====== 如何选取Threshold for CAM

i = 11
cam = s_cam[i]
x = x_shot[i]

### ===== 可视化
cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam)).data.numpy()
cam_img = np.uint8(255 * cam)
cam_img= cv2.resize(cam_img, (80,80))

invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                               transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                               ])
inv_x = invTrans(x)
import matplotlib.pyplot as plt
plt.imshow(inv_x.permute(1, 2, 0))
img = inv_x.permute(1, 2, 0).numpy()*255
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
heatmap = cv2.applyColorMap(cv2.resize(cam_img, (80, 80)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM.jpg', result)
cv2.imwrite('img.jpg',img)



# normalize between 0 and 1, then softmax
T = 0.5
cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam))
y = cam.unsqueeze(dim=0)/T
out = F.softmax(y.view(1, -1), dim=1).view(y.shape[1], -1).numpy()
cam_array = cam.data.numpy()

# nomalize by mean and std and then softmax
T = 2
cam1 = (cam - torch.mean(cam))/torch.std(cam)
y = cam1.unsqueeze(dim=0)/T
out1 = F.softmax(y.view(1, -1), dim=1).view(y.shape[1], -1).numpy()

# Keep the top 50th percentile, rescale between 0 and 1 then softmax

cam = s_cam[i]
cam1 = cam.flatten()
val, idx = torch.topk(cam1, 12)
topk = torch.zeros_like(cam1)
topk[idx] = val
cam = topk.view(*cam.shape)

T = 0.5
cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam))
y = cam.unsqueeze(dim=0)/T
out = F.softmax(y.view(1, -1), dim=1).view(y.shape[1], -1).numpy()
cam_array = cam.data.numpy()


#




z = []
for i_cam in cam:
    z.append(np.average(i_cam))
print(i, ':', z)
print(np.average((np.array(z))))


def normalize_cam1(cam):
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return cam_img
cam_norm = normalize_cam1(cam[0])

T = 0.5
y = torch.tensor(cam_norm).unsqueeze(dim=0)/T
out = F.softmax(y.view(1, -1), dim=1).view(y.shape[1], -1).numpy()




invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                               transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                               ])
inv_x = invTrans(x[0])
import matplotlib.pyplot as plt
plt.imshow(inv_x.permute(1, 2, 0))
img = inv_x.permute(1, 2, 0).numpy()*255
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
heatmap = cv2.applyColorMap(cv2.resize(cam_img, (80, 80)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM.jpg', result)
cv2.imwrite('img.jpg',img)





