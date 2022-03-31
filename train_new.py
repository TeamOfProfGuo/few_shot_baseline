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
    svname = 'classifier_{}'.format(config['train_dataset'])
    svname += '_' + config['model_args']['encoder']
    clsfr = config['model_args']['classifier']
    if clsfr != 'linear-classifier':
        svname += '-' + clsfr
if args.tag is not None:
    svname += '_' + args.tag
save_path = os.path.join('./save', svname)
utils.ensure_path(save_path)
utils.set_log_path(save_path)
writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

# yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

####============================================= Dataset ==================================================####

# train
n_train_way = 5
n_query = 15
n_train_shot = 5
ep_per_batch = 1
config['model_args']['classifier_args'] = {'n_classes': n_train_way}

train_dataset = dataset.make(config['train_dataset'], **config['train_dataset_args'])  # 返回x:tensor[3,80,80],y:int
train_sampler = CategoriesSampler(
            train_dataset.label, config['train_batches'],
            n_train_way, n_train_shot + n_query,
            ep_per_batch=ep_per_batch)  # 生成每个batch idx: [320] = 4(ep)*5(way)*16(n_shot+n_query)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=8, pin_memory=True) #共200个batch

utils.log('train dataset: {} (x{}), {}'.format(
    train_dataset[0][0].shape, len(train_dataset), train_dataset.n_classes))


#### ======================================== Model and Optimizer  ========================================####

if config.get('load'):
    model_sv = torch.load(config['load'])
    model = models.load(model_sv)
else:
    model = models.make(config['model'], **config['model_args'])

if config.get('load_encoder'):
    encoder = models.load(torch.load(config['load_encoder'], map_location=lambda storage, location: storage)).encoder  # classifier模型with pretrained params
    model.encoder.load_state_dict(encoder.state_dict())

utils.log('num params: {}'.format(utils.compute_n_params(model)))
for name, param in model.named_parameters():
    if param.requires_grad:
        if 'encoder' in name:
            param.requires_grad = False

optimizer, lr_scheduler = utils.make_optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                                               config['optimizer'], **config['optimizer_args'])

# optimizer, lr_scheduler = utils.make_optimizer(model.parameters(), config['optimizer'], **config['optimizer_args'])

###==== set up

max_epoch = config['max_epoch']
save_epoch = config.get('save_epoch')
max_va = 0.
timer_used = utils.Timer()
timer_epoch = utils.Timer()

for epoch in range(1, max_epoch + 1 + 1):
    if epoch == max_epoch + 1:
        if not config.get('epoch_ex'):
            break
        train_dataset.transform = train_dataset.default_transform
        train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)

    timer_epoch.s()
    aves_keys = ['tl', 'ta', 'vl', 'va']  # train_loss, train_acc, val_loss, val_acc
    # if eval_fs:
    #     for n_shot in n_shots:
    #         aves_keys += ['fsa-' + str(n_shot)]
    aves = {k: utils.Averager() for k in aves_keys}

    ###==== train
    model.train()
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

    for i, (data, label) in enumerate(train_loader):  # data[400,3,80,80],_[400]
        if i>=2:
            break
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        x_shot, x_query = fs.split_shot_query(
            data, n_train_way, n_train_shot, n_query,
            ep_per_batch=ep_per_batch)  # x_shot:[4,5,5,3,80,80], x_query:[4,75,3,80,80]

def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()
model.classifier.apply(weight_reset)

x_shot, x_query = x_shot.squeeze(0), x_query.squeeze(0) # [5,5,3,80,80], way,shot x_query[75, 3, 80, 80]
x_shot = x_shot.view(n_train_way*n_train_shot, *x_shot.shape[-3:]) # [25,3,80,80]
s_label = fs.make_nk_label(n_train_way, n_train_shot, ep_per_batch=ep_per_batch)
q_label = fs.make_nk_label(n_train_way, n_query, ep_per_batch=ep_per_batch)  # label for query:[300]

init_weight, init_bias = model.state_dict()['classifier.linear.weight'], model.state_dict()['classifier.linear.bias']
#用support image训练classifier weight
model.inner_loop(x_shot, s_label, init_weight=False)
new_weight, new_bias = model.state_dict()['classifier.linear.weight'], model.state_dict()['classifier.linear.bias']

i = 2
x = x_shot[i:i+1]
y = s_label[i:i+1]
cam, cls_id = model.get_CAM(x, )

def normalize_cam(cam, size_upsample):
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, size_upsample)
    return cam_img

cam_img = normalize_cam(cam[1], (80, 80))

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


        logits = model(data)
        loss = F.cross_entropy(logits, label)
        acc = utils.compute_acc(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        aves['tl'].add(loss.item())
        aves['ta'].add(acc)

        logits = None;
        loss = None

    #### ===== eval
    if eval_val:
        model.eval()
        for data, label in tqdm(val_loader, desc='val', leave=False):
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            with torch.no_grad():
                logits = model(data)
                loss = F.cross_entropy(logits, label)
                acc = utils.compute_acc(logits, label)

            aves['vl'].add(loss.item())
            aves['va'].add(acc)

    if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
        fs_model.eval()
        for i, n_shot in enumerate(n_shots):
            np.random.seed(0)
            for data, _ in tqdm(fs_loaders[i], desc='fs-' + str(n_shot),
                                leave=False):  # data:[320,3,80,80],320=4(ep)*5*(1+15)
                # x_shot:[4,5(way),1(shot),3,80,90], x_query:[4,75(n_q*way),3,80,80]
                x_shot, x_query = fs.split_shot_query(data.cuda(), n_way, n_shot, n_query, ep_per_batch=4)
                label = fs.make_nk_label(n_way, n_query,
                                         ep_per_batch=4).cuda()  # label for query only (based on order)
                with torch.no_grad():
                    logits = fs_model(x_shot, x_query).view(-1, n_way)  # [300, 5]
                    acc = utils.compute_acc(logits, label)
                aves['fsa-' + str(n_shot)].add(acc)

    ###==== post each epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    for k, v in aves.items():
        aves[k] = v.item()

    t_epoch = utils.time_str(timer_epoch.t())
    t_used = utils.time_str(timer_used.t())
    t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)

    if epoch <= max_epoch:
        epoch_str = str(epoch)
    else:
        epoch_str = 'ex'
    log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(epoch_str, aves['tl'], aves['ta'])
    writer.add_scalars('loss', {'train': aves['tl']}, epoch)
    writer.add_scalars('acc', {'train': aves['ta']}, epoch)

    if eval_val:
        log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
        writer.add_scalars('loss', {'val': aves['vl']}, epoch)
        writer.add_scalars('acc', {'val': aves['va']}, epoch)

    if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
        log_str += ', fs'
        for n_shot in n_shots:
            key = 'fsa-' + str(n_shot)
            log_str += ' {}: {:.4f}'.format(n_shot, aves[key])
            writer.add_scalars('acc', {key: aves[key]}, epoch)

    if epoch <= max_epoch:
        log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
    else:
        log_str += ', {}'.format(t_epoch)
    utils.log(log_str)

    if config.get('_parallel'):
        model_ = model.module
    else:
        model_ = model

    training = {
        'epoch': epoch,
        'optimizer': config['optimizer'],
        'optimizer_args': config['optimizer_args'],
        'optimizer_sd': optimizer.state_dict(),
    }
    save_obj = {
        'file': __file__,
        'config': config,

        'model': config['model'],
        'model_args': config['model_args'],
        'model_sd': model_.state_dict(),

        'training': training,
    }
    if epoch <= max_epoch:
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj, os.path.join(
                save_path, 'epoch-{}.pth'.format(epoch)))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))
    else:
        torch.save(save_obj, os.path.join(save_path, 'epoch-ex.pth'))

    writer.flush()



