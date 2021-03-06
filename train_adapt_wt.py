# encoding:utf-8

import argparse
import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms

import dataset
import models
import utils
import utils.few_shot as fs
from dataset.samplers import CategoriesSampler

from train_mean import extract_mean


def main(config):
    svname = args.name
    if svname is None:
        svname = 'adapt_{}'.format(config['train_dataset'])
        svname += '_' + config['model_args']['encoder']
        clsfr = config['model_args']['classifier']
        if clsfr != 'linear-classifier':
            svname += '-' + clsfr
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)

    utils.log('svname {}, save_path {}'.format(svname, save_path))

    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    ####============================================= Dataset ==================================================####

    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']

    n_train_way = config['n_train_way'] if config.get('n_train_way') is not None else n_way
    n_train_shot = config['n_train_shot'] if config.get('n_train_shot') is not None else n_shot
    ep_per_batch = config['ep_per_batch'] if config.get('ep_per_batch') is not None else 1

    config['model_args']['classifier_args'] = {'n_classes': n_train_way}

    # meta train
    train_dataset = dataset.make(config['train_dataset'], **config['train_dataset_args'])  # 返回x:tensor[3,80,80],y:int
    utils.log('train dataset: {} (x{}), (classes {})'.format(
        train_dataset[0][0].shape, len(train_dataset), train_dataset.n_classes))

    train_sampler = CategoriesSampler(
                train_dataset.label, config['train_batches'],
                n_train_way, n_train_shot + n_query,
                ep_per_batch=ep_per_batch)  # 生成每个batch idx: [100] = 1(ep)*5(way)*20(n_shot+n_query)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=8, pin_memory=True) #共200个batch

    # val
    val_dataset = dataset.make(config['val_dataset'], **config['val_dataset_args'])
    utils.log('val dataset: {} (x{}), {} classes'.format(
        val_dataset[0][0].shape, len(val_dataset), val_dataset.n_classes))

    if config.get('visualize_datasets'):
        utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    val_sampler = CategoriesSampler(
        val_dataset.label, 100,
        n_way, n_shot + n_query,
        ep_per_batch=ep_per_batch)  # 生成每个batch idx: [320] = 4*5*(1+15)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)  # 共200个batch

    # tval
    if config.get('tval_dataset'):
        tval_dataset = dataset.make(config['tval_dataset'], **config['tval_dataset_args'])
        utils.log('tval dataset: {} (x{}), {} classes'.format(
            tval_dataset[0][0].shape, len(tval_dataset), tval_dataset.n_classes))

        if config.get('visualize_datasets'):
            utils.visualize_dataset(tval_dataset, 'tval_dataset', writer)
        tval_sampler = CategoriesSampler(
            tval_dataset.label, 100,
            n_way, n_shot + n_query,
            ep_per_batch=ep_per_batch)  # 生成每个batch idx: [320] = 4*5*(1+15)
        tval_loader = DataLoader(tval_dataset, batch_sampler=tval_sampler, num_workers=8, pin_memory=True)
    else:
        tval_loader = None

    #### ======================================== Model and Optimizer  ========================================####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

    # get mean feature from base_train dataset
    mean_list = extract_mean(config['train_dataset'], config['model_args']['encoder'], config.get('load_encoder'))
    model.get_base_mean(mean_list)

    if config.get('load_encoder'):
        pretrained_dict = torch.load(config['load_encoder'])  # classifier模型with pretrained params
        pretrained_dict = {k: v for k, v in pretrained_dict['model_sd'].items() if 'encoder' in k}

        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'encoder' in name:
                param.requires_grad = False

    param_list = []  # 在meta train过程中只更新 meta参数（thresh, temp, tp)
    for name, param in model.named_parameters():
        if name in ['thresh', 'tp', 'temp', 'down_mid.weight'] or 'down_mid' in name:
            param_list.append(param)
    optimizer, lr_scheduler = utils.make_optimizer(param_list, config['optimizer'], **config['optimizer_args'])

    ###==== set up

    max_epoch = config['max_epoch']
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves_keys = ['tl', 'ta0', 'ta', 'ta1', 'ta2', 'tap',
                     'vl', 'va0', 'va', 'va1', 'va2', 'vap',
                     'tvl','tva0','tva', 'tva1', 'tva2', 'tvap']  # 'a0': base_classifier Acc, 'a': Localized Proto Acc, 'l': loss
        aves = {k: utils.Averager() for k in aves_keys}

        ###==== train
        meta_args = {'n_way': 5, 'n_shot': 5}
        for i, (data, label) in enumerate(train_loader):  # data[400,3,80,80],_[400]
            # model.train()  # 准备重新训练 model.classifier module 已经写在 outer_loop中
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()

            # x_shot:[4,5,5,3,80,80], x_query:[4,75,3,80,80]
            x_shot, x_query = fs.split_shot_query(data, n_train_way, n_train_shot, n_query, ep_per_batch=ep_per_batch)
            y_shot = fs.make_nk_label(n_train_way, n_train_shot, ep_per_batch=ep_per_batch).cuda()
            y_query = fs.make_nk_label(n_train_way, n_query, ep_per_batch=ep_per_batch).cuda()  # label for query:[300]

            logits0, logits, sub1, sub2 = model.outer_loop(x_shot, x_query, y_shot, meta_args)  # [75, 5], sub1 low level
            loss = F.cross_entropy(logits, y_query)
            acc0 = utils.compute_acc(logits0, y_query)
            acc = utils.compute_acc(logits, y_query)
            acc1 = utils.compute_acc(sub1, y_query)
            acc2 = utils.compute_acc(sub2, y_query)

            p1 = F.softmax(sub1 * 2.0, dim=-1)      # temperature = 2.0 可以调节
            p2 = F.softmax(sub2 * 2.0, dim=-1)
            p = (p1 + p2)/2
            accp = utils.compute_acc(p, y_query)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta0'].add(acc0)
            aves['ta'].add(acc)
            aves['ta1'].add(acc1)
            aves['ta2'].add(acc2)
            aves['tap'].add(accp)

            if i%10==0 or (epoch==1 and i%2==0):
                t_used = utils.time_str(timer_used.t())
                log_msg = 'epoch {}, episode {}, Acc0 {:.4f}| Acc {:.4f}| Acc1 {:.4f}| Acc2 {:.4f}| AccP {:.4f}| ' \
                          'QLoss {:.4f} AllTime {} thresh {:.4f} tp {:.4f}'.format(
                    epoch, i, acc0, acc, acc1, acc2, accp,
                    loss.item(), t_used, model.thresh, model.tp)
                if isinstance(model.down_mid, nn.Parameter):
                     log_msg += ', wt_low {:.4f}'.format(model.down_mid)
                utils.log(log_msg)

        t_epoch = utils.time_str(timer_epoch.t())
        utils.log('=========finish epoch {}========== \n '
                  'Epoch Acc0 {:.4f}| Acc {:.4f}| Acc1 {:.4f}| Acc2 {:.4f}| AccP {:.4f}| '
                  'Query Loss {:.4f}, EpochTime {}, thresh {:.4f}, tp {:.4f}'.format(
            epoch, aves['ta0'].v, aves['ta'].v, aves['ta1'].v, aves['ta2'].v, aves['tap'].v,
            aves['tl'].v, t_epoch, model.thresh, model.tp ))

        # ========= eval
        model.eval()

        for name, loader, name_l, name_a in [ ('tval', tval_loader, 'tvl', 'tva'), ('val', val_loader, 'vl', 'va') ]:

            if (config.get('tval_dataset') is None) and name == 'tval':
                continue

            np.random.seed(0)
            for i, (data, label) in enumerate(loader):  # data:[320, 3, 80, 80]

                x_shot, x_query = fs.split_shot_query(data.cuda(), n_way, n_shot, n_query, ep_per_batch=ep_per_batch)
                y_shot = fs.make_nk_label(n_way, n_shot, ep_per_batch=ep_per_batch).cuda()
                y_query = fs.make_nk_label(n_way, n_query, ep_per_batch=ep_per_batch).cuda()  # label for query:[300]

                logits0, logits, sub1, sub2 = model.outer_loop(x_shot, x_query, y_shot, meta_args)  # [75, 5]
                loss = F.cross_entropy(logits, y_query)
                acc0 = utils.compute_acc(logits0, y_query)
                acc = utils.compute_acc(logits, y_query)
                acc1 = utils.compute_acc(sub1, y_query)
                acc2 = utils.compute_acc(sub2, y_query)

                p1 = F.softmax(sub1 * 2.0, dim=-1)  # temperature = 2.0 可以调节
                p2 = F.softmax(sub2 * 2.0, dim=-1)
                p = (p1 + p2) / 2
                accp = utils.compute_acc(p, y_query)

                aves[name_l].add(loss.item())   # loss
                aves[name_a].add(acc)           # localized cls acc
                aves[name_a+'0'].add(acc0)      # base cls acc
                aves[name_a+'1'].add(acc1)
                aves[name_a+'2'].add(acc2)
                aves[name_a+'p'].add(accp)
        utils.log('=========finish epoch {}========== \n '
                  'Val Acc0 {:.4f} |Acc {:.4f} |Acc1 {:.4f}| Acc2 {:.4f}| AccP {:.4f}| loss {:.4f} || '
                  'TVal Acc0 {:.4f} |Acc {:.4f} |Acc1 {:.4f}| Acc2 {:.4f}| AccP {:.4f}| loss {:.4f}'.format(
            epoch, aves['va0'].v, aves['va'].v, aves['va1'].v, aves['va2'].v, aves['vap'].v, aves['vl'].v,
                   aves['tva0'].v, aves['tva'].v,  aves['tva1'].v,  aves['tva2'].v,  aves['tvap'].v, aves['tvl'].v
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_adapt_mini.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)
