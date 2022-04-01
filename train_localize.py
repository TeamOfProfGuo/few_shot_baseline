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


def main(config):
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
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)

    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    ####============================================= Dataset ==================================================####

    # meta train
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
                ep_per_batch=ep_per_batch)  # 生成每个batch idx: [100] = 1(ep)*5(way)*20(n_shot+n_query)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=8, pin_memory=True) #共200个batch

    utils.log('train dataset: {} (x{}), (classes {})'.format(
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

    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves_keys = ['ca', 'la']  # 'ca': base_classifier Acc, 'la': Localized Proto Acc
        aves = {k: utils.Averager() for k in aves_keys}

        ###==== train
        meta_args = {'n_way': 5, 'n_shot': 5}
        for i, (data, label) in enumerate(train_loader):  # data[400,3,80,80],_[400]
            model.train()  # 准备重新训练 model.classifier module
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()

            x_shot, x_query = fs.split_shot_query(data, n_train_way, n_train_shot, n_query,
                                                  ep_per_batch=ep_per_batch)  # x_shot:[4,5,5,3,80,80], x_query:[4,75,3,80,80]
            x_shot, x_query = x_shot.squeeze(0), x_query.squeeze(0)  # [5,5,3,80,80], way,shot x_query[75, 3, 80, 80]
            x_shot = x_shot.view(n_train_way * n_train_shot, *x_shot.shape[-3:])  # [25,3,80,80]
            y_shot = fs.make_nk_label(n_train_way, n_train_shot, ep_per_batch=ep_per_batch).cuda()
            y_query = fs.make_nk_label(n_train_way, n_query, ep_per_batch=ep_per_batch).cuda()  # label for query:[300]

            pred0, pred = model.outer_loop(x_shot, x_query, y_shot, y_query, meta_args)
            acc0 = (pred0 == y_query).float().sum() / len(y_query)
            acc = (pred == y_query).float().sum() / len(y_query)
            aves['ca'].add(acc0)
            aves['la'].add(acc)

            t_used = utils.time_str(timer_used.t())
            utils.log('epoch {}, episode {}, Classifier Acc {:.4f}, Localized Classifier Acc {:.4f}'
                      'AllTime {}'.format(epoch, i, acc0, acc, t_used))

        t_epoch = utils.time_str(timer_epoch.t())
        utils.log('=========finish epoch {}========== \n '
            'Overall Classifier Acc {:.4f}, Localized Classifier Acc {:.4f} EpochTime {}'.format(
            epoch, aves['ca'].v, aves['la'].v, t_epoch))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_localize_mini.yaml')
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
