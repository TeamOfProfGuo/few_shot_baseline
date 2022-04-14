

# encoding:utf-8

# 这个file里面我只关心model的encoder所产出的feature的均值， 只与backbone的类型相关，所用的数据相关

import os
import pickle
import numpy as np
from torch.utils.data import DataLoader

import dataset
import models
from utils.few_shot import *


# =========== 计算pretrain过程中所有feature vector的mean
def extract_feature(train_data, encoder, encoder_path):
    # return out_mean, mid_mean

    fpath = encoder_path.replace('pth', 'plk')  # 存储base train过程中所有feature的mean/avg
    if os.path.isfile(fpath):
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        return data

    # load train_data
    train_dataset = dataset.make(train_data, **{'split':'train'})
    base_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # load model
    model = models.make(encoder, **{'aux':True})      # 返回中间层 ：f4,f3
    pretrained_dict = torch.load(encoder_path)  # map_location=lambda storage, location: storage)  # classifier模型with pretrained params
    pretrained_dict = {k.replace('encoder.',''): v for k, v in pretrained_dict['model_sd'].items() if 'encoder' in k}
    model.load_state_dict(pretrained_dict)

    # get training mean
    model.eval()
    out_mean, mid_mean= [], []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(base_loader):
            out, mid = model(inputs)
            out_mean.append(out.cpu().data.numpy())
            mid_mean.append(mid.cpu().data.numpy())

    out_mean = np.concatenate(out_mean, axis=0).mean(0)
    mid_mean = np.concatenate(mid_mean, axis=0).mean(0)
    all_info = [out_mean, mid_mean]
    with open(fpath, 'wb') as f:
        pickle.dump(all_info, f)
    return all_info

if __name__ == '__main__':
    out_mean, mid_mean = extract_feature('mini-imagenet', 'resnet12', './save/classifier_mini-imagenet_resnet12/epoch-last.pth')
    print('out mean shape'.format(out_mean.shape))