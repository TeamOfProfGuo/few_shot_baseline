# encoding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register


@register('meta-baseline')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos', temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]   # [4, 5, 1] [ep, way, shot]
        query_shape = x_query.shape[:-3] # [4, 75]   [ep, way*n_q]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)    # [20, 3, 80, 80]
        x_query = x_query.view(-1, *img_shape)  # [300, 3, 80, 80]
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))   # [320, 512]
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]  #[20, 512], [300, 512]
        x_shot = x_shot.view(*shot_shape, -1)    # [4, 5, 1, 512]
        x_query = x_query.view(*query_shape, -1) # [4, 75, 512]

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)    # [4, 5, 512]   已经在shot维度取平均
            x_query = F.normalize(x_query, dim=-1)  # [4, 75, 512]
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits(x_query, x_shot, metric=metric, temp=self.temp)
        return logits

