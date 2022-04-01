import torch
import torch.nn.functional as F


def split_shot_query(data, way, shot, query, ep_per_batch=1):
    img_shape = data.shape[1:]
    data = data.view(ep_per_batch, way, shot + query, *img_shape)
    x_shot, x_query = data.split([shot, query], dim=2)
    x_shot = x_shot.contiguous()
    x_query = x_query.contiguous().view(ep_per_batch, way * query, *img_shape)
    return x_shot, x_query


def make_nk_label(n, k, ep_per_batch=1):
    label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
    label = label.repeat(ep_per_batch)
    return label


def weighted_feat(feat, cam, T=0.5, method='percentile'):  # feat[512,5,5] cam [5, 5]
    # normalize cam between [0, 1]
    cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam))
    weight = F.softmax(torch.flatten(cam) / T, dim=0)  # .view(cam.shape)  # [hw]
    ch = feat.shape[0]
    out = torch.matmul(feat.view(ch, -1), weight)
    return out