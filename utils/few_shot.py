# encoding:utf-8
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


def weighted_feat(feat, cam, T=0.5, norm='scale', thresh=None, method='percentile'):  # feat[512,5,5] cam [5, 5]
    if cam.dim() == 2:
        if not cam.shape[-2:] == feat.shape[-2:]:
            cam = F.interpolate(cam.reshape(1, 1, *cam.shape), size=feat.shape[-2:], mode='bilinear', align_corners=True)
            cam = cam.squeeze()
        if norm == 'norm': # normalize cam between [0, 1]
            cam = (cam - torch.mean(cam)) / torch.std(cam)
        else:
            cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam))
        if thresh is not None:
            cam = torch.relu(cam-thresh)

        weight = F.softmax(torch.flatten(cam) / T, dim=0)  # .view(cam.shape)  # [hw]
        ch = feat.shape[0]
        out = torch.matmul(feat.view(ch, -1), weight)
        return out

    elif cam.dim() == 3:  # cam: [N, h, w], feat: [25, 256, 10, 10]
        Nm = cam.shape[0]
        N, ch, h, w= feat.shape
        if not cam.shape[-2:] == feat.shape[-2:]:
            cam = F.interpolate(cam.unsqueeze(1), size=feat.shape[-2:], mode='bilinear', align_corners=True).squeeze(1)#[25,10,10]
        if norm == 'norm':
            cam = ( cam - torch.mean(cam,dim=[1,2]).view(Nm, 1, 1) )/torch.std(cam, dim=[1,2]).view(Nm, 1, 1)
        else:
            cam_min = torch.min(cam.view(Nm,-1), dim=-1)[0].view(Nm, 1, 1)
            cam_max = torch.max(cam.view(Nm,-1), dim=-1)[0].view(Nm, 1, 1)
            cam = ( cam - cam_min ) / (cam_max-cam_min)      # [25, 10, 10]
        if thresh is not None:
            cam = torch.relu( cam - thresh*torch.ones_like(cam) )

        weight = F.softmax( cam.view(Nm,-1)/T, dim=-1 ).unsqueeze(-1) # [25, 100, 1]   / query: [375,100,1]
        if N<Nm: # only needed for query img, support img的Nm=N
            feat = feat.unsqueeze(1).expand(N, int(Nm/N), ch, h, w) # query [75,5way,256,10,10]
            feat = feat.contiguous().view(Nm,ch,h,w)  # query [375,256,10,10]
        out = torch.bmm( feat.view(Nm, ch, -1), weight ).squeeze(-1)     # [25, 256, 100] * [25,100,1]-> [25,256,1]
        return out  # 返回每个img的weighted feature, [25,256] / query[375,256]
