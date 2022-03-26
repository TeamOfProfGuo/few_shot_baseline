# encoding:utf-8
import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, ep_per_batch=1):
        """
            生成一个batch generator, (batch中每个iter可以是不同的任务）
            batch中每一个iteration, 随机选取n_way个class, 每个cls选取n_per(n_s+n_q)个数据的idx,最终返回整个batch中所有数据的idx
            label: list of labels for all testing data
            n_batch:
            n_cls:  n_way
            n_per:  n_shot+n_query
            ep_per_batch: episode per batch/相当于batch_size
        """
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.ep_per_batch = ep_per_batch

        label = np.array(label)
        self.catlocs = []
        for c in range(max(label) + 1):
            self.catlocs.append(np.argwhere(label == c).reshape(-1))  # list of list, each sublist对应cls为idx的image idx

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                classes = np.random.choice(len(self.catlocs), self.n_cls, replace=False)  # 随机选n_way个class
                for c in classes:
                    l = np.random.choice(self.catlocs[c], self.n_per, replace=False) # query+support img
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)   # [5, 16] 5=n_way, 16=n_s+n_q
                batch.append(episode)
            batch = torch.stack(batch) # [200, 5, 16] bs * n_cls * n_per
            yield batch.view(-1) # [bs*n_cls*n_per]

