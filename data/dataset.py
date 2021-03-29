import random
import numpy as np
import torch

from config import cfg

# dataset import
for dataset in (cfg.trainset + cfg.testset):
    exec('from %s import %s'%(dataset, dataset))

class DataManager:
    def __init__(self, mode='train'):
        if mode == 'train':
            db_list = cfg.trainset
        elif mode == 'test':
            db_list = cfg.testset
        else:
            assert 0, "Invalid dataset mode"

        self.dbs = []
        for dataset in db_list:
            self.dbs.append(eval(dataset)(mode))

        self.db_num = len(self.dbs)
        self.db_len_cumsum = np.cumsum([len(db) for db in self.dbs])

    def __len__(self):
        return self.db_len_cumsum[-1]

    def __getitem__(self, index):
        for i in range(self.db_num):
            if index < self.db_len_cumsum[i]:
                db_idx = i
                break
                
        if db_idx == 0:
            data_idx = index
        else:
            data_idx = index - self.db_len_cumsum[db_idx-1]
            
        return self.dbs[db_idx][data_idx]

    def collate_fn(self, batch):
        inputs = [x[0] for x in batch]
        targets = [x[1] for x in batch]
        return torch.stack(inputs, 0), targets