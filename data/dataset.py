import random
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

    def __len__(self):
        return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        return self.dbs[0][index]
