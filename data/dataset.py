import random
from config import cfg

from utils import logger

# Datasets
from Human36M import Human36M

class DataManager:
    def __init__(self, mode='train'):
        if mode == 'train':
            db_list = cfg.trainset
        elif mode == 'test':
            db_list = cfg.testset
        else:
            assert 0, "Invalid dataset mode"

        # load datasets
        logger.info("Load datasets...")
        self.dbs = []
        for i in range(len(db_list)):
            self.dbs.append(eval(db_list[i])(mode))
        self.db_num = len(self.dbs)

    def __len__(self):
        return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        return self.dbs[0][index]
