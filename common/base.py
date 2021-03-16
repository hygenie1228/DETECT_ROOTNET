import torch
from torch.utils.data import DataLoader
import json

from dataset import DataManager
from config import cfg

class Trainer:
    def __init__(self):
        self.dataloader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.start_epoch = 0

    def set_trainer(self):
        self._build_dataloader()

    def _build_dataloader(self):
        dataset = DataManager(mode='train')
        
        self.dataloader = DataLoader(
            dataset,
            batch_size = cfg.batch_size,
            num_workers = cfg.num_worker,
            shuffle = cfg.shuffle
        )

    def _build_model(self):
        self.model = MaskRCNN()
        self.model.cuda()
        self.model.train()
        print(self.model)       

    def _set_optimizer(self):
        params = []
        for key, value in dict(self.model.named_parameters()).items():
            if value.requires_grad:
                params += [{'params': [value],'lr': self.lr, 'weight_decay': cfg.weight_decay}]

        self.optimizer = torch.optim.SGD(params, momentum=cfg.momentum)

    def _set_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2], gamma=0.1)

    def save_model(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, cfg.save_model_path)


    def load_model(self):
        checkpoint = torch.load(cfg.load_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1

        return epoch
