import torch
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import DataParallel

from dataset import DataManager
from nets import Model
from utils import logger
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
        self._build_model()

    def _build_dataloader(self):
        logger.info("Load datasets...")
        dataset = DataManager(mode='train')
        
        self.dataloader = DataLoader(
            dataset,
            batch_size = cfg.num_gpus * cfg.batch_size,
            num_workers = cfg.num_worker,
            shuffle = cfg.shuffle,
            drop_last = True,
            collate_fn = dataset.collate_fn
        )

    def _build_model(self):
        logger.info("Build model...")

        self.model = Model()
        self.model = DataParallel(self.model).cuda()
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
