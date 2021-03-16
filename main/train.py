import argparse
import torch
import torch.backends.cudnn as cudnn

from config import cfg
from base import Trainer
from utils import logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg.set_args(args.gpu)
    cudnn.benchmark = True
    logger.info("Using GPU: %s"%args.gpu)

    # set trainer
    trainer = Trainer()
    trainer.set_trainer()

    # train model
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        for i, (inputs, targets) in enumerate(trainer.dataloader):   
            print(inputs.shape)

if __name__ == "__main__":
    main()