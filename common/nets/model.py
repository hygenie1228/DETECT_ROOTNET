import torch
from torch import nn

from config import cfg
from nets.resnet import resnet50

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Networks
        self.backbone = resnet50(pretrained=True)

        

