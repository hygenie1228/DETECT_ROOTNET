from torch import nn

from config import cfg
from nets.layer import make_conv_layers
from utils import Transform

class PositionNet(nn.Module):
    def __init__(self):
        super(PositionNet, self).__init__()
        self.joint_num = 25                 # ???? 어떻게 정함? smplx 기준으로 하는 이유?
        self.hm_shape = cfg.output_hm_shape
        self.conv = make_conv_layers([2048, self.joint_num*self.hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)
        # grouped convolution 쓰면?

    def forward(self, img_feat):
        depth, height, width = self.hm_shape

        heatmap = self.conv(img_feat)
        heatmap = heatmap.view(-1, self.joint_num, depth, height, width)
        joint_img = Transform.soft_argmax_3d(heatmap)

        return joint_img