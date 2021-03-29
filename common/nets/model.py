import torch
from torch import nn
from torch.nn import functional as F
import copy

from config import cfg
from nets.resnet import resnet50
from nets.positionnet import PositionNet
from nets.rotationnet import RotationNet
from structure import smpl
from utils import Vis

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Networks
        self.backbone = resnet50(pretrained=True)
        self.body_position_net = PositionNet()
        self.body_rotation_net = RotationNet()

        # SMPL layer
        self.smpl_layer = copy.deepcopy(smpl.layer['neutral']).cuda()

    def forward(self, inputs, targets):
        body_img = F.interpolate(inputs, cfg.input_body_shape)
        img_feat = self.backbone(body_img)
        
        body_joint_img = self.body_position_net(img_feat)
        root_pose, body_pose, shape, cam_trans = self.body_rotation_net(img_feat, body_joint_img.detach())

        # final output
        joint_img, joint_cam, mesh_cam = self.get_coord(root_pose, body_pose, shape, cam_trans)

    def get_coord(self, root_pose, body_pose, shape, cam_trans):
        batch_size = root_pose.shape[0]
        
        output = self.smpl_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, transl=cam_trans)
        print(output)
        # TODO

        mesh_cam = output.vertices[0].detach().cpu().numpy()
        Vis.save_obj(mesh_cam, smpl.layer['neutral'].faces, './outputs/output.obj')
        
        return joint_proj, joint_cam, mesh_cam
