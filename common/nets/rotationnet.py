import torch
from torch import nn
from torch.nn import functional as F

from config import cfg
from nets.layer import make_conv_layers, make_linear_layers
from utils import Transform

class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()
        self.joint_num = 25

        self.body_conv = make_conv_layers([2048, 512], kernel=1, stride=1, padding=0)
        self.root_pose_out = make_linear_layers([self.joint_num*515, 6], relu_final=False)
        self.body_pose_out = make_linear_layers([self.joint_num*515, 23*6], relu_final=False)
        self.shape_out = make_linear_layers([2048, 10], relu_final=False)
        self.cam_out = make_linear_layers([2048, 3], relu_final=False)

    def sample_image_feature_joint(self, img_feat, joint_xy, width, height):
        joint_num = joint_xy.shape[1]
        img_feat_joints = []
        for j in range(joint_num):
            img_feat_joints.append(self.sample_image_feature(img_feat, joint_xy[:,j,:], width, height))
        img_feat_joints = torch.stack(img_feat_joints,1)
        return img_feat_joints

    def sample_image_feature(self, img_feat, xy, width, height):
        x = xy[:,0] / width * 2 - 1
        y = xy[:,1] / height * 2 - 1
        grid = torch.stack((x,y),1)[:,None,None,:]
        img_feat = F.grid_sample(img_feat, grid, align_corners=True)[:,:,0,0] # (batch_size, channel_dim)
        return img_feat

    def forward(self, img_feat, joint_img):
        batch_size = img_feat.shape[0]

        # shape parameter
        shape_param = self.shape_out(img_feat.mean((2,3)))

        # camera parameter
        cam_param = self.cam_out(img_feat.mean((2,3)))

        # body pose parameter
        body_img_feat = self.body_conv(img_feat)
        body_img_feat = self.sample_image_feature_joint(body_img_feat, joint_img[:,:,:2], cfg.output_hm_shape[2]-1, cfg.output_hm_shape[1]-1)
        body_feat = torch.cat((body_img_feat, joint_img), 2) # batch_size, joint_num (body), 512+3

        root_pose = self.root_pose_out(body_feat.view(batch_size, -1))
        body_pose = self.body_pose_out(body_feat.view(batch_size, -1))

        root_pose = Transform.rot6d_to_axis_angle(root_pose)
        body_pose = Transform.rot6d_to_axis_angle(body_pose.reshape(-1,6)).reshape(body_pose.shape[0],-1) # (N, J_R*3)
        cam_trans = Transform.get_camera_trans(cam_param)

        print("---")
        print(root_pose.shape)
        print(body_pose.shape)
        return root_pose, body_pose, shape_param, cam_trans

        

    
    