import math
import numpy as np
import torch
from torch.nn import functional as F
import torchgeometry as tgm

from config import cfg

class Transform:
    def world2cam(world_coord, R, t):
        cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
        return cam_coord

    def cam2pixel(cam_coord, f, c):
        x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
        y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
        z = cam_coord[:,2]
        return np.stack((x,y,z),1)

    def soft_argmax_3d(heatmap3d):
        batch_size = heatmap3d.shape[0]
        depth, height, width = heatmap3d.shape[2:]
        heatmap3d = heatmap3d.reshape((batch_size, -1, depth*height*width))
        heatmap3d = F.softmax(heatmap3d, 2)
        heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))

        accu_x = heatmap3d.sum(dim=(2,3))
        accu_y = heatmap3d.sum(dim=(2,4))
        accu_z = heatmap3d.sum(dim=(3,4))

        accu_x = accu_x * torch.arange(width).float().cuda()[None,None,:]
        accu_y = accu_y * torch.arange(height).float().cuda()[None,None,:]
        accu_z = accu_z * torch.arange(depth).float().cuda()[None,None,:]

        accu_x = accu_x.sum(dim=2, keepdim=True)
        accu_y = accu_y.sum(dim=2, keepdim=True)
        accu_z = accu_z.sum(dim=2, keepdim=True)

        coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
        return coord_out

    def rot6d_to_axis_angle(x):
        batch_size = x.shape[0]

        x = x.view(-1,3,2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2)
        rot_mat = torch.stack((b1, b2, b3), dim=-1) # 3x3 rotation matrix
        
        rot_mat = torch.cat([rot_mat,torch.zeros((batch_size,3,1)).cuda().float()],2) # 3x4 rotation matrix
        axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1,3) # axis-angle
        axis_angle[torch.isnan(axis_angle)] = 0.0
        return axis_angle

    def get_camera_trans(cam_param):
        # camera translation
        t_xy = cam_param[:,:2]
        gamma = torch.sigmoid(cam_param[:,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0]*cfg.focal[1]*cfg.camera_3d_size*cfg.camera_3d_size/(cfg.input_body_shape[0]*cfg.input_body_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:,None]),1)
        return cam_trans