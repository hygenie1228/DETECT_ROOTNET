import numpy as np
import torch
import cv2
import random

from config import cfg
from structure import smpl
from .transform import Transform
from .vis import Vis

class Img:
    def load_img(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        return img.astype(np.float32)

    def rotate_2d(pt_2d, rot_rad):
        x, y = pt_2d
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        xx = x * cs - y * sn
        yy = x * sn + y * cs
        return np.array([xx, yy])

    def get_augmentation_cfg(mode):
        if mode == 'train':
            scale_factor = 0.25
            rot_factor = 30
            color_factor = 0.2

            scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
            rot = np.clip(np.random.randn(), -2.0,
                        2.0) * rot_factor if random.random() <= 0.6 else 0
            c_up = 1.0 + color_factor
            c_low = 1.0 - color_factor
            color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)], dtype=np.float32)
            do_flip = random.random() <= 0.5
        else:
            scale = 1.0
            rot = 0.0
            color_scale = np.array([1, 1, 1])
            do_flip = False

        return scale, rot, color_scale, do_flip

    def get_trans(bbox, output_shape, scale, rot):
        # src
        src_w = bbox[2] * scale
        src_h = bbox[3] * scale
        src_center = bbox[0:2]
        # dst
        dst_w = output_shape[1]
        dst_h = output_shape[0]
        dst_center = np.array([dst_w, dst_h])/2
        
        rot_rad = np.pi * rot / 180
        src_downdir = Img.rotate_2d(np.array([0, src_h/2]), rot_rad)
        src_rightdir = Img.rotate_2d(np.array([src_w/2, 0]), rot_rad)
        
        dst_downdir = np.array([0, dst_h/2])
        dst_rightdir = np.array([dst_w/2, 0])

        # set src, dst
        src = np.zeros((3, 2))
        src[0, :] = src_center
        src[1, :] = src_center + src_downdir
        src[2, :] = src_center + src_rightdir

        dst = np.zeros((3, 2))
        dst[0, :] = dst_center
        dst[1, :] = dst_center + dst_downdir
        dst[2, :] = dst_center + dst_rightdir
        
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans

    def augmentation(img, bbox, mode):
        scale, rot, color_scale, do_flip = Img.get_augmentation_cfg(mode)
        img_height, img_width, img_channels = img.shape
        bbox = Box.xywh_to_cxcywh(bbox)

        # flip
        if do_flip:
            img = img[:, ::-1, :]
            bbox[0] = img_width - bbox[0] - 1
        
        # color scaling
        img = np.clip(img * color_scale[None,None,:], 0, 255)
        
        # warping
        trans = Img.get_trans(bbox, cfg.input_img_shape, scale, rot)
        img_patch = cv2.warpAffine(img, trans, cfg.input_img_shape[::-1], flags=cv2.INTER_LINEAR)
        return img_patch, trans, rot, do_flip

class Box:
    def xyxy_to_cxcywh(box):
        w = box[2] - box[0]
        h = box[3] - box[1]
        c_x = box[0] + w/2
        c_y = box[1] + h/2
        return np.array([c_x, c_y, w, h])

    def xywh_to_cxcywh(box):
        c_x = box[0] + box[2]/2
        c_y = box[1] + box[3]/2
        w = box[2]
        h = box[3]
        return np.array([c_x, c_y, w, h])

    def preprocess_box(box, img_shape):
        x, y, w, h = box

        # clamp box
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((img_shape[1] - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((img_shape[0] - 1, y1 + np.max((0, h - 1))))
        box = np.array([x1, y1, x2, y2])

        # to cxcywh
        box = Box.xyxy_to_cxcywh(box)
        aspect_ratio = cfg.input_img_shape[1]/cfg.input_img_shape[0]

        if box[2] > aspect_ratio * box[3]:
            box[3] = box[2] / aspect_ratio
        elif box[2] < aspect_ratio * box[3]:
            box[2] = box[3] * aspect_ratio
        
        # to xywh
        box[2] = box[2] * 1.25
        box[3] = box[3] * 1.25
        box[0] = box[0] - box[2] / 2
        box[1] = box[1] - box[3] / 2
        return box

class Keypoint:
    joint_num = 29 # original: 24. manually add nose, L/R eye, L/R ear
    joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 
                'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 
                'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 
                'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 
                'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 
                'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear')
    flip_pairs = ((1,2), (4,5), (7,8), (10,11), (13,14), (16,17), 
                (18,19), (20,21), (22,23), (25,26), (27,28) )
    skeleton = ((0,1), (1,4), (4,7), (7,10), (0,2), (2,5), (5,8), (8,11), (0,3), (3,6), 
                (6,9), (9,14), (14,17), (17,19), (19,21), (21,23), (9,13), (13,16), (16,18), (18,20), 
                (20,22), (9,12), (12,24), (24,15), (24,25), (24,26), (25,27), (26,28))

    def to_other_db(src_joint, src_name):
        src_joint_num = len(src_name)
        dst_joint_num = len(Keypoint.joints_name)

        new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]))
        for src_idx in range(len(src_name)):
            name = src_name[src_idx]
            if name in Keypoint.joints_name:
                dst_idx = Keypoint.joints_name.index(name)
                new_joint[dst_idx] = src_joint[src_idx]
        return new_joint

    def parameter_to_other_db(joints_name, skeleton, flip_pairs, eval_joint):
        new_skeleton = []
        for pair in skeleton:
            src_name_0, src_name_1 = joints_name[pair[0]], joints_name[pair[1]]
            if src_name_0 in Keypoint.joints_name and src_name_1 in Keypoint.joints_name :
                dst_i0 = Keypoint.joints_name.index(src_name_0)
                dst_i1 = Keypoint.joints_name.index(src_name_1)
                new_skeleton.append((dst_i0, dst_i1))

        new_flip_pairs = []
        for pair in flip_pairs:
            src_name_0, src_name_1 = joints_name[pair[0]], joints_name[pair[1]]
            if src_name_0 in Keypoint.joints_name and src_name_1 in Keypoint.joints_name :
                dst_i0 = Keypoint.joints_name.index(src_name_0)
                dst_i1 = Keypoint.joints_name.index(src_name_1)
                new_flip_pairs.append((dst_i0, dst_i1))

        new_eval_joint = []
        for joint in eval_joint:
            src_name = joints_name[joint]
            if src_name in Keypoint.joints_name :
                dst_i = Keypoint.joints_name.index(src_name)
                new_eval_joint.append(dst_i)

        return Keypoint.joints_name, new_skeleton, new_flip_pairs, new_eval_joint
    
    def preprocess_keypoint(joint_img, joint_cam, img_shape, flip_pairs, img_trans, rot, do_flip):
        joint_img, joint_cam= joint_img.copy(), joint_cam.copy()

        joint_cam = (joint_cam - joint_cam[0, None, :])
        joint_img[:,2] = (joint_cam[:,2] / (cfg.body_3d_size / 2) + 1)/2 * cfg.output_hm_shape[0]

        # flip
        if do_flip:
            joint_cam[:,0] = -joint_cam[:,0]
            joint_img[:,0] = img_shape[1] - 1 - joint_img[:,0]
            for pair in flip_pairs:
                joint_img[pair[0],:], joint_img[pair[1],:] = joint_img[pair[1],:].copy(), joint_img[pair[0],:].copy()
                joint_cam[pair[0],:], joint_cam[pair[1],:] = joint_cam[pair[1],:].copy(), joint_cam[pair[0],:].copy()
        
        # rotation - joint_cam
        rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
                                [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                [0, 0, 1]])
        joint_cam = np.dot(rot_aug_mat, joint_cam.transpose(1,0)).transpose(1,0)

        # affine - joint_img
        joint_img_xy1 = np.concatenate((joint_img[:,:2], np.ones_like(joint_img[:,:1])),1)
        joint_img[:,:2] = np.dot(img_trans, joint_img_xy1.transpose(1,0)).transpose(1,0)
        return joint_img, joint_cam

class Human:
    def preprocess_smpl(human_model_param, cam_param, img_shape, do_flip, img_trans, rot):
        pose, shape, trans = torch.tensor(human_model_param['pose']), torch.tensor(human_model_param['shape']), torch.tensor(human_model_param['trans'])
        pose = pose.view(-1, 3)
        shape = shape.view(1, -1)
        trans = trans.view(1, -1)
        
        if 'gender' in human_model_param:
            gender = human_model_param['gender']
        else:
            gender = 'neutral'

        # extrinsic transform
        if 'R' in cam_param:
            R = np.array(cam_param['R'], dtype=np.float32).reshape(3,3)
            root_pose = pose[smpl.orig_root_joint_idx,:].numpy()
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose))
            pose[smpl.orig_root_joint_idx] = torch.from_numpy(root_pose).view(3)

        # get mesh
        root_pose = pose[smpl.orig_root_joint_idx].view(1, 3)
        body_pose = torch.cat((pose[:smpl.orig_root_joint_idx,:], pose[smpl.orig_root_joint_idx+1:,:])).view(1,-1)
        output = smpl.layer[gender](betas=shape, body_pose=body_pose, global_orient=root_pose, transl=trans)
        mesh_cam = output.vertices[0].numpy()

        # get joint
        joint_cam = np.dot(smpl.joint_regressor, mesh_cam)
        joint_img = Transform.cam2pixel(joint_cam, cam_param['f'], cam_param['c'])

        # joint preprocessing
        joint_img, joint_cam = Keypoint.preprocess_keypoint(joint_img, joint_cam, img_shape, smpl.flip_pairs, img_trans, rot, do_flip)
        joint_vis = np.ones((smpl.joint_num,))

        joint_cam = Keypoint.to_other_db(joint_cam, smpl.joints_name)
        joint_img = Keypoint.to_other_db(joint_img, smpl.joints_name)
        joint_vis = Keypoint.to_other_db(joint_vis, smpl.joints_name)

        # pose & shape preprocessing
        pose = pose.reshape(-1)
        return joint_img, joint_cam, joint_vis, pose, shape, mesh_cam

    def preprocessing_pose_shape(self, pose, shape, do_flip, rot):
        if do_flip:
            for pair in human_model.orig_flip_pairs:
                pose[pair[0], :], pose[pair[1], :] = pose[pair[1], :].clone(), pose[pair[0], :].clone()
                if human_model_type == 'smplx':
                    param_valid[pair[0]], param_valid[pair[1]] = param_valid[pair[1]].copy(), param_valid[pair[0]].copy()
            pose[:,1:3] *= -1 # multiply -1 to y and z axis of axis-angle
        
        # rotate root pose
        pose = pose.numpy()
        root_pose = pose[human_model.orig_root_joint_idx,:]
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,root_pose))
        pose[human_model.orig_root_joint_idx] = root_pose.reshape(3)
        
        # change to mean shape if beta is too far from it
        shape[(shape.abs() > 3).any(dim=1)] = 0.
        shape = shape.numpy().reshape(-1)

        return pose, shape



