import numpy as np
import cv2
import random

from config import cfg

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
            color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
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
    def preprocess_keypoint(joint_img, joint_cam, img_shape, flip_pairs, img_trans, rot, do_flip):
        joint_img, joint_cam= joint_img.copy(), joint_cam.copy()

        joint_cam = (joint_cam - joint_cam[0, None, :]) / 1000
        joint_img = np.concatenate((joint_img[:,:2], joint_cam[:,2:]), 1)
        joint_img[:,2] = (joint_img[:,2] / (cfg.body_3d_size / 2) + 1)/2. * cfg.output_hm_shape[0]

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