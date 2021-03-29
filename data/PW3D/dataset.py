import json
import copy
import os.path as osp
import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

from config import cfg
from utils import Transform, Img, Box, Keypoint, Human, Vis
from structure import smpl

class PW3D(Dataset):
    def __init__(self, mode):
        # Path
        cur_dir = osp.dirname(osp.abspath(__file__))
        self.annot_path = osp.join(cur_dir, 'data')
        self.image_path = osp.join(cur_dir, 'imageFiles')
        
        # Human parameter
        self.joint_num = 24
        self.joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 
                            'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 
                            'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 
                            'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 
                            'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand')
        self.skeleton = ((0,1), (1,4), (4,7), (7,10), (0,2), (2,5), (5,8), (8,11), (0,3), (3,6), 
                        (6,9), (9,14), (14,17), (17,19), (19,21), (21,23), (9,13), (13,16), (16,18), (18,20), 
                        (20,22), (9,12), (12,15))
        self.flip_pairs = ((1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), (22,23))
        self.eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)

        self.mode = mode
        self.transform = transforms.ToTensor()
        
        # Load db
        self.db = self.load_data()

    def __len__(self):
        if cfg.for_debug:
            return 100
        else:
            return len(self.db)
    
    def __getitem__(self, index):
        data = copy.deepcopy(self.db[index])

        # load image & augmentation
        img = Img.load_img(data['img_path'])
        
        if cfg.for_debug:
            Vis.visualize_boxes(img, [data['bbox']], "./outputs/debug1.jpg")
        img, img_trans, rot, do_flip = Img.augmentation(img, data['bbox'], self.mode)
        img = self.transform(img) / 255

        if self.mode == 'train':
            joint_img, joint_cam, joint_vis = data['joint_img'], data['joint_cam'], data['joint_vis']
            img_shape, cam_param, smpl_param = data['img_shape'], data['cam_param'], data['smpl_param']

            # preprocess joint coordinates
            joint_cam = joint_cam / 1000
            joint_img, joint_cam = Keypoint.preprocess_keypoint(joint_img, joint_cam, img_shape, self.flip_pairs, img_trans, rot, do_flip) 

            # preprocess smpl coordinates
            smpl_joint_img, smpl_joint_cam, smpl_joint_vis, smpl_pose, smpl_shape, smpl_mesh_cam = \
                    Human.preprocess_smpl(smpl_param, cam_param, img_shape, do_flip, img_trans, rot)
            
        else:
            joint_img, joint_cam, joint_vis = None, None, None
            cam_param, smpl_param = None, None
        
        # visualize
        if cfg.for_debug:
            Vis.visualize_skeleton(img, joint_img, joint_vis, self.skeleton, "./outputs/debug2.jpg")
            Vis.visualize_skeleton(img, smpl_joint_img, smpl_joint_vis, self.skeleton, "./outputs/debug2_1.jpg")
            Vis.visualize_3d_skeleton(joint_img, joint_vis, self.skeleton, "./outputs/debug3.jpg")

            mesh_img = Transform.cam2pixel(smpl_mesh_cam, np.array(cam_param['f']), np.array(cam_param['c']))
            mesh_img, mesh_cam = Keypoint.preprocess_keypoint(mesh_img, smpl_mesh_cam, img_shape, (), img_trans, rot, do_flip)
            mesh_vis = np.ones((smpl_mesh_cam.shape[0]))
            Vis.visualize_keypoints(img, mesh_img, mesh_vis, "./outputs/debug4.jpg")
            Vis.save_obj(mesh_cam, smpl.layer['neutral'].faces, './outputs/debug5.obj')

        targets = {
            'joint_img' : joint_img,
            'joint_cam' : joint_cam,
            'joint_vis' : joint_vis,
            'cam_param' : cam_param,
            'smpl_pose' : smpl_pose,
            'smpl_shape': smpl_shape,
            'smpl_mesh_cam' : smpl_mesh_cam
            }

        return img, targets

    def get_subject(self):
        if self.mode == 'train':
            subject_list = [1, 5, 6, 7, 8]
        elif self.mode == 'test':
            subject_list = [9, 11]
        else:
            assert 0, "Unknown subset"
        return subject_list

    def get_sampling_ratio(self):
        if self.mode == 'train':
            return 5
        elif self.mode == 'test':
            return 64
        else:
            assert 0, "Unknown subset"

    def load_data(self):
        anns = {}
        cameras = {}
        joints = {}

        with open(osp.join(self.annot_path, '3DPW_' + self.mode + '.json'),'r') as f:
            annot = json.load(f)
        for k,v in annot.items():
            anns[k] = v
                
        imgs = {}
        for img in anns['images']:
            imgs[img['id']] = img
        
        db = []
        for ann in anns['annotations']:
            image_id = ann['image_id']
            img = imgs[image_id]

            # get image info
            sequence_name = img['sequence']
            img_name = img['file_name']
            img_path = osp.join(self.image_path, sequence_name, img_name)
            img_shape = (img['height'], img['width']) 

            # camera parameter
            cam_param = img['cam_param']
            f, c = np.array(cam_param['focal']), np.array(cam_param['princpt'])
            cam_param = {'f': f, 'c': c}
            
            # get joint_cam, joint_img
            joint_cam = np.array(ann['joint_cam']).reshape(-1,3)
            joint_cam = joint_cam[:self.joint_num, :]
            joint_img = Transform.cam2pixel(joint_cam, f, c)
            joint_vis = np.ones((self.joint_num,))

            # to standard db
            joint_cam = Keypoint.to_other_db(joint_cam, self.joints_name)
            joint_img = Keypoint.to_other_db(joint_img, self.joints_name)
            joint_vis = Keypoint.to_other_db(joint_vis, self.joints_name)

            # pre-process box
            bbox = Box.preprocess_box(np.array(ann['bbox']), img_shape)

            # smlpl parameter
            smpl_param = ann['smpl_param']

            db.append({
                'img_path' : img_path,
                'img_shape' : img_shape,
                'bbox' : bbox,
                'joint_img' : joint_img,
                'joint_cam' : joint_cam,
                'joint_vis' : joint_vis,
                'cam_param' : cam_param,
                'smpl_param' : smpl_param
            })

        # parameter to standard db
        self.joints_name, self.skeleton, self.flip_pairs, self.eval_joint = Keypoint.parameter_to_other_db(self.joints_name, self.skeleton, self.flip_pairs, self.eval_joint)
        return db