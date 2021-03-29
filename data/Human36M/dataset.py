import json
import copy
import os.path as osp
import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

from config import cfg
from utils import Transform, Img, Box, Keypoint, Vis

class Human36M(Dataset):
    def __init__(self, mode):
        # Path
        cur_dir = osp.dirname(osp.abspath(__file__))
        self.annot_path = osp.join(cur_dir, 'annotations')
        self.image_path = osp.join(cur_dir, 'images')
        
        # Human parameter
        self.joint_num = 17
        self.joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 
                            'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Head', 
                            'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 
                            'R_Elbow', 'R_Wrist')
        self.skeleton = ((0,1), (1,2), (2,3), (0,4), (4,5), (5,6), (0,7), (7,8), (8,9),
                        (9,10), (8,11), (11,12), (12,13) ,(8,14), (14,15), (15,16))
        self.flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
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
        Vis.visualize_boxes(img, [data['bbox']], "./outputs/debug1.jpg")
        img, img_trans, rot, do_flip = Img.augmentation(img, data['bbox'], self.mode)
        img = self.transform(img) / 255
        
        if self.mode == 'train':
            joint_img, joint_cam, joint_vis = data['joint_img'], data['joint_cam'], data['joint_vis']
            img_shape = data['img_shape']
            cam_param = data['cam_param']

            # modify joint coordinates
            joint_img, joint_cam = Keypoint.preprocess_keypoint(joint_img, joint_cam, img_shape, self.flip_pairs, img_trans, rot, do_flip)            
        else:
            joint_img, joint_cam, joint_vis = None, None, None
            cam_param = None

        Vis.visualize_keypoints(img, joint_img, joint_vis, "./outputs/debug2.jpg")
        Vis.visualize_skeleton(img, joint_img, joint_vis, self.skeleton, "./outputs/debug3.jpg")

        targets = {
            'joint_img' : joint_img,
            'joint_cam' : joint_cam,
            'joint_vis' : joint_vis,
            'cam_param' : cam_param
            }

        return img, targets

    def get_subject(self):
        if self.mode == 'train':
            if cfg.for_debug:
                subject_list = [1, 5]
            else:
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

        subject_list = self.get_subject()
        sampling_ratio = self.get_sampling_ratio()

        for subject in subject_list:
            # data load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'),'r') as f:
                annot = json.load(f)
            if len(anns) == 0:
                for k,v in annot.items(): anns[k] = v
            else:
                for k,v in annot.items(): anns[k] += v

            # camera load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'),'r') as f:
                cameras[str(subject)] = json.load(f)

            # joint coordinate load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'),'r') as f:
                joints[str(subject)] = json.load(f)
                
        imgs = {}
        for img in anns['images']:
            imgs[img['id']] = img
        
        db = []
        for ann in anns['annotations']:
            image_id = ann['image_id']
            img = imgs[image_id]

            # get image info
            subject = img['subject']
            action_idx = img['action_idx']
            subaction_idx = img['subaction_idx']
            frame_idx = img['frame_idx']

            # sampling image
            if frame_idx % sampling_ratio != 0: continue
            
            img_path = osp.join(self.image_path, img['file_name'])
            img_shape = (img['height'], img['width']) 

            # camera parameter
            cam_idx = img['cam_idx']
            cam_param = cameras[str(subject)][str(cam_idx)]
            R,t,f,c = np.array(cam_param['R']), np.array(cam_param['t']), np.array(cam_param['f']), np.array(cam_param['c'])
            cam_param = {'R': R, 't': t, 'f': f, 'c': c}

            # get joint_cam, joint_img
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)])
            joint_cam = Transform.world2cam(joint_world, R, t)
            joint_img = Transform.cam2pixel(joint_cam, f, c)
            joint_vis = np.ones((self.joint_num,))

            # to standard db
            joint_cam = Keypoint.to_other_db(joint_cam, self.joints_name)
            joint_img = Keypoint.to_other_db(joint_img, self.joints_name)
            joint_vis = Keypoint.to_other_db(joint_vis, self.joints_name)

            # pre-process box
            bbox = Box.preprocess_box(np.array(ann['bbox']), img_shape)
            
            db.append({
                'img_path' : img_path,
                'img_shape' : img_shape,
                'bbox' : bbox,
                'joint_img' : joint_img,
                'joint_cam' : joint_cam,
                'joint_vis' : joint_vis,
                'cam_param' : cam_param
            })

        # parameter to standard db
        self.joints_name, self.skeleton, self.flip_pairs, self.eval_joint = Keypoint.parameter_to_other_db(self.joints_name, self.skeleton, self.flip_pairs, self.eval_joint)
        return db