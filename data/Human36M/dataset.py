import json
import copy
import os.path as osp
import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

from utils import Transform, Box, Img, Keypoint, Vis

class Human36M(Dataset):
    def __init__(self, mode):
        # Path
        cur_dir = osp.dirname(osp.abspath(__file__))
        self.annot_path = osp.join(cur_dir, 'annotations')
        self.image_path = osp.join(cur_dir, 'images')
        
        # Parameter
        self.joint_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Head', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.joint_adjacency = ((0, 1), (0, 4))
        self.joint_flip_pair = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.mode = mode
        
        self.db = self.load_data()
        self.transform = transforms.ToTensor()

    def __len__(self):
        return 1
        #return len(self.db)
    
    def __getitem__(self, index):
        data = copy.deepcopy(self.db[index])

        # load image & augmentation
        img = Img.load_img(data['img_path'])
        Vis.visualize_boxes(img, [data['bbox']], "./debug1.jpg")
        img, img_trans, rot, do_flip = Img.augmentation(img, data['bbox'], self.mode)
        img = self.transform(img) / 255
        
        # modify joint coordinates
        joint_img, joint_cam, img_shape = data['joint_img'], data['joint_cam'], data['img_shape']
        joint_img, joint_cam = Keypoint.preprocess_keypoint(joint_img, joint_cam, img_shape, self.joint_flip_pair, img_trans, rot, do_flip)

        Vis.visualize_image(img, "./debug2.jpg")
        Vis.visualize_keypoints(img, joint_img, "./debug3.jpg")
        
        targets = {
            'joint_img': data['joint_img']
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
        smpl_params = {}

        subject_list = self.get_subject()
        sampling_ratio  =self.get_sampling_ratio()

        for subject in subject_list:
            # data load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'),'r') as f:
                annot = json.load(f)
            for k,v in annot.items():
                    anns[k] = v

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

            # project world coordinate to cam, image coordinate space
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)])
            joint_cam = Transform.world2cam(joint_world, R, t)
            joint_img = Transform.cam2pixel(joint_cam, f, c)

            # pre-process box
            bbox = Box.preprocess_box(np.array(ann['bbox']), img_shape)
            
            db.append({
                'img_path' : img_path,
                'img_shape' : img_shape,
                'bbox' : bbox,
                'joint_img' : joint_img,
                'joint_cam' : joint_cam,
                'cam_param' : cam_param
            })

        return db