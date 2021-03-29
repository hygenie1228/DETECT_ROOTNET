import os
import os.path as osp
import sys

class Config:
    # Path
    root_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    human_model_path = osp.join(root_dir, 'common', 'structure', 'human_model_files')

    # Dataset
    # Human36M, PW3D
    trainset = ['PW3D']
    testset = ['Human36M']
    train_same_len = False

    # Train parameter
    batch_size = 48
    num_worker = 16
    shuffle = False

    end_epoch = 10
    lr = 1e-4

    # Image, keypoints info
    input_img_shape = (512, 384)
    input_body_shape = (256, 192)
    output_hm_shape = (8, 8, 6)
    body_3d_size = 2
    camera_3d_size = 2.5

    # Virtual focal lengths & principal point position
    focal = (5000, 5000)
    princpt = (input_body_shape[1]/2, input_body_shape[0]/2)
    
    # Debug
    for_debug = True

    def set_args(self, gpus):
        gpus = str(gpus)
        self.num_gpus = len(gpus.split(','))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        # for rendering
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

cfg = Config()
sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
sys.path.insert(0, osp.join(cfg.root_dir, 'data'))