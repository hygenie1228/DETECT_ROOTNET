import os
import os.path as osp
import sys

class Config:
    # Path
    root_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))

    # Dataset
    trainset = ['Human36M']
    testset = ['Human36M']
    
    # Train parameter
    batch_size = 48
    num_worker = 16
    shuffle = False

    end_epoch = 10
    lr = 1e-4

    # Image, keypoints info
    input_img_shape = (512, 384) 
    output_hm_shape = (8, 8, 6)
    body_3d_size = 2
    

    def set_args(self, gpus):
        gpus = str(gpus)
        self.num_gpus = len(gpus.split(','))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

cfg = Config()
sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
sys.path.insert(0, osp.join(cfg.root_dir, 'data'))