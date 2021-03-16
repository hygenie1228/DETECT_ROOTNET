import numpy as np

class Transform:
    def world2cam(world_coord, R, t):
        cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
        return cam_coord

    def cam2pixel(cam_coord, f, c):
        x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
        y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
        z = cam_coord[:,2]
        return np.stack((x,y,z),1)