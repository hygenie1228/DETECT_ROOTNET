import numpy as np
import cv2
import pyrender
import trimesh
import matplotlib.pyplot as plt

class Vis:
    def visualize_image(img, output_path):
        img = (img*255).cpu().numpy().transpose(1, 2, 0)
        img = img.copy()

        cv2.imwrite(output_path, img)

    def visualize_boxes(img, boxes, output_path):
        img = img.copy()
        
        img = Vis.draw_box(img, boxes, (255,255,255), 2)
        cv2.imwrite(output_path, img)

    def visualize_skeleton(img, keypoints, valids, kps_lines, output_path):
        img = (img*255).cpu().numpy().transpose(1, 2, 0)
        new_img = img.copy()

        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

        # draw the skeleton lines
        for i in range(len(kps_lines)):
            i1, i2 = kps_lines[i]
            p1 = (int(keypoints[i1][0]), int(keypoints[i1][1]))
            p2 = (int(keypoints[i2][0]), int(keypoints[i2][1]))

            if valids[i1] * valids[i2] == 1:
                cv2.line(new_img, p1, p2, color=colors[i], thickness=2)
            if valids[i1] == 1:
                cv2.circle(new_img, p1, radius=3, color=colors[i], thickness=2)
            if valids[i2] == 1:
                cv2.circle(new_img, p2, radius=3, color=colors[i], thickness=2)

        new_img = cv2.addWeighted(img, 0, new_img, 1.0, 0)
        cv2.imwrite(output_path, new_img)

    def visualize_3d_skeleton(kpt_3d, valids, kps_lines, output_path):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
        colors = np.array([np.array([c[2], c[1], c[0]]).reshape(1, -1) for c in colors])

        # draw the skeleton lines
        for i in range(len(kps_lines)):
            i1, i2 = kps_lines[i]
            
            x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
            y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
            z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

            if valids[i1] * valids[i2] == 1:
                ax.plot(x, z, -y, c=np.squeeze(colors[i]), linewidth=2)
            if valids[i1] == 1:
                ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[i], marker='o')
            if valids[i2] == 1:
                ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[i], marker='o')

        ax.set_title('3D skeleton')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Z Label')
        ax.set_zlabel('Y Label')
        ax.set_xlim(0,512)
        ax.set_ylim(-5,5)
        ax.set_zlim(0,384)
        
        plt.savefig(output_path)
        plt.close(fig)
        cv2.waitKey(0)

    def visualize_keypoints(img, keypoints, valids, output_path):
        img = (img*255).cpu().numpy().transpose(1, 2, 0)
        new_img = img.copy()

        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(keypoints) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

        # draw the keypoints
        for i in range(len(keypoints)):
            if valids[i] == 1:
                p = (int(keypoints[i][0]), int(keypoints[i][1]))
                cv2.circle(new_img, p, radius=3, color=colors[i], thickness=3)

        new_img = cv2.addWeighted(img, 0, new_img, 1.0, 0)
        cv2.imwrite(output_path, new_img)

    def draw_box(img, boxes, color, thickness):
        for box in boxes:
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[0] + box[2])
            y_max = int(box[1] + box[3])
            
            pos1 = (x_min, y_min)
            pos2 = (x_min, y_max)
            pos3 = (x_max, y_min)
            pos4 = (x_max, y_max)
            
            img = cv2.line(img, pos1, pos2, color, thickness) 
            img = cv2.line(img, pos1, pos3, color, thickness) 
            img = cv2.line(img, pos2, pos4, color, thickness) 
            img = cv2.line(img, pos3, pos4, color, thickness) 
        return img

    def save_obj(v, f, output_path):
        obj_file = open(output_path, 'w')
        for i in range(len(v)):
            obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
        for i in range(len(f)):
            obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
        obj_file.close()


    def render_mesh(img, mesh, face, cam_param, output_path):
        img = (img*255).cpu().numpy().transpose(1, 2, 0)
        img = img.copy()

        # mesh
        mesh = trimesh.Trimesh(mesh, face)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
        
        scene.add(mesh, 'mesh')
        
        focal, princpt = cam_param['f'], cam_param['c']
        camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
        scene.add(camera)
        
        # renderer
        renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
        
        # light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)
        
        # render
        rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        rgb = rgb[:,:,:3].astype(np.float32)
        valid_mask = (depth > 0)[:,:,None]

        # save to image
        img = rgb * valid_mask + img * (1-valid_mask)
        cv2.imwrite(output_path, img)
        