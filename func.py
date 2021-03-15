import numpy as np
import cv2
import matplotlib.pyplot as plt

class Vis:
    def visualize_image(img, output_path):
        img = (img*255).cpu().numpy().transpose(1, 2, 0)
        img = img.copy()

        cv2.imwrite(output_path, img)

    def visualize_boxes(img, boxes, output_path):
        #img = (img*255).cpu().numpy().transpose(1, 2, 0)
        img = img.copy()
        
        img = Vis.draw_box(img, boxes, (255,255,255), 2)
        cv2.imwrite(output_path, img)

    def visualize_skeleton(img, keypoints, output_path):
        img = (img*255).cpu().numpy().transpose(1, 2, 0)
        new_img = img.copy()

        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(keypoints) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

        # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

        new_img = cv2.addWeighted(img, 0, new_img, 1.0, 0)
        cv2.imwrite(output_path, new_img)

    def visualize_keypoints(img, keypoints, output_path):
        img = (img*255).cpu().numpy().transpose(1, 2, 0)
        new_img = img.copy()

        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(keypoints) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

        # draw the keypoints
        for i in range(len(keypoints)):
            p = (int(keypoints[i][0]), int(keypoints[i][1]))
            print(p)
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