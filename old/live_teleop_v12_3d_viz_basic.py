import cv2
import torch
import numpy as np
import time
import mediapipe as mp
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hamba.models import load_hamba
from hamba.utils.renderer import cam_crop_to_full

# --- CONFIGURATION ---
CHECKPOINT_PATH = "ckpts/hamba/checkpoints/hamba.ckpt"
RESCALE_FACTOR = 2.0
IMAGE_SIZE = 256
FOCAL_LENGTH = 1000.0 
DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- 3D VISUALIZER CLASS ---
class Visualizer3D:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=-90, azim=-90) # Look from "Camera" perspective
        
        # MANO Bone connections (adjusted for 22 points where 0 is Forearm)
        # 0=Forearm, 1=Wrist, 2-5=Thumb, 6-9=Index, 10-13=Middle, 14-17=Ring, 18-21=Pinky
        self.bones = [
            (0, 1), # Forearm -> Wrist
            (1, 2), (2, 3), (3, 4), (4, 5),       # Thumb
            (1, 6), (6, 7), (7, 8), (8, 9),       # Index
            (1, 10), (10, 11), (11, 12), (12, 13), # Middle
            (1, 14), (14, 15), (15, 16), (16, 17), # Ring
            (1, 18), (18, 19), (19, 20), (20, 21)  # Pinky
        ]

    def update(self, hands_dict):
        self.ax.cla()
        # Set consistent bounds so the plot doesn't jump around
        self.ax.set_xlim(-0.2, 0.2) 
        self.ax.set_ylim(-0.2, 0.2) 
        self.ax.set_zlim(0.0, 1.0) # Z is depth in meters
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z (Depth)')

        for label, skeleton in hands_dict.items():
            # skeleton shape: (22, 3)
            xs = skeleton[:, 0]
            ys = skeleton[:, 1]
            zs = skeleton[:, 2]
            
            # Pick color
            color = 'green' if label == "Right" else 'red'
            
            # Plot Joints
            self.ax.scatter(xs, ys, zs, c=color, marker='o')
            
            # Plot Bones
            for (start, end) in self.bones:
                self.ax.plot([xs[start], xs[end]], 
                             [ys[start], ys[end]], 
                             [zs[start], zs[end]], color=color)
            
            # Label the Forearm (0) vs Wrist (1)
            self.ax.text(xs[0], ys[0], zs[0], "Forearm", fontsize=8)
            self.ax.text(xs[1], ys[1], zs[1], "Wrist", fontsize=8)

        plt.draw()
        plt.pause(0.001)

# --- 1 EURO FILTER ---
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0, freq=30.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, x, t=None):
        if t is None: t = time.time()
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

def get_crop_params(bbox, img_shape):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    center_x, center_y = x1 + w/2, y1 + h/2
    s = max(w, h) * RESCALE_FACTOR
    x1, y1 = int(center_x - s/2), int(center_y - s/2)
    x2, y2 = int(center_x + s/2), int(center_y + s/2)
    img_h, img_w = img_shape[:2]
    pad_left, pad_top = max(0, -x1), max(0, -y1)
    pad_right, pad_bottom = max(0, x2 - img_w), max(0, y2 - img_h)
    return x1, y1, x2, y2, pad_left, pad_top, pad_right, pad_bottom

def crop_image(img_rgb, crop_params):
    x1, y1, x2, y2, pad_left, pad_top, pad_right, pad_bottom = crop_params
    if any([pad_left, pad_top, pad_right, pad_bottom]):
        img_padded = cv2.copyMakeBorder(img_rgb, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
        crop = img_padded[y1+pad_top : y2+pad_top, x1+pad_left : x2+pad_left]
    else:
        crop = img_rgb[y1:y2, x1:x2]
    try:
        return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    except:
        return None

def synthesize_forearm(skeleton_21):
    wrist = skeleton_21[0]
    middle_base = skeleton_21[9]
    forearm_point = wrist + (wrist - middle_base)
    skeleton_22 = np.vstack([forearm_point[None, :], skeleton_21])
    return skeleton_22

def main():
    device = torch.device('cuda')
    print(f"--> Engine: Hamba (OFLEX) on {device}")
    
    model, model_cfg = load_hamba(CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()

    print("--> Detector: MediaPipe Hands")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                           min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)

    filters_skeleton = {"Left": OneEuroFilter(beta=0.05), "Right": OneEuroFilter(beta=0.05)}
    
    # Initialize 3D Plotter
    vis3d = Visualizer3D()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened(): return

    print("\n=== TELEOP V12 (3D VISUALIZATION) ===")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        t0 = time.time()
        
        frame = cv2.flip(frame, 1)
        img_rgb = frame[:, :, ::-1].copy()
        h, w, _ = img_rgb.shape
        results = hands.process(img_rgb)

        batch_images = []
        batch_is_right = []
        visual_info = [] 

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_label = results.multi_handedness[idx].classification[0].label
                if mp_label == "Left":
                    label_text = "Left"
                    is_right_flag = 1
                else:
                    label_text = "Right"
                    is_right_flag = 0
                
                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_list) * w), int(max(x_list) * w)
                y_min, y_max = int(min(y_list) * h), int(max(y_list) * h)
                bbox = [x_min, y_min, x_max, y_max]

                crop_params = get_crop_params(bbox, img_rgb.shape)
                crop_img = crop_image(img_rgb, crop_params)
                
                if crop_img is not None:
                    img_norm = (crop_img.astype(np.float32) / 255.0 - DEFAULT_MEAN) / DEFAULT_STD
                    img_chw = img_norm.transpose(2, 0, 1)
                    batch_images.append(img_chw)
                    batch_is_right.append(is_right_flag)
                    
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    bbox_size = max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * RESCALE_FACTOR
                    
                    visual_info.append({
                        'label': label_text,
                        'bbox': bbox,
                        'center': [center_x, center_y],
                        'scale': bbox_size,
                        'img_size': [h, w]
                    })

        # Dictionary to store data for the 3D Plotter
        hands_to_plot = {}

        if len(batch_images) > 0:
            tensor_batch = torch.from_numpy(np.stack(batch_images)).to(device)
            is_right_batch = torch.tensor(batch_is_right, dtype=torch.float32).to(device)
            batch = {'img': tensor_batch, 'right': is_right_batch}

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    out = model(batch)

            # Get Camera-Space Keypoints (Not just relative to wrist, but relative to camera)
            # We need to perform the crop-to-full projection manually for keypoints
            pred_cam = out['pred_cam'] # (N, 3)
            keypoints_3d_crop = out['pred_keypoints_3d'].detach().cpu().numpy() # (N, 21, 3)
            
            fps = 1 / (time.time() - t0)
            print(f"\rFPS: {fps:.0f} | ", end="")
            
            for i in range(len(batch_images)):
                info = visual_info[i]
                label = info['label']

                # --- PROJECT KEYPOINTS TO CAMERA SPACE ---
                # This is critical for 3D Visualization to look correct
                # 1. Recover global translation
                box_center_t = torch.tensor([info['center']], dtype=torch.float32).to(device)
                box_size_t = torch.tensor([info['scale']], dtype=torch.float32).to(device)
                img_size_t = torch.tensor([info['img_size']], dtype=torch.float32).to(device)
                
                # We use the renderer utility to get the translation vector
                cam_t = cam_crop_to_full(pred_cam[i:i+1], box_center_t, box_size_t, img_size_t, FOCAL_LENGTH).detach().cpu().numpy()[0]
                
                # 2. Add translation to the relative keypoints
                # The raw Hamba output is centered around the wrist/root.
                # We add the camera translation to move it into "World" space (Camera space).
                raw_21_cam_space = keypoints_3d_crop[i] + cam_t
                
                # Filter and synthesize forearm
                flat_21 = raw_21_cam_space.flatten()
                smoothed_flat = filters_skeleton[label](flat_21)
                smoothed_21 = smoothed_flat.reshape(21, 3)
                skeleton_22 = synthesize_forearm(smoothed_21)
                
                # Store for plotting
                hands_to_plot[label] = skeleton_22

                print(f"[{label}: 22pts OK] ", end="")
                
                color = (0, 255, 0) if label == "Right" else (0, 0, 255)
                bbox = info['bbox']
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        else:
            print(f"\rFPS: {1 / (time.time() - t0):.0f} | Searching...", end="")

        # Update 3D Plot
        vis3d.update(hands_to_plot)

        cv2.imshow('Hamba Teleop V12', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()