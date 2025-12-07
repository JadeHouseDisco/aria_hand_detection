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

import threading

# --- ARIA SDK IMPORTS ---
import aria.sdk as aria_sdk  # Project Aria Client SDK

# --- CONFIGURATION ---
CHECKPOINT_PATH = "ckpts/hamba/checkpoints/hamba.ckpt"
IMAGE_SIZE = 256
FOCAL_LENGTH = 1000.0  # keep as before for now
DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PALM_SCALE_MULTIPLIER = 3.8

ARIA_IP = "192.168.1.2"  # set from companion app Wi-Fi page

# --- SMART SCALE FILTER (The Fix) ---
class SmartScaleFilter:
    def __init__(self, grow_alpha=0.8, shrink_alpha=0.05):
        self.val = None
        self.grow_alpha = grow_alpha
        self.shrink_alpha = shrink_alpha

    def __call__(self, x):
        if self.val is None:
            self.val = x
            return x

        if x > self.val:
            self.val = self.grow_alpha * x + (1 - self.grow_alpha) * self.val
        else:
            self.val = self.shrink_alpha * x + (1 - self.shrink_alpha) * self.val

        return self.val

# --- 3D VISUALIZER ---
class Visualizer3D:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_proj_type('ortho')
        self.ax.view_init(elev=30, azim=45)
        self.ax.set_box_aspect([1, 1, 1])
        self.bones = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (1, 6), (6, 7), (7, 8), (8, 9),
            (1, 10), (10, 11), (11, 12), (12, 13),
            (1, 14), (14, 15), (15, 16), (16, 17),
            (1, 18), (18, 19), (19, 20), (20, 21)
        ]

    def update(self, hands_dict):
        self.ax.cla()
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(1.5, 2.5)
        self.ax.set_zlim(0.0, 1.0)
        self.ax.set_xlabel('X (Left/Right)')
        self.ax.set_ylabel('Y (Depth)')
        self.ax.set_zlabel('Z (Up/Down)')
        self.ax.scatter([0], [0], [0], c='blue', marker='^', s=50)

        for label, skeleton in hands_dict.items():
            xs = skeleton[:, 0]
            ys = -skeleton[:, 1]
            zs = skeleton[:, 2]
            color = 'green' if label == "Right" else 'red'
            self.ax.scatter(xs, zs, ys, c=color, marker='o', s=20)
            for (start, end) in self.bones:
                self.ax.plot(
                    [xs[start], xs[end]],
                    [zs[start], zs[end]],
                    [ys[start], ys[end]],
                    color=color,
                )
            self.ax.text(xs[1], zs[1], ys[1], label, fontsize=8)

        plt.draw()
        plt.pause(0.001)

# --- STANDARD FILTER (For Position/Rotation) ---
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
        if t is None:
            t = time.time()
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x
        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev
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

# --- HELPERS ---
def get_stable_crop_params(landmarks, img_shape):
    img_h, img_w = img_shape[:2]
    wrist = np.array([landmarks[0].x * img_w, landmarks[0].y * img_h])
    middle_mcp = np.array([landmarks[9].x * img_w, landmarks[9].y * img_h])
    palm_size = np.linalg.norm(wrist - middle_mcp)
    s = palm_size * PALM_SCALE_MULTIPLIER
    center_x, center_y = middle_mcp[0], middle_mcp[1]
    return s, center_x, center_y

def compute_crop_coords(center_x, center_y, s, img_shape):
    img_h, img_w = img_shape[:2]
    x1 = int(center_x - s / 2)
    y1 = int(center_y - s / 2)
    x2 = int(center_x + s / 2)
    y2 = int(center_y + s / 2)
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_w)
    pad_bottom = max(0, y2 - img_h)
    return x1, y1, x2, y2, pad_left, pad_top, pad_right, pad_bottom

def crop_image(img_rgb, crop_params):
    x1, y1, x2, y2, pad_left, pad_top, pad_right, pad_bottom = crop_params
    if any([pad_left, pad_top, pad_right, pad_bottom]):
        img_padded = cv2.copyMakeBorder(
            img_rgb,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        crop = img_padded[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left]
    else:
        crop = img_rgb[y1:y2, x1:x2]
    try:
        return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    except Exception:
        return None

def synthesize_forearm(skeleton_21):
    wrist = skeleton_21[0]
    middle_base = skeleton_21[9]
    forearm_point = wrist + (wrist - middle_base)
    skeleton_22 = np.vstack([forearm_point[None, :], skeleton_21])
    return skeleton_22


class AriaImageListener:
    def __init__(self):
        self.rgb_image = None
        # Initialize with zeroes
        self.accel = np.array([0.0, 0.0, 0.0])
        self.gyro = np.array([0.0, 0.0, 0.0])
        self.lock = threading.Lock()
        
    # --- RGB CALLBACK (Working) ---
    def on_image_received(self, image, record):
        if record.camera_id == aria_sdk.CameraId.Rgb:
            with self.lock:
                rotated_img = np.rot90(image, -1)
                self.rgb_image = rotated_img.copy()

    # --- IMU CALLBACK (The Fix) ---
    # Based on visualizer.py, the method MUST be named 'on_imu_received'
    # Arguments: (self, samples, imu_idx)
    def on_imu_received(self, samples, imu_idx):
        """
        Callback for IMU data.
        samples: A list/sequence of MotionData objects.
        imu_idx: 0 (Left) or 1 (Right).
        """
        if not samples:
            return

        # We take the last sample in the batch (most recent)
        latest = samples[-1]
        
        with self.lock:
            # According to visualizer.py, the attributes are .accel_msec2 and .gyro_radsec
            self.accel = np.array(latest.accel_msec2)
            self.gyro = np.array(latest.gyro_radsec)

    # --- DUCK TYPING SAFETY ---
    # Some C++ wrappers might look for the CamelCase version.
    # We map it to the snake_case version just to be 100% safe.
    def onImuReceived(self, samples, imu_idx):
        self.on_imu_received(samples, imu_idx)
        
    # --- IMAGE ALIAS ---
    def onImageReceived(self, image, record):
        self.on_image_received(image, record)

    # --- HELPER ---
    def get_latest_data(self):
        with self.lock:
            img = None if self.rgb_image is None else self.rgb_image.copy()
            acc = self.accel.copy()
            gyr = self.gyro.copy()
            return img, acc, gyr
        

def setup_aria_stream(ip_address: str, profile_name: str):
    # 1. Device Client (Same as before)
    device_client = aria_sdk.DeviceClient()
    client_cfg = aria_sdk.DeviceClientConfig()
    client_cfg.ip_v4_address = ip_address
    device_client.set_client_config(client_cfg)
    device = device_client.connect()

    # 2. Streaming Manager (Same as before)
    streaming_manager = device.streaming_manager
    streaming_config = aria_sdk.StreamingConfig()
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_config.streaming_interface = aria_sdk.StreamingInterface.WifiStation
    streaming_manager.streaming_config = streaming_config
    streaming_manager.start_streaming()

    # 3. Streaming Client Setup
    streaming_client = streaming_manager.streaming_client
    sub_config = streaming_client.subscription_config
    
    # --- CHANGED: Subscribe to RGB AND IMU ---
    sub_config.subscriber_data_type = (
        aria_sdk.StreamingDataType.Rgb | 
        aria_sdk.StreamingDataType.Imu
    )

    # Configure queues
    sub_config.message_queue_size[aria_sdk.StreamingDataType.Rgb] = 1
    # IMU comes very fast (800Hz or 1kHz). 
    # We set a small queue because we only care about the "current" gravity vector.
    sub_config.message_queue_size[aria_sdk.StreamingDataType.Imu] = 10

    options = aria_sdk.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    sub_config.security_options = options
    streaming_client.subscription_config = sub_config

    # 4. Attach Observer
    listener_logic = AriaImageListener()
    
    # Pass the raw object (Duck Typing)
    streaming_client.set_streaming_client_observer(listener_logic)
    
    print("--> Subscribing to RGB + IMU streams...")
    streaming_client.subscribe()

    return device_client, device, streaming_manager, streaming_client, listener_logic



def main():
    device = torch.device('cuda')
    print(f"--> Engine: Hamba (OFLEX) on {device}")

    model, model_cfg = load_hamba(CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()

    print("--> Detector: MediaPipe Hands")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0,
    )

    filters_skeleton = {"Left": OneEuroFilter(beta=0.05), "Right": OneEuroFilter(beta=0.05)}
    filters_pose_trans = {"Left": OneEuroFilter(beta=0.05), "Right": OneEuroFilter(beta=0.05)}

    filters_scale = {
        "Left": SmartScaleFilter(grow_alpha=0.8, shrink_alpha=0.02),
        "Right": SmartScaleFilter(grow_alpha=0.8, shrink_alpha=0.02),
    }

    vis3d = Visualizer3D()

    print(f"--> Connecting to Aria over Wi-Fi at {ARIA_IP}")
    device_client, aria_device, streaming_manager, streaming_client, aria_observer = setup_aria_stream(
        ARIA_IP, profile_name="default"
    )

    print("\n=== TELEOP V17 (SMART SCALE, ARIA RGB) ===")

    try:
        while True:
            t0 = time.time()

            # 1. Get Data
            raw_bgr, accel, gyro = aria_observer.get_latest_data()
            
            if raw_bgr is None:
                time.sleep(0.01)
                continue

            # 2. RGB Conversion
            img_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            frame = raw_bgr.copy()
            h, w, _ = img_rgb.shape

            # 3. Detect Hands
            results = hands.process(img_rgb)

            batch_images = []
            batch_is_right = []
            visual_info = []

            # --- PREPARE DATA CONTAINERS ---
            hands_to_plot = {}
            hand_status_text = [] # We will collect strings here, then join them later

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_label = results.multi_handedness[idx].classification[0].label
                    
                    if mp_label == "Left":
                        label_text = "Right"
                        is_right_flag = 1
                    else:
                        label_text = "Left"
                        is_right_flag = 0

                    raw_s, c_x, c_y = get_stable_crop_params(hand_landmarks.landmark, img_rgb.shape)
                    smooth_s = filters_scale[label_text](raw_s)
                    crop_rect_params = compute_crop_coords(c_x, c_y, smooth_s, img_rgb.shape)
                    crop_img = crop_image(img_rgb, crop_rect_params)

                    if crop_img is not None:
                        img_norm = (crop_img.astype(np.float32) / 255.0 - DEFAULT_MEAN) / DEFAULT_STD
                        img_chw = img_norm.transpose(2, 0, 1)
                        batch_images.append(img_chw)
                        batch_is_right.append(is_right_flag)

                        visual_info.append({
                            "label": label_text,
                            "center": [c_x, c_y],
                            "scale": smooth_s,
                            "img_size": [h, w],
                            "crop_rect": crop_rect_params[0:4],
                        })

            # 4. Inference
            if len(batch_images) > 0:
                tensor_batch = torch.from_numpy(np.stack(batch_images)).to(device)
                is_right_batch = torch.tensor(batch_is_right, dtype=torch.float32).to(device)
                batch = {"img": tensor_batch, "right": is_right_batch}

                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        out = model(batch)

                pred_cam = out["pred_cam"]
                keypoints_3d_crop = out["pred_keypoints_3d"].detach().cpu().numpy()

                for i in range(len(batch_images)):
                    info = visual_info[i]
                    label = info["label"]

                    box_center_t = torch.tensor([info["center"]], dtype=torch.float32).to(device)
                    box_size_t = torch.tensor([info["scale"]], dtype=torch.float32).to(device)
                    img_size_t = torch.tensor([info["img_size"]], dtype=torch.float32).to(device)

                    cam_t = cam_crop_to_full(
                        pred_cam[i : i + 1], box_center_t, box_size_t, img_size_t, FOCAL_LENGTH
                    ).detach().cpu().numpy()[0]
                    
                    smooth_trans = filters_pose_trans[label](cam_t)

                    raw_21_cam_space = keypoints_3d_crop[i] + cam_t
                    flat_21 = raw_21_cam_space.flatten()
                    smoothed_flat = filters_skeleton[label](flat_21)
                    smoothed_21 = smoothed_flat.reshape(21, 3)
                    skeleton_22 = synthesize_forearm(smoothed_21)

                    hands_to_plot[label] = skeleton_22

                    # --- COLLECT STATUS TEXT INSTEAD OF PRINTING IMMEDIATELY ---
                    hand_status_text.append(f"[{label} Z:{smooth_trans[2]:.2f}m]")

                    # Draw visualization
                    color = (0, 255, 0) if label == "Right" else (0, 0, 255)
                    x1, y1, x2, y2 = info["crop_rect"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    scale_len = int(info["scale"] / 4)
                    cv2.line(frame, (x1, y1 - 5), (x1 + scale_len, y1 - 5), color, 3)

            # 5. Visualization Update
            vis3d.update(hands_to_plot)
            cv2.imshow("Hamba Teleop V17 - Aria", frame)
            
            # 6. --- FINAL FORMATTED PRINT ---
            fps = 1 / (time.time() - t0)
            
            # Format IMU (Fixed width :5.1f ensures numbers don't jump around)
            imu_str = f"X:{accel[0]:5.1f} Y:{accel[1]:5.1f} Z:{accel[2]:5.1f}"
            
            # Format Hands
            if hand_status_text:
                hands_str = " ".join(hand_status_text)
            else:
                hands_str = "Searching..."

            # Print single line with padding spaces at the end to wipe previous long lines
            print(f"\rFPS: {fps:3.0f} | Accel: [{imu_str}] | {hands_str:<40}", end="")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        try:
            streaming_client.unsubscribe()
        except Exception:
            pass
        try:
            streaming_manager.stop_streaming()
        except Exception:
            pass
        try:
            device_client.disconnect(aria_device)
        except Exception:
            pass

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()