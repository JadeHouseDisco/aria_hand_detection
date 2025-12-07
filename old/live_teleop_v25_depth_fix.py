import cv2
import torch
import numpy as np
import time
import mediapipe as mp
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import socket
import json
from scipy.spatial.transform import Rotation

# --- ARIA SDK IMPORTS ---
import aria.sdk as aria_sdk

# --- HAMBA IMPORTS ---
from hamba.models import load_hamba
from hamba.utils.renderer import cam_crop_to_full

from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)

from projectaria_tools.core import data_provider

# --- CONFIGURATION ---
CHECKPOINT_PATH = "ckpts/hamba/checkpoints/hamba.ckpt"
IMAGE_SIZE = 256
ARIA_IP = "172.16.0.186"
VISUALIZATION = True
DISPLAY = True

# --- OPTIMIZATION CONSTANTS ---
# We want the output linear image to be 1408x1408 with 110 degree FOV.
# This results in the focal length of approx 493.0
TARGET_WIDTH = 1408
TARGET_HEIGHT = 1408
TARGET_FOV = 110  # Degrees
FOCAL_LENGTH = (TARGET_WIDTH / 2.0) / np.tan(np.deg2rad(TARGET_FOV / 2.0)) # ~493.0

DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PALM_SCALE_MULTIPLIER = 4.0

DEPTH_SCALE = 1.0
DEPTH_OFFSET = 0.2


class Visualizer3D:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_proj_type('ortho')
        
        # --- VIEW SETTINGS ---
        # elev=20: Slightly above looking down (like eyes looking at hands)
        # azim=-90: Orients the X-axis (Forward) to point "into" the screen
        self.ax.view_init(elev=20, azim=135) # 180 for fpv
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
        
        # --- AXIS LIMITS (IN METERS) ---
        # X (Forward): 0.0 to 1.0 (Arm's reach is usually ~0.8m)
        self.ax.set_xlim(0.0, 1.0)
        
        # Y (Left): -0.5 (Right) to 0.5 (Left)
        # Note: Since Positive Y is Left, -0.5 is 0.5m to your right.
        self.ax.set_ylim(-0.5, 0.5)
        
        # Z (Up): -0.5 (Down) to 0.5 (Up)
        self.ax.set_zlim(-0.5, 0.5)
        
        # --- LABELS ---
        self.ax.set_xlabel('X (Forward)')
        self.ax.set_ylabel('Y (Left)')
        self.ax.set_zlabel('Z (Up)')
        
        # Draw "Head/Camera" at Origin
        self.ax.scatter([0], [0], [0], c='black', marker='o', s=100, label="Head")

        for label, skeleton in hands_dict.items():
            # Coordinate System is already: X=Fwd, Y=Left, Z=Up
            xs = skeleton[:, 0]
            ys = skeleton[:, 1]  # REMOVED NEGATIVE SIGN (Keep it Left)
            zs = skeleton[:, 2]
            
            color = 'green' if label == "Right" else 'red'
            self.ax.scatter(xs, ys, zs, c=color, marker='o', s=20)
            
            for (start, end) in self.bones:
                self.ax.plot(
                    [xs[start], xs[end]],
                    [ys[start], ys[end]],
                    [zs[start], zs[end]],
                    color=color,
                )
            self.ax.text(xs[1], ys[1], zs[1], label, fontsize=8)

        plt.draw()
        plt.pause(0.001)


# --- UDP SENDER ---
class UDPSender:
    def __init__(self, ip="127.0.0.1", port=5555):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (ip, port)

    def send_data(self, skeleton_array, wrist_pos, wrist_quat):
        # skeleton_array: (22, 3) flattened or list
        # wrist_pos: [x, y, z]
        # wrist_quat: [x, y, z, w]
        data = {
            "joints": skeleton_array.tolist(),
            "wrist_pos": wrist_pos.tolist(),
            "wrist_quat": wrist_quat.tolist()
        }
        # Use try/except to prevent crashing on network blips
        try:
            message = json.dumps(data)
            self.sock.sendto(message.encode(), self.addr)
        except Exception as e:
            print(f"UDP Send Error: {e}")

# --- FILTERS (Keep your existing ones) ---
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

# --- HELPERS (Keep crop/synthesize logic) ---
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
            img_rgb, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
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
    return np.vstack([forearm_point[None, :], skeleton_21])

# --- HEAD POSE ---
class HeadPoseEstimator:
    def __init__(self, alpha=0.90):
        self.alpha = alpha
        self.last_ts = None
        
        # Euler angles in radians
        # Pitch: Rotation around Y (Lateral axis) -> Looking Up/Down
        # Roll:  Rotation around X (Forward axis) -> Tilting Head Left/Right
        # Yaw:   Rotation around Z (Vertical axis) -> Turning Head Left/Right
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0 

    def remap_axes(self, accel, gyro):
        """
        Maps Raw Aria Sensor Frame to Target Visualizer Frame.
        
        Raw Aria Frame (Based on your description):
        - Raw X: Right (+)
        - Raw Y: Down (+)
        - Raw Z: Forward (+)
        
        Target Frame (Your requirement):
        - Vis X: Forward (+)
        - Vis Y: Left (+)
        - Vis Z: Up (+)
        """
        raw_ax, raw_ay, raw_az = accel
        raw_gx, raw_gy, raw_gz = gyro

        # --- 1. ACCELEROMETER MAPPING ---
        # Target X (Fwd)  <== Raw Z (Fwd)
        ax = raw_az
        
        # Target Y (Left) <== -Raw X (Right)
        ay = -raw_ax
        
        # Target Z (Up)   <== -Raw Y (Down)
        az = -raw_ay

        # --- 2. GYROSCOPE MAPPING ---
        # Must match the accel permutation
        # Rotation around Target X (Roll) <== Rotation around Raw Z
        gx = raw_gz
        
        # Rotation around Target Y (Pitch) <== Rotation around -Raw X
        gy = -raw_gx
        
        # Rotation around Target Z (Yaw) <== Rotation around -Raw Y
        gz = -raw_gy
        
        return np.array([ax, ay, az]), np.array([gx, gy, gz])

    def process(self, raw_accel, raw_gyro, timestamp):
        if self.last_ts is None:
            self.last_ts = timestamp
            return np.eye(3)

        dt = timestamp - self.last_ts
        self.last_ts = timestamp

        # 1. REMAP AXES
        accel, gyro = self.remap_axes(raw_accel, raw_gyro)

        # 2. INTEGRATE GYRO
        # In this frame:
        # gyro[0] is X-axis (Roll)
        # gyro[1] is Y-axis (Pitch)
        # gyro[2] is Z-axis (Yaw)
        self.roll  += gyro[0] * dt 
        self.pitch += gyro[1] * dt
        self.yaw   += gyro[2] * dt

        # 3. ACCEL CORRECTIONS (Gravity Vector Analysis)
        
        # Pitch (Rotation around Y-axis):
        # We look at the relationship between X (Forward) and Z (Up).
        # atan2(forward_accel, up_accel)
        acc_pitch = math.atan2(accel[0], accel[2])
        
        # Roll (Rotation around X-axis):
        # We look at the relationship between Y (Left) and Z (Up).
        # atan2(left_accel, up_accel)
        # Note: We negate accel[1] if the tilt direction is inverted relative to standard math
        acc_roll = math.atan2(accel[1], accel[2])

        # 4. COMPLEMENTARY FILTER
        self.pitch = self.alpha * self.pitch + (1 - self.alpha) * acc_pitch
        self.roll  = self.alpha * self.roll  + (1 - self.alpha) * acc_roll

        return self.get_rotation_matrix()

    def get_rotation_matrix(self):
        c_y = math.cos(self.yaw)
        s_y = math.sin(self.yaw)
        c_p = math.cos(self.pitch)
        s_p = math.sin(self.pitch)
        c_r = math.cos(self.roll)
        s_r = math.sin(self.roll)

        # --- ROTATION MATRICES FOR TARGET FRAME ---
        # Frame: X=Forward, Y=Left, Z=Up
        
        # 1. Roll (Rotation about X axis)
        R_x = np.array([[1, 0, 0], 
                        [0, c_r, -s_r], 
                        [0, s_r, c_r]])
        
        # 2. Pitch (Rotation about Y axis)
        R_y = np.array([[c_p, 0, s_p], 
                        [0, 1, 0], 
                        [-s_p, 0, c_p]])
        
        # 3. Yaw (Rotation about Z axis)
        R_z = np.array([[c_y, -s_y, 0], 
                        [s_y, c_y, 0], 
                        [0, 0, 1]])
        
        # Order: Yaw -> Pitch -> Roll (Standard Z-Y-X Euler Sequence)
        R = R_z @ R_y @ R_x
        return R

# --- ARIA LISTENER (Kept exactly the same) ---
class AriaImageListener:
    def __init__(self):
        self.rgb_image = None
        self.accel = np.array([0.0, 0.0, 0.0])
        self.gyro = np.array([0.0, 0.0, 0.0])
        self.timestamp_sec = 0.0 
        self.lock = threading.Lock()
        
    def on_image_received(self, image, record):
        if record.camera_id == aria_sdk.CameraId.Rgb:
            with self.lock:
                self.rgb_image = image.copy()

    def on_imu_received(self, samples, imu_idx):
        if not samples: return
        latest = samples[-1]
        with self.lock:
            self.accel = np.array(latest.accel_msec2)
            self.gyro = np.array(latest.gyro_radsec)
            self.timestamp_sec = latest.capture_timestamp_ns * 1e-9
    
    def onImuReceived(self, samples, imu_idx): self.on_imu_received(samples, imu_idx)
    def onImageReceived(self, image, record): self.on_image_received(image, record)
    
    def get_latest_data(self):
        with self.lock:
            img = None if self.rgb_image is None else self.rgb_image.copy()
            acc = self.accel.copy()
            gyr = self.gyro.copy()
            ts = self.timestamp_sec
            return img, acc, gyr, ts

def setup_aria_stream(ip_address: str, profile_name: str):
    device_client = aria_sdk.DeviceClient()
    client_cfg = aria_sdk.DeviceClientConfig()
    client_cfg.ip_v4_address = ip_address
    device_client.set_client_config(client_cfg)
    device = device_client.connect()

    streaming_manager = device.streaming_manager
    streaming_config = aria_sdk.StreamingConfig()
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_config.profile_name = profile_name
    streaming_config.streaming_interface = aria_sdk.StreamingInterface.WifiStation
    streaming_manager.streaming_config = streaming_config
    streaming_manager.start_streaming()

    streaming_client = streaming_manager.streaming_client
    sub_config = streaming_client.subscription_config
    
    sub_config.subscriber_data_type = (
        aria_sdk.StreamingDataType.Rgb | aria_sdk.StreamingDataType.Imu
    )
    sub_config.message_queue_size[aria_sdk.StreamingDataType.Rgb] = 1
    sub_config.message_queue_size[aria_sdk.StreamingDataType.Imu] = 1

    options = aria_sdk.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    sub_config.security_options = options
    streaming_client.subscription_config = sub_config

    listener_logic = AriaImageListener()
    streaming_client.set_streaming_client_observer(listener_logic)
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
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        model_complexity=1,
    )

    # --- FILTERS ---
    filters_skeleton = {"Left": OneEuroFilter(beta=0.05), "Right": OneEuroFilter(beta=0.05)}
    filters_pose_trans = {"Left": OneEuroFilter(beta=0.05), "Right": OneEuroFilter(beta=0.05)}
    
    # NEW: Filter for Rodrigues Vectors (3 values)
    filters_pose_rot = {"Left": OneEuroFilter(beta=0.05), "Right": OneEuroFilter(beta=0.05)}

    filters_scale = {
        "Left": SmartScaleFilter(grow_alpha=0.8, shrink_alpha=0.8),
        "Right": SmartScaleFilter(grow_alpha=0.8, shrink_alpha=0.8),
    }

    print(f"--> Connecting to Aria over Wi-Fi at {ARIA_IP}")
    device_client, aria_device, streaming_manager, streaming_client, aria_observer = setup_aria_stream(
        ARIA_IP, profile_name="profile22"
    )

    print("--> Fetching Device Calibration...")
    # 1. Get the factory calibration JSON from the device
    sensors_calib_json = streaming_manager.sensors_calibration()
    sensors_calib = device_calibration_from_json_string(sensors_calib_json)
    
    # 2. Get the specific Src Calibration for the RGB Camera
    src_calib = sensors_calib.get_camera_calib("camera-rgb")

    # 3. Define the Dst (Target) Calibration
    # We want a Linear (Pinhole) camera.
    # Width/Height: 1408 (Native RGB resolution)
    # Focal Length: 493.0 (Matches your config and the 110 deg FOV math)
    dst_calib = get_linear_camera_calibration(TARGET_WIDTH, TARGET_HEIGHT, FOCAL_LENGTH, "camera-rgb")

    pose_estimator = HeadPoseEstimator(alpha=0.98)
    udp_sender = UDPSender()

    if VISUALIZATION:
        print("--> Starting 3D Visualizer...")
        vis3d = Visualizer3D()

    P = np.array([
            [ 0,  0,  1],
            [-1,  0,  0],
            [ 0, -1,  0]
        ], dtype=np.float32)
    
    P_yz_swap = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])

    print("\n=== TELEOP RUNNING (6D MODE) ===")
    
    try:
        while True:
            raw_img, accel, gyro, ts_imu = aria_observer.get_latest_data()
            if raw_img is None or ts_imu == 0:
                continue

            # --- OFFICIAL UNDISTORTION ---
            # Transforms the Fisheye 'raw_img' into a Linear Pinhole image
            # using the exact parameters of your device.
            # Note: raw_img is usually RGB from the SDK, but check orientation.
            img_rectified = distort_by_calibration(raw_img, dst_calib, src_calib)

            img_display = np.rot90(img_rectified, -1)

            if DISPLAY:
                # Show the Rectified Image (The one MediaPipe is processing)
                # Convert to BGR for OpenCV
                debug_frame = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
                cv2.imshow("Rectified (Input to AI)", debug_frame)

            # --- PREPARE FOR PIPELINE ---
            # 1. For MediaPipe/Hamba (Needs RGB)
            img_rgb = img_display.copy()

            # 2. Process Head Rotation (R_head)
            # This is crucial for stabilizing the wrist against head movement
            R_head = pose_estimator.process(accel, gyro, ts_imu)

            # 3. Detect Hands
            results = hands.process(img_rgb)

            batch_images = []
            batch_is_right = []
            visual_info = []
            if VISUALIZATION:
                hands_to_plot = {}

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                    # Draw landmarks on debug frame
                    if DISPLAY:
                         mp.solutions.drawing_utils.draw_landmarks(debug_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                         cv2.imshow("Rectified (Input to AI)", debug_frame)

                    mp_label = results.multi_handedness[idx].classification[0].label
                    # Swap Label because Aria Image is Rotated 90 degrees
                    label_text = "Right" if mp_label == "Left" else "Left"
                    is_right_flag = 1 if label_text == "Right" else 0

                    raw_s, c_x, c_y = get_stable_crop_params(hand_landmarks.landmark, img_rgb.shape)
                    smooth_s = filters_scale[label_text](raw_s)

                    crop_rect_params = compute_crop_coords(c_x, c_y, smooth_s, img_rgb.shape)
                    crop_img = crop_image(img_rgb, crop_rect_params)

                    if crop_img is not None:

                        # DEBUG CROP
                        if DISPLAY:
                             cv2.imshow(f"Crop {label_text}", cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

                        img_norm = (crop_img.astype(np.float32) / 255.0 - DEFAULT_MEAN) / DEFAULT_STD
                        img_chw = img_norm.transpose(2, 0, 1)
                        batch_images.append(img_chw)
                        batch_is_right.append(is_right_flag)
                        
                        visual_info.append({
                            "label": label_text,
                            "center": [c_x, c_y],
                            "scale": raw_s,           # <--- raw for geometry
                            "smooth_scale": smooth_s, # <--- only for visualization if needed
                            "img_size": img_rgb.shape[:2],
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
                # A. Rotation (Global Orient)
                # Ensure we handle shape correctly. Hamba usually returns [N, 1, 3, 3] or [N, 3, 3]
                wrist_rot_matrix = out['pred_mano_params']['global_orient'].detach().cpu().numpy()
                keypoints_3d_crop = out["pred_keypoints_3d"].detach().cpu().numpy()

                for i in range(len(batch_images)):
                    info = visual_info[i]
                    label = info["label"]

                    # --- 1. RECOVER TRANSLATION (Camera Space) ---
                    box_center_t = torch.tensor([info["center"]], dtype=torch.float32).to(device)
                    box_size_t   = torch.tensor([info["scale"]],  dtype=torch.float32).to(device)
                    img_size_t   = torch.tensor([info["img_size"]], dtype=torch.float32).to(device)

                    full_cam_t = cam_crop_to_full(
                        pred_cam[i:i+1],
                        box_center_t,
                        box_size_t,
                        img_size_t,
                        FOCAL_LENGTH,
                    ).detach().cpu().numpy()[0]
                    
                    # Filter Translation (in Camera Space)
                    smooth_cam_t = filters_pose_trans[label](full_cam_t)

                    # --- 2. RECOVER ROTATION (Camera Space) ---
                    raw_rot_mat = wrist_rot_matrix[i]
                    if raw_rot_mat.ndim == 3: 
                        raw_rot_mat = raw_rot_mat[0]
                    
                    raw_rot_vec, _ = cv2.Rodrigues(raw_rot_mat)
                    smooth_rot_vec = filters_pose_rot[label](raw_rot_vec.flatten())
                    smooth_rot_mat_cam, _ = cv2.Rodrigues(smooth_rot_vec)

                    # --- 3. APPLY COORDINATE CHANGE + HEAD STABILIZATION ---
                    
                    # A. POSITION: 
                    # 1. Permute Vector (Cam -> Target Frame)
                    t_target = P @ smooth_cam_t 
                    # 2. Apply Head Rotation (Target Frame -> Gravity Aligned)
                    world_pos = R_head @ t_target 
                    
                    # B. ROTATION (WRIST):
                    # 1. Permute Matrix (Similarity Transform: P * R * P_inv)
                    # This aligns the wrist's local axes to the new coordinate system
                    rot_mat_target = P @ smooth_rot_mat_cam @ P.T
                    # 2. Apply Head Rotation
                    world_rot_mat = R_head @ rot_mat_target @ R_head.T
                    
                    world_quat = Rotation.from_matrix(world_rot_mat).as_quat() # [x, y, z, w]

                    if VISUALIZATION:
                        # --- 4. SKELETON (Keypoints) ---
                        # Calculate raw 3D points in original camera space
                        points_cam = keypoints_3d_crop[i] + smooth_cam_t
                        
                        # 1. Permute Points (N, 3) -> Transpose for matrix mult -> (3, N)
                        # (P @ [x,y,z].T).T  is the same as your previous np.stack approach
                        points_target = (P @ points_cam.T)
                        
                        # 2. Apply Head Rotation
                        points_world = (R_head @ points_target).T
                        
                        flat_21 = points_world.flatten()
                        smoothed_flat = filters_skeleton[label](flat_21)
                        smoothed_21 = smoothed_flat.reshape(21, 3)
                        skeleton_22 = synthesize_forearm(smoothed_21)

                        hands_to_plot[label] = skeleton_22
                    
                    # --- 4. SKELETON (Keypoints) ---
                    # Calculate raw 3D points in original camera space
                    points_wrist = keypoints_3d_crop[i]
                    
                    # 1. Permute Points (N, 3) -> Transpose for matrix mult -> (3, N)
                    # (P @ [x,y,z].T).T  is the same as your previous np.stack approach
                    points_target_wrist = (P @ points_wrist.T)
                    
                    # 2. Apply Head Rotation
                    points_world_wrist = (R_head @ points_target_wrist).T
                    
                    flat_21 = points_world_wrist.flatten()
                    smoothed_flat = filters_skeleton[label](flat_21)
                    smoothed_21 = smoothed_flat.reshape(21, 3)
                    skeleton_22 = synthesize_forearm(smoothed_21)

                    # 5. SEND DATA
                    if label == "Right":
                        print(f"Depth (X): {world_pos[0]:.3f} | Left/Right (Y): {world_pos[1]:.3f} | Up/Down (Z): {world_pos[2]:.3f}")
                        udp_sender.send_data(skeleton_22, world_pos, world_quat)

            if VISUALIZATION:
                vis3d.update(hands_to_plot)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        try: streaming_client.unsubscribe()
        except: pass
        try: streaming_manager.stop_streaming()
        except: pass
        try: device_client.disconnect(aria_device)
        except: pass

if __name__ == "__main__":
    main()