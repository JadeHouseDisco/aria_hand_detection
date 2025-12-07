import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import torch
import numpy as np
import time
import gc
gc.set_threshold(700, 10, 10)

import mediapipe as mp
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import socket
import json
from scipy.spatial.transform import Rotation

import aria.sdk as aria_sdk

from hamba.models import load_hamba
from hamba.utils.renderer import cam_crop_to_full

from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)

# --- CONFIGURATION ---
CHECKPOINT_PATH = "ckpts/hamba/checkpoints/hamba.ckpt"
# ARIA_IP = "192.168.1.2"
# ARIA_IP = "172.16.0.186"
ARIA_IP = "192.168.0.124"
VISUALIZATION = False
DISPLAY = True

# --- OPTIMIZATION CONSTANTS ---
TARGET_WIDTH = 1024
TARGET_HEIGHT = 1024
TARGET_FOV = 110  # Degrees
FOCAL_LENGTH = (TARGET_WIDTH / 2.0) / np.tan(np.deg2rad(TARGET_FOV / 2.0)) # ~493.0
VIRTUAL_FOCAL_LENGTH = 5000.0
IMAGE_SIZE = 256

DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PALM_SCALE_MULTIPLIER = 4.0
DETECT_SCALE = 0.5

# --- VISUALIZER ---
class Visualizer3D:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_proj_type('ortho')
        
        # View: Looking from slightly above
        self.ax.view_init(elev=20, azim=135) 
        self.ax.set_box_aspect([1, 1, 1])
        
        # --- UPDATED BONES FOR 22 POINTS ---
        # 0: Forearm, 1: Wrist
        # Previous 0 becomes 1, Previous 1 becomes 2, etc.
        self.bones = [
            (0, 1),   # Forearm -> Wrist (NEW)
            
            # Thumb (Wrist -> CMC -> MCP -> IP -> Tip)
            (1, 2), (2, 3), (3, 4), (4, 5),
            
            # Index
            (1, 6), (6, 7), (7, 8), (8, 9),
            
            # Middle
            (1, 10), (10, 11), (11, 12), (12, 13),
            
            # Ring
            (1, 14), (14, 15), (15, 16), (16, 17),
            
            # Pinky
            (1, 18), (18, 19), (19, 20), (20, 21)
        ]

    def update(self, hands_dict):
        # Optimization: If no hands, don't clear/redraw to save flicker
        if not hands_dict:
            plt.pause(0.001)
            return

        self.ax.cla()
        
        # Fixed limits to stop the camera from jumping around
        self.ax.set_xlim(0.0, 1.0)  # Forward
        self.ax.set_ylim(-0.5, 0.5) # Left/Right
        self.ax.set_zlim(-0.5, 0.5) # Up/Down
        
        self.ax.set_xlabel('X (Forward)')
        self.ax.set_ylabel('Y (Left)')
        self.ax.set_zlabel('Z (Up)')
        
        # Draw Head
        self.ax.scatter([0], [0], [0], c='black', marker='o', s=100, label="Head")

        for label, skeleton in hands_dict.items():
            xs = skeleton[:, 0]
            ys = skeleton[:, 1]
            zs = skeleton[:, 2]
            
            color = 'green' if "Right" in label else 'red'
            
            # Draw Joints
            self.ax.scatter(xs, ys, zs, c=color, marker='o', s=20)
            
            # Draw Bones
            for (start, end) in self.bones:
                self.ax.plot(
                    [xs[start], xs[end]],
                    [ys[start], ys[end]],
                    [zs[start], zs[end]],
                    color=color,
                )
            
            # Label the Wrist (Index 1)
            self.ax.text(xs[1], ys[1], zs[1], label, fontsize=8)

        plt.draw()
        # Shortest possible pause to allow GUI update
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


# --- FILTERS ---
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
    
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.
    
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
        Representation: (w, x, y, z)
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
        """
        Returns sqrt(max(0, x))
        """
        ret = torch.sqrt(torch.clamp(x, min=0))
        return ret

    # We compute 4 potential solutions (candidates) to avoid numerical instability
    # near singularities. We will pick the one with the largest diagonal term.
    
    # 1. Solution based on Trace (Good for small angles)
    trace = m00 + m11 + m22
    sq_trace = _sqrt_positive_part(1 + trace) * 0.5
    
    # 2. Solution based on m00 (Good if rotation is mostly around X)
    sq_m00 = _sqrt_positive_part(1 + m00 - m11 - m22) * 0.5
    
    # 3. Solution based on m11 (Good if rotation is mostly around Y)
    sq_m11 = _sqrt_positive_part(1 - m00 + m11 - m22) * 0.5
    
    # 4. Solution based on m22 (Good if rotation is mostly around Z)
    sq_m22 = _sqrt_positive_part(1 - m00 - m11 + m22) * 0.5

    # Stack the four candidates: Shape (4, Batch)
    # We construct the full quaternion for all 4 cases
    
    # Case 0: Trace is largest (w is dominant)
    quat0 = torch.stack([sq_trace, 
                         (m21 - m12) / (4 * sq_trace + 1e-8), 
                         (m02 - m20) / (4 * sq_trace + 1e-8), 
                         (m10 - m01) / (4 * sq_trace + 1e-8)], dim=-1)

    # Case 1: m00 is largest (x is dominant)
    quat1 = torch.stack([(m21 - m12) / (4 * sq_m00 + 1e-8), 
                         sq_m00, 
                         (m01 + m10) / (4 * sq_m00 + 1e-8), 
                         (m02 + m20) / (4 * sq_m00 + 1e-8)], dim=-1)

    # Case 2: m11 is largest (y is dominant)
    quat2 = torch.stack([(m02 - m20) / (4 * sq_m11 + 1e-8), 
                         (m01 + m10) / (4 * sq_m11 + 1e-8), 
                         sq_m11, 
                         (m12 + m21) / (4 * sq_m11 + 1e-8)], dim=-1)
    
    # Case 3: m22 is largest (z is dominant)
    quat3 = torch.stack([(m10 - m01) / (4 * sq_m22 + 1e-8), 
                         (m02 + m20) / (4 * sq_m22 + 1e-8), 
                         (m12 + m21) / (4 * sq_m22 + 1e-8), 
                         sq_m22], dim=-1)

    # Now, explicitly select the best candidate per batch item
    candidates = torch.stack([quat0, quat1, quat2, quat3], dim=0) # (4, Batch, 4)
    
    # Find which denominator was largest
    vals = torch.stack([sq_trace, sq_m00, sq_m11, sq_m22], dim=0) # (4, Batch)
    best_idx = torch.argmax(vals, dim=0) # (Batch,)

    # Gather the best quaternion for each element in the batch
    # This is effectively: result[b] = candidates[best_idx[b], b, :]
    
    # Advanced gather logic for efficient GPU usage:
    batch_indices = torch.arange(matrix.shape[0], device=matrix.device)
    quaternions = candidates[best_idx, batch_indices, :]
    
    return quaternions
    

# --- HEAD POSE ---
class HeadPoseEstimator:
    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.last_ts = None
        
        # State: Euler angles in radians
        # Frame: X=Forward, Y=Left, Z=Up
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0 

        # --- CALIBRATION CONFIG ---
        self.is_calibrated = False
        self.calib_buffer_accel = []
        self.CALIB_FRAMES = 20 # Number of IMU packets to average on start

    def update(self, accel, gyro, timestamp):
        """
        Updates the internal state. fast. Does NOT return a matrix.
        Call this in your batch loop.
        """
        if self.last_ts is None:
            self.last_ts = timestamp
            return

        dt = timestamp - self.last_ts
        self.last_ts = timestamp

        # --- PHASE 1: CALIBRATION ---
        if not self.is_calibrated:
            self._run_calibration(accel)
            return

        # --- PHASE 2: NORMAL RUNNING ---
        
        # 1. Integrate Gyro
        self.roll  += gyro[0] * dt 
        self.pitch += gyro[1] * dt
        self.yaw   += gyro[2] * dt

        # 2. Accel Corrections (Gravity Vector)
        acc_pitch = math.atan2(accel[0], accel[2])
        acc_roll = math.atan2(accel[1], accel[2])

        # 3. Complementary Filter
        self.pitch = self.alpha * self.pitch + (1 - self.alpha) * acc_pitch
        self.roll  = self.alpha * self.roll  + (1 - self.alpha) * acc_roll

    def _run_calibration(self, accel):
        """Internal helper to handle startup calibration."""
        self.calib_buffer_accel.append(accel)
        
        if len(self.calib_buffer_accel) >= self.CALIB_FRAMES:
            # Average the noise
            avg = np.mean(self.calib_buffer_accel, axis=0)
            
            # Snap to reality
            self.pitch = math.atan2(avg[0], avg[2])
            self.roll  = math.atan2(avg[1], avg[2])
            self.yaw   = 0.0 # Reset Yaw to 0 (Forward) on start
            
            self.is_calibrated = True
            print(f"Calibration Done. Pitch: {math.degrees(self.pitch):.1f}, Roll: {math.degrees(self.roll):.1f}")

    def get_rotation_matrix(self):
        """
        Constructs the matrix. Heavy. Call this ONLY when you are ready to render.
        """
        # If not calibrated yet, return Identity (no rotation)
        if not self.is_calibrated:
            return np.eye(3)

        c_y = math.cos(self.yaw)
        s_y = math.sin(self.yaw)
        c_p = math.cos(self.pitch)
        s_p = math.sin(self.pitch)
        c_r = math.cos(self.roll)
        s_r = math.sin(self.roll)

        # R_x (Roll)
        R_x = np.array([[1, 0, 0], 
                        [0, c_r, -s_r], 
                        [0, s_r, c_r]])
        
        # R_y (Pitch)
        R_y = np.array([[c_p, 0, s_p], 
                        [0, 1, 0], 
                        [-s_p, 0, c_p]])
        
        # R_z (Yaw)
        R_z = np.array([[c_y, -s_y, 0], 
                        [s_y, c_y, 0], 
                        [0, 0, 1]])
        
        # Z-Y-X Sequence
        return R_z @ R_y @ R_x
    

# --- ARIA LISTENER ---
class AriaImageListener:
    def __init__(self):
        self.rgb_image = None
        self.lock = threading.Lock()
        
        # CHANGED: Instead of storing one value, we store a list (queue)
        # Format: List of tuples (accel_np, gyro_np, timestamp_sec)
        self.imu_queue = [] 

    def on_image_received(self, image, record):
        if record.camera_id == aria_sdk.CameraId.Rgb:
            with self.lock:
                # We still only want the LATEST image, so overwriting here is fine
                self.rgb_image = image.copy()

    def on_imu_received(self, samples, imu_idx):
        # FILTER: Only use the Right IMU (Index 0)
        if imu_idx != 0:
            return

        if not samples: return
        
        new_data = []
        for s in samples:
            acc = np.array(s.accel_msec2)
            gyr = np.array(s.gyro_radsec)
            ts = s.capture_timestamp_ns * 1e-9
            new_data.append((acc, gyr, ts))

        with self.lock:
            self.imu_queue.extend(new_data)
    
    # Boilerplate overrides
    def onImuReceived(self, samples, imu_idx): self.on_imu_received(samples, imu_idx)
    def onImageReceived(self, image, record): self.on_image_received(image, record)
    
    def get_data_queue(self):
        """
        Returns:
        1. The latest RGB image (or None)
        2. A LIST of all IMU packets received since the last call.
        """
        with self.lock:
            img = None if self.rgb_image is None else self.rgb_image.copy()
            
            # COPY the current queue to return it, then CLEAR the internal queue
            # This ensures we don't process the same IMU data twice.
            imu_batch = list(self.imu_queue)
            self.imu_queue.clear()
            
            return img, imu_batch
        

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
    sub_config.message_queue_size[aria_sdk.StreamingDataType.Rgb] = 2
    sub_config.message_queue_size[aria_sdk.StreamingDataType.Imu] = 100

    options = aria_sdk.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    sub_config.security_options = options
    streaming_client.subscription_config = sub_config

    listener_logic = AriaImageListener()
    streaming_client.set_streaming_client_observer(listener_logic)
    streaming_client.subscribe()

    return device_client, device, streaming_manager, streaming_client, listener_logic


def report(label, t_start, t_end):
        dt = t_end - t_start
        # if dt > 0.2:  # 200 ms is already pretty bad for teleop
        print(f"[STALL] {label}: {dt*1000:.1f} ms")


def main():
    device = torch.device('cuda')
    print(f"--> Engine: Hamba (OFLEX) on {device}")

    model, model_cfg = load_hamba(CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()

    # --- WARMUP (CRITICAL STEP) ---
    # We run a fake batch through the model NOW so the compilation happens 
    # while the app is loading, not when the user is waiting.
    print("Warming up model...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            # Create a dummy input matching your batch shape
            # Shape: (Batch=1, Channels=3, Height=256, Width=256) -> Check your model's expected size!
            # Assuming Hamba uses 256x256 or similar. Adjust if needed.
            dummy_img = torch.randn(1, 3, 256, 256).to(device)
            dummy_right = torch.tensor([1.0]).to(device) # Dummy 'is_right' flag
            
            dummy_batch = {"img": dummy_img, "right": dummy_right}
            
            # This call triggers the compilation
            _ = model(dummy_batch)

    print("--> Detector: MediaPipe Hands")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        model_complexity=0,
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
        ARIA_IP, profile_name="profile5"
    )

    print("--> Fetching Device Calibration...")
    # 1. Get the factory calibration JSON from the device
    sensors_calib_json = streaming_manager.sensors_calibration()
    sensors_calib = device_calibration_from_json_string(sensors_calib_json)
    
    # 2. Get the specific Src Calibration for the RGB Camera
    src_calib = sensors_calib.get_camera_calib("camera-rgb")

    # 3. Define the Dst (Target) Calibration
    # We want a Linear (Pinhole) camera.
    # Width/Height: 1024 (downsampled pinhole)
    # Focal length is recomputed from TARGET_FOV
    dst_calib = get_linear_camera_calibration(TARGET_WIDTH, TARGET_HEIGHT, FOCAL_LENGTH, "camera-rgb")

    head_pose_estimator = HeadPoseEstimator(alpha=0.98)
    udp_sender = UDPSender()

    rot_mat_tensor = torch.tensor([
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0]
            ], dtype=torch.float32, device=device)
    rot_mat_T = rot_mat_tensor.T

    if VISUALIZATION:
        print("--> Starting 3D Visualizer...")
        vis3d = Visualizer3D()

    t6 = time.time()
    t7 = time.time()

    try:
        with torch.no_grad():
            while True:

                t0 = time.time()
                raw_img, imu_batch = aria_observer.get_data_queue()
                t1 = time.time()

                if imu_batch:
                    for (accel, gyro, ts) in imu_batch:
                        
                        # --- MAPPING: Aria Right IMU -> HeadPoseEstimator ---
                        # Input (Aria): [0:Down, 1:Right, 2:Back]
                        # Target (Est): [0:Fwd,  1:Left,  2:Up]

                        accel_mapped = np.array([
                            -accel[2],  # Est X (Fwd)  = -Aria Z (Back)
                            -accel[1],  # Est Y (Left) = -Aria Y (Right)
                            -accel[0]   # Est Z (Up)   = -Aria X (Down) -> Gravity is now correctly on Z!
                        ])

                        # Gyro must follow the exact same permutation
                        gyro_mapped = np.array([
                            -gyro[2],   # Roll (around Fwd)
                            -gyro[1],   # Pitch (around Left)
                            -gyro[0]    # Yaw (around Up)
                        ])

                        # Update the estimator
                        head_pose_estimator.update(accel_mapped, gyro_mapped, ts)
                t2 = time.time()

                if raw_img is None:
                    time.sleep(0.001)
                    continue

                # 2. GET RESULT (Once per frame)
                # Now that the math is up to date, we calculate the matrix ONCE.
                R_head = head_pose_estimator.get_rotation_matrix()
                t3 = time.time()

                # --- OFFICIAL UNDISTORTION ---
                # Transforms the Fisheye 'raw_img' into a Linear Pinhole image
                # using the exact parameters of your device.
                # Note: raw_img is usually RGB from the SDK, but check orientation.
                img_undistorted = distort_by_calibration(raw_img, dst_calib, src_calib)
                img_rgb = np.rot90(img_undistorted, -1)
                t4 = time.time()

                if DISPLAY:
                    # Show the Rectified Image (The one MediaPipe is processing)
                    # Convert to BGR for OpenCV
                    debug_frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Rectified (Input to AI)", debug_frame)

                # --- STEP 1: CREATE LOW-RES COPY ---
                # We use 0.5 scale (704x704). This is 4x faster to process (pixel count).
                # CRITICAL: We maintain square aspect ratio!
                input_h, input_w = img_rgb.shape[:2]
                detect_w, detect_h = int(input_w * DETECT_SCALE), int(input_h * DETECT_SCALE)

                # Resize strictly for detection. 
                # INTER_LINEAR is fast and good enough for detection.
                img_detection = cv2.resize(img_rgb, (detect_w, detect_h), interpolation=cv2.INTER_LINEAR)

                # --- PREPARE FOR PIPELINE ---
                # Detect Hands
                results = hands.process(img_detection)

                batch_images = []
                batch_is_right = []
                visual_info = []

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

                        crop_rect_params = compute_crop_coords(c_x, c_y, raw_s, img_rgb.shape)
                        crop_img = crop_image(img_rgb, crop_rect_params)

                        # Flip Left Hand Horizontally to Match Right Hand Orientation
                        if label_text == "Left":
                            crop_img = cv2.flip(crop_img, 1) # 1 = Horizontal Flip
                        else:
                            crop_img = crop_img

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
                                "img_size": img_rgb.shape[:2],
                            })
                t5 = time.time()

                # 4. Inference
                if len(batch_images) > 0:
                    tensor_batch = torch.from_numpy(np.stack(batch_images)).to(device, non_blocking=True)
                    is_right_batch = torch.tensor(batch_is_right, dtype=torch.float32).to(device, non_blocking=True)
                    batch = {"img": tensor_batch, "right": is_right_batch}

                    with torch.no_grad():
                        with torch.amp.autocast('cuda'):
                            out = model(batch)

                    pred_verts = out['pred_vertices'] # Shape: (Batch, 778, 3)
                    j_regressor = model.mano.J_regressor.to(device) # Shape: (16, 778)

                    # 2. Get the 16 Skeletal Joints (Regressed)
                    # Shape: (Batch, 16, 3)
                    regressed_joints = torch.matmul(j_regressor, pred_verts)

                    # 3. Get the 5 Fingertips (From Vertices)
                    # Thumb: 744, Index: 320, Middle: 443, Ring: 555, Pinky: 672
                    tip_indices = [744, 320, 443, 555, 672]
                    fingertip_verts = pred_verts[:, tip_indices, :] # Shape: (Batch, 5, 3)

                    joints_3d = torch.stack([
                        # --- WRIST ---
                        regressed_joints[:, 0],   # 0: Wrist
                        
                        # --- THUMB (CMC, MCP, IP, Tip) ---
                        regressed_joints[:, 13],  # 1: Thumb CMC (Mano ID 13)
                        regressed_joints[:, 14],  # 2: Thumb MCP (Mano ID 14)
                        regressed_joints[:, 15],  # 3: Thumb IP  (Mano ID 15)
                        fingertip_verts[:, 0],    # 4: Thumb Tip (Vertex 744)

                        # --- INDEX (MCP, PIP, DIP, Tip) ---
                        regressed_joints[:, 1],   # 5: Index MCP (Mano ID 1)
                        regressed_joints[:, 2],   # 6: Index PIP (Mano ID 2)
                        regressed_joints[:, 3],   # 7: Index DIP (Mano ID 3)
                        fingertip_verts[:, 1],    # 8: Index Tip (Vertex 320)

                        # --- MIDDLE (MCP, PIP, DIP, Tip) ---
                        regressed_joints[:, 4],   # 9: Middle MCP (Mano ID 4)
                        regressed_joints[:, 5],   # 10: Middle PIP (Mano ID 5)
                        regressed_joints[:, 6],   # 11: Middle DIP (Mano ID 6)
                        fingertip_verts[:, 2],    # 12: Middle Tip (Vertex 443)

                        # --- RING (MCP, PIP, DIP, Tip) ---
                        regressed_joints[:, 10],  # 13: Ring MCP (Mano ID 10)
                        regressed_joints[:, 11],  # 14: Ring PIP (Mano ID 11)
                        regressed_joints[:, 12],  # 15: Ring DIP (Mano ID 12)
                        fingertip_verts[:, 3],    # 16: Ring Tip (Vertex 555)

                        # --- PINKY (MCP, PIP, DIP, Tip) ---
                        regressed_joints[:, 7],   # 17: Pinky MCP (Mano ID 7)
                        regressed_joints[:, 8],   # 18: Pinky PIP (Mano ID 8)
                        regressed_joints[:, 9],   # 19: Pinky DIP (Mano ID 9)
                        fingertip_verts[:, 4],    # 20: Pinky Tip (Vertex 672)
                    ], dim=1)

                    wrist_position = out["pred_cam_t"]
                    wrist_orientation_matrix = out['pred_mano_params']['global_orient'].squeeze(1)

                    # --- START: FOCAL LENGTH CORRECTION (VECTORIZED) ---

                    # 1. Pre-process list data into Tensors (Do this once per batch)
                    # Assuming visual_info is a list of dicts. We extract data for the whole batch.
                    labels = [info["label"] for info in visual_info]
                    scales = torch.tensor([info["scale"] for info in visual_info], device=device)
                    centers = torch.tensor([info["center"] for info in visual_info], device=device) # Shape (N, 2)

                    # 2. Create a mask for "Right" hands
                    right_hand_mask = torch.tensor([l == "Right" for l in labels], device=device, dtype=torch.bool)

                    # 3. Only proceed if there is at least one right hand to fix
                    if right_hand_mask.any():
                        # Extract only the data we need using the mask
                        # t_coords shape: (M, 3) where M is number of "Right" samples
                        t_coords = wrist_position[right_hand_mask] 
                        t_x, t_y, t_z = t_coords[:, 0], t_coords[:, 1], t_coords[:, 2]

                        img_h, img_w = img_rgb.shape[:2]
                        pp_x = img_w / 2.0
                        pp_y = img_h / 2.0
                        
                        masked_scales = scales[right_hand_mask]
                        masked_cx = centers[right_hand_mask, 0] - pp_x
                        masked_cy = centers[right_hand_mask, 1] - pp_y

                        # 4. Perform Vectorized Math (Computes all M samples at once)
                        # Pre-calculate common ratio to simplify equations
                        # Ratio represents the scaling difference between real and virtual setup
                        # Formula: Z_real = Z_virt * (F_real / F_virt) * (ImgSize / Scale)
                        z_factor = (FOCAL_LENGTH / VIRTUAL_FOCAL_LENGTH) * (IMAGE_SIZE / masked_scales)
                        z_real = t_z * z_factor

                        # Common projection term: (Z_real / F_real)
                        proj_factor = z_real / FOCAL_LENGTH

                        # Re-projection logic
                        # X_real = (Z_real / F_real) * ( x_virt_projected + c_x )
                        scale_ratio = masked_scales / IMAGE_SIZE
                        
                        # Calculate X and Y
                        # Note: t_x / t_z is the normalized image coordinate
                        x_term = (VIRTUAL_FOCAL_LENGTH * (t_x / t_z)) * scale_ratio
                        x_real = proj_factor * (x_term + masked_cx)

                        y_term = (VIRTUAL_FOCAL_LENGTH * (t_y / t_z)) * scale_ratio
                        y_real = proj_factor * (y_term + masked_cy)

                        # 5. Assign back to original tensor
                        # We stack the results and place them back into the wrist_position tensor
                        # only at the indices where the mask is True.
                        corrected_pos = torch.stack([x_real, y_real, z_real], dim=1).to(wrist_position.dtype)
                        wrist_position[right_hand_mask] = corrected_pos

                    # --- END FOCAL LENGTH CORRECTION ---


                    # --- START: FLIP BACK LEFT HANDS ---
                    
                    # 1. Create a boolean mask for Left Hands
                    # batch['right'] is 1 for Right, 0 for Left. 
                    # We want indices where it is < 0.5 (meaning 0)
                    is_left_mask = batch['right'] < 0.5 

                    if is_left_mask.any():
                        # 2. Flip Position X-Axis
                        # Multiply X (index 0) by -1 for only the Left Hands
                        
                        # Flip Joints (Batch, 21, 3)
                        joints_3d[is_left_mask, :, 0] *= -1
                        
                        # Flip Wrist Position (Batch, 3)
                        wrist_position[is_left_mask, 0] *= -1

                        # 3. Flip Orientation Matrix
                        # To reflect a rotation matrix across the X-axis (YZ plane), 
                        # we calculate R_new = M @ R_old @ M, where M is diag(-1, 1, 1).
                        # Mathematically, this just means negating specific cells:
                        # [ r00  r01  r02 ]      [  r00  -r01  -r02 ]
                        # [ r10  r11  r12 ]  ->  [ -r10   r11   r12 ]
                        # [ r20  r21  r22 ]      [ -r20   r21   r22 ]

                        # We negate row 0 (excluding 0,0) and column 0 (excluding 0,0)
                        
                        # Negate elements [0,1] and [0,2]
                        wrist_orientation_matrix[is_left_mask, 0, 1] *= -1
                        wrist_orientation_matrix[is_left_mask, 0, 2] *= -1
                        
                        # Negate elements [1,0] and [2,0]
                        wrist_orientation_matrix[is_left_mask, 1, 0] *= -1
                        wrist_orientation_matrix[is_left_mask, 2, 0] *= -1

                    # --- END FLIP BACK ---

                    
                    # --- START: ROTATE TO TARGET FRAME ---

                    # Use the pre-calculated Transpose for points
                    joints_3d = torch.matmul(joints_3d, rot_mat_T)
                    wrist_position = torch.matmul(wrist_position, rot_mat_T)
                    
                    # Use the normal matrix for orientation
                    wrist_orientation_matrix = torch.matmul(rot_mat_tensor, wrist_orientation_matrix)

                    # --- END ROTATE TO TARGET FRAME ---


                    # --- START: HEAD MOVEMENT COMPENSATION ---

                    # 1. Convert R_head to Tensor ONCE
                    # Note: R_head is float64 in numpy, PyTorch usually wants float32
                    R_head_tensor = torch.from_numpy(R_head).float().to(device)

                    # 2. Apply transformations
                    # For points: We need the Transpose (R_head.T)
                    # matmul(A, B) -> A @ B
                    joints_3d = torch.matmul(joints_3d, R_head_tensor.T)
                    wrist_position = torch.matmul(wrist_position, R_head_tensor.T)

                    # For orientation matrix: We use normal R_head
                    wrist_orientation_matrix = torch.matmul(R_head_tensor, wrist_orientation_matrix)

                    # --- END HEAD MOVEMENT COMPENSATION ---


                    # --- START: CONVERT TO QUATERNION ---
                    
                    # Input: wrist_orientation_matrix (Batch, 3, 3)
                    # Output: wrist_quat (Batch, 4) in format [w, x, y, z]
                    wrist_orientation_quat = matrix_to_quaternion(wrist_orientation_matrix)

                    # --- END CONVERT TO QUATERNION ---

                    
                    # --- START: VISUALIZATION PREP ---
                    if VISUALIZATION:
                        hands_to_plot = {}
                        
                        # 1. Calculate Global Position (Broadcasting)
                        # joints_3d: (Batch, 21, 3)
                        global_skeleton_tensor = joints_3d + wrist_position.unsqueeze(1)
                        
                        # 2. Move to CPU/Numpy for Matplotlib
                        global_skeleton_np = global_skeleton_tensor.detach().cpu().numpy()
                        
                        # 3. Iterate over the batch to populate dictionary
                        # batch_is_right comes from your earlier loop
                        for i in range(len(batch_images)):
                            # Determine label
                            is_r = batch_is_right[i] > 0.5 # Assuming 1.0 is Right
                            label = f"Right_{i}" if is_r else f"Left_{i}"
                            
                            hands_to_plot[label] = global_skeleton_np[i]

                        # 4. Call Update
                        # Note: This might slow down your loop to ~15 FPS due to Matplotlib
                        vis3d.update(hands_to_plot)
                        
                    # --- END VISUALIZATION PREP ---
                    t6 = time.time()


                    # --- START: EXPORT AND VISUALIZE ---
                    
                    # Move Batch to CPU (Perform this sync once for both Viz and UDP)
                    # We use .detach().cpu().numpy() to get raw python-friendly numbers
                    batch_skeleton_np = joints_3d.detach().cpu().numpy()
                    batch_wrist_pos_np = wrist_position.detach().cpu().numpy()
                    batch_wrist_quat_np = wrist_orientation_quat.detach().cpu().numpy()

                    # 3. Iterate through the batch
                    for i in range(len(batch_images)):
                        # Determine label (Right vs Left)
                        is_right = batch_is_right[i] > 0.5
                        label = "Right" if is_right else "Left"
                        
                        # Extract individual data for this hand
                        current_skeleton = batch_skeleton_np[i]   # Shape (21, 3)
                        current_pos = batch_wrist_pos_np[i]       # Shape (3,)
                        current_quat = batch_wrist_quat_np[i]     # Shape (4,) [w, x, y, z]

                        # --- A. UDP SENDING ---
                        # Logic: Only send "Right" hand (or modify to send both if needed)
                        if label == "Right":
                            
                            # Fix Quaternion Order: [w, x, y, z] -> [x, y, z, w]
                            # This is crucial for Unity / ROS / Standard Game Engines
                            quat_xyzw = current_quat[[1, 2, 3, 0]]
                            
                            # Print debug info (Optional)
                            if VISUALIZATION:
                                print(f"Sending Right Hand: Pos {current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}")
                                print(f"Quat {quat_xyzw[0]:.3f}, {quat_xyzw[1]:.3f}, {quat_xyzw[2]:.3f}, {quat_xyzw[3]:.3f}")
                            
                            # Send via UDP
                            udp_sender.send_data(current_skeleton, current_pos, quat_xyzw)
                            t7 = time.time()

                if cv2.waitKey(1) & 0xFF == ord('q'): break

                report("queue", t0, t1)
                report("imu", t1, t2)
                report("R_head", t2, t3)
                report("distort", t3, t4)
                report("mediapipe", t4, t5)
                report("hamba", t5, t6)
                report("cpu+udp", t6, t7)

    except KeyboardInterrupt:
        print("\nCtrl+C pressed. Stopping...")

    except Exception as e:
        # Catch other runtime errors so we still clean up the glasses
        print(f"\nRuntime Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("Cleaning up resources...")
        
        # 1. Stop Aria Streaming (Most Important)
        if streaming_client:
            try: 
                streaming_client.unsubscribe()
                print("Unsubscribed.")
            except Exception as e: print(f"Error unsubscribing: {e}")

        if streaming_manager:
            try: 
                streaming_manager.stop_streaming()
                print("Streaming stopped.")
            except Exception as e: print(f"Error stopping stream: {e}")

        if device_client and aria_device:
            try: 
                device_client.disconnect(aria_device)
                print("Aria disconnected.")
            except Exception as e: print(f"Error disconnecting: {e}")

        # 2. Close UI Windows
        try:
            cv2.destroyAllWindows()
            plt.close('all') # Close the matplotlib window
            print("UI closed.")
        except: 
            pass
            
        print("Cleanup Complete. Exiting.")

if __name__ == "__main__":
    main()