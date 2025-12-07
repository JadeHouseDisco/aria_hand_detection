import cv2
import torch
import numpy as np
import time
import mediapipe as mp
import math
from hamba.models import load_hamba
from hamba.utils import recursive_to

# --- CONFIGURATION ---
CHECKPOINT_PATH = "ckpts/hamba/checkpoints/hamba.ckpt"
RESCALE_FACTOR = 2.0
IMAGE_SIZE = 256
DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- 1 EURO FILTER CLASS ---
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
        if t_e <= 0: return self.x_prev # Avoid divide by zero on fast loops

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

# Helper functions
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

def main():
    device = torch.device('cuda')
    print(f"--> Engine: Hamba (OFLEX) on {device}")
    
    model, model_cfg = load_hamba(CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()

    print("--> Detector: MediaPipe Hands (2-Hand Mode)")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                           min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)

    # Initialize FILTERS (One for Left, One for Right)
    # Tune 'beta' to control lag vs jitter. 
    # Higher beta = More responsive, more jitter. Lower beta = Smoother, more lag.
    # 0.05 is a good starting point for hands.
    filters = {
        "Left": OneEuroFilter(min_cutoff=0.5, beta=0.05), 
        "Right": OneEuroFilter(min_cutoff=0.5, beta=0.05)
    }

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened(): return

    print("\n=== TELEOP V8 (SMOOTHED) ===")
    
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
                
                # [USER FIX applied]
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
                    visual_info.append((bbox, label_text))

        if len(batch_images) > 0:
            tensor_batch = torch.from_numpy(np.stack(batch_images)).to(device)
            is_right_batch = torch.tensor(batch_is_right, dtype=torch.float32).to(device)
            batch = {'img': tensor_batch, 'right': is_right_batch}

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    out = model(batch)

            # [NEW] Get 3D Keypoints instead of Pose Angles
            # Hamba calculates 3D positions of joints relative to the camera/root
            keypoints_3d = out['pred_keypoints_3d'].detach().cpu().numpy() # Expected Shape: (N, 21, 3)
            
            fps = 1 / (time.time() - t0)
            print(f"\rFPS: {fps:.0f} | ", end="")
            
            for i in range(len(batch_images)):
                label = visual_info[i][1]
                box = visual_info[i][0]
                
                # Get the raw skeleton for this hand
                raw_skeleton = keypoints_3d[i] # Shape: (21, 3)
                
                # DEBUG: Print the shape to confirm what Hamba gives us
                print(f"[{label} Hand Output Shape: {raw_skeleton.shape}] ", end="")
                
                # --- FILTERING STEP (Updated for XYZ) ---
                # 1. Flatten (21*3 = 63 numbers)
                flat_skeleton = raw_skeleton.flatten()
                # 2. Filter
                smoothed_skeleton_flat = filters[label](flat_skeleton)
                # 3. Reshape back to (21, 3)
                smoothed_skeleton = smoothed_skeleton_flat.reshape(-1, 3)
                
                color = (0, 255, 0) if label == "Right" else (0, 0, 255)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        else:
            print(f"\rFPS: {1 / (time.time() - t0):.0f} | Searching...", end="")

        cv2.imshow('Hamba Teleop V8 (Smoothed)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()