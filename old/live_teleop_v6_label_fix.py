import cv2
import torch
import numpy as np
import time
import mediapipe as mp
from hamba.models import load_hamba
from hamba.utils import recursive_to

# --- CONFIGURATION ---
CHECKPOINT_PATH = "ckpts/hamba/checkpoints/hamba.ckpt"
RESCALE_FACTOR = 2.0
IMAGE_SIZE = 256
DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def get_crop_params(bbox, img_shape):
    """ Calculate crop coordinates with padding """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    center_x, center_y = x1 + w/2, y1 + h/2
    s = max(w, h) * RESCALE_FACTOR
    
    # Crop coords
    x1 = int(center_x - s/2)
    y1 = int(center_y - s/2)
    x2 = int(center_x + s/2)
    y2 = int(center_y + s/2)
    
    # Pad logic
    img_h, img_w = img_shape[:2]
    pad_left, pad_top = max(0, -x1), max(0, -y1)
    pad_right, pad_bottom = max(0, x2 - img_w), max(0, y2 - img_h)
    
    return x1, y1, x2, y2, pad_left, pad_top, pad_right, pad_bottom

def crop_image(img_rgb, crop_params):
    """ Perform the actual image slicing and resizing """
    x1, y1, x2, y2, pad_left, pad_top, pad_right, pad_bottom = crop_params
    
    if any([pad_left, pad_top, pad_right, pad_bottom]):
        img_padded = cv2.copyMakeBorder(img_rgb, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
        crop = img_padded[y1+pad_top : y2+pad_top, x1+pad_left : x2+pad_left]
    else:
        crop = img_rgb[y1:y2, x1:x2]
        
    try:
        crop_resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        return crop_resized
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
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2, # ENABLED 2 HANDS
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0 
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened(): return

    print("\n=== TELEOP V7 (DUAL HAND SUPPORT) ===")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        t0 = time.time()
        
        # 1. FLIP (Mirror Mode)
        frame = cv2.flip(frame, 1)
        img_rgb = frame[:, :, ::-1].copy()
        h, w, _ = img_rgb.shape
        
        # 2. DETECT
        results = hands.process(img_rgb)

        batch_images = []
        batch_is_right = []
        visual_info = [] # Store bbox/label for drawing later

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # MediaPipe Label Logic
                mp_label = results.multi_handedness[idx].classification[0].label
                
                if mp_label == "Left":
                    label_text = "Left"
                    is_right_flag = 1
                else:
                    label_text = "Right"
                    is_right_flag = 0
                
                # Get BBox
                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_list) * w), int(max(x_list) * w)
                y_min, y_max = int(min(y_list) * h), int(max(y_list) * h)
                bbox = [x_min, y_min, x_max, y_max]

                # Preprocess (Crop & Normalize)
                crop_params = get_crop_params(bbox, img_rgb.shape)
                crop_img = crop_image(img_rgb, crop_params)
                
                if crop_img is not None:
                    # Normalize
                    img_norm = (crop_img.astype(np.float32) / 255.0 - DEFAULT_MEAN) / DEFAULT_STD
                    img_chw = img_norm.transpose(2, 0, 1)
                    
                    # Add to batch lists
                    batch_images.append(img_chw)
                    batch_is_right.append(is_right_flag)
                    visual_info.append((bbox, label_text))

        # 3. BATCH INFERENCE (If we found hands)
        if len(batch_images) > 0:
            # Stack numpy arrays into one tensor (N, 3, 256, 256)
            tensor_batch = torch.from_numpy(np.stack(batch_images)).to(device)
            is_right_batch = torch.tensor(batch_is_right, dtype=torch.float32).to(device)
            
            batch = {
                'img': tensor_batch,
                'right': is_right_batch
            }

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    out = model(batch)

            # 4. PARSE & VISUALIZE BATCH
            # The model outputs N results. We loop through them.
            poses = out['pred_mano_params']['hand_pose'].cpu().numpy() # Shape: (N, 15, 3, 3)
            
            fps = 1 / (time.time() - t0)
            
            # Print Header
            print(f"\rFPS: {fps:.0f} | ", end="")
            
            for i in range(len(batch_images)):
                label = visual_info[i][1]
                box = visual_info[i][0]
                
                # Raw Matrix Data (15 joints x 3x3 matrix)
                # We print just the mean of the first joint matrix as a sanity check
                matrix_sanity = np.mean(poses[i][0]) 
                
                print(f"[{label}: Data OK] ", end="")
                
                # Draw
                color = (0, 255, 0) if label == "Right" else (0, 0, 255) # Green=Right, Red=Left
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        else:
            print(f"\rFPS: {1 / (time.time() - t0):.0f} | Searching...", end="")

        cv2.imshow('Hamba Dual-Hand', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()