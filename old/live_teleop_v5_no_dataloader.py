import cv2
import torch
import numpy as np
import time
import mediapipe as mp
from hamba.models import load_hamba
from hamba.utils.renderer import cam_crop_to_full

# --- CONFIGURATION ---
CHECKPOINT_PATH = "ckpts/hamba/checkpoints/hamba.ckpt"
RESCALE_FACTOR = 2.0
IMAGE_SIZE = 256
DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_frame(img_rgb, bbox, device):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    center_x, center_y = x1 + w/2, y1 + h/2
    s = max(w, h) * RESCALE_FACTOR
    
    # Crop coords
    x1, y1 = int(center_x - s/2), int(center_y - s/2)
    x2, y2 = int(center_x + s/2), int(center_y + s/2)
    
    # Pad logic
    img_h, img_w = img_rgb.shape[:2]
    pad_left, pad_top = max(0, -x1), max(0, -y1)
    pad_right, pad_bottom = max(0, x2 - img_w), max(0, y2 - img_h)
    
    if any([pad_left, pad_top, pad_right, pad_bottom]):
        img_padded = cv2.copyMakeBorder(img_rgb, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
        crop = img_padded[y1+pad_top : y2+pad_top, x1+pad_left : x2+pad_left]
    else:
        crop = img_rgb[y1:y2, x1:x2]
        
    try:
        crop_resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    except:
        return None, None

    img_tensor = (crop_resized.astype(np.float32) / 255.0 - DEFAULT_MEAN) / DEFAULT_STD
    img_tensor = img_tensor.transpose(2, 0, 1) # CHW
    tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device)
    
    # Meta data for potential 3D projection
    box_center = torch.tensor([[center_x, center_y]], dtype=torch.float32).to(device)
    box_size = torch.tensor([s], dtype=torch.float32).to(device)
    img_size = torch.tensor([[img_h, img_w]], dtype=torch.float32).to(device)
    
    return tensor, (box_center, box_size, img_size)

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
        max_num_hands=1, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0 
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened(): return

    print("\n=== TELEOP V5 (FIXED LABELS & DATA STREAM) ===")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        t0 = time.time()
        
        # Flip frame horizontally to act like a mirror (more natural for teleop)
        frame = cv2.flip(frame, 1)
        
        img_rgb = frame[:, :, ::-1].copy()
        h, w, _ = img_rgb.shape
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_label = results.multi_handedness[0].classification[0].label
            
            if mp_label == "Left":
                label_text = "Left"
                is_right = 1
            else:
                label_text = "Right"
                is_right = 0
            
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_list) * w), int(max(x_list) * w)
            y_min, y_max = int(min(y_list) * h), int(max(y_list) * h)
            bbox = [x_min, y_min, x_max, y_max]

            batch_img, meta = preprocess_frame(img_rgb, bbox, device)
            
            if batch_img is not None:
                batch = {
                    'img': batch_img,
                    'right': torch.tensor([is_right], dtype=torch.float32).to(device)
                }

                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        out = model(batch)

                # [FIX] Data Extraction & Print
                # These are the 45 numbers (15 joints * 3 rotations) you send to the robot
                pose_params = out['pred_mano_params']['hand_pose'].cpu().numpy()[0]
                
                # Calculate FPS
                fps = 1 / (time.time() - t0)
                
                # Print Stream (Overwrites previous line for clean look)
                # Printing the first 3 values of the pose as a sanity check
                print(f"\rFPS: {fps:.0f} | Hand: {label_text} | Data: {pose_params[:3]}...", end="")
                
                # Visualization
                color = (0, 255, 0) if is_right else (0, 0, 255)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, f"{label_text} Hand", (bbox[0], bbox[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        else:
            print(f"\rFPS: {1 / (time.time() - t0):.0f} | Searching...", end="")

        cv2.imshow('Hamba Teleop', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()