import cv2
import torch
import numpy as np
import time
import mediapipe as mp
from hamba.models import load_hamba
from hamba.datasets.vitdet_dataset import ViTDetDataset
from hamba.utils import recursive_to

# --- CONFIGURATION ---
SHOW_VISUALIZATION = True  # Set to False for MAX SPEED (Headless mode)
CHECKPOINT_PATH = "ckpts/hamba/checkpoints/hamba.ckpt"
RESCALE_FACTOR = 2.0

def main():
    # 1. SETUP DEVICE
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"--> Engine: Hamba (OFLEX) on {device}")
    
    # 2. LOAD HAMBA (The Heavy Lifter)
    model, model_cfg = load_hamba(CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()

    # 3. LOAD MEDIAPIPE (The Fast Spotter)
    print("--> Detector: MediaPipe Hands (CPU Optimized)")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0 # 0=Lite (Fastest), 1=Full
    )

    # 4. WEBCAM SETUP
    cap = cv2.VideoCapture(0)
    # Optimize Camera Buffer
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error opening webcam")
        return

    print("\n=== ULTRA-FAST TELEOP STARTED ===")
    print(f"Visualization: {'ON' if SHOW_VISUALIZATION else 'OFF (Headless)'}")
    
    # Warmup GPU
    dummy_batch = None 

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        t0 = time.time()
        
        # MediaPipe needs RGB
        img_rgb = frame[:, :, ::-1].copy()
        h, w, _ = img_rgb.shape

        # --- A. FAST DETECTION (MediaPipe) ---
        results = hands.process(img_rgb)

        bboxes = []
        is_right_list = []
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get Handedness (Left/Right)
                # Note: MediaPipe assumes mirrored image by default, checking label
                label = results.multi_handedness[idx].classification[0].label
                # MediaPipe Output: "Left" means it appears left in the image
                # Hamba Expects: 0 for Left Hand, 1 for Right Hand
                is_right = 1 if label == "Right" else 0
                
                # Get Bounding Box from Landmarks
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x < x_min: x_min = x
                    if x > x_max: x_max = x
                    if y < y_min: y_min = y
                    if y > y_max: y_max = y
                
                # Pack for Hamba
                bboxes.append([x_min, y_min, x_max, y_max])
                is_right_list.append(is_right)

        # --- B. HAMBA INFERENCE ---
        if len(bboxes) > 0:
            boxes = np.stack(bboxes)
            right = np.stack(is_right_list)
            
            # Create Dummy Keypoints (Hamba dataset needs them for cropping logic, but we rely on bbox)
            # We create a zero array of shape (N, 21, 3)
            dummy_keypoints = np.zeros((len(bboxes), 21, 3))

            dataset = ViTDetDataset(model_cfg, frame, boxes, right, 
                                  rescale_factor=RESCALE_FACTOR, 
                                  keypoints_2d_arr=dummy_keypoints)
            
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(bboxes), shuffle=False, num_workers=0)

            for batch in dataloader:
                batch = recursive_to(batch, device)
                
                with torch.no_grad():
                    # Mixed Precision for RTX 4080
                    with torch.amp.autocast('cuda'):
                        out = model(batch)

                # --- C. DATA OUTPUT (Simulate Robot Command) ---
                # MANO Params: 
                # global_orient (1, 3), hand_pose (1, 45), betas (1, 10), transl (1, 3)
                mano_params = out['pred_mano_params']
                pose = mano_params['hand_pose'].cpu().numpy() # The finger joints
                
                # Calculate Loop Frequency
                fps = 1 / (time.time() - t0)
                
                # Print Data Stream (Overwrite line for clean terminal)
                print(f"\r[FPS: {fps:.1f}] Hands Detected: {len(bboxes)} | Sending Pose Data...", end="")

                # --- D. VISUALIZATION (Optional) ---
                if SHOW_VISUALIZATION:
                    # Draw simple bbox to show it's working
                    for box in bboxes:
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        else:
            print(f"\r[FPS: {1 / (time.time() - t0):.1f}] Searching...", end="")

        if SHOW_VISUALIZATION:
            cv2.imshow('Hamba Ultra-Fast', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()