import cv2
import torch
import numpy as np
import os
from pathlib import Path
import time
from hamba.models import load_hamba
from hamba.datasets.vitdet_dataset import ViTDetDataset
from hamba.utils import recursive_to
from vitpose_model import ViTPoseModel

# FAST DETECTOR IMPORTS
from detectron2 import model_zoo
from detectron2.config import get_cfg
from hamba.utils.utils_detectron2 import DefaultPredictor_Lazy

def expand_bbox(bbox, scale=1.5, img_shape=None):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    center_x, center_y = x1 + w/2, y1 + h/2
    w *= scale
    h *= scale
    x1 = center_x - w/2
    y1 = center_y - h/2
    x2 = center_x + w/2
    y2 = center_y + h/2
    if img_shape is not None:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_shape[1], x2)
        y2 = min(img_shape[0], y2)
    return np.array([x1, y1, x2, y2])

def main():
    checkpoint_path = "ckpts/hamba/checkpoints/hamba.ckpt"
    rescale_factor = 2.0
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print(f"--> Loading Hamba on {device}...")
    model, model_cfg = load_hamba(checkpoint_path)
    model = model.to(device)
    model.eval()

    print("--> Loading RegNetY (FAST Body Detector)...")
    # Use RegNetY instead of ViTDet for speed
    try:
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    except Exception as e:
        print(f"Error loading RegNetY: {e}")
        print("Please ensure you have internet access to download the config/weights.")
        return

    print("--> Loading ViTPose...")
    cpm = ViTPoseModel(device)

    # WEBCAM
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Disable internal buffer to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    tracked_bbox = None 
    tracking_patience = 0 
    
    print("\n=== HIGH SPEED TELEOP STARTED ===")

    while True:
        ret, frame = cap.read()
        if not ret: break
        t0 = time.time()
        
        img_rgb = frame[:, :, ::-1].copy()
        
        # --- DETECTION LOGIC ---
        det_instances = None
        
        if tracked_bbox is None:
            # Run RegNetY (Faster than ViTDet)
            det_out = detector(frame) # Detectron expects BGR
            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
            pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores = det_instances.scores[valid_idx].cpu().numpy()
            
            if len(pred_bboxes) > 0:
                detection_input = [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
            else:
                detection_input = None
        else:
            # Tracking Mode
            x1, y1, x2, y2 = tracked_bbox
            detection_input = [np.array([[x1, y1, x2, y2, 1.0]])]

        if detection_input is not None:
            vitposes_out = cpm.predict_pose(img_rgb, detection_input)
        else:
            vitposes_out = []

        final_bboxes = []
        is_right = []
        keypoints_2d_list = []

        # Visualize 2D Keypoints directly on frame (Fast)
        display_frame = frame.copy()
        
        for vitposes in vitposes_out:
            # We only care about the Right Hand for this demo
            right_hand_keyp = vitposes['keypoints'][-21:]
            valid_r = right_hand_keyp[:,2] > 0.1
            
            if sum(valid_r) > 3:
                bbox = [right_hand_keyp[valid_r,0].min(), right_hand_keyp[valid_r,1].min(),
                        right_hand_keyp[valid_r,0].max(), right_hand_keyp[valid_r,1].max()]
                final_bboxes.append(bbox)
                is_right.append(1)
                keypoints_2d_list.append(right_hand_keyp)
                
                tracked_bbox = expand_bbox(bbox, scale=3.0, img_shape=img_rgb.shape)

                # DRAW 2D SKELETON (Instant visualization)
                for kp in right_hand_keyp:
                    if kp[2] > 0.1:
                        cv2.circle(display_frame, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)

        # --- HAMBA INFERENCE (Only run if hand found) ---
        if len(final_bboxes) > 0:
            boxes = np.stack(final_bboxes)
            right = np.stack(is_right)
            keypoints_2d_arr = np.stack(keypoints_2d_list)

            dataset = ViTDetDataset(model_cfg, frame, boxes, right, rescale_factor=rescale_factor, keypoints_2d_arr=keypoints_2d_arr)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(final_bboxes), shuffle=False, num_workers=0)

            for batch in dataloader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    # FP16 Autocast for speed boost on RTX 4080
                    with torch.cuda.amp.autocast(enabled=True):
                        out = model(batch)

                # --- TELEOP DATA EXTRACTION ---
                # This is where you would send data to your robot
                # Shape: (Batch_Size, 16) -> 1 Global Orient + 15 Hand Joints
                mano_pose = out['pred_mano_params']['hand_pose'].detach().cpu().numpy() 
                # Shape: (Batch_Size, 10) -> Hand Shape
                mano_shape = out['pred_mano_params']['betas'].detach().cpu().numpy()
                
                # Printing simple debug info instead of rendering 3D mesh
                # print(f"Sending Robot Command: Pose Mean={np.mean(mano_pose):.4f}")

        else:
            if tracked_bbox is not None:
                tracking_patience += 1
                if tracking_patience > 5:
                    tracked_bbox = None
                    tracking_patience = 0
                    print("Lost tracking.")

        # UI
        fps = 1 / (time.time() - t0)
        color = (0, 255, 0) if tracked_bbox is not None else (0, 0, 255)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Hamba Teleop V3 (Fast)', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): tracked_bbox = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()