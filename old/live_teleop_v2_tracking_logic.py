import cv2
import torch
import numpy as np
import os
from pathlib import Path
import time
from hamba.models import load_hamba
from hamba.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamba.utils.renderer import Renderer, cam_crop_to_full
from hamba.utils import recursive_to
from vitpose_model import ViTPoseModel

# Detectron2 imports
from detectron2.config import LazyConfig
from hamba.utils.utils_detectron2 import DefaultPredictor_Lazy
import hamba

LIGHT_BLUE = (0, 0.278, 0.671)

def expand_bbox(bbox, scale=1.5, img_shape=None):
    # Expand the box by a scale factor to ensure we don't lose the hand
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
    # --- CONFIGURATION ---
    checkpoint_path = "ckpts/hamba/checkpoints/hamba.ckpt"
    rescale_factor = 2.0
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print(f"--> Loading Hamba model on {device}...")
    model, model_cfg = load_hamba(checkpoint_path)
    model = model.to(device)
    model.eval()

    print("--> Loading Detectron2 (Heavy Detector)...")
    cfg_path = Path(hamba.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.1
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    print("--> Loading ViTPose (Keypoints)...")
    cpm = ViTPoseModel(device)

    print("--> Initializing Renderer...")
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    cap = cv2.VideoCapture(0)
    # Force lower resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Tracking Variables
    tracked_bbox = None # If we have a box, we skip Detectron2
    tracking_patience = 0 # How many frames we tolerate losing the hand
    
    print("\n=== TELEOP V2 STARTED ===")
    print("   [Mode: DETECTING] - Press 'r' to reset tracking, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        t0 = time.time()
        
        img_cv2 = frame
        img_rgb = img_cv2[:, :, ::-1].copy()
        img_h, img_w = img_rgb.shape[:2]

        # --- LOGIC: DETECTION VS TRACKING ---
        det_instances = None
        
        if tracked_bbox is None:
            # SLOW MODE: Run full body detection
            det_out = detector(img_cv2)
            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
            pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores = det_instances.scores[valid_idx].cpu().numpy()
            
            if len(pred_bboxes) > 0:
                # Just take the most confident person
                detection_input = [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
            else:
                detection_input = None
        else:
            # FAST MODE: Use the previous box expanded
            # We fake a detection input for ViTPose using the tracked box
            x1, y1, x2, y2 = tracked_bbox
            score = 1.0
            # Format: [[x1, y1, x2, y2, score]]
            detection_input = [np.array([[x1, y1, x2, y2, score]])]

        # --- VITPOSE (Always runs, but input changes) ---
        if detection_input is not None:
            vitposes_out = cpm.predict_pose(img_rgb, detection_input)
        else:
            vitposes_out = []

        # Process ViTPose results to find HANDS
        final_bboxes = []
        is_right = []
        keypoints_2d_list = []

        found_hand_this_frame = False
        
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Check Right Hand (Prioritize Right for Teleop usually)
            valid_r = right_hand_keyp[:,2] > 0.1
            if sum(valid_r) > 3:
                # Get bbox of the HAND, not the person
                bbox = [right_hand_keyp[valid_r,0].min(), right_hand_keyp[valid_r,1].min(),
                        right_hand_keyp[valid_r,0].max(), right_hand_keyp[valid_r,1].max()]
                final_bboxes.append(bbox)
                is_right.append(1)
                keypoints_2d_list.append(right_hand_keyp)
                found_hand_this_frame = True
                
                # Update tracking box (Expand hand box to include some context)
                tracked_bbox = expand_bbox(bbox, scale=3.0, img_shape=img_rgb.shape) 

        # --- RENDERING LOGIC (Fixing the Overlap Issue) ---
        final_visual = img_rgb.astype(np.float32) / 255.0

        if len(final_bboxes) > 0:
            # Stack data for batch processing
            boxes = np.stack(final_bboxes)
            right = np.stack(is_right)
            keypoints_2d_arr = np.stack(keypoints_2d_list)

            dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=rescale_factor, keypoints_2d_arr=keypoints_2d_arr)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(final_bboxes), shuffle=False, num_workers=0)

            for batch in dataloader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model(batch)

                # 3D Projection Math
                multiplier = (2*batch['right']-1)
                pred_cam = out['pred_cam']
                pred_cam[:,1] = multiplier*pred_cam[:,1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                verts = out['pred_vertices'].detach().cpu().numpy()
                is_right_batch = batch['right'].cpu().numpy()
                verts[:,:,0] = ((2*is_right_batch-1)[:,None] * verts[:,:,0])

                # Render ALL hands in the batch at once (or iterate and accumulate)
                # Renderer handles multiple internally if passed as list, but let's accumulate to be safe
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(0, 0, 0),
                    focal_length=scaled_focal_length,
                )
                
                # Use the batch renderer
                cam_view = renderer.render_rgba_multiple(verts, cam_t=pred_cam_t_full, render_res=img_size[0].cpu().numpy(), is_right=is_right_batch, **misc_args)
                
                # Composite onto our accumulating frame
                valid_mask = (cam_view[:, :, -1] > 0)[:, :, np.newaxis]
                final_visual = final_visual * (1 - valid_mask) + cam_view[:, :, :3] * valid_mask

        else:
            # If we lost the hand, count patience
            if tracked_bbox is not None:
                tracking_patience += 1
                if tracking_patience > 5: # Lost hand for 5 frames
                    tracked_bbox = None # Reset to slow detection
                    tracking_patience = 0
                    print("Lost tracking. Resetting detector.")

        # Convert back to BGR for OpenCV (Add .copy() to fix memory layout)
        final_bgr = (255 * final_visual).astype(np.uint8)[:, :, ::-1].copy()
        
        # UI Info
        fps = 1 / (time.time() - t0)
        mode_color = (0, 255, 0) if tracked_bbox is not None else (0, 0, 255)
        mode_text = "TRACKING" if tracked_bbox is not None else "DETECTING"
        cv2.putText(final_bgr, f"FPS: {fps:.1f} | {mode_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
        
        # Draw Tracking Box Debug
        if tracked_bbox is not None:
            x1, y1, x2, y2 = tracked_bbox.astype(int)
            cv2.rectangle(final_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)

        cv2.imshow('Hamba Teleop V2', final_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): tracked_bbox = None # Force reset

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
