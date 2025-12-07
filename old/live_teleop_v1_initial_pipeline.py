import cv2
import torch
import numpy as np
import os
from pathlib import Path
import argparse
from hamba.models import load_hamba
from hamba.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamba.utils.renderer import Renderer, cam_crop_to_full
from hamba.utils import recursive_to
from vitpose_model import ViTPoseModel

# Detectron2 imports (for bounding box detection)
from detectron2.config import LazyConfig
from hamba.utils.utils_detectron2 import DefaultPredictor_Lazy
import hamba

LIGHT_BLUE = (0, 0.278, 0.671)

def main():
    # --- CONFIGURATION ---
    checkpoint_path = "ckpts/hamba/checkpoints/hamba.ckpt"
    rescale_factor = 2.0
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print(f"--> Loading Hamba model on {device}...")
    model, model_cfg = load_hamba(checkpoint_path)
    model = model.to(device)
    model.eval()

    print("--> Loading Detectron2 (Body Detector)...")
    # Load VitDet exactly as demo.py does
    cfg_path = Path(hamba.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.1
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    print("--> Loading ViTPose (Hand Keypoints)...")
    cpm = ViTPoseModel(device)

    print("--> Initializing Renderer...")
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # --- WEBCAM SETUP ---
    cap = cv2.VideoCapture(0)
    # Set resolution (optional, lower is faster)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("\n=== TELEOP STARTED (Press 'q' to quit) ===")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. PREPARE IMAGE
        img_cv2 = frame
        img_rgb = img_cv2[:, :, ::-1].copy() # BGR -> RGB

        # 2. DETECT BODY (Detectron2)
        det_out = detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # If no person found, just show original frame
        if len(pred_bboxes) == 0:
            cv2.imshow('Hamba Teleop', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # 3. DETECT HAND KEYPOINTS (ViTPose)
        vitposes_out = cpm.predict_pose(
            img_rgb,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []
        keypoints_2d_list = []

        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Check Left Hand
            valid_l = left_hand_keyp[:,2] > 0.1
            if sum(valid_l) > 3:
                bbox = [left_hand_keyp[valid_l,0].min(), left_hand_keyp[valid_l,1].min(),
                        left_hand_keyp[valid_l,0].max(), left_hand_keyp[valid_l,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
                keypoints_2d_list.append(left_hand_keyp)

            # Check Right Hand
            valid_r = right_hand_keyp[:,2] > 0.1
            if sum(valid_r) > 3:
                bbox = [right_hand_keyp[valid_r,0].min(), right_hand_keyp[valid_r,1].min(),
                        right_hand_keyp[valid_r,0].max(), right_hand_keyp[valid_r,1].max()]
                bboxes.append(bbox)
                is_right.append(1)
                keypoints_2d_list.append(right_hand_keyp)

        if len(bboxes) == 0:
            cv2.imshow('Hamba Teleop', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # 4. HAMBA INFERENCE
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        keypoints_2d_arr = np.stack(keypoints_2d_list)

        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=rescale_factor, keypoints_2d_arr=keypoints_2d_arr)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(bboxes), shuffle=False, num_workers=0)

        # Process batch (usually just 1 or 2 hands)
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            # 5. RENDERING (Overlay mesh on frame)
            # Prepare rendering data
            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Prepare arrays for renderer
            verts = out['pred_vertices'].detach().cpu().numpy()
            is_right_batch = batch['right'].cpu().numpy()
            # Flip left hands for correct rendering
            verts[:,:,0] = ((2*is_right_batch-1)[:,None] * verts[:,:,0])
            
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(0, 0, 0),
                focal_length=scaled_focal_length,
            )
            
            # Render view
            cam_view = renderer.render_rgba_multiple(verts, cam_t=pred_cam_t_full, render_res=img_size[0].cpu().numpy(), is_right=is_right_batch, **misc_args)

            # Overlay logic (Fast version)
            input_img = img_rgb.astype(np.float32) / 255.0
            valid_mask = (cam_view[:, :, -1] > 0)[:, :, np.newaxis]
            input_img_overlay = input_img * (1 - valid_mask) + cam_view[:, :, :3] * valid_mask
            final_img = (255 * input_img_overlay).astype(np.uint8)
            
            # Convert RGB back to BGR for OpenCV display
            final_img_bgr = final_img[:, :, ::-1]
            
            cv2.imshow('Hamba Teleop', final_img_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
