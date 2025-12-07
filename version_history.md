Phase 1: Getting it to Run (The Heavy Era)

    live_teleop_v1_initial_pipeline.py: The first script. Connected the webcam but ran the heavy Detectron2 detector every frame (<1 FPS).

    live_teleop_v2_tracking_logic.py: Added "Tracking Mode" to skip detection on most frames and fixed the flickering window bug.

    live_teleop_v3_regnety_fast.py: Swapped the huge ViTDet for RegNetY and removed 3D rendering for speed (Green dots only).

Phase 2: The Speed Breakthrough (The Hybrid Era)

    live_teleop_v4_mediapipe_hybrid.py: Replaced Detectron2 with MediaPipe (CPU) as the spotter.

    live_teleop_v5_no_dataloader.py: CRITICAL SPEED FIX. Removed the PyTorch DataLoader overhead. This hit the 30 FPS hardware limit.

    live_teleop_v6_label_fix.py: Fixed the Left/Right label swapping and added raw data printing.

Phase 3: Robot Data Formatting (The Math Era)

    live_teleop_v7_dual_hands_batch.py: Added logic to detect and batch-process two hands simultaneously.

    live_teleop_v8_one_euro_smooth.py: Added the 1€ Filter to remove jitter and shaking.

    live_teleop_v9_robot_format_22pt.py: Added vector math to synthesize the Forearm point, creating the 22x3 array your robot needs.

    live_teleop_v10_6d_wrist_pose.py: Calculated the full 6D Pose (Rotation Matrix + Global Translation) for the wrist.

    live_teleop_v11_dual_hand_6d.py: Combined Dual Hands + 6D Pose + 22-point skeleton into one feature-complete script.

Phase 4: Visualization & Debugging (The plotting Era)

    live_teleop_v12_3d_viz_basic.py: Added the Matplotlib 3D stick figure window.

    live_teleop_v13_3d_viz_isometric.py: Switched the plot to Isometric (Orthographic) view for engineering accuracy.

    live_teleop_v14_raw_physics_chirality_fix.py: Removed the input image mirroring to fix the "Right Hand Flipping" physics bug.

    live_teleop_v15_stable_palm.py: Attempted to stabilize depth using palm-distance cropping (Current buggy version).