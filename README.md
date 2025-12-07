# Live Teleop (Left Hand Mode Control)

This project implements a teleoperation system using **Project Aria glasses** and a **Hamba** (Hand Model) inference engine. It streams RGB and IMU data from the Aria glasses, detects hands, and sends skeletal data via UDP for downstream control (e.g., robot teleoperation).

## Features

- **Project Aria Integration**: Wireless streaming of RGB images and IMU data.
- **Head Pose Estimation**: Uses IMU (Accelerometer + Gyroscope) to estimate head rotation (Pitch, Roll, Yaw) and stabilize the hand coordinate system.
- **Hybrid Hand Tracking**:
  - **Right Hand**: High-fidelity 3D reconstruction using the **Hamba** model.
  - **Left Hand**: Used as a **Mode Controller** via MediaPipe (no heavy inference).
- **Mode Switching**: Detects discrete gestures on the left hand to switch control modes (0-5).
- **Visualization**: Optional 3D skeletal visualization using Matplotlib.

## Prerequisites

- **Hardware**: Project Aria Glasses (on the same Wi-Fi network).
- **Software**:
  - Python 3.8+
  - [Project Aria Tools](https://facebookresearch.github.io/projectaria_tools/docs/intro)
  - PyTorch (CUDA supported)
  - MediaPipe
  - OpenCV, NumPy, Matplotlib, SciPy

## Configuration

Open `live_teleop_v30_left_hand.py` and adjust the following constants:

```python
CHECKPOINT_PATH = "ckpts/hamba/checkpoints/hamba.ckpt"  # Path to Hamba model weights
ARIA_IP = "192.168.0.124"                               # IP address of your Aria glasses
VISUALIZATION = False                                   # Set True to see 3D skeleton plot
DISPLAY = True                                          # Set True to see camera feed debug window
```

## Usage

1. Ensure your Aria glasses are powered on and connected to the network.
2. Run the script:
   ```bash
   python live_teleop_v30_left_hand.py
   ```

## Controls

### Left Hand Modes
The system detects the number of extended fingers on the **Left Hand** to trigger modes:

- **Mode 0**: Fist (All closed)
- **Mode 1**: Index extended
- **Mode 2**: Index + Middle extended
- **Mode 3**: Index + Middle + Ring (Default)
- **Mode 4**: Four fingers (No thumb)
- **Mode 5**: Open Palm (All 5 fingers)

### Right Hand Teleop
The **Right Hand** is tracked in 3D. The system outputs:
- **Joint Angles**: 21-point skeleton.
- **Wrist Position**: Absolute 3D position relative to the camera (stabilized).
- **Wrist Orientation**: Quaternion (x, y, z, w).

## Output (UDP)

Data is sent via UDP to `127.0.0.1:5555` (default) as a JSON string:

```json
{
  "joints": [[x, y, z], ...],     // 21 points
  "wrist_pos": [x, y, z],         // Position in meters
  "wrist_quat": [x, y, z, w],     // Orientation quaternion
  "mode": 3                       // Current detected mode
}
```
