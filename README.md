# NeuroDrive: Real-Time Driver Alertness Detection System

**NeuroDrive** is a modular computer vision system designed to detect driver drowsiness and distraction using eye aspect ratio (EAR) and head pose estimation in real time.

## Features

- **Facial Landmark Detection** (MediaPipe)
- **Eye Aspect Ratio (EAR)** Calculation
- **Head Pose Estimation** (Yaw, Pitch, Roll)
- **Distraction Detection** (Looking Left/Right/Up/Down)
- **Logging Support** (Event logs, EAR values, pose info)
- **Modular Codebase** (`modules/` structure)
- **Video File or Webcam Support**

## How It Works

The system processes each video frame to:
1. Detect facial landmarks using MediaPipe.
2. Calculate EAR for both eyes to determine eye openness.
3. Estimate head pose (yaw, pitch) via 3D-2D projection.
4. Determine attention state: Awake / Distracted.
5. Overlay results on the video + log significant events.


## Usage

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/NeuroDrive.git
cd NeuroDrive
