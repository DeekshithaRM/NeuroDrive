import numpy as np
import cv2

# 3D model points (approximate in mm)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),         # Nose tip
    (-30.0, -30.0, -30.0),   # Left eye
    (30.0, -30.0, -30.0),    # Right eye
    (-40.0, 30.0, -30.0),    # Left mouth
    (40.0, 30.0, -30.0),     # Right mouth
    (0.0, 60.0, -30.0)       # Chin
])

YAW_THRESHOLD = 25
PITCH_THRESHOLD = 15
UPWARD_PITCH_THRESHOLD = -10

def smooth_pose_status(current_status, history):
    history.append(current_status)
    return max(set(history), key=history.count)

def get_head_pose_angles(frame, landmarks):
    h, w = frame.shape[:2]

    image_points = np.array([
        [landmarks[1].x * w, landmarks[1].y * h],     # Nose tip
        [landmarks[33].x * w, landmarks[33].y * h],   # Left eye
        [landmarks[263].x * w, landmarks[263].y * h], # Right eye
        [landmarks[61].x * w, landmarks[61].y * h],   # Left mouth
        [landmarks[291].x * w, landmarks[291].y * h], # Right mouth
        [landmarks[199].x * w, landmarks[199].y * h], # Chin
    ], dtype="double")

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs
    )

    if not success:
        return 0, 0, 0

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch, yaw, roll = angles

    pitch = max(-90, min(90, pitch))
    yaw = max(-90, min(90, yaw))
    roll = max(-90, min(90, roll))

    return pitch, yaw, roll

def draw_head_direction_arrow(frame, yaw, pitch):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    length = 100

    dx = int(length * np.sin(np.radians(yaw)))
    dy = int(length * np.sin(np.radians(pitch)))
    end_point = (center[0] + dx, center[1] - dy)

    color = (0, 255, 0) if -YAW_THRESHOLD < yaw < YAW_THRESHOLD and UPWARD_PITCH_THRESHOLD < pitch < PITCH_THRESHOLD else (0, 0, 255)

    cv2.arrowedLine(frame, center, end_point, color, 4, tipLength=0.3)
    cv2.circle(frame, center, 5, (255, 0, 0), -1)