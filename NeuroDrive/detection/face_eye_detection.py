import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque
import logging

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing specs
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# 3D model points (approximate in mm)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),         # Nose tip
    (-30.0, -30.0, -30.0),   # Left eye
    (30.0, -30.0, -30.0),    # Right eye
    (-40.0, 30.0, -30.0),    # Left mouth
    (40.0, 30.0, -30.0),     # Right mouth
    (0.0, 60.0, -30.0)       # Chin
])

# Thresholds
YAW_THRESHOLD = 25
PITCH_THRESHOLD = 15
UPWARD_PITCH_THRESHOLD = -10

# Pose smoothing history
pose_history = deque(maxlen=5)


def smooth_pose_status(current_status, history):
    history.append(current_status)
    return max(set(history), key=history.count)


def euclidean(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def get_ear(landmarks):
    left = [landmarks[i] for i in LEFT_EYE]
    right = [landmarks[i] for i in RIGHT_EYE]

    left_ear = (euclidean(left[1], left[5]) + euclidean(left[2], left[4])) / (2.0 * euclidean(left[0], left[3]))
    right_ear = (euclidean(right[1], right[5]) + euclidean(right[2], right[4])) / (2.0 * euclidean(right[0], right[3]))

    return left_ear, right_ear


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
        logger.warning("solvePnP failed to estimate head pose.")
        return 0, 0, 0

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch, yaw, roll = angles

    pitch = max(-90, min(90, pitch))
    yaw = max(-90, min(90, yaw))
    roll = max(-90, min(90, roll))

    return pitch, yaw, roll


def detect_face_landmarks(frame):
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    left_ear = None
    right_ear = None
    avg_ear = None
    distraction_status = "No Face Detected"

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                drawing_spec,
                drawing_spec
            )

            try:
                left = [landmarks.landmark[i] for i in LEFT_EYE]
                left_ear = (euclidean(left[1], left[5]) + euclidean(left[2], left[4])) / (2.0 * euclidean(left[0], left[3]))
                left_ear = round(left_ear, 3)
            except Exception:
                left_ear = None

            try:
                right = [landmarks.landmark[i] for i in RIGHT_EYE]
                right_ear = (euclidean(right[1], right[5]) + euclidean(right[2], right[4])) / (2.0 * euclidean(right[0], right[3]))
                right_ear = round(right_ear, 3)
            except Exception:
                right_ear = None

            valid_ears = [ear for ear in [left_ear, right_ear] if ear is not None]
            if valid_ears:
                avg_ear = round(sum(valid_ears) / len(valid_ears), 3)

            logger.info(f"Detected face. Left EAR: {left_ear}, Right EAR: {right_ear}, Avg EAR: {avg_ear}")

            cv2.putText(frame, f"L: {left_ear} R: {right_ear}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            pitch, yaw, roll = get_head_pose_angles(frame, landmarks.landmark)

            if yaw < -YAW_THRESHOLD:
                distraction_status_raw = "Looking Right"
            elif yaw > YAW_THRESHOLD:
                distraction_status_raw = "Looking Left"
            elif pitch > PITCH_THRESHOLD:
                distraction_status_raw = "Looking Down"
            elif pitch < UPWARD_PITCH_THRESHOLD:
                distraction_status_raw = "Looking Up"
            else:
                distraction_status_raw = "Looking Forward"

            distraction_status = smooth_pose_status(distraction_status_raw, pose_history)
            logger.info(f"Distraction detected: {distraction_status}")

            draw_head_direction_arrow(frame, yaw, pitch)

            cv2.putText(frame, f"Yaw: {round(yaw,1)} Pitch: {round(pitch,1)}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
            cv2.putText(frame, f"Pose: {distraction_status}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        logger.warning("No face detected in this frame.")

    return frame, left_ear, right_ear, avg_ear, distraction_status


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