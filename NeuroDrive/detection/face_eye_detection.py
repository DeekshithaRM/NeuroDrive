import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize MediaPipe
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

def euclidean(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_ear(landmarks):
    left = [landmarks[i] for i in LEFT_EYE]
    right = [landmarks[i] for i in RIGHT_EYE]

    left_ear = (euclidean(left[1], left[5]) + euclidean(left[2], left[4])) / (2.0 * euclidean(left[0], left[3]))
    right_ear = (euclidean(right[1], right[5]) + euclidean(right[2], right[4])) / (2.0 * euclidean(right[0], right[3]))

    return (left_ear + right_ear) / 2.0

def get_head_pose_angles(frame, landmarks):
    h, w = frame.shape[:2]

    # 2D image points
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

    dist_coeffs = np.zeros((4, 1))  # Assume no lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs
    )

    if not success:
        return 0, 0, 0  # fallback

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    return angles[0], angles[1], angles[2]  # pitch, yaw, roll

def detect_face_landmarks(frame):
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    ear = None
    distraction_status = "Unknown"

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                drawing_spec,
                drawing_spec
            )

            ear = round(get_ear(landmarks.landmark), 3)
            cv2.putText(frame, f"EAR: {ear}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Head pose analysis
            pitch, yaw, roll = get_head_pose_angles(frame, landmarks.landmark)

            # Simple distraction logic
            if abs(yaw) > 15:
                distraction_status = "Looking Sideways"
            elif pitch > 10:
                distraction_status = "Looking Down"
            else:
                distraction_status = "Looking Forward"

            # Show angles
            cv2.putText(frame, f"Yaw: {round(yaw,1)} Pitch: {round(pitch,1)}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

            # Show distraction status
            cv2.putText(frame, f"Pose: {distraction_status}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return frame, ear, distraction_status