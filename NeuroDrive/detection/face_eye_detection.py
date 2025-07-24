import cv2
import mediapipe as mp
import math

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Draw utils
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Eye landmark indices (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_ear(landmarks):
    # Left Eye
    left = [landmarks[i] for i in LEFT_EYE]
    left_ear = (euclidean(left[1], left[5]) + euclidean(left[2], left[4])) / (2.0 * euclidean(left[0], left[3]))

    # Right Eye
    right = [landmarks[i] for i in RIGHT_EYE]
    right_ear = (euclidean(right[1], right[5]) + euclidean(right[2], right[4])) / (2.0 * euclidean(right[0], right[3]))

    return (left_ear + right_ear) / 2.0

def detect_face_landmarks(frame):
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    ear = None
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                drawing_spec,
                drawing_spec
            )

            # Compute EAR
            ear = round(get_ear(landmarks.landmark), 3)

            # show EAR on frame
            cv2.putText(frame, f"EAR: {ear}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame, ear