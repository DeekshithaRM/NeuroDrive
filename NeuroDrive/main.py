

from detection.face_eye_detection import detect_face_landmarks
from ui.overlay import draw_ui_overlay  # Your function
import cv2

# Simulated eye status sequence (False = closed, True = open)
simulated_eye_states = [False, True, True, True, False, False, True]
frame_index = 0
closed_count = 0
threshold = 2  # Number of consecutive closed-eye frames to be considered drowsy

def get_driver_status(eyes_closed, closed_count):
    """
    Returns current driver status based on eye state and closed count.
    """
    if eyes_closed:
        closed_count += 1
        if closed_count >= threshold:
            return "Drowsy", closed_count
    else:
        closed_count = 0
    return "Active", closed_count

def main():
    global frame_index, closed_count

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Face/eye landmark detection (optional visual overlay)
        frame = detect_face_landmarks(frame)

        # Simulated eye state
        if frame_index < len(simulated_eye_states):
            eyes_closed = not simulated_eye_states[frame_index]  # False means eyes are open
        else:
            eyes_closed = False

        # Determine driver status
        status, closed_count = get_driver_status(eyes_closed, closed_count)

        # Run UI overlay display
        color = (0, 255, 0) if status == "Active" else (0, 0, 255)

        # --- Overlay logic directly here instead of inside overlay.py ---
        banner_height = 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], banner_height), (50, 50, 50), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.circle(frame, (50, 30), 20, color, -1)
        cv2.putText(frame, f"Driver Status: {status}", (90, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show frame
        cv2.imshow('Driver Monitor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()