import cv2
import winsound
from detection.face_eye_detection import detect_face_landmarks
from ui.overlay import draw_ui_overlay

# Thresholds
CLOSED_EAR_THRESHOLD = 0.25  # below this = eyes closed
CONSEC_FRAMES_THRESHOLD = 20  # ~2 sec if ~10 fps

# State variables
closed_count = 0

def get_driver_status(eyes_closed, closed_count):
    """
    Updates closed eye frame count and returns current status.
    """
    if eyes_closed:
        closed_count += 1
        if closed_count >= CONSEC_FRAMES_THRESHOLD:
            return "Drowsy", closed_count
    else:
        closed_count = 0
    return "Active", closed_count

def main():
    global closed_count
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        overlay = frame.copy()

        # Step 1: Get EAR from face landmarks
        frame, ear = detect_face_landmarks(overlay)

        # Step 2: Use EAR to determine if eyes are closed
        if ear is not None:
            eyes_closed = ear < CLOSED_EAR_THRESHOLD
        else:
            eyes_closed = False  # no face, assume awake

        # Step 3: Get driver status
        status, closed_count = get_driver_status(eyes_closed, closed_count)

        # Optional: Print EAR & status for debugging
        print(f"EAR: {ear}, Eyes Closed: {eyes_closed}, Status: {status}")

        # Play beep alert when driver is detected as drowsy
        if status == "Drowsy":
              winsound.Beep(1000, 500)  # frequency=1000 Hz, duration=500 ms


        # Step 4: Overlay UI
        frame = draw_ui_overlay(frame, status)

        # Step 5: Show frame
        cv2.imshow("Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
