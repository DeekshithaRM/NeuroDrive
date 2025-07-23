import cv2
from detection.face_eye_detection import detect_face_landmarks
from ui.overlay import draw_ui_overlay
import pygame
import os

def play_alarm():
    pygame.mixer.init()
    alarm_path = os.path.join("assets", "alert.wav")
    pygame.mixer.music.load(alarm_path)
    pygame.mixer.music.play()

# Thresholds
CLOSED_EAR_THRESHOLD = 0.25  # below this = eyes closed
CONSEC_FRAMES_THRESHOLD = 20  # ~2 sec if ~10 fps

# State variables
closed_count = 0

def get_driver_status(eyes_closed, closed_count):
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
        frame, ears = detect_face_landmarks(overlay)

        # Step 2: Use EAR to determine if eyes are closed
        if ears is not None:
            left_ear, right_ear = ears
            eyes_closed = left_ear < CLOSED_EAR_THRESHOLD and right_ear < CLOSED_EAR_THRESHOLD
            print(f"Left EAR: {left_ear:.3f}, Right EAR: {right_ear:.3f}, Eyes Closed: {eyes_closed}")
        else:
            eyes_closed = True  # no face, assume awake
            print("No face detected. Assuming eyes closed.")

        # Step 3: Get driver status
        status, closed_count = get_driver_status(eyes_closed, closed_count)

        # Step 4: Draw UI and play sound
        if status == "Drowsy":
            draw_ui_overlay(frame, status="Drowsy")
            play_alarm()
        else:
            draw_ui_overlay(frame, status="Active")

        # Step 5: Show frame
        cv2.imshow("Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources after loop ends
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
