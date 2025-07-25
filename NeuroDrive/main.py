import cv2
from detection.face_eye_detection import detect_face_landmarks
from ui.overlay import draw_ui_overlay
import pygame
import os

def play_alarm():
    pygame.mixer.init()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    alarm_path = os.path.join(BASE_DIR, "assets", "alert.wav")
    pygame.mixer.music.load(alarm_path)
    pygame.mixer.music.play(-1)  # loop

def stop_alarm():
    pygame.mixer.music.stop()

# Thresholds
CLOSED_EAR_THRESHOLD = 0.25
CONSEC_FRAMES_THRESHOLD = 20  # adjust based on FPS

# State variables
closed_count = 0
alarm_playing = False

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
    global closed_count, alarm_playing
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        overlay = frame.copy()

        # Step 1: Get left and right EAR
        frame, (left_ear, right_ear) = detect_face_landmarks(overlay)

        # Step 2: Check if both eyes are closed
        if left_ear is not None and right_ear is not None:
            eyes_closed = (
                left_ear < CLOSED_EAR_THRESHOLD and right_ear < CLOSED_EAR_THRESHOLD
            )
        else:
            eyes_closed = False  # if landmarks not detected

        # Step 3: Get driver status
        status, closed_count = get_driver_status(eyes_closed, closed_count)

        # Debug info
        print(
            f"L_EAR: {left_ear}, R_EAR: {right_ear}, Eyes Closed: {eyes_closed}, Status: {status}"
        )

        # Step 4: Handle alarm logic
        if status == "Drowsy":
            if not alarm_playing:
                play_alarm()
                alarm_playing = True
        else:
            if alarm_playing:
                stop_alarm()
                alarm_playing = False

        # Step 5: Overlay UI with EAR
        if left_ear is not None and right_ear is not None:
            avg_ear = (left_ear + right_ear) / 2
        else:
            avg_ear = None

        frame = draw_ui_overlay(frame, status, avg_ear)

        # Step 6: Show frame
        cv2.imshow("Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()

if __name__ == "__main__":
    main()  