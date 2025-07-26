import cv2
from detection.face_eye_detection import detect_face_landmarks
from ui.overlay import draw_ui_overlay
import pygame
import os

def play_alarm():
    try:
        pygame.mixer.init()
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        alarm_path = os.path.join(BASE_DIR, "assets", "alert.wav")
        pygame.mixer.music.load(alarm_path)
        pygame.mixer.music.play(-1)
    except Exception as e:
        print("Failed to play alarm:", e)

def stop_alarm():
    if pygame.mixer.get_init():
        pygame.mixer.music.stop()

CLOSED_EAR_THRESHOLD = 0.25
CONSEC_FRAMES_THRESHOLD = 20

closed_count = 0
alarm_playing = False

def get_driver_status(eyes_closed, closed_count):
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
        frame, left_ear, right_ear, distraction_status = detect_face_landmarks(overlay)

        if left_ear is None or right_ear is None:
            status = "No Face Detected"
            avg_ear = None
            distraction_status = "No Face Detected"

            if not alarm_playing:
                play_alarm()
                alarm_playing = True
        else:
            avg_ear = (left_ear + right_ear) / 2
            eyes_closed = left_ear < CLOSED_EAR_THRESHOLD and right_ear < CLOSED_EAR_THRESHOLD
            status, closed_count = get_driver_status(eyes_closed, closed_count)

            if status == "Drowsy":
                if not alarm_playing:
                    play_alarm()
                    alarm_playing = True
            else:
                if alarm_playing:
                    stop_alarm()
                    alarm_playing = False

            # Debug info
            print(f"L_EAR: {left_ear}, R_EAR: {right_ear}, avg: {avg_ear}, Eyes Closed: {eyes_closed}, Status: {status}, Pose: {distraction_status}")

        frame = draw_ui_overlay(frame, status, avg_ear, distraction_status)

        cv2.imshow("Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()

if __name__ == "__main__":
    main()