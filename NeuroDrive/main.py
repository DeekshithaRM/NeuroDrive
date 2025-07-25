import cv2
from detection.face_eye_detection import detect_face_landmarks
from ui.overlay import draw_ui_overlay
import pygame
import os

# ---------- Alarm Setup ----------
def play_alarm():
    try:
        pygame.mixer.init()
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        alarm_path = os.path.join(BASE_DIR, "assets", "alert.wav")
        print("[DEBUG] Loading alarm:", alarm_path)
        pygame.mixer.music.load(alarm_path)
        pygame.mixer.music.play(-1)
        print("[DEBUG] Alarm playing...")
    except Exception as e:
        print("[ERROR] Failed to play alarm:", e)

def stop_alarm():
    try:
        pygame.mixer.music.stop()
        print("[DEBUG] Alarm stopped.")
    except Exception as e:
        print("[ERROR] Failed to stop alarm:", e)

# ---------- Constants ----------
CLOSED_EAR_THRESHOLD = 0.25
CONSEC_FRAMES_THRESHOLD = 20

# ---------- Globals ----------
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
            print("[ERROR] Failed to grab frame")
            break

        overlay = frame.copy()

        # Face detection & EAR
        frame, left_ear, right_ear, distraction_status = detect_face_landmarks(overlay)

        if left_ear is None or right_ear is None:
            status = "No Face Detected"
            avg_ear = None

            if not alarm_playing:
                print("[DEBUG] No face detected — playing alarm.")
                play_alarm()
                alarm_playing = True

        else:
            eyes_closed = left_ear < CLOSED_EAR_THRESHOLD and right_ear < CLOSED_EAR_THRESHOLD
            status, closed_count = get_driver_status(eyes_closed, closed_count)
            avg_ear = (left_ear + right_ear) / 2

            # Terminal print
            print(f"L_EAR: {left_ear:.3f}, R_EAR: {right_ear:.3f}, Eyes Closed: {eyes_closed}, Status: {status}, Pose: {distraction_status}")

            if status == "Drowsy":
                if not alarm_playing:
                    print("[DEBUG] Drowsy detected — playing alarm.")
                    play_alarm()
                    alarm_playing = True
            else:
                if alarm_playing:
                    print("[DEBUG] Driver active — stopping alarm.")
                    stop_alarm()
                    alarm_playing = False

        # ---------- UI ----------
        frame = draw_ui_overlay(frame, status, avg_ear, distraction_status)

        cv2.imshow("Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()

if __name__ == "__main__":
    main()
