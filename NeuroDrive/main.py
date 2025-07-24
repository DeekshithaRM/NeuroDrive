import cv2
from detection.face_eye_detection import detect_face_landmarks
from ui.overlay import draw_ui_overlay
import pygame
import os
import time

# Alarm sound logic
def play_alarm():
    pygame.mixer.init()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    alarm_path = os.path.join(BASE_DIR, "assets", "alert.wav")
    pygame.mixer.music.load(alarm_path)
    pygame.mixer.music.play(-1)  # loop indefinitely

def stop_alarm():
    pygame.mixer.music.stop()

# Thresholds
CLOSED_EAR_THRESHOLD = 0.25
CONSEC_FRAMES_THRESHOLD = 20  # ~2 sec if ~10 fps

# State variables
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

    # FPS tracking
    prev_time = time.time()
    smoothed_fps = 0
    fps_values = []  # Store FPS to calculate average later

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        overlay = frame.copy()

        # Step 1: Get EAR from face landmarks
        frame, ears = detect_face_landmarks(overlay)

        if ears is not None:
            left_ear, right_ear = ears
            eyes_closed = left_ear < CLOSED_EAR_THRESHOLD and right_ear < CLOSED_EAR_THRESHOLD
            print(f"Left EAR: {left_ear:.3f}, Right EAR: {right_ear:.3f}, Eyes Closed: {eyes_closed}")

            status, closed_count = get_driver_status(eyes_closed, closed_count)
        else:
            print("No face detected.")
            status = "No Face"
            eyes_closed = False
            draw_ui_overlay(frame, status=status)

            # FPS calculation even when face not detected
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time
            smoothed_fps = 0.9 * smoothed_fps + 0.1 * fps
            fps_values.append(fps)

            # Show FPS
            cv2.putText(frame, f"FPS: {smoothed_fps:.1f}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Driver Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Step 2: Alarm logic
        if status == "Drowsy":
            draw_ui_overlay(frame, status="Drowsy")
            if not alarm_playing:
                play_alarm()
                alarm_playing = True
        else:
            draw_ui_overlay(frame, status="Active")
            if alarm_playing:
                stop_alarm()
                alarm_playing = False

        # Step 3: FPS calculation and tracking
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time
        smoothed_fps = 0.9 * smoothed_fps + 0.1 * fps
        fps_values.append(fps)

        # Show FPS on frame
        cv2.putText(frame, f"FPS: {smoothed_fps:.1f}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Step 4: Show frame
        cv2.imshow("Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()

    # Print average FPS
    if fps_values:
        avg_fps = sum(fps_values) / len(fps_values)
        print(f"\nAverage FPS during session: {avg_fps:.2f}")

if __name__ == "__main__":
    main()
