
import cv2
from detection.face_eye_detection import detect_face_landmarks
from ui.overlay import draw_ui_overlay
import pygame
import os
import time
from performance.fps_checker import FPSCounter
from performance.latency_checker import measure_latency
from performance.trigger_timer import TriggerTimer

drowsy_start_time = None

pose_history = []
def smooth_pose_status(new_status, history, window=5):
    history.append(new_status)
    if len(history) > window:
        history.pop(0)
    # Return most frequent recent status
    return max(set(history), key=history.count)

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

trigger_logged = False  # Global flag to track if trigger time was already printed
from datetime import datetime  # Make sure this is at the top

def main():
    global closed_count, alarm_playing

    cap = cv2.VideoCapture('media/sample 1_drowsy.mp4')

    fps_counter = FPSCounter()
    trigger_timer = TriggerTimer()


    while True:
        ret, frame = cap.read()
        fps_counter.update()
        t1 = time.time()

        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow('Frame', frame)

    # Increase this value to slow down playback (1000 ms = 1 second)
        if cv2.waitKey(150) & 0xFF == ord('q'):  # 100 ms = 10 FPS
             break

        overlay = frame.copy()
        frame, left_ear, right_ear, avg_ear, distraction_status_raw = detect_face_landmarks(overlay)
        t2 = time.time()
        latency = measure_latency(t1, t2)

        distraction_status = smooth_pose_status(distraction_status_raw, pose_history)


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
            # Track time before alert triggers
            if eyes_closed:
                if closed_count == 1:
                    drowsy_start_time = time.time()  # start timing
                elif closed_count == CONSEC_FRAMES_THRESHOLD:
                    if drowsy_start_time is not None:
                        time_to_alert = time.time() - drowsy_start_time
                        print(f"⏱️ Time before alert triggered: {time_to_alert:.2f} seconds")
                        drowsy_start_time = None  # reset after alert
            else:
                drowsy_start_time = None  # reset if eyes open before threshold

            if status == "Drowsy":
                trigger_timer.start()
                if not trigger_logged:
                    print("Trigger Time:", datetime.now().strftime("%H:%M:%S"))
                    trigger_logged = True
                if not alarm_playing:
                    play_alarm()
                    alarm_playing = True

            else:
                trigger_timer.end()
                trigger_logged = False  # Reset when driver is no longer drowsy
                if alarm_playing:
                    stop_alarm()
                    alarm_playing = False


            # Debug info
            print(f"L_EAR: {left_ear}, R_EAR: {right_ear}, avg: {avg_ear}, Eyes Closed: {eyes_closed}, Status: {status}, Pose: {distraction_status}")
            print(f"FPS: {fps_counter.get_fps():.2f}, Latency: {latency*1000:.2f} ms")

        frame = draw_ui_overlay(frame, status, avg_ear, distraction_status)

        cv2.imshow("Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()


    
if __name__ == "__main__":
    print("Running performance tests...\n")
    main()
    