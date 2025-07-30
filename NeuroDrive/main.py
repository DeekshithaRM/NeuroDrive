import cv2
import pygame
import os
import time

from collections import deque

from detection.face_eye_detection import detect_face_landmarks
from ui.overlay import draw_ui_overlay
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

# Sound configuration
def play_alarm():
    try:
        pygame.mixer.init()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        alarm_path = os.path.join(base_dir, "assets", "alert.wav")
        pygame.mixer.music.load(alarm_path)
        pygame.mixer.music.play(-1)
    except Exception as e:
        logger.error(f"Failed to play alarm: {e}")

def stop_alarm():
    if pygame.mixer.get_init():
        pygame.mixer.music.stop()

# EAR threshold settings
CLOSED_EAR_THRESHOLD = 0.25
CONSEC_FRAMES_THRESHOLD = 20

closed_count = 0
alarm_playing = False

pose_history = deque(maxlen=5)

def smooth_pose_status(new_status, history):
    history.append(new_status)
    return max(set(history), key=history.count)

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

    if not cap.isOpened():
        logger.error("Camera not accessible.")
        return
    prev_time = time.time()
    trigger_start_time = None
    fps_list = []
    latency_list = []



    while True:
        ret, frame = cap.read()
        start_time = time.time()  # Start time for latency measurement
        fps = 1.0 / (start_time - prev_time)
        prev_time = start_time

        if not ret:
            logger.warning("Failed to grab frame.")
            break

        overlay = frame.copy()
        annotated_frame, left_ear, right_ear, avg_ear, distraction_status = detect_face_landmarks(overlay)
        latency = (time.time() - start_time) * 1000  # in milliseconds
        fps_list.append(fps)
        latency_list.append(latency)



        if avg_ear is None:
            status = "No Face Detected"
            distraction_status = "No Face Detected"
            if not alarm_playing:
                play_alarm()
                alarm_playing = True
        else:
            eyes_closed = left_ear < CLOSED_EAR_THRESHOLD and right_ear < CLOSED_EAR_THRESHOLD
            status, closed_count = get_driver_status(eyes_closed, closed_count)

            distraction_status = smooth_pose_status(distraction_status, pose_history)

            if status == "Drowsy":
                if trigger_start_time is None:
                    trigger_start_time = time.time()  # Start counting when drowsiness begins
                if not alarm_playing:
                    play_alarm()
                    alarm_playing = True
            else:
                if trigger_start_time:
                    time_to_trigger = time.time() - trigger_start_time
                    logger.info(f"⏱ Time taken to trigger drowsiness: {time_to_trigger:.2f} seconds")
                    trigger_start_time = None
                if alarm_playing:
                    stop_alarm()
                    alarm_playing = False


            logger.info(f"EAR: {avg_ear:.3f}, Status: {status}, Pose: {distraction_status}")
            logger.info(f"FPS: {fps:.2f}, Latency: {latency:.2f} ms")


        final_frame = draw_ui_overlay(overlay, status, avg_ear if avg_ear else None, distraction_status)

        cv2.imshow("Driver Monitor", final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()
    if fps_list and latency_list:
        avg_fps = sum(fps_list) / len(fps_list)
        avg_latency = sum(latency_list) / len(latency_list)
        logger.info(f"📊 Average FPS: {avg_fps:.2f}")
        logger.info(f"📊 Average Latency: {avg_latency:.2f} ms")


if __name__ == "__main__":
    main()
