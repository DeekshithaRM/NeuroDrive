import cv2
import numpy as np
from datetime import datetime

def draw_ui_overlay(frame, status="Distracted", avg_ear=None, distraction_status=None):
    color = (0, 255, 0) if status == "Active" else (0, 0, 255)

    # Banner overlay
    banner_height = 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], banner_height), (50, 50, 50), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Status circle
    cv2.circle(frame, (50, 30), 20, color, -1)

    # Status text
    status_text = f"Driver Status: {status}"
    cv2.putText(frame, status_text, (90, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Optional: Show EAR value
    if avg_ear is not None:
        ear_text = f"EAR: {avg_ear:.2f}"
        cv2.putText(frame, ear_text, (frame.shape[1] - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    # Optional: Date and Time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, now, (frame.shape[1] - 250, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    return frame
