import cv2
import numpy as np
from datetime import datetime
import os

def draw_ui_overlay(frame, status="Drowsy", avg_ear=None):
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
    text_x = 90
    text_y = 40
    cv2.putText(frame, status_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- Emoji beside the text ---
    emoji_path = f"{status.lower()}.png"  # e.g., "drowsy.png"
    if os.path.exists(emoji_path):
        emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
        if emoji is not None:
            emoji = cv2.resize(emoji, (32, 32))
            if emoji.shape[2] == 4:  # has alpha channel
                x_offset = text_x + 250  # position close to text
                y_offset = text_y - 25
                for c in range(3):
                    frame[y_offset:y_offset+32, x_offset:x_offset+32, c] = \
                        (emoji[:, :, 3] / 255.0) * emoji[:, :, c] + \
                        (1.0 - emoji[:, :, 3] / 255.0) * frame[y_offset:y_offset+32, x_offset:x_offset+32, c]
    else:
        print(f"⚠️ Emoji not found at path: {emoji_path}")

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

