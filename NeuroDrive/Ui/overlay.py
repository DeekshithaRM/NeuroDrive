import cv2
import numpy as np

def run_driver_monitor(status="Drowsy"):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
        cv2.putText(frame, f"Driver Status: {status}", (90, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Driver Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
run_driver_monitor("Active")
# run_driver_monitor("Drowsy")
