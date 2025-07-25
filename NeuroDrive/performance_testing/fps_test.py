import cv2
import time

cap = cv2.VideoCapture(0)
prev_time = 0
fps_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    fps_list.append(fps)

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("FPS Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Average FPS: {sum(fps_list)/len(fps_list):.2f}")
