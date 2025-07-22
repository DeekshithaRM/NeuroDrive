import cv2
from detection.face_eye_detection import detect_face_landmarks  
import time


closed_count = 0
threshold = 2  

simulated_eye_states = [False, True, True, True, False, False, True]
frame_index = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    frame = detect_face_landmarks(frame)

 
    if frame_index < len(simulated_eye_states):
        eyes_closed = simulated_eye_states[frame_index]
    else:
        eyes_closed = False

    print(f"Eyes closed: {eyes_closed}")

    if eyes_closed:
        closed_count += 1
        if closed_count >= threshold:
            print("Drowsy")
    else:
        closed_count = 0

    frame_index += 1

    cv2.imshow('Driver Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
