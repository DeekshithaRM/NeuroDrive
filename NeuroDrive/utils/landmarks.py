import math

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_ear(landmarks, side='both'):
    left = [landmarks[i] for i in LEFT_EYE]
    right = [landmarks[i] for i in RIGHT_EYE]

    left_ear = (euclidean(left[1], left[5]) + euclidean(left[2], left[4])) / (2.0 * euclidean(left[0], left[3]))
    right_ear = (euclidean(right[1], right[5]) + euclidean(right[2], right[4])) / (2.0 * euclidean(right[0], right[3]))

    if side == 'left':
        return round(left_ear, 3)
    elif side == 'right':
        return round(right_ear, 3)
    else:
        return round(left_ear, 3), round(right_ear, 3)