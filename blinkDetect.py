import dlib
import cv2
import time
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread
import playsound3
import queue

# Constants
FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 460
BLINK_THRESH = 0.27
BLINK_TIME = 0.15  # 150ms
DROWSY_TIME = 1.5  # 1.5 seconds
ALARM_ON = False
GAMMA = 1.5
threadStatusQ = queue.Queue()

# Load dlib models
modelPath = "C:/Users/arunkunta/Downloads/shape_predictor_68_face_landmarks.dat"

sound_path = "alarm.wav"

detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(modelPath)
except Exception as e:
    print(f"Error loading dlib model: {e}")
    exit()

# Eye Landmark Indices
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def sound_alert(path, threadStatusQ):
    while True:
        if not threadStatusQ.empty():
            if threadStatusQ.get():
                break
        playsound3.playsound(path)

def get_landmarks(image):
    small_img = cv2.resize(image, None, fx=1.0/FACE_DOWNSAMPLE_RATIO, fy=1.0/FACE_DOWNSAMPLE_RATIO)
    faces = detector(small_img, 0)
    if len(faces) == 0:
        return None
    face = dlib.rectangle(int(faces[0].left() * FACE_DOWNSAMPLE_RATIO),
                          int(faces[0].top() * FACE_DOWNSAMPLE_RATIO),
                          int(faces[0].right() * FACE_DOWNSAMPLE_RATIO),
                          int(faces[0].bottom() * FACE_DOWNSAMPLE_RATIO))
    return [(p.x, p.y) for p in predictor(image, face).parts()]

# Start Video Capture
capture = cv2.VideoCapture(0)
print("Calibrating...")

total_time = 0.0
dummy_frames = 50  # Adjust for calibration speed
valid_frames = 0

while valid_frames < dummy_frames:
    ret, frame = capture.read()
    if not ret:
        continue
    start_time = time.time()
    landmarks = get_landmarks(frame)
    if landmarks is None:
        continue
    total_time += time.time() - start_time
    valid_frames += 1

spf = total_time / dummy_frames  # Seconds per frame
falseBlinkLimit = BLINK_TIME / spf
drowsyLimit = DROWSY_TIME / spf
print(f"Calibration Complete! SPF: {spf * 1000:.2f} ms")

# Main Loop
while True:
    try:
        ret, frame = capture.read()
        if not ret:
            break

        landmarks = get_landmarks(frame)
        if landmarks is None:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        
        left_eye = [landmarks[i] for i in LEFT_EYE]
        right_eye = [landmarks[i] for i in RIGHT_EYE]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        if ear < BLINK_THRESH:
            drowsy += 1
            if drowsy >= drowsyLimit:
                cv2.putText(frame, "DROWSINESS ALERT!", (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not ALARM_ON:
                    ALARM_ON = True
                    threadStatusQ.put(False)
                    Thread(target=sound_alert, args=(sound_path, threadStatusQ), daemon=True).start()
        else:
            drowsy = 0
            ALARM_ON = False
        
        for i in LEFT_EYE + RIGHT_EYE:
            cv2.circle(frame, landmarks[i], 2, (0, 255, 0), -1)
        
        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    except Exception as e:
        print(f"Error: {e}")

capture.release()
cv2.destroyAllWindows()
