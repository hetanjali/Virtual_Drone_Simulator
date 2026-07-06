import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
prev_time = 0
delay = 0.5

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'hand_landmarker.task'
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

tipIds = [4, 8, 12, 16, 20]
frame_count = 0
command = ""

while True:
    success, img = cap.read()
    if not success:
        continue

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    timestamp_ms = frame_count * 33  # approximate for 30fps
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    frame_count += 1

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            h, w, c = img.shape
            landmark_points = []
            for landmark in hand_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmark_points.append((cx, cy))

            if len(landmark_points) != 0:
                fingers = []

                # Thumb (check handedness)
                handedness = 'Right'  # default
                if results.handedness:
                    handedness = results.handedness[0][0].category_name

                if handedness == 'Right':
                    if landmark_points[4][0] > landmark_points[3][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:  # Left
                    if landmark_points[4][0] < landmark_points[3][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Other 4 fingers
                for id in range(1, 5):
                    tip_idx = tipIds[id]
                    pip_idx = tipIds[id] - 2
                    if landmark_points[tip_idx][1] < landmark_points[pip_idx][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                totalFingers = fingers.count(1)

                # Gesture → Command mapping
                current_time = time.time()
                if current_time - prev_time > delay:
                    prev_time = current_time
                    if totalFingers == 0:
                        command = "STOP"
                    elif totalFingers == 1:
                        command = "UP"
                    elif totalFingers == 2:
                        command = "DOWN"
                    elif totalFingers == 3:
                        command = "LEFT"
                    elif totalFingers == 4:
                        command = "RIGHT"
                    elif totalFingers == 5:
                        command = "TAKEOFF"
                    else:
                        command = ""

                cv2.putText(img, f'Command: {command}', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                cv2.putText(img, f'Fingers: {totalFingers}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

                # Draw landmarks
                connections = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),
                    (0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20)
                ]
                for start, end in connections:
                    cv2.line(img, landmark_points[start], landmark_points[end], (0,255,0), 2)
                for point in landmark_points:
                    cv2.circle(img, point, 5, (0,0,255), -1)

    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()