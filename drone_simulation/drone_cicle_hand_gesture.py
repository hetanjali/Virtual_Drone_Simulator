import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pygame
import time
import os

# -------- PYGAME SETUP --------
pygame.init()
width, height = 800, 600
win = pygame.display.set_mode((width, height))
pygame.display.set_caption("Gesture Drone")

x, y = width // 2, height // 2
speed = 5

# -------- MEDIAPIPE SETUP --------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Get the model path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'hand_landmarker.task')

try:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1
    )
    landmarker = HandLandmarker.create_from_options(options)
except Exception as e:
    print(f"Error loading model from {model_path}: {e}")
    print("Make sure hand_landmarker.task exists in the parent directory")
    exit(1)

cap = cv2.VideoCapture(0)

tipIds = [4, 8, 12, 16, 20]

prev_time = 0
delay = 0.3
command = "STOP"

running = True
frame_count = 0

while running:
    pygame.time.delay(30)

    # -------- PYGAME EXIT --------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # -------- CAMERA --------
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    timestamp_ms = frame_count * 33  # approximate for 30fps
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    frame_count += 1

    if results.hand_landmarks:
        for hand_landmarks_idx, hand_landmarks in enumerate(results.hand_landmarks):
            handedness = 'Right'
            if results.handedness and len(results.handedness) > hand_landmarks_idx:
                handedness = results.handedness[hand_landmarks_idx][0].category_name
            
            lmList = []
            h, w, c = img.shape

            for id, landmark in enumerate(hand_landmarks):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                lmList.append([id, cx, cy, landmark.z])

            if len(lmList) != 0:
                fingers = []

                # Thumb
                if handedness == 'Right':
                    if lmList[4][1] > lmList[3][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    if lmList[4][1] < lmList[3][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Other fingers
                for i in range(1, 5):
                    if lmList[tipIds[i]][3] < lmList[tipIds[i]-2][3]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                totalFingers = fingers.count(1)

                # -------- COMMAND LOGIC --------
                current_time = time.time()
                if current_time - prev_time > delay:

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

                    prev_time = current_time

                cv2.putText(img, f'Command: {command}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    cv2.imshow("Camera", img)

    # -------- DRONE MOVEMENT --------
    if command == "UP":
        y -= speed
    elif command == "DOWN":
        y += speed
    elif command == "LEFT":
        x -= speed
    elif command == "RIGHT":
        x += speed

    # Keep drone on screen
    x = max(20, min(width - 20, x))
    y = max(20, min(height - 20, y))

    # -------- DRAW DRONE --------
    win.fill((0, 0, 0))
    pygame.draw.circle(win, (0, 255, 0), (x, y), 20)
    pygame.display.update()

    if cv2.waitKey(1) & 0xFF == 27:
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
pygame.quit()