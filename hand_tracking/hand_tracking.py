import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    timestamp_ms = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        # Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # Perform hand landmarks detection
        hand_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += 33  # Approximate 30 FPS

        # Draw landmarks if detected
        if hand_landmarker_result.hand_landmarks:
            for hand_landmark in hand_landmarker_result.hand_landmarks:
                mp.tasks.vision.drawing_utils.draw_landmarks(
                    img,
                    hand_landmark,
                    mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS)

        cv2.imshow("Hand Tracking", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()