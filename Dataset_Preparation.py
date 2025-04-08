import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import urllib.request

# IP Webcam URL
url = 'http://192.168.8.32:8080/shot.jpg'

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Dataset setup
data = []
label = input("Enter the label for this recording (e.g., ONE, TWO, A, B): ")
frame_count = 0
max_frames = 500  # Number of frames to capture

while True:
    # ✅ Capture frame using urllib
    try:
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)
    except:
        print("Failed to grab frame from IP Webcam")
        continue

    if frame is None:
        print("Empty frame received")
        continue

    frame = cv2.flip(frame, 1)  # Optional: flip for mirror image
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 21 landmarks (x, y)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            # Append landmarks with the label
            data.append(landmarks + [label])
            frame_count += 1

            # Show capture progress
            cv2.putText(frame, f'Captured: {frame_count}/{max_frames}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if frame_count >= max_frames:
                break

    # ✅ Resize the frame for better view
    resized_frame = cv2.resize(frame, (800, 600))  # Adjust size if needed
    cv2.imshow('Hand Landmark Capture', resized_frame)

    if cv2.waitKey(1) & 0xFF == 27 or frame_count >= max_frames:
        break

cv2.destroyAllWindows()

# ✅ Save the dataset as CSV
df = pd.DataFrame(data)
df.to_csv(f'{label}_dataset.csv', index=False)
print(f'Dataset for {label} saved as {label}_dataset.csv')
