import cv2
import mediapipe as mp
import numpy as np
import urllib.request
from tensorflow.keras.models import load_model

# ✅ Load trained model
model = load_model('hand_gesture_model.h5')

# ✅ Use the EXACT label order from your dataset
labels = ['42', 'A', 'B', 'C', 'D', 'EIGHT', 'E', 'FIVE', 'FOUR', 'F', 'G', 'H', 'I', 'J', 'K',
          'L', 'M', 'NINE', 'N', 'ONE', 'O', 'P', 'Q', 'R', 'SEVEN', 'SIX', 'S', 'TEN', 'THREE',
          'TWO', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# ✅ Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ✅ IP Webcam URL
url = 'http://192.168.8.32:8080/shot.jpg'

while True:
    try:
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)
    except:
        print("Failed to grab frame")
        continue

    if frame is None:
        continue

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    prediction_text = "No Hand Detected"
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            if len(landmarks) == 42:
                # ✅ Normalize landmarks
                landmarks_np = np.array(landmarks).reshape(1, -1)
                landmarks_np = landmarks_np / np.max(landmarks_np)

                # ✅ Predict
                prediction = model.predict(landmarks_np)[0]
                class_id = np.argmax(prediction)

                # ✅ Safety check to avoid IndexError
                if class_id >= len(labels):
                    print(f"⚠ Predicted class {class_id} is out of range!")
                    class_id = 0  # Fallback to '42' or unknown

                gesture = labels[class_id]
                confidence = np.max(prediction) * 100
                prediction_text = f"Gesture: {gesture} ({confidence:.2f}%)"

    # ✅ Resize and display
    resized_frame = cv2.resize(frame, (800, 600))
    cv2.putText(resized_frame, prediction_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 2)
    cv2.imshow("Real-Time Gesture Recognition", resized_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
