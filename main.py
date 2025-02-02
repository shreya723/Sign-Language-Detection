import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
labels_dict = {27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9', 36:'Thank You'}  # Update as needed

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot capture frame.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    frame_data = []
    x_left, y_left, x_right, y_right = [], [], [], []

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            x_, y_ = [], []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            # Normalize landmarks
            for lm in hand_landmarks.landmark:
                frame_data.append(lm.x - min(x_))
                frame_data.append(lm.y - min(y_))

            # Assign to left or right based on index
            if idx == 0:
                x_left, y_left = x_, y_
            elif idx == 1:
                x_right, y_right = x_, y_

        # Pad missing hand with zeros
        if not x_left:
            frame_data.extend([0.0] * 42)
        if not x_right:
            frame_data.extend([0.0] * 42)

        # Ensure 84 features
        if len(frame_data) == 84:
            prediction = model.predict([np.asarray(frame_data)])
            predicted_class = int(prediction[0])
            predicted_action = labels_dict.get(predicted_class, "Unknown")

            cv2.putText(frame, f"Action: {predicted_action}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)

    else:
        cv2.putText(frame, "No hands detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
