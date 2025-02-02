import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

expected_num_landmarks = 21 * 4  # 21 landmarks * 2 (x, y) * 2 hands

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        data_aux = []
        x_left, y_left, x_right, y_right = [], [], [], []

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                x_, y_ = [], []

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                # Normalize landmarks relative to min x, y
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                # Assign left or right hand data based on index
                if idx == 0:
                    x_left, y_left = x_, y_
                elif idx == 1:
                    x_right, y_right = x_, y_

            # Pad with zeros if one hand is missing
            if not x_left:
                data_aux.extend([0.0] * 42)  # Add zeros for left hand
            if not x_right:
                data_aux.extend([0.0] * 42)  # Add zeros for right hand

        if len(data_aux) == expected_num_landmarks:
            data.append(data_aux)
            labels.append(dir_)

# Save data to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
