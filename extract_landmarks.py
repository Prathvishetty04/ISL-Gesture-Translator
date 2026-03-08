import cv2
import mediapipe as mp
import os
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

dataset_path = "dataset"

data = []
labels = []

for gesture in os.listdir(dataset_path):

    gesture_path = os.path.join(dataset_path, gesture)

    for img_name in os.listdir(gesture_path):

        img_path = os.path.join(gesture_path, img_name)

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                features = []

                for lm in hand_landmarks.landmark:
                    features.append(lm.x)
                    features.append(lm.y)

                data.append(features)
                labels.append(gesture)

data = np.array(data)
labels = np.array(labels)

np.save("X_data.npy", data)
np.save("y_labels.npy", labels)

print("Landmark extraction complete")
print("Total samples:", len(data))