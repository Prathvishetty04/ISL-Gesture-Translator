import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

dataset_path = "dataset"

X = []
y = []

for label in os.listdir(dataset_path):

    label_path = os.path.join(dataset_path, label)

    for img_name in os.listdir(label_path):

        img_path = os.path.join(label_path, img_name)

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                wrist = hand_landmarks.landmark[0]

                features = []

                for lm in hand_landmarks.landmark:

                    features.append(lm.x - wrist.x)
                    features.append(lm.y - wrist.y)
                    features.append(lm.z)

                X.append(features)
                y.append(label)

np.save("X_data.npy", np.array(X))
np.save("y_labels.npy", np.array(y))

print("Feature extraction done")