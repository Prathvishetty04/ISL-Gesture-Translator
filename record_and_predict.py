import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3

# load model
model = pickle.load(open("gesture_model.pkl","rb"))

# speech engine
engine = pyttsx3.init('sapi5')
engine.setProperty('rate',150)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

recording = False
frames = []

print("Press R to start recording")
print("Press S to stop recording")
print("Press ESC to exit")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # draw skeleton
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,
                                   hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

    if recording:
        frames.append(frame)
        cv2.putText(frame,"RECORDING",
                    (10,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,0,255),3)

    cv2.imshow("ISL Recognition",frame)

    key = cv2.waitKey(1)

    if key == ord('r'):
        recording = True
        frames = []
        print("Recording started")

    if key == ord('s'):

        recording = False
        print("Processing video...")

        predictions = []

        for frame in frames:

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:

                    features = []

                    for lm in hand_landmarks.landmark:
                        features.append(lm.x)
                        features.append(lm.y)

                    pred = model.predict([features])[0]
                    predictions.append(pred)

        # remove duplicates to detect gesture change
        sentence = []

        for p in predictions:
            if len(sentence) == 0 or p != sentence[-1]:
                sentence.append(p)

        final_sentence = " ".join(sentence)

        print("Sentence:", final_sentence)

        # speak full sentence
        engine.say(final_sentence)
        engine.runAndWait()

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()