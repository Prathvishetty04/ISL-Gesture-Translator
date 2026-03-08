import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3

# Load trained model
model = pickle.load(open("gesture_model.pkl", "rb"))

# Text to speech engine
engine = pyttsx3.init()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

recording = False
frames = []
sentence = []

print("Press R to start recording")
print("Press S to stop recording")
print("Press C to clear sentence")
print("Press ESC to exit")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Draw hand skeleton
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    if recording:
        frames.append(frame)
        cv2.putText(frame, "RECORDING...", (10,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)

    cv2.putText(frame, "Sentence: " + " ".join(sentence),
                (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2)

    cv2.imshow("ISL Recognition System", frame)

    key = cv2.waitKey(1)

    # Start recording
    if key == ord('r'):
        recording = True
        frames = []
        print("Recording started")

    # Stop recording and process
    if key == ord('s'):

        recording = False
        print("Processing gesture...")

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

        if predictions:

            final_prediction = max(set(predictions), key=predictions.count)

            print("Predicted Gesture:", final_prediction)

            sentence.append(final_prediction)

            # Speak the word
            engine.say(final_prediction)
            engine.runAndWait()

    # Clear sentence
    if key == ord('c'):
        sentence = []
        print("Sentence cleared")

    # Exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()