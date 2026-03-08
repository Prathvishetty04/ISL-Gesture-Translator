import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import threading
import time
from collections import deque

# -----------------------------
# Load model
# -----------------------------

model = pickle.load(open("gesture_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# -----------------------------
# Speech engine
# -----------------------------

engine = pyttsx3.init()
speech_lock = threading.Lock()

def speak(text):

    def run():

        if speech_lock.locked():
            return

        with speech_lock:
            engine.say(text)
            engine.runAndWait()

    threading.Thread(target=run).start()

# -----------------------------
# MediaPipe
# -----------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -----------------------------
# UI
# -----------------------------

st.title("🤟 Indian Sign Language Translator")

st.write("Real-time gesture recognition with sentence generation")

start_camera = st.button("Start Camera")
clear_button = st.button("Clear Sentence")

col1,col2 = st.columns([2,1])

with col1:
    frame_placeholder = st.image([])

with col2:
    gesture_box = st.empty()
    sentence_box = st.empty()
    generated_box = st.empty()
    history_box = st.empty()

# -----------------------------
# State variables
# -----------------------------

prediction_history = deque(maxlen=10)

sentence = []
gesture_history = []

last_prediction = ""
last_spoken_sentence = ""

prediction = "None"
confidence = 0.0

cooldown_time = 2
last_added_time = 0

# -----------------------------
# Sentence generator
# -----------------------------

INTENT_MAP = {
    ("help",): "I need help.",
    ("help","yes"): "Yes, I need help.",
    ("hello",): "Hello.",
    ("hello","help"): "Hello, I need help.",
    ("hello","help","yes"): "Yes, I need help.",
    ("thanks",): "Thank you.",
    ("please","help"): "Please help me."
}

def generate_sentence(words):

    words = list(dict.fromkeys(words))

    key = tuple(sorted(words))

    if key in INTENT_MAP:
        return INTENT_MAP[key]

    return " ".join(words)

# -----------------------------
# Clear button
# -----------------------------

if clear_button:

    sentence.clear()
    gesture_history.clear()

# -----------------------------
# Camera loop
# -----------------------------

if start_camera:

    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        ret,frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # -----------------------------
                # Feature extraction
                # -----------------------------

                wrist = hand_landmarks.landmark[0]

                features = []

                for lm in hand_landmarks.landmark:

                    features.append(lm.x - wrist.x)
                    features.append(lm.y - wrist.y)
                    features.append(lm.z)

                features = scaler.transform([features])

                # -----------------------------
                # Prediction
                # -----------------------------

                probs = model.predict_proba(features)[0]

                confidence = np.max(probs)

                prediction = model.classes_[np.argmax(probs)]

                # -----------------------------
                # Stability filter
                # -----------------------------

                prediction_history.append(prediction)

                stable_prediction = None

                if prediction_history.count(prediction) > 7:
                    stable_prediction = prediction

                # -----------------------------
                # Sentence builder
                # -----------------------------

                if stable_prediction:

                    current_time = time.time()

                    if stable_prediction != last_prediction and current_time - last_added_time > cooldown_time:

                        sentence.append(stable_prediction)
                        gesture_history.append(stable_prediction)

                        gesture_history = gesture_history[-5:]

                        last_added_time = current_time

                    last_prediction = stable_prediction

                # -----------------------------
                # Draw prediction
                # -----------------------------

                cv2.putText(
                    frame,
                    f"{prediction} ({confidence:.2f})",
                    (10,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    3
                )

        # -----------------------------
        # Display camera
        # -----------------------------

        frame_placeholder.image(frame,channels="BGR")

        # -----------------------------
        # UI panels
        # -----------------------------

        if prediction == "None":
            gesture_box.write("### Current Gesture\nNo hand detected")
        else:
            gesture_box.write(f"### Current Gesture\n{prediction} ({confidence:.2f})")

        sentence_box.write("### Sentence")
        sentence_box.write(" → ".join(sentence))

        history_box.write("### Gesture History")
        history_box.write(" → ".join(gesture_history))

        # -----------------------------
        # Sentence generation
        # -----------------------------

        if len(sentence) >= 3:

            final_sentence = generate_sentence(sentence)

            generated_box.write("### Generated Sentence")
            generated_box.write(final_sentence)

            if final_sentence != last_spoken_sentence:

                speak(final_sentence)

                last_spoken_sentence = final_sentence

            sentence.clear()

    cap.release()