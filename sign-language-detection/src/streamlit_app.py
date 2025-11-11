import os
import time
import cv2
import joblib
import numpy as np
import streamlit as st
from utils.hand_landmarks import HandLandmarkExtractor

st.set_page_config(page_title="Sign Language Detection", layout="centered")

MODEL_PATH = os.path.join("..", "models", "sign_classifier.joblib")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it with train_classifier.py")
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["label_encoder"]

st.title("🖐️ Sign Language Detection")
st.markdown("Real-time hand sign recognition using MediaPipe + scikit-learn.")

model, le = load_model()
extractor = HandLandmarkExtractor()

# Simple webcam loop using OpenCV
run = st.checkbox("Enable webcam", value=False)
frame_window = st.image([], caption="Webcam")
label_text = st.empty()

cap = None
if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        run = False

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("No frame captured.")
        break

    results = extractor.process(frame)
    vis = frame.copy()
    vis = extractor.draw_landmarks(vis, results)

    pred_text = "No hand"
    if results and results.multi_hand_landmarks:
        feats, _ = extractor.landmarks_to_features(results)
        if feats is not None:
            feats = feats.reshape(1, -1)
            pred = model.predict(feats)[0]
            label = le.inverse_transform([pred])[0]
            pred_text = label

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    frame_window.image(vis_rgb, caption="Webcam", channels="RGB")
    label_text.markdown(f"### Prediction: **{pred_text}**")

    # small delay to keep UI responsive
    time.sleep(0.01)

if cap is not None:
    cap.release()
