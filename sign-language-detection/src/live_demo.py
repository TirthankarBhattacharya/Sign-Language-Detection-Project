import os
import cv2
import joblib
import numpy as np
from utils.hand_landmarks import HandLandmarkExtractor

MODEL_PATH = os.path.join("..", "models", "sign_classifier.joblib")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it with train_classifier.py")
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["label_encoder"]

def main():
    model, le = load_model()
    extractor = HandLandmarkExtractor()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = extractor.process(frame)
        vis = frame.copy()
        vis = extractor.draw_landmarks(vis, results)

        text = "No hand"
        if results and results.multi_hand_landmarks:
            feats, _ = extractor.landmarks_to_features(results)
            if feats is not None:
                feats = feats.reshape(1, -1)
                pred = model.predict(feats)[0]
                label = le.inverse_transform([pred])[0]
                proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(feats).max()
                text = f"{label}" + (f" ({proba:.2f})" if proba is not None else "")

        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(vis, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Sign Language - Live Demo", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
