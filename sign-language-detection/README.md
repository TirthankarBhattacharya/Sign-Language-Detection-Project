# Sign Language Detection (Lightweight, Real-Time)

A complete, college-ready project for hand sign recognition using **MediaPipe Hands** for landmark detection
and a **scikit-learn** classifier. Runs on CPU and standard laptops—no GPU or heavy training required.

## What You Can Demo
- Live webcam detection (`python src/live_demo.py`)
- One-click Streamlit demo (`streamlit run src/streamlit_app.py`)
- Collect your own data and train your own model (4–6 signs in ~10 minutes)

---

## Project Structure
```text
sign-language-detection/
├─ data/
│  ├─ raw/              # raw collection CSVs
│  └─ processed/        # merged dataset files
├─ models/
│  └─ (your trained model will be saved here)
├─ src/
│  ├─ collect_data.py   # capture landmarks with labels using your webcam
│  ├─ train_classifier.py   # trains a classifier from collected data
│  ├─ live_demo.py      # real-time webcam prediction using your trained model
│  ├─ streamlit_app.py  # nice UI demo
│  └─ utils/
│     └─ hand_landmarks.py  # landmark extraction helpers
├─ requirements.txt
└─ README.md
```

---

## Quick Start

1. **Install dependencies (Python 3.9–3.11 recommended)**
   ```bash
   pip install -r requirements.txt
   ```

2. **Collect data for your signs** (example signs: `Hello`, `Yes`, `No`, `Thanks`)
   ```bash
   python src/collect_data.py --label Hello --samples 200
   python src/collect_data.py --label Yes --samples 200
   python src/collect_data.py --label No --samples 200
   python src/collect_data.py --label Thanks --samples 200
   ```
   - Press **space** to capture a sample when your hand pose looks correct.
   - Press **q** to quit early.
   - Repeat for each label. CSV files will appear in `data/raw/`.

3. **Train the model**
   ```bash
   python src/train_classifier.py --test-size 0.2
   ```
   - This creates `models/sign_classifier.joblib` and prints accuracy.

4. **Run the real-time demo**
   ```bash
   python src/live_demo.py
   ```
   - A window will open. Raise your hand and hold your sign—prediction appears at the top.

5. **Optional: Streamlit demo UI**
   ```bash
   streamlit run src/streamlit_app.py
   ```

---

## Tips for Best Results
- Keep your hand well-lit and centered; avoid fast motion while capturing samples.
- Collect **balanced** data: roughly the same number of samples for each label.
- Include small variations (distance to camera, angle, slight rotation).
- If doing two-handed signs, start with single-handed ones first; the included pipeline supports one hand.

## Notes
- This project uses 21 hand landmarks (x, y, z) → 63 features + handedness.
- Model: `StandardScaler` → `RandomForestClassifier` (easy to train, robust).
- You can swap the classifier in `train_classifier.py` (e.g., SVM/KNN) if you want.

## Credits
- Hand tracking by [MediaPipe Hands].
- Built for an easy, impressive college demo on 2025-08-18.
