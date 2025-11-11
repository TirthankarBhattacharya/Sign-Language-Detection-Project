import os
import glob
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    parser = argparse.ArgumentParser(description="Train a sign classifier from collected CSVs.")
    parser.add_argument("--rawdir", default=os.path.join("..", "data", "raw"))
    parser.add_argument("--outdir", default=os.path.join("..", "models"))
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    csv_files = glob.glob(os.path.join(args.rawdir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.rawdir}. Collect data first.")

    dfs = [pd.read_csv(p) for p in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values
    y = df["label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=args.test_size, stratify=y_enc, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print("Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump({"model": pipe, "label_encoder": le, "feature_names": feature_cols},
                os.path.join(args.outdir, "sign_classifier.joblib"))
    print(f"Saved model to {os.path.join(args.outdir, 'sign_classifier.joblib')}")

if __name__ == "__main__":
    main()
