import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from feature_extraction import extract_features


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    dataset_root = base / "dataset"
    model_path = base / "models" / "voice_model.pkl"

    X, y = [], []

    for label, folder in enumerate(["human", "ai"]):
        path = dataset_root / folder
        if not path.exists():
            print(f"No folder found at {path}; skipping.")
            continue

        for file in path.glob("*.mp3"):
            features = extract_features(file)
            X.append(features)
            y.append(label)

    if not X:
        print("No .mp3 files found in dataset/human or dataset/ai; add data and rerun.")
        return

    X_arr = np.vstack(X)
    y_arr = np.array(y)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_arr, y_arr)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model trained and saved to {model_path}")


if __name__ == "__main__":
    main()
