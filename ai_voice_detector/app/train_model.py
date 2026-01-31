import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:  # allow both `python -m ai_voice_detector.app.train_model` and direct execution
    from .feature_extraction import extract_features
except ImportError:  # pragma: no cover - fallback for script-style execution
    from feature_extraction import extract_features


def load_dataset(dataset_root: Path) -> Tuple[List[np.ndarray], List[int]]:
    features_list: List[np.ndarray] = []
    labels: List[int] = []

    for label, folder in enumerate(["human", "ai"]):
        path = dataset_root / folder
        if not path.exists():
            print(f"⚠️  Missing folder: {path}")
            continue

        for audio_file in path.glob("*.mp3"):
            features = extract_features(audio_file)
            features_list.append(features)
            labels.append(label)

    return features_list, labels


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "dataset"
    model_path = project_root / "models" / "voice_model.pkl"

    X_list, y_list = load_dataset(dataset_root)
    if not X_list:
        raise SystemExit("No .mp3 files found under dataset/human or dataset/ai")

    X = np.vstack(X_list)
    y = np.array(y_list)

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model trained and saved to {model_path}")


if __name__ == "__main__":
    main()
