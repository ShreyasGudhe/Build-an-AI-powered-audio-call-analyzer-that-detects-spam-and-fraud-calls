"""Train a supervised language classifier from dataset/lang_<code> folders.

Each subfolder name after "lang_" is treated as the language code (e.g., lang_en, lang_hi).
Outputs a pickle with the model and class list at models/language_model.pkl.
"""
from pathlib import Path
import pickle
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

try:
    from .feature_extraction import extract_features
except ImportError:  # pragma: no cover - fallback for script-style execution
    from feature_extraction import extract_features


AUDIO_PATTERNS = ("*.mp3", "*.wav", "*.m4a", "*.flac")


def load_language_dataset(dataset_root: Path) -> Tuple[List[np.ndarray], List[str]]:
    features_list: List[np.ndarray] = []
    labels: List[str] = []

    for folder in sorted(dataset_root.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        if not name.startswith("lang_"):
            continue
        lang_code = name.split("lang_", 1)[-1].strip()
        if not lang_code:
            continue

        files = []
        for pattern in AUDIO_PATTERNS:
            files.extend(folder.glob(pattern))

        if not files:
            continue

        file_count = 0
        for audio_file in files:
            try:
                # Use language-specific feature extraction
                feats = extract_features(audio_file, for_language=True)
                features_list.append(feats)
                labels.append(lang_code)
                file_count += 1
            except Exception as e:
                # Skip files that fail feature extraction
                print(f"Warning: Failed to extract features from {audio_file}: {e}")
                continue
        
        print(f"Loaded {file_count} files for language: {lang_code}")

    return features_list, labels


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "dataset"
    model_path = project_root / "models" / "language_model.pkl"

    X_list, y_list = load_language_dataset(dataset_root)
    if not X_list:
        raise SystemExit(
            "No audio files found under dataset/lang_<code> (e.g., dataset/lang_en)."
        )

    X = np.vstack(X_list)
    y = np.array(y_list)

    # Print dataset statistics
    unique_langs, counts = np.unique(y, return_counts=True)
    print(f"\nDataset Statistics:")
    for lang, count in zip(unique_langs, counts):
        print(f"  {lang}: {count} samples")
    print(f"Total samples: {len(y)}\n")

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    # Use a more robust classifier for better language detection
    from sklearn.ensemble import RandomForestClassifier
    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Handle imbalanced datasets
            random_state=42,
            n_jobs=-1
        ),
    )
    clf.fit(X, y_enc)

    # Calculate and display training accuracy
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, X, y_enc, cv=min(5, len(X_list)), scoring='accuracy')
    print(f"Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    payload = {"model": clf, "classes": encoder.classes_.tolist()}

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"\nLanguage model trained and saved to {model_path}")
    print(f"Supported languages: {', '.join(encoder.classes_.tolist())}")


if __name__ == "__main__":
    main()
