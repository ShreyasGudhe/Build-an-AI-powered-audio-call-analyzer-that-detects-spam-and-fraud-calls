import pickle
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

from faster_whisper import WhisperModel

from .feature_extraction import extract_features


base_dir = Path(__file__).resolve().parents[1]
model_path = base_dir / "models" / "voice_model.pkl"
with model_path.open("rb") as f:
    model = pickle.load(f)

LANGUAGE_NAMES = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "ml": "Malayalam",
    "te": "Telugu",
}

_language_executor = ThreadPoolExecutor(max_workers=1)
_language_model: Optional[WhisperModel] = None
_language_future: Optional[Future[WhisperModel]] = None


def _load_language_model() -> WhisperModel:
    return WhisperModel("tiny", device="cpu", compute_type="int8")


def detect_language(audio_path: Path) -> Tuple[str, float]:
    """Detect language code and probability using Whisper; fallback to unknown."""
    global _language_future, _language_model

    if _language_model is None:
        if _language_future is None:
            _language_future = _language_executor.submit(_load_language_model)
        if _language_future.done():
            try:
                _language_model = _language_future.result()
            except Exception:
                _language_model = None
                _language_future = None
                return "unknown", 0.0
        else:
            return "unknown", 0.0

    try:
        _, info = _language_model.transcribe(
            str(audio_path), beam_size=1, without_timestamps=True
        )
        code = info.language or "unknown"
        probability = float(info.language_probability or 0.0)
        return code, probability
    except Exception:
        return "unknown", 0.0


def resolve_language_name(code: str) -> str:
    if code == "unknown":
        return "Unknown"
    return LANGUAGE_NAMES.get(code, code.upper())


def predict_audio(audio_path) -> Dict[str, object]:
    path = Path(audio_path)
    features = extract_features(path)
    prob = model.predict_proba([features])[0]
    prediction = model.predict([features])[0]

    label = "AI-generated" if prediction == 1 else "Human"
    confidence = max(prob)

    explanation = (
        "Synthetic spectral patterns detected"
        if label == "AI-generated"
        else "Natural pitch and prosody detected"
    )

    lang_code, lang_prob = detect_language(path)

    return {
        "label": label,
        "confidence": float(confidence),
        "explanation": explanation,
        "language_code": lang_code,
        "language_name": resolve_language_name(lang_code),
        "language_confidence": round(float(lang_prob), 2),
    }
