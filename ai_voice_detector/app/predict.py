import os
import io
import pickle
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, Optional, Tuple, List

import requests
import numpy as np

from faster_whisper import WhisperModel
from faster_whisper.transcribe import decode_audio

from .feature_extraction import extract_features


base_dir = Path(__file__).resolve().parents[1]
model_path = base_dir / "models" / "voice_model.pkl"
with model_path.open("rb") as f:
    model = pickle.load(f)

lang_model_path = base_dir / "models" / "language_model.pkl"
_lang_clf = None
_lang_classes: Optional[List[str]] = None
_lang_lock = Lock()

LANGUAGE_NAMES = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "ml": "Malayalam",
    "te": "Telugu",
}

SUPPORTED_LANGUAGE_CODES = set(LANGUAGE_NAMES)

LANGUAGE_ALIASES = {
    # Normalize ISO-639-3 variants to the supported ISO-639-1 codes.
    # IMPORTANT: Do not map other languages into supported ones; return "unknown" instead
    # of mislabeling the language name.
    "hin": "hi",  # Hindi ISO-639-3 → hi
    "tam": "ta",  # Tamil ISO-639-3 → ta
    "tel": "te",  # Telugu ISO-639-3 → te
    "eng": "en",  # English ISO-639-3 → en
    "mal": "ml",  # Malayalam ISO-639-3 → ml
}


def _predict_language_via_classifier(audio_path: Path) -> Optional[Tuple[str, float]]:
    clf, classes = _ensure_language_classifier()
    if clf is None or not classes:
        return None

    try:
        # Use language-specific features for better accuracy
        features = extract_features(audio_path, for_language=True)
        probs = clf.predict_proba([features])[0]
        idx = int(np.argmax(probs))
        code = _normalize_code(classes[idx])
        prob = float(probs[idx])
        
        # Strict validation: check if result is clear enough
        sorted_probs = sorted(probs, reverse=True)
        if len(sorted_probs) > 1:
            margin = sorted_probs[0] - sorted_probs[1]
            # Require good separation between top two choices
            if margin < 0.15:
                return None  # Too ambiguous
        
        # Must meet minimum probability
        if prob < 0.35:
            return None
            
    except Exception:
        return None

    if code not in SUPPORTED_LANGUAGE_CODES:
        return None
    return code, prob

_language_model: Optional[WhisperModel] = None
_language_lock = Lock()
_MAX_AUDIO_SECONDS = 30
_MAX_LANG_SECONDS = int(os.getenv("MAX_LANG_SECONDS", "8"))
_WHISPER_SAMPLE_RATE = 16000
# Language decision parameters
_MIN_LANG_PROB = float(os.getenv("MIN_LANG_PROB", "0.50"))  # minimum confidence to accept a language
_PROB_MARGIN = float(os.getenv("LANG_PROB_MARGIN", "0.15"))    # winner must beat runner-up by this margin

# Optional external detectors (configure via environment)
LANG_API_URL = os.getenv("LANGUAGE_API_URL")
LANG_API_TOKEN = os.getenv("LANGUAGE_API_TOKEN")
DETECTLANG_API_KEY = os.getenv("DETECTLANG_API_KEY")
DETECTLANG_API_URL = os.getenv("DETECTLANG_API_URL", "https://ws.detectlanguage.com/0.2/detect")


def _load_language_model() -> WhisperModel:
    size = os.getenv("WHISPER_MODEL_SIZE", "base")
    try:
        return WhisperModel(size, device="cpu", compute_type="int8")
    except Exception:
        return WhisperModel("tiny", device="cpu", compute_type="int8")


def _ensure_language_model() -> WhisperModel:
    global _language_model
    if _language_model is not None:
        return _language_model

    with _language_lock:
        if _language_model is None:
            _language_model = _load_language_model()
    return _language_model


def _ensure_language_classifier():
    global _lang_clf, _lang_classes
    if _lang_clf is not None and _lang_classes is not None:
        return _lang_clf, _lang_classes

    if not lang_model_path.exists():
        return None, None

    with _lang_lock:
        if _lang_clf is not None and _lang_classes is not None:
            return _lang_clf, _lang_classes
        try:
            payload = pickle.load(lang_model_path.open("rb"))
            if isinstance(payload, dict):
                clf = payload.get("model")
                classes = payload.get("classes")
            else:
                clf = payload
                classes = getattr(payload, "classes_", None)

            if clf is None or classes is None:
                return None, None

            _lang_clf = clf
            _lang_classes = list(classes)
        except Exception:
            return None, None

    return _lang_clf, _lang_classes


def _normalize_code(code: str) -> str:
    norm = (code or "unknown").lower()
    return LANGUAGE_ALIASES.get(norm, norm)


def _detect_language_via_api(audio_path: Path) -> Optional[Tuple[str, float]]:
    if not LANG_API_URL:
        return None

    headers = {}
    if LANG_API_TOKEN:
        headers["Authorization"] = f"Bearer {LANG_API_TOKEN}"

    files = {
        "file": (audio_path.name, audio_path.open("rb"), "audio/mpeg"),
    }

    try:
        resp = requests.post(LANG_API_URL, headers=headers, files=files, timeout=20)
        if not resp.ok:
            return None
        data = resp.json()
    except Exception:
        return None

    code = (
        data.get("language")
        or data.get("language_code")
        or data.get("detected_language")
    )
    prob = (
        data.get("language_probability")
        or data.get("probability")
        or data.get("confidence")
        or 0.0
    )

    if not code:
        return None

    normalized = _normalize_code(str(code))
    probability = float(prob or 0.0)
    return normalized, probability


def _prepare_audio_for_lang(audio: np.ndarray, sr: int) -> np.ndarray:
    """Trim leading/trailing silence and cap duration for faster, stable language detection."""
    if audio.size == 0:
        return audio

    # Simple energy-based VAD
    frame = max(1, int(0.02 * sr))  # 20 ms
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    # Normalize
    max_abs = np.max(np.abs(audio)) or 1.0
    a = audio / max_abs
    # Compute moving RMS
    window = np.ones(frame) / frame
    rms = np.sqrt(np.convolve(a ** 2, window, mode="same") + 1e-8)
    thr = 0.02  # empirical threshold
    idx = np.where(rms > thr)[0]
    if idx.size > 0:
        start = max(0, idx[0] - frame)
        end = min(len(a), idx[-1] + frame)
        a = a[start:end]
    # Cap duration
    max_frames = _WHISPER_SAMPLE_RATE * _MAX_LANG_SECONDS
    if a.shape[0] > max_frames:
        a = a[:max_frames]
    return a.astype(np.float32)


def _transcribe_snippet(audio_path: Path, max_seconds: int = None) -> Optional[str]:
    """Transcribe a short snippet to text for downstream language detection APIs."""
    text, _, _ = _transcribe_snippet_with_language(audio_path)
    return text


def _transcribe_snippet_with_language(
    audio_path: Path | np.ndarray,
    model: WhisperModel | None = None,
) -> Tuple[Optional[str], Optional[str], float]:
    """Transcribe a prepared audio snippet and return text plus Whisper's language guess."""
    try:
        model = model or _ensure_language_model()
    except Exception:
        return None, None, 0.0

    try:
        if isinstance(audio_path, np.ndarray):
            audio = audio_path
        else:
            audio = decode_audio(audio_path, sampling_rate=_WHISPER_SAMPLE_RATE)
    except Exception:
        return None, None, 0.0

    if audio.size == 0:
        return None, None, 0.0

    # Prepare audio quickly for language-related tasks
    audio = _prepare_audio_for_lang(audio, _WHISPER_SAMPLE_RATE)

    try:
        # Use a light beam search and VAD to stabilize short-snippet language ID
        segments, info = model.transcribe(
            audio,
            beam_size=3,
            without_timestamps=True,
            vad_filter=True,
        )
    except Exception:
        return None, None, 0.0

    texts = [seg.text.strip() for seg in segments if getattr(seg, "text", "").strip()]
    text = " ".join(texts)[:500] if texts else None

    lang = _normalize_code(getattr(info, "language", None) or "unknown")
    lang_prob_raw = getattr(info, "language_probability", 0.0)
    try:
        lang_prob = float(lang_prob_raw or 0.0)
    except Exception:
        lang_prob = 0.0

    return text, lang, lang_prob


def _script_hint_from_text(text: str) -> Optional[str]:
    """Return a supported language code if text strongly matches a native script."""
    if not text:
        return None

    def _ratio_in_range(txt: str, start: int, end: int) -> float:
        chars = [ch for ch in txt if ch.isalpha()]
        if not chars:
            return 0.0
        in_block = sum(1 for ch in chars if start <= ord(ch) <= end)
        return in_block / max(1, len(chars))

    # Unicode blocks
    mal_ratio = _ratio_in_range(text, 0x0D00, 0x0D7F)  # Malayalam
    tam_ratio = _ratio_in_range(text, 0x0B80, 0x0BFF)  # Tamil
    tel_ratio = _ratio_in_range(text, 0x0C00, 0x0C7F)  # Telugu
    dev_ratio = _ratio_in_range(text, 0x0900, 0x097F)  # Devanagari (Hindi)

    ratios = {"ml": mal_ratio, "ta": tam_ratio, "te": tel_ratio, "hi": dev_ratio}
    best_code = max(ratios, key=ratios.get)
    # Require substantial native script presence (at least 30%)
    if ratios[best_code] >= 0.30 and best_code in SUPPORTED_LANGUAGE_CODES:
        return best_code
    return None


def _detect_language_via_libretranslate_text(text: str) -> Optional[Tuple[str, float]]:
    if not text:
        return None

    try:
        resp = requests.post(
            "https://libretranslate.de/detect", json={"q": text}, timeout=15
        )
        if not resp.ok:
            return None
        data = resp.json()
    except Exception:
        return None

    if not isinstance(data, list) or not data:
        return None

    best = data[0]
    code = best.get("language")
    prob = best.get("confidence", 0.0)
    if not code:
        return None

    normalized = _normalize_code(str(code))
    probability = float(prob or 0.0)
    return normalized, probability


def _detect_language_via_libretranslate(audio_path: Path) -> Optional[Tuple[str, float]]:
    """Detect language using LibreTranslate's /detect endpoint by sending transcribed text."""
    text = _transcribe_snippet(audio_path)
    if not text:
        return None

    return _detect_language_via_libretranslate_text(text)


def _detect_language_via_detectlanguage_text(text: str) -> Optional[Tuple[str, float]]:
    """Detect language using detectlanguage.com API with API key (text-based)."""
    if not DETECTLANG_API_KEY or not text:
        return None

    headers = {
        "Authorization": f"Bearer {DETECTLANG_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"q": text}

    try:
        resp = requests.post(
            DETECTLANG_API_URL, json=payload, headers=headers, timeout=15
        )
        if not resp.ok:
            return None
        data = resp.json()
    except Exception:
        return None

    results = (
        data.get("data", {}).get("detections", []) if isinstance(data, dict) else []
    )
    if not results:
        return None

    best = results[0]
    code = best.get("language")
    prob = best.get("confidence", 0.0)
    reliable = best.get("isReliable", True)
    if not code or not reliable:
        return None

    normalized = _normalize_code(str(code))
    try:
        probability = float(prob or 0.0)
    except Exception:
        probability = 0.0
    probability = max(0.0, min(1.0, probability))
    return normalized, probability


def _detect_language_via_detectlanguage(audio_path: Path) -> Optional[Tuple[str, float]]:
    """Detect language using detectlanguage.com API with API key."""
    if not DETECTLANG_API_KEY:
        return None

    text = _transcribe_snippet(audio_path)
    if not text:
        return None

    return _detect_language_via_detectlanguage_text(text)


def _select_supported_language(
    primary_code: str,
    primary_prob: float,
    alternatives: Iterable[Tuple[str, float]] | None,
) -> Tuple[str, float]:
    all_candidates: Dict[str, float] = {}

    def register(candidate_code: str, prob: float) -> None:
        normalized = _normalize_code(candidate_code)
        if normalized == "unknown":
            return
        if normalized not in SUPPORTED_LANGUAGE_CODES:
            return
        stored = all_candidates.get(normalized, 0.0)
        if prob > stored:
            all_candidates[normalized] = prob

    register(primary_code, primary_prob)

    if alternatives:
        for code, prob in alternatives:
            register(code, float(prob))

    if not all_candidates:
        normalized = _normalize_code(primary_code)
        return normalized if normalized in SUPPORTED_LANGUAGE_CODES else "unknown", primary_prob

    # Sort to evaluate margin between best and runner-up
    sorted_candidates: List[Tuple[str, float]] = sorted(
        all_candidates.items(), key=lambda item: item[1], reverse=True
    )
    best_code, best_prob = sorted_candidates[0]
    runner_prob = sorted_candidates[1][1] if len(sorted_candidates) > 1 else 0.0

    # Strict validation
    if best_prob < _MIN_LANG_PROB:
        return "unknown", best_prob
    
    # Require clear winner
    if len(sorted_candidates) > 1 and best_prob - runner_prob < _PROB_MARGIN:
        return "unknown", best_prob

    return best_code, best_prob


def detect_language(audio_path: Path) -> Tuple[str, float]:
    """Detect one of the supported languages quickly and robustly.

    Strategy:
    1) If available, use the supervised language classifier trained on labeled data (most reliable).
    2) Whisper `detect_language()` on a short, voiced snippet (fast).
    3) Combine classifier and Whisper results for better accuracy.
    4) If uncertain, transcribe ONCE and apply: script heuristic → text-based APIs.
    5) Finally, optional file-based API (if configured).
    """
    
    all_detections = {}

    # Prefer supervised classifier when present (most reliable for our specific languages)
    clf_guess = _predict_language_via_classifier(audio_path)
    if clf_guess:
        code0, prob0 = clf_guess
        if code0 in SUPPORTED_LANGUAGE_CODES:
            all_detections[code0] = prob0
            # Only return immediately if very confident
            if prob0 >= 0.75:
                return code0, prob0

    try:
        model = _ensure_language_model()
    except Exception:
        # If no Whisper model and no classifier, return unknown
        if not all_detections:
            return "unknown", 0.0
        # Use classifier result as fallback
        best = max(all_detections.items(), key=lambda x: x[1])
        return best[0], best[1] if best[1] >= _MIN_LANG_PROB else ("unknown", best[1])

    try:
        audio = decode_audio(audio_path, sampling_rate=_WHISPER_SAMPLE_RATE)
        if audio.size == 0:
            if all_detections:
                best = max(all_detections.items(), key=lambda x: x[1])
                return best[0], best[1] if best[1] >= _MIN_LANG_PROB else ("unknown", best[1])
            return "unknown", 0.0

        # Use trimmed, short snippet for fast language ID
        audio = _prepare_audio_for_lang(audio, _WHISPER_SAMPLE_RATE)

        detected = model.detect_language(audio)
        if len(detected) == 2:
            code, probability = detected
            alternatives = None
        else:
            code, probability, alternatives = detected
    except Exception:
        # Use classifier result as fallback
        if all_detections:
            best = max(all_detections.items(), key=lambda x: x[1])
            return best[0], best[1] if best[1] >= _MIN_LANG_PROB else ("unknown", best[1])
        return "unknown", 0.0

    code = _normalize_code(code)
    probability = float(probability or 0.0)
    
    # Add Whisper detections to our collection
    if code in SUPPORTED_LANGUAGE_CODES:
        all_detections[code] = max(all_detections.get(code, 0.0), probability)
    
    filtered_code, filtered_prob = _select_supported_language(
        code, probability, alternatives
    )

    if filtered_code in SUPPORTED_LANGUAGE_CODES:
        all_detections[filtered_code] = max(all_detections.get(filtered_code, 0.0), filtered_prob)
    
    # Combine multiple detections - if classifier and Whisper agree, boost confidence
    if len(all_detections) > 0:
        best_code = max(all_detections.items(), key=lambda x: x[1])[0]
        best_prob = all_detections[best_code]
        
        # If both methods detected same language, boost confidence
        if clf_guess and clf_guess[0] == best_code:
            # Both agree - this is strong signal
            boosted_prob = min(0.95, best_prob + 0.25)
            if boosted_prob >= _MIN_LANG_PROB:
                return best_code, boosted_prob
        
        # Single method detection - be more strict
        if best_prob >= _MIN_LANG_PROB:
            return best_code, best_prob

    # If Whisper is uncertain, transcribe once and try stronger heuristics/APIs.
    text, tx_lang, tx_prob = _transcribe_snippet_with_language(audio, model)

    # Try Whisper's language from the transcription metadata; sometimes more stable
    if tx_lang and tx_lang in SUPPORTED_LANGUAGE_CODES and tx_prob >= 0.60:
        return tx_lang, float(tx_prob or 0.0)

    if text:
        hinted = _script_hint_from_text(text)
        if hinted:
            # Script detection is reliable but require substantial native script presence
            return hinted, 0.85

        detectlang_guess = _detect_language_via_detectlanguage_text(text)
        if detectlang_guess:
            code2, prob2 = detectlang_guess
            if code2 in SUPPORTED_LANGUAGE_CODES and prob2 >= 0.70:
                return code2, prob2

        libre_guess = _detect_language_via_libretranslate_text(text)
        if libre_guess:
            code2, prob2 = libre_guess
            if code2 in SUPPORTED_LANGUAGE_CODES and prob2 >= 0.70:
                return code2, prob2

    # Last resort: file-based API (if configured)
    api_guess = _detect_language_via_api(audio_path)
    if api_guess:
        code2, prob2 = api_guess
        if code2 in SUPPORTED_LANGUAGE_CODES and prob2 >= 0.70:
            return code2, prob2

    return "unknown", float(filtered_prob or 0.0)


def resolve_language_name(code: str) -> str:
    if code == "unknown":
        return "Unknown"
    return LANGUAGE_NAMES.get(code, code.upper())


def predict_audio(audio_path) -> Dict[str, object]:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
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


def predict_audio_bytes(audio_bytes: bytes, name_hint: str = "audio.mp3") -> Dict[str, object]:
    """Predict from in-memory audio bytes (avoids writing a temp file)."""
    # librosa can read file-like objects; attach a name to help format detection
    buf = io.BytesIO(audio_bytes)
    buf.name = name_hint

    features = extract_features(buf)
    prob = model.predict_proba([features])[0]
    prediction = model.predict([features])[0]

    label = "AI-generated" if prediction == 1 else "Human"
    confidence = max(prob)

    explanation = (
        "Synthetic spectral patterns detected"
        if label == "AI-generated"
        else "Natural pitch and prosody detected"
    )

    # For language detection, we still need a Path; write a short-lived temp file
    suffix = Path(name_hint).suffix or ".mp3"
    tmp_path = Path(f"temp_lang_{os.getpid()}_{id(buf)}{suffix}")
    try:
        tmp_path.write_bytes(audio_bytes)
        lang_code, lang_prob = detect_language(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {
        "label": label,
        "confidence": float(confidence),
        "explanation": explanation,
        "language_code": lang_code,
        "language_name": resolve_language_name(lang_code),
        "language_confidence": round(float(lang_prob), 2),
    }
