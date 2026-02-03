import os
import io
import pickle
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, Optional, Tuple, List

import requests
import numpy as np
import tempfile
import soundfile as sf

from faster_whisper import WhisperModel
from faster_whisper.transcribe import decode_audio
from pydub import AudioSegment
from pydub.silence import split_on_silence

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


# Fraud detection keywords - Enhanced categories
FRAUD_KEYWORDS = {
    "financial": ["otp", "cvv", "pin", "password", "account number", "card number", "ifsc", "upi", "net banking"],
    "identity": ["aadhaar", "pan", "passport", "voter id", "driving license", "kyc", "verification", "verify now"],
    "urgency": ["urgent", "immediately", "right now", "expire", "suspend", "block", "freeze", "deactivate"],
    "authority": ["police", "cyber cell", "income tax", "rbi", "court", "legal action", "arrest warrant", "fir"],
    "rewards": ["lottery", "prize", "won", "reward", "cashback", "refund", "claim now", "congratulations"],
    "payment": ["payment", "pay now", "transfer", "deposit", "remittance", "wire transfer", "transaction"],
    "threat": ["bank blocked", "account blocked", "suspended", "legal consequences", "penalty", "fine"],
}


def detect_fraud_keywords(audio_path: Path) -> Tuple[bool, str, List[str], Dict[str, int]]:
    """Enhanced transcribe audio and check for fraud-related keywords with categorization.
    
    Returns:
        (is_fraud, transcribed_text, detected_keywords, category_counts): 
        is_fraud is True if fraud keywords detected, with details on what was found.
    """
    try:
        text = _transcribe_snippet(audio_path, max_seconds=_MAX_AUDIO_SECONDS)
        if not text:
            return False, "", [], {}
        
        text_lower = text.lower()
        detected_keywords = []
        category_counts = {cat: 0 for cat in FRAUD_KEYWORDS.keys()}
        
        for category, keywords in FRAUD_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_keywords.append(keyword)
                    category_counts[category] += 1
        
        # Calculate fraud risk based on detection patterns
        is_fraud = len(detected_keywords) > 0
        
        # High-risk patterns: multiple categories or high-risk single keywords
        high_risk_keywords = ["otp", "cvv", "pin", "password", "account number", "police", "arrest warrant"]
        has_high_risk = any(kw in text_lower for kw in high_risk_keywords)
        multiple_categories = sum(1 for count in category_counts.values() if count > 0) >= 2
        
        # Enhanced fraud detection: mark as fraud if high-risk patterns detected
        is_fraud = is_fraud and (has_high_risk or multiple_categories or len(detected_keywords) >= 2)
        
        return is_fraud, text, list(set(detected_keywords)), category_counts
    except Exception:
        return False, "", [], {}


def predict_audio(audio_path) -> Dict[str, object]:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    # 1. AI vs Human voice detection using trained model
    features = extract_features(path)
    prob = model.predict_proba([features])[0]
    prediction = model.predict([features])[0]
    
    voice_type = "AI-generated" if prediction == 1 else "Human"
    voice_confidence = max(prob)
    voice_explanation = (
        "Synthetic spectral patterns detected"
        if voice_type == "AI-generated"
        else "Natural pitch and prosody detected"
    )
    
    # 2. Enhanced fraud keyword detection
    is_fraud, transcribed_text, detected_keywords, category_counts = detect_fraud_keywords(path)
    
    # Calculate fraud risk level based on findings
    fraud_risk_level = None
    fraud_confidence = 0.90
    if is_fraud:
        num_categories = sum(1 for count in category_counts.values() if count > 0)
        total_keywords = len(detected_keywords)
        
        if num_categories >= 3 or total_keywords >= 5:
            fraud_risk_level = "CRITICAL"
            fraud_confidence = 0.98
        elif num_categories >= 2 or total_keywords >= 3:
            fraud_risk_level = "HIGH"
            fraud_confidence = 0.95
        elif num_categories >= 1 or total_keywords >= 2:
            fraud_risk_level = "MEDIUM"
            fraud_confidence = 0.90
        else:
            fraud_confidence = 0.85
    
    # Build fraud explanation
    if is_fraud:
        fraud_label = "Fraud"
        fraud_explanation = f"Detected {len(detected_keywords)} fraud keywords across {sum(1 for c in category_counts.values() if c > 0)} categories: "
        fraud_explanation += ", ".join([f"{cat.title()} ({count})" for cat, count in category_counts.items() if count > 0])
    else:
        fraud_label = "Safe Call"
        fraud_explanation = "No fraud indicators detected in the audio content"
    
    # 3. Detect language
    lang_code, lang_prob = detect_language(path)

    # Combined result with both voice type and fraud detection
    return {
        # Fraud detection results
        "classification": fraud_label,
        "label": fraud_label,
        "confidence": float(fraud_confidence),
        "risk_level": fraud_risk_level,
        "explanation": fraud_explanation,
        "detected_patterns": detected_keywords if is_fraud else [],
        "fraud_categories": {k: v for k, v in category_counts.items() if v > 0} if is_fraud else {},
        
        # Voice type detection (AI vs Human)
        "voice_type": voice_type,
        "voice_confidence": round(float(voice_confidence), 2),
        "voice_explanation": voice_explanation,
        
        # Language detection
        "language_code": lang_code,
        "language_name": resolve_language_name(lang_code),
        "language": resolve_language_name(lang_code),
        "language_confidence": round(float(lang_prob), 2),
        
        # Transcript
        "transcribed_text": transcribed_text[:500] if transcribed_text else "",
    }


def predict_audio_bytes(audio_bytes: bytes, name_hint: str = "audio.mp3") -> Dict[str, object]:
    """Predict from in-memory audio bytes (avoids writing a temp file)."""
    # Write temp file for fraud detection and language detection
    suffix = Path(name_hint).suffix or ".mp3"
    tmp_path = Path(f"temp_fraud_{os.getpid()}_{id(audio_bytes)}{suffix}")
    try:
        tmp_path.write_bytes(audio_bytes)
        
        # 1. AI vs Human voice detection using trained model
        buf = io.BytesIO(audio_bytes)
        buf.name = name_hint
        features = extract_features(buf)
        prob = model.predict_proba([features])[0]
        prediction = model.predict([features])[0]
        
        voice_type = "AI-generated" if prediction == 1 else "Human"
        voice_confidence = max(prob)
        voice_explanation = (
            "Synthetic spectral patterns detected"
            if voice_type == "AI-generated"
            else "Natural pitch and prosody detected"
        )
        
        # 2. Enhanced fraud keyword detection
        is_fraud, transcribed_text, detected_keywords, category_counts = detect_fraud_keywords(tmp_path)
        
        # Calculate fraud risk level
        fraud_risk_level = None
        fraud_confidence = 0.90
        if is_fraud:
            num_categories = sum(1 for count in category_counts.values() if count > 0)
            total_keywords = len(detected_keywords)
            
            if num_categories >= 3 or total_keywords >= 5:
                fraud_risk_level = "CRITICAL"
                fraud_confidence = 0.98
            elif num_categories >= 2 or total_keywords >= 3:
                fraud_risk_level = "HIGH"
                fraud_confidence = 0.95
            elif num_categories >= 1 or total_keywords >= 2:
                fraud_risk_level = "MEDIUM"
                fraud_confidence = 0.90
            else:
                fraud_confidence = 0.85
        
        if is_fraud:
            fraud_label = "Fraud"
            fraud_explanation = f"Detected {len(detected_keywords)} fraud keywords across {sum(1 for c in category_counts.values() if c > 0)} categories"
        else:
            fraud_label = "Safe Call"
            fraud_explanation = "No fraud indicators detected in the audio content"
        
        # 3. Language detection
        lang_code, lang_prob = detect_language(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {
        # Fraud detection results
        "classification": fraud_label,
        "label": fraud_label,
        "confidence": float(fraud_confidence),
        "risk_level": fraud_risk_level,
        "explanation": fraud_explanation,
        "detected_patterns": detected_keywords if is_fraud else [],
        "fraud_categories": {k: v for k, v in category_counts.items() if v > 0} if is_fraud else {},
        
        # Voice type detection (AI vs Human)
        "voice_type": voice_type,
        "voice_confidence": round(float(voice_confidence), 2),
        "voice_explanation": voice_explanation,
        
        # Language detection
        "language_code": lang_code,
        "language_name": resolve_language_name(lang_code),
        "language": resolve_language_name(lang_code),
        "language_confidence": round(float(lang_prob), 2),
        
        # Transcript
        "transcribed_text": transcribed_text[:500] if transcribed_text else "",
    }


def analyze_conversation(audio_bytes: bytes) -> dict:
    """
    Analyze a two-way conversation from audio bytes.
    Separates speakers and analyzes each for fraud/spam indicators.
    
    Returns:
        dict: Analysis of both sides of the conversation including:
              - speaker_1: Analysis for first speaker
              - speaker_2: Analysis for second speaker (if detected)
              - conversation_summary: Overall assessment
              - is_fraud: Whether fraud detected from either side
              - fraud_source: Which speaker (if any) showed fraud indicators
    """
    try:
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            tmp_file.write(audio_bytes)
        
        # Load audio with pydub for speaker separation
        try:
            audio = AudioSegment.from_file(str(tmp_path))
        except Exception as e:
            # If conversion fails, try direct analysis
            return {
                "error": f"Could not process audio: {str(e)}",
                "fallback_analysis": predict_audio_bytes(audio_bytes)
            }
        
        # Split audio based on silence to identify different speakers
        # This is a simple heuristic - longer silence suggests turn-taking
        chunks = split_on_silence(
            audio,
            min_silence_len=800,  # 800ms of silence
            silence_thresh=audio.dBFS - 16,  # 16dB below average
            keep_silence=400  # Keep 400ms of silence for context
        )
        
        if len(chunks) == 0:
            # No clear speaker separation, analyze as single audio
            return {
                "speakers_detected": 1,
                "speaker_1": predict_audio_bytes(audio_bytes),
                "conversation_summary": {
                    "is_fraud": predict_audio_bytes(audio_bytes)["classification"] == "Fraud",
                    "fraud_source": "speaker_1" if predict_audio_bytes(audio_bytes)["classification"] == "Fraud" else None,
                    "overall_risk": predict_audio_bytes(audio_bytes).get("risk_level", "LOW")
                }
            }
        
        # Group chunks by speaker (simple alternating pattern assumption)
        # In real scenarios, you''d use voice embeddings to cluster
        speaker_1_chunks = []
        speaker_2_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i % 2 == 0:
                speaker_1_chunks.append(chunk)
            else:
                speaker_2_chunks.append(chunk)
        
        # Combine chunks for each speaker
        speaker_1_audio = sum(speaker_1_chunks) if speaker_1_chunks else None
        speaker_2_audio = sum(speaker_2_chunks) if speaker_2_chunks else None
        
        result = {
            "speakers_detected": 2 if speaker_2_audio else 1,
            "conversation_analysis": True
        }
        
        # Analyze speaker 1
        if speaker_1_audio:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as s1_file:
                s1_path = Path(s1_file.name)
                speaker_1_audio.export(str(s1_path), format='wav')
                
            with s1_path.open('rb') as f:
                speaker_1_bytes = f.read()
            
            result["speaker_1"] = predict_audio_bytes(speaker_1_bytes)
            result["speaker_1"]["duration_seconds"] = len(speaker_1_audio) / 1000.0
            result["speaker_1"]["speaking_turns"] = len(speaker_1_chunks)
            
            s1_path.unlink()
        
        # Analyze speaker 2
        if speaker_2_audio:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as s2_file:
                s2_path = Path(s2_file.name)
                speaker_2_audio.export(str(s2_path), format='wav')
                
            with s2_path.open('rb') as f:
                speaker_2_bytes = f.read()
            
            result["speaker_2"] = predict_audio_bytes(speaker_2_bytes)
            result["speaker_2"]["duration_seconds"] = len(speaker_2_audio) / 1000.0
            result["speaker_2"]["speaking_turns"] = len(speaker_2_chunks)
            
            s2_path.unlink()
        
        # Generate conversation summary
        speaker_1_fraud = result.get("speaker_1", {}).get("classification") == "Fraud"
        speaker_2_fraud = result.get("speaker_2", {}).get("classification") == "Fraud" if "speaker_2" in result else False
        
        speaker_1_ai = result.get("speaker_1", {}).get("voice_type") == "AI-generated"
        speaker_2_ai = result.get("speaker_2", {}).get("voice_type") == "AI-generated" if "speaker_2" in result else False
        
        # Determine overall fraud risk
        fraud_sources = []
        if speaker_1_fraud:
            fraud_sources.append("speaker_1")
        if speaker_2_fraud:
            fraud_sources.append("speaker_2")
        
        # Get highest risk level
        risk_levels = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        speaker_1_risk = result.get("speaker_1", {}).get("risk_level", "LOW")
        speaker_2_risk = result.get("speaker_2", {}).get("risk_level", "LOW") if "speaker_2" in result else "LOW"
        
        overall_risk = speaker_1_risk if risk_levels.get(speaker_1_risk, 0) >= risk_levels.get(speaker_2_risk, 0) else speaker_2_risk
        
        result["conversation_summary"] = {
            "is_fraud": speaker_1_fraud or speaker_2_fraud,
            "fraud_detected_from": fraud_sources if fraud_sources else None,
            "overall_risk_level": overall_risk if fraud_sources else "SAFE",
            "ai_voices_detected": speaker_1_ai or speaker_2_ai,
            "ai_voice_from": [s for s, is_ai in [("speaker_1", speaker_1_ai), ("speaker_2", speaker_2_ai)] if is_ai],
            "recommendation": _generate_conversation_recommendation(speaker_1_fraud, speaker_2_fraud, speaker_1_ai, speaker_2_ai, overall_risk)
        }
        
        # Cleanup
        tmp_path.unlink()
        
        return result
        
    except Exception as e:
        return {
            "error": f"Conversation analysis failed: {str(e)}",
            "fallback_analysis": predict_audio_bytes(audio_bytes)
        }


def _generate_conversation_recommendation(s1_fraud: bool, s2_fraud: bool, s1_ai: bool, s2_ai: bool, risk_level: str) -> str:
    """Generate actionable recommendation based on conversation analysis"""
    if s1_fraud and s2_fraud:
        return "CRITICAL: Both parties showing fraud indicators - terminate call immediately and report"
    elif s1_fraud:
        return f"WARNING: Caller showing fraud patterns ({risk_level} risk) - exercise extreme caution"
    elif s2_fraud:
        return f"WARNING: Recipient showing fraud patterns ({risk_level} risk) - verify their identity"
    elif s1_ai or s2_ai:
        speakers = []
        if s1_ai:
            speakers.append("caller")
        if s2_ai:
            speakers.append("recipient")
        return f"ALERT: AI-generated voice detected from {'' and ''.join(speakers)} - potential deepfake scam"
    else:
        return "No immediate fraud indicators detected - conversation appears legitimate"
