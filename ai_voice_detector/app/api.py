import base64
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .predict import predict_audio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    """Simple root endpoint to verify the API is running."""
    return {"status": "ok", "endpoint": "/detect-voice"}

@app.post("/detect-voice")
def detect_voice(data: dict):
    if "audio" not in data:
        raise HTTPException(status_code=400, detail="Missing 'audio' field")

    audio_base64 = data["audio"]

    temp_path = Path(f"temp_{uuid.uuid4()}.mp3")
    try:
        temp_path.write_bytes(base64.b64decode(audio_base64))
    except Exception as exc:  # pragma: no cover - defensive
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=400, detail="Invalid audio payload") from exc

    try:
        prediction = predict_audio(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return {
        "classification": prediction["label"],
        "confidence": round(float(prediction["confidence"]), 2),
        "explanation": prediction["explanation"],
        "language": prediction["language_name"],
        "language_confidence": prediction["language_confidence"],
        "language_code": prediction["language_code"],
    }
