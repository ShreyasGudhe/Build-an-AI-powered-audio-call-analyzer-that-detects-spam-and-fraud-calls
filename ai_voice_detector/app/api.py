import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .predict import predict_audio, predict_audio_bytes

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
    try:
        audio_bytes = base64.b64decode(audio_base64)
        if not audio_bytes:
            raise ValueError("empty audio")
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail="Invalid audio payload") from exc

    prediction = predict_audio_bytes(audio_bytes)

    return {
        "classification": prediction["label"],
        "confidence": round(float(prediction["confidence"]), 2),
        "explanation": prediction["explanation"],
        "language": prediction["language_name"],
        "language_confidence": prediction["language_confidence"],
        "language_code": prediction["language_code"],
    }
