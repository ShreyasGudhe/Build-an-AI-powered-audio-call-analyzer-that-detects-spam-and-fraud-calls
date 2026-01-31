# AI Voice Detector

Project skeleton mirroring requested layout. Replace placeholder logic with real audio processing and model code.

## Structure
- dataset/
  - human/: human speech audio samples
  - ai/: AI-generated speech samples
- app/: core code (feature extraction, training, prediction, API hook)
- models/: serialized model artifacts
- requirements.txt: Python deps

## Quick start
1. Create/activate a virtualenv.
2. Install deps: `pip install -r requirements.txt`.
3. Train: `python -m app.train_model` (after adding real dataset + code).
4. API: `python -m uvicorn ai_voice_detector.app.api:app --host 127.0.0.1 --port 8000`.
5. Frontend (optional): `cd frontend && npm install && npm run dev`.

### Language detection
- The API now returns both the voice classification and a language guess among English, Tamil, Hindi, Malayalam, and Telugu using a Whisper language detector (`faster-whisper` package).
- The detector downloads a pretrained model the first time it runs; keep the server terminal open to reuse the cached model.
