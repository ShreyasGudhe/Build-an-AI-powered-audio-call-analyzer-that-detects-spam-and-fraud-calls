# Sample Fraud Call Dataset Guide

## Dataset Structure

The fraud detection model requires two types of audio samples:

### 1. Fraud Calls (`dataset/fraud_calls/`)
Examples of fraudulent or spam calls including:
- IRS/Tax scam calls
- Tech support scams
- Bank/financial fraud
- Prize/lottery scams
- Social security scams
- Robocalls
- Phishing attempts

### 2. Legitimate Calls (`dataset/legitimate_calls/`)
Examples of normal, non-fraudulent calls:
- Customer service calls
- Business conversations
- Personal phone calls
- Appointment confirmations
- Professional inquiries

## Audio Format Requirements

- **Format**: MP3, WAV, M4A, FLAC, or OGG
- **Duration**: 10-60 seconds (optimal: 30 seconds)
- **Quality**: Clear audio with audible speech
- **Sample Rate**: 16kHz or higher recommended

## Dataset Size

For training an effective fraud detection model:
- **Minimum**: 50 samples per category (100 total)
- **Recommended**: 200+ samples per category (400+ total)
- **Optimal**: 500+ samples per category (1000+ total)

## Data Collection Tips

### Fraud Call Sources:
1. **Public Datasets**:
   - Search for "scam call recordings" on audio repositories
   - Academic fraud detection datasets
   - Open source scam call databases

2. **YouTube/Public Videos**:
   - Many creators share recordings of scam calls
   - Extract audio using tools like youtube-dl

3. **Personal Recordings**:
   - Record spam calls you receive (check local laws)
   - Share with consent from participants

### Legitimate Call Sources:
1. **Simulated Calls**:
   - Record sample business conversations
   - Create mock customer service scenarios

2. **Public Domain Audio**:
   - Professional call recordings
   - Training call examples

3. **Personal Archives**:
   - Saved voicemails
   - Recorded business calls (with consent)

## Data Organization

Place audio files directly in the respective directories:

```
dataset/
├── fraud_calls/
│   ├── scam_001.mp3
│   ├── scam_002.wav
│   ├── robocall_001.mp3
│   └── ...
└── legitimate_calls/
    ├── business_001.mp3
    ├── customer_service_001.wav
    ├── personal_001.mp3
    └── ...
```

## Sample Data Generation Script

If you need synthetic data for testing, run:

```bash
python test_fraud_detection.py --generate-samples
```

This will create sample audio files with different characteristics for testing purposes.

## Training the Model

Once you have sufficient data:

```bash
python -m app.train_fraud_model dataset/fraud_calls dataset/legitimate_calls
```

Or use the batch script:
```bash
train_fraud_model.bat
```

## Data Privacy & Ethics

⚠️ **Important Considerations**:

1. **Privacy**: Never share recordings with personal information
2. **Consent**: Ensure you have permission to record/use audio
3. **Legal**: Follow local laws regarding call recording
4. **Anonymization**: Remove identifying information before training
5. **Balanced Dataset**: Maintain similar quality across both categories

## Pre-trained Model

If you don't have access to training data, the system includes:
- Pre-configured fraud pattern detection (keyword-based)
- Audio behavioral analysis (no training required)
- Speech pattern analysis (works without custom model)

These features work immediately without a trained model!

## Next Steps

1. Collect or generate audio samples
2. Organize them into the appropriate directories
3. Run the training script
4. Test the model with new audio samples
5. Iterate and improve with more diverse data
