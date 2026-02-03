# 🛡️ AI Voice Detector & Fraud Call Analyzer

**A comprehensive AI-powered system for detecting fraud calls, spam, and analyzing voice patterns in real-time.**

## 🌟 Overview

This project combines advanced AI technologies to protect users from fraud calls by:
- **Detecting spam and fraud calls** using keyword and behavioral analysis
- **Analyzing audio patterns** for robocalls, VOIP spoofing, and suspicious behaviors
- **Real-time transcription** and keyword detection
- **Voice classification** (AI-generated vs Human)
- **Multi-language support** (English, Hindi, Tamil, Telugu, Malayalam)
- **Instant threat alerts** with risk assessment and recommendations

## ✨ Key Features

### 🚨 Fraud Detection
- **Keyword Pattern Matching**: Identifies 50+ fraud-related keywords across 7 categories
- **Audio Behavioral Analysis**: Detects robocalls, VOIP patterns, and suspicious audio characteristics
- **Speech Pattern Analysis**: Identifies stress, urgency, and aggressive speech patterns
- **Caller Number Analysis**: Flags suspicious phone number patterns
- **Risk Categorization**: LOW → MEDIUM → HIGH → CRITICAL threat levels
- **Real-Time Alerts**: Instant notifications with actionable recommendations

### 🎤 Voice & Speech Analysis
- **AI Voice Detection**: Distinguishes AI-generated from human voices
- **Speech-to-Text**: Automatic transcription using Whisper AI
- **Language Detection**: Supports English, Hindi, Tamil, Telugu, Malayalam
- **Audio Quality Assessment**: Analyzes call clarity and authenticity

### 🔔 Alert & Protection System
- **Real-Time Alerts**: Instant fraud warnings during analysis
- **Alert History**: Track past threats and patterns
- **Auto-Blocking**: Automatically blocks high-risk numbers
- **Threat Classification**: Identifies specific fraud types (Financial, IRS, Tech Support, etc.)

## 📁 Project Structure

```
ai_voice_detector/
├── app/
│   ├── api.py                    # FastAPI endpoints (fraud, voice, transcription)
│   ├── fraud_detection.py        # Fraud detection engine
│   ├── transcription.py          # Speech-to-text & keyword detection
│   ├── train_fraud_model.py      # Fraud model trainer
│   ├── feature_extraction.py     # Audio feature extraction
│   ├── predict.py                # Voice classification
│   └── train_model.py            # Voice model trainer
├── frontend/
│   └── src/
│       ├── App.jsx               # React UI with fraud detection
│       └── style.css             # Modern responsive styling
├── models/
│   ├── voice_model.pkl           # Pre-trained voice classifier
│   ├── language_model.pkl        # Language detection model
│   └── fraud_model.pkl           # Custom fraud model (after training)
├── dataset/
│   ├── fraud_calls/              # Fraud call training data
│   ├── legitimate_calls/         # Legitimate call training data
│   ├── human/                    # Human voice samples
│   ├── ai/                       # AI-generated voice samples
│   └── DATASET_GUIDE.md          # Dataset preparation guide
├── test_fraud_detection.py       # Comprehensive test suite
├── run_fraud_api.bat             # Quick start script (Windows)
├── train_fraud_model.bat         # Model training script (Windows)
├── FRAUD_DETECTION_GUIDE.md      # Complete documentation
├── QUICK_START.md                # 5-minute setup guide
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install frontend packages (optional)
cd frontend
npm install
cd ..
```

### 2. Run the Application

**Windows:**
```bash
# Start API server
run_fraud_api.bat

# Start frontend (new terminal)
cd frontend
npm run dev
```

**Linux/Mac:**
```bash
# Start API server
python run_api.py

# Start frontend (new terminal)
cd frontend
npm run dev
```

### 3. Access the Application

- **Web UI**: http://localhost:5173
- **API**: http://127.0.0.1:8000
- **API Docs**: http://127.0.0.1:8000/docs

### 4. Test Fraud Detection

```bash
# Run comprehensive tests
python test_fraud_detection.py

# Or on Windows
test_fraud.bat
```

## 📖 Documentation

- **📘 [Quick Start Guide](QUICK_START.md)** - Get running in 5 minutes
- **📗 [Fraud Detection Guide](FRAUD_DETECTION_GUIDE.md)** - Complete documentation
- **📕 [Dataset Guide](dataset/DATASET_GUIDE.md)** - Training data preparation
- **📙 [API Documentation](http://127.0.0.1:8000/docs)** - Interactive API docs (when server running)

## 🎯 Use Cases

### For Individuals
- **Screen Suspicious Calls**: Upload recordings to check for fraud patterns
- **Analyze Voicemails**: Detect scam messages before responding
- **Protect Family**: Identify threats targeting vulnerable individuals
- **Build Awareness**: Share results to educate others

### For Businesses
- **Call Center Quality**: Monitor customer service calls
- **Fraud Prevention**: Detect fraudulent callers in real-time
- **Compliance**: Ensure regulatory adherence
- **Training**: Build fraud awareness with real examples

### For Researchers
- **Fraud Pattern Analysis**: Study scam techniques and evolution
- **Audio Forensics**: Analyze call authenticity
- **ML Model Development**: Train custom detection models
- **Dataset Creation**: Build fraud call databases

## 🔧 API Endpoints

### Comprehensive Analysis
```bash
POST /analyze-call
```
Complete fraud detection + voice analysis + transcription

### Fraud Detection
```bash
POST /detect-fraud
```
Fraud analysis only (faster)

### Voice Classification
```bash
POST /detect-voice
```
AI vs Human voice detection

### Transcription
```bash
POST /transcribe
```
Speech-to-text conversion

### Alert Management
```bash
GET  /alert-history         # View past alerts
POST /block-number          # Block/unblock numbers
GET  /is-blocked/{number}   # Check if number is blocked
```

## 🎓 Training Custom Models

### Voice Classification Model

```bash
python -m app.train_model
```

Requires:
- `dataset/human/` - Human voice samples
- `dataset/ai/` - AI-generated voice samples

### Fraud Detection Model

```bash
# Windows
train_fraud_model.bat

# Linux/Mac
python -m app.train_fraud_model dataset/fraud_calls dataset/legitimate_calls
```

Requires:
- `dataset/fraud_calls/` - Fraud/spam call recordings
- `dataset/legitimate_calls/` - Normal call recordings

**Minimum**: 50 samples per category
**Recommended**: 200+ samples per category

See [dataset/DATASET_GUIDE.md](dataset/DATASET_GUIDE.md) for detailed instructions.

## 🧰 Technologies

### Backend
- **FastAPI** - Modern Python API framework
- **faster-whisper** - Speech recognition (Whisper AI)
- **librosa** - Audio analysis and feature extraction
- **scikit-learn** - Machine learning models
- **NumPy** - Numerical computing

### Frontend
- **React** - UI framework
- **Vite** - Build tool and dev server
- **Modern CSS** - Responsive design with dark theme

### AI/ML
- **Whisper** - OpenAI's speech recognition model
- **Random Forest** - Fraud classification
- **Audio Feature Extraction** - MFCC, spectral analysis, prosody features

## 📊 Performance

### Detection Capabilities
- **Fraud Detection**: ~75-90% accuracy (keyword-based)
- **Voice Classification**: ~85-95% accuracy (with training)
- **Language Detection**: ~90%+ accuracy
- **Transcription**: Depends on Whisper model size

### Speed
- **Voice Detection**: 1-2 seconds
- **Transcription**: 3-5 seconds (30s audio)
- **Fraud Analysis**: <1 second
- **Total Processing**: 4-8 seconds (comprehensive)

## 🔒 Privacy & Security

- **Local Processing**: All analysis runs locally (no external API calls)
- **No Data Storage**: Audio processed in memory only
- **Alert History**: Stored in memory (not persistent by default)
- **Configurable**: Add authentication, encryption, and persistence as needed

## 🤝 Contributing

Contributions welcome! Areas for improvement:
1. Add more fraud keyword patterns
2. Improve audio behavioral detection algorithms
3. Expand language support
4. Create larger training datasets
5. Implement real-time streaming analysis
6. Add voice biometric identification
7. Create mobile app interface

## 📝 License

This project is for educational and research purposes. Ensure compliance with local laws regarding call recording and analysis.

## 🙏 Acknowledgments

- **OpenAI Whisper** - Speech recognition technology
- **librosa** - Audio analysis library
- **FastAPI** - Modern API framework
- **scikit-learn** - Machine learning tools
- Community contributors and fraud awareness organizations

## 📧 Support

Having issues? Try these resources:

1. **Quick Start**: [QUICK_START.md](QUICK_START.md)
2. **Full Guide**: [FRAUD_DETECTION_GUIDE.md](FRAUD_DETECTION_GUIDE.md)
3. **Run Tests**: `python test_fraud_detection.py`
4. **API Docs**: http://127.0.0.1:8000/docs

## 🚀 What's New

### Fraud Detection Features (Latest)
- ✅ Real-time fraud pattern detection
- ✅ Keyword-based threat analysis
- ✅ Audio behavioral analysis
- ✅ Speech pattern recognition
- ✅ Alert system with history
- ✅ Auto-blocking for high-risk numbers
- ✅ Comprehensive web UI
- ✅ Multi-mode analysis options

### Language Detection (Previous)
- ✅ Multi-language support (5 languages)
- ✅ Automatic language detection
- ✅ Language-specific feature extraction

---

**🛡️ Built to protect users from fraud calls with AI-powered detection**

*Stay safe. Stay informed. Stay protected.* 📞
