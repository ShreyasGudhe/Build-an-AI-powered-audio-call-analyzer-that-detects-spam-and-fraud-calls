# 🚀 Quick Start Guide - AI Call Fraud Analyzer

Get up and running in 5 minutes!

## ⚡ Fast Setup

### Step 1: Install Dependencies (2 minutes)

```bash
# Install Python packages
pip install -r requirements.txt

# Install frontend packages
cd frontend
npm install
cd ..
```

### Step 2: Start the Application (1 minute)

**Option A - Windows:**
```bash
# Terminal 1: Start API
run_fraud_api.bat

# Terminal 2: Start Frontend
cd frontend
npm run dev
```

**Option B - Linux/Mac:**
```bash
# Terminal 1: Start API
python run_api.py

# Terminal 2: Start Frontend
cd frontend
npm run dev
```

### Step 3: Open and Test (2 minutes)

1. Open browser to: **http://localhost:5173**
2. Click "Select audio file"
3. Choose any audio file with speech
4. Click "Analyze Call"
5. View comprehensive fraud analysis!

## 🎯 What You Can Do Immediately

### Without Training Any Models

The system works out-of-the-box with:

✅ **Keyword-Based Fraud Detection**
- Identifies 50+ fraud-related keywords and phrases
- Categorizes threats (financial scam, impersonation, etc.)
- No training required!

✅ **Audio Behavioral Analysis**
- Robocall detection
- VOIP/spoofed call identification
- Speech pattern analysis
- Works on any audio!

✅ **Real-Time Transcription**
- Converts speech to text
- Multi-language support
- Powered by Whisper AI

✅ **Voice Classification**
- AI-generated vs Human voice
- Uses pre-trained models

✅ **Alert System**
- Real-time threat notifications
- Risk level categorization
- Action recommendations

## 🧪 Test It Out

### Quick Test (No Audio Files Needed)

```bash
# Run comprehensive tests
python test_fraud_detection.py

# Or on Windows
test_fraud.bat
```

This will:
- Test fraud keyword detection
- Demonstrate alert system
- Show risk assessment
- Verify all components

### Test with Audio Files

1. Find any audio file with speech
2. Upload via the web interface
3. See immediate fraud analysis!

**Test Scenarios:**
- Upload a recorded customer service call → Should show LOW risk
- Upload audio saying "IRS calling, provide SSN" → HIGH risk
- Upload AI-generated voice → Detected as AI

## 📡 API Testing

### Using curl:

```bash
# Test API health
curl http://127.0.0.1:8000/

# Analyze audio (need base64 encoded audio)
curl -X POST http://127.0.0.1:8000/analyze-call \
  -H "Content-Type: application/json" \
  -d '{"audio":"BASE64_AUDIO_DATA"}'
```

### Using Browser:
Visit **http://127.0.0.1:8000/docs** for interactive API documentation!

## 🎨 UI Features

### Analysis Modes

**🔍 Full Analysis** (Recommended)
- Complete fraud detection
- Voice classification
- Full transcription
- All in one analysis

**🚨 Fraud Only**
- Skip voice analysis
- Faster results
- Focus on fraud patterns

**🎤 Voice Only**
- AI vs Human detection
- No fraud analysis
- Quick classification

### Understanding Results

**Risk Levels:**
- 🟢 **LOW** (0-35%): Likely safe
- 🟡 **MEDIUM** (35-55%): Be cautious
- 🟠 **HIGH** (55-75%): High risk, hang up
- 🔴 **CRITICAL** (75-100%): Block immediately

**Fraud Indicators:**
- Keywords: Fraud-related terms detected
- Behavior: Audio pattern anomalies
- Speech Patterns: Stress/urgency detected
- Caller Number: Suspicious number pattern

## 🎓 Optional: Train Custom Model

Want even better accuracy? Train with your own data:

### Step 1: Prepare Data

```bash
# Create directories (already exists)
dataset/
├── fraud_calls/        # Add fraud recordings here
└── legitimate_calls/   # Add normal call recordings here
```

Add at least 50 audio files per category (more = better).

### Step 2: Train

```bash
# Windows
train_fraud_model.bat

# Linux/Mac
python -m app.train_fraud_model dataset/fraud_calls dataset/legitimate_calls
```

### Step 3: Done!
The trained model is automatically used for even more accurate fraud detection.

## 💡 Usage Tips

### Best Results
1. **Audio Quality**: Use clear audio with audible speech
2. **Duration**: 10-60 seconds works best
3. **Format**: MP3, WAV, M4A, FLAC, OGG all supported
4. **Caller Number**: Add if available for better analysis

### Common Use Cases

**📞 Analyze Suspicious Call**
1. Record the call (check local laws!)
2. Upload to analyzer
3. Get instant fraud assessment
4. Follow recommended action

**🔍 Check Voicemail**
1. Save voicemail as audio file
2. Upload for analysis
3. Identify scam messages

**📊 Monitor Call Center**
1. Upload customer service calls
2. Ensure quality standards
3. Detect potential fraud attempts

**🛡️ Protect Others**
1. Analyze known scam calls
2. Share results
3. Build awareness

## 🐛 Quick Fixes

**API won't start?**
```bash
# Check if port 8000 is free
netstat -ano | findstr :8000

# Try different port
uvicorn app.api:app --port 8001
```

**Frontend won't start?**
```bash
# Clear npm cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

**Slow transcription?**
```bash
# Use faster (smaller) model
# Edit app/transcription.py line 21:
# Change model_size="base" to model_size="tiny"
```

## 📚 Learn More

- **Full Documentation**: [FRAUD_DETECTION_GUIDE.md](FRAUD_DETECTION_GUIDE.md)
- **Dataset Guide**: [dataset/DATASET_GUIDE.md](dataset/DATASET_GUIDE.md)
- **API Documentation**: http://127.0.0.1:8000/docs (when server running)

## 🎉 You're Ready!

The fraud analyzer is now running and ready to protect you from scam calls!

**Next Steps:**
- Upload your first audio file
- Test different analysis modes
- Review the alert history
- Train with custom data (optional)
- Integrate into your workflow

---

**Questions?** Run `python test_fraud_detection.py` to verify everything works!

🛡️ Stay safe from fraud calls! 📞
