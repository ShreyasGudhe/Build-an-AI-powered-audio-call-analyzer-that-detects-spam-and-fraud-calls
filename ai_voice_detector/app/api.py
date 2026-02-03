import base64
import io
import tempfile
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from .predict import predict_audio, predict_audio_bytes, analyze_conversation
from .fraud_detection import FraudDetector, RealTimeAlertSystem
from .transcription import SpeechTranscriber

app = FastAPI(title="AI Voice & Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize fraud detection systems
fraud_detector = FraudDetector()
alert_system = RealTimeAlertSystem()
transcriber = SpeechTranscriber(model_size="base")

@app.get("/")
def health_check():
    """Simple root endpoint to verify the API is running."""
    return {
        "status": "ok",
        "endpoints": {
            "voice_detection": "/detect-voice",
            "fraud_detection": "/detect-fraud",
            "comprehensive_analysis": "/analyze-call",
            "conversation_analysis": "/analyze-conversation",
            "alert_history": "/alert-history",
            "block_number": "/block-number",
        }
    }

@app.post("/detect-voice")
def detect_voice(data: dict):
    """Voice and fraud detection endpoint (enhanced with AI/Human + fraud keyword detection)"""
    if "audio" not in data:
        raise HTTPException(status_code=400, detail="Missing 'audio' field")

    audio_base64 = data["audio"]
    try:
        audio_bytes = base64.b64decode(audio_base64)
        if not audio_bytes:
            raise ValueError("empty audio")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid audio payload") from exc

    prediction = predict_audio_bytes(audio_bytes)

    return {
        # Fraud detection
        "classification": prediction["classification"],
        "confidence": round(float(prediction["confidence"]), 2),
        "explanation": prediction["explanation"],
        "risk_level": prediction.get("risk_level"),
        "detected_patterns": prediction.get("detected_patterns", []),
        "fraud_categories": prediction.get("fraud_categories", {}),
        
        # Voice type (AI vs Human)
        "voice_type": prediction.get("voice_type", "Unknown"),
        "voice_confidence": prediction.get("voice_confidence", 0),
        "voice_explanation": prediction.get("voice_explanation", ""),
        
        # Language
        "language": prediction.get("language_name", prediction.get("language", "Unknown")),
        "language_confidence": prediction.get("language_confidence", 0),
        "language_code": prediction.get("language_code", "unknown"),
        
        # Transcript
        "transcribed_text": prediction.get("transcribed_text", ""),
    }

@app.post("/detect-fraud")
def detect_fraud(data: dict):
    """
    Enhanced fraud detection in audio call with keyword analysis + AI/Human voice detection
    
    Required: audio (base64 encoded)
    Optional: caller_number
    """
    if "audio" not in data:
        raise HTTPException(status_code=400, detail="Missing 'audio' field")
    
    try:
        audio_base64 = data["audio"]
        audio_bytes = base64.b64decode(audio_base64)
        if not audio_bytes:
            raise ValueError("empty audio")
        
        # Use enhanced prediction
        prediction = predict_audio_bytes(audio_bytes)
        
        caller_number = data.get("caller_number")
        
        # Format response for fraud detection
        return {
            # Fraud results
            "classification": prediction["classification"],
            "confidence": round(float(prediction["confidence"]), 2),
            "risk_level": prediction.get("risk_level"),
            "explanation": prediction["explanation"],
            "detected_patterns": prediction.get("detected_patterns", []),
            "fraud_categories": prediction.get("fraud_categories", {}),
            
            # Voice type (AI vs Human)
            "voice_type": prediction.get("voice_type", "Unknown"),
            "voice_confidence": prediction.get("voice_confidence", 0),
            "voice_explanation": prediction.get("voice_explanation", ""),
            
            # Language & transcript
            "transcribed_text": prediction.get("transcribed_text", ""),
            "language_name": prediction.get("language_name", "Unknown"),
            "language_confidence": prediction.get("language_confidence", 0),
            
            # Caller info
            "caller_number": caller_number,
            "recommended_action": "Block this number immediately" if prediction.get("risk_level") in ["CRITICAL", "HIGH"] else 
                                 "Monitor this number" if prediction.get("risk_level") == "MEDIUM" else
                                 "No action needed",
            "should_block": prediction.get("risk_level") in ["CRITICAL", "HIGH"] or prediction.get("voice_type") == "AI-generated",
        }
            
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Fraud detection failed: {str(exc)}")

@app.post("/analyze-call")
def analyze_call(data: dict):
    """
    Comprehensive call analysis: Fraud detection + AI/Human voice + transcription
    
    Required: audio (base64 encoded)
    Optional: caller_number
    """
    if "audio" not in data:
        raise HTTPException(status_code=400, detail="Missing 'audio' field")
    
    try:
        audio_base64 = data["audio"]
        audio_bytes = base64.b64decode(audio_base64)
        if not audio_bytes:
            raise ValueError("empty audio")
        
        # Enhanced prediction with fraud keyword detection + AI/Human voice detection
        prediction = predict_audio_bytes(audio_bytes)
        
        caller_number = data.get("caller_number")
        
        # Build comprehensive response
        is_fraud = prediction["classification"] == "Fraud"
        risk_level = prediction.get("risk_level")
        voice_type = prediction.get("voice_type", "Unknown")
        is_ai_voice = voice_type == "AI-generated"
        
        fraud_detection_result = {
            "classification": prediction["classification"],
            "confidence": round(float(prediction["confidence"]), 2),
            "risk_level": risk_level,
            "explanation": prediction["explanation"],
            "detected_patterns": prediction.get("detected_patterns", []),
            "fraud_categories": prediction.get("fraud_categories", {}),
            "transcribed_text": prediction.get("transcribed_text", ""),
            "language_name": prediction.get("language_name", "Unknown"),
            "language_confidence": prediction.get("language_confidence", 0),
            "recommended_action": "Block immediately" if risk_level in ["CRITICAL", "HIGH"] or is_ai_voice else 
                                 "Monitor" if risk_level == "MEDIUM" else "Safe",
        }
        
        # Voice detection (AI vs Human)
        voice_detection_result = {
            "classification": voice_type,
            "confidence": prediction.get("voice_confidence", 0),
            "explanation": prediction.get("voice_explanation", ""),
            "is_ai_generated": is_ai_voice,
        }
        
        # Transcription details
        transcription_result = {
            "text": prediction.get("transcribed_text", ""),
            "language": prediction.get("language_name", "Unknown"),
            "confidence": prediction.get("language_confidence", 0),
        }
        
        # Overall assessment
        overall_assessment = {
            "is_suspicious": is_fraud or risk_level in ["HIGH", "CRITICAL", "MEDIUM"] or is_ai_voice,
            "should_block": risk_level in ["CRITICAL", "HIGH"] or is_ai_voice,
            "summary": _generate_summary_enhanced(prediction)
        }
        
        return {
            "fraud_detection": fraud_detection_result,
            "voice_detection": voice_detection_result,
            "transcription": transcription_result,
            "overall_assessment": overall_assessment,
            "caller_number": caller_number
        }
            
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Call analysis failed: {str(exc)}")

@app.get("/alert-history")
def get_alert_history(limit: int = 50):
    """Get recent fraud alert history"""
    history = alert_system.get_alert_history(limit=limit)
    return {
        "total_alerts": len(history),
        "alerts": history
    }

@app.post("/block-number")
def block_number(data: dict):
    """Block or unblock a phone number"""
    if "number" not in data:
        raise HTTPException(status_code=400, detail="Missing 'number' field")
    
    number = data["number"]
    action = data.get("action", "block")  # "block" or "unblock"
    
    if action == "block":
        alert_system.block_number(number)
        return {"status": "blocked", "number": number}
    elif action == "unblock":
        alert_system.unblock_number(number)
        return {"status": "unblocked", "number": number}
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Use 'block' or 'unblock'")

@app.get("/is-blocked/{number}")
def check_blocked(number: str):
    """Check if a number is blocked"""
    is_blocked = alert_system.is_blocked(number)
    return {
        "number": number,
        "is_blocked": is_blocked
    }

@app.post("/transcribe")
def transcribe_audio_endpoint(data: dict):
    """Transcribe audio to text"""
    if "audio" not in data:
        raise HTTPException(status_code=400, detail="Missing 'audio' field")
    
    try:
        audio_base64 = data["audio"]
        audio_bytes = base64.b64decode(audio_base64)
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            language = data.get("language")
            result = transcriber.transcribe(tmp_path, language=language)
            return result
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(exc)}")

def _generate_summary_enhanced(prediction: dict) -> str:
    """Generate a human-readable summary of the enhanced analysis"""
    parts = []
    
    # Voice type (AI vs Human)
    voice_type = prediction.get("voice_type", "Unknown")
    if voice_type == "AI-generated":
        parts.append("🤖 AI Voice")
    elif voice_type == "Human":
        parts.append("👤 Human Voice")
    
    # Fraud status
    if prediction["classification"] == "Fraud":
        parts.append(f"⚠️ FRAUD")
        if prediction.get("risk_level"):
            parts.append(f"Risk: {prediction['risk_level']}")
    else:
        parts.append("✓ Safe")
    
    # Keywords
    if prediction.get("detected_patterns"):
        num_keywords = len(prediction["detected_patterns"])
        parts.append(f"{num_keywords} fraud keyword(s)")
    
    # Language
    if prediction.get("language_name") and prediction["language_name"] != "Unknown":
        parts.append(f"{prediction['language_name']}")
    
    return " | ".join(parts)

def _generate_summary(voice_result: dict, fraud_alert: dict, language: str) -> str:
    """Generate a human-readable summary of the analysis (legacy)"""
    parts = []
    
    # Voice type
    if voice_result.get("label") == "AI-generated" or voice_result.get("classification") == "AI-generated":
        parts.append("⚠️ AI-generated voice detected")
    elif voice_result.get("classification") == "Fraud":
        parts.append("⚠️ Fraud detected")
    else:
        parts.append("✓ Human voice detected")
    
    # Fraud risk
    risk = fraud_alert.get("risk_level")
    if risk == "CRITICAL":
        parts.append("🚨 CRITICAL FRAUD RISK")
    elif risk == "HIGH":
        parts.append("⚠️ HIGH fraud risk")
    elif risk == "MEDIUM":
        parts.append("⚡ MEDIUM fraud risk")
    elif risk:
        parts.append("✓ LOW fraud risk")
    
    # Language
    if language and language != "unknown":
        parts.append(f"Language: {language}")
    
    # Threat type
    if fraud_alert.get("threat_type"):
        parts.append(f"Threat: {fraud_alert['threat_type']}")
    
    return " | ".join(parts)


@app.post("/analyze-conversation")
def analyze_conversation_endpoint(data: dict):
    """
    Analyze a two-way conversation between caller and receiver.
    Detects fraud/spam from both sides of the call.
    
    Expected input:
    {
        "audio": "base64_encoded_audio_data"
    }
    
    Returns analysis for both speakers including:
    - Individual fraud detection for each speaker
    - AI vs Human classification for each speaker
    - Language detection for each speaker
    - Overall conversation assessment
    - Actionable recommendations
    """
    if "audio" not in data:
        raise HTTPException(status_code=400, detail="Missing ''audio'' field")
    
    audio_base64 = data["audio"]
    try:
        audio_bytes = base64.b64decode(audio_base64)
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio data")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {str(e)}")
    
    try:
        # Analyze conversation with speaker separation
        result = analyze_conversation(audio_bytes)
        
        # Enhance response with formatted summary
        if "conversation_summary" in result:
            summary_parts = []
            
            # Overall status
            if result["conversation_summary"]["is_fraud"]:
                risk = result["conversation_summary"]["overall_risk_level"]
                summary_parts.append(f"FRAUD DETECTED ({risk})")
                
                # Which speaker(s)
                fraud_from = result["conversation_summary"]["fraud_detected_from"]
                if fraud_from:
                    speakers = " & ".join([s.replace("_", " ").title() for s in fraud_from])     
                    summary_parts.append(f"From: {speakers}")
            else:
                summary_parts.append("No Fraud Detected")
            
            # AI voice detection
            if result["conversation_summary"]["ai_voices_detected"]:
                ai_from = result["conversation_summary"]["ai_voice_from"]
                speakers = " & ".join([s.replace("_", " ").title() for s in ai_from])
                summary_parts.append(f"AI Voice: {speakers}")
            
            # Number of speakers
            summary_parts.append(f"{result['speakers_detected']} Speaker(s)")
            
            result["conversation_summary"]["formatted_summary"] = " | ".join(summary_parts)       
        
        # Add individual speaker summaries
        for speaker_key in ["speaker_1", "speaker_2"]:
            if speaker_key in result:
                speaker_data = result[speaker_key]
                speaker_summary = []
                
                # Voice type
                if speaker_data.get("voice_type") == "AI-generated":
                    speaker_summary.append("AI Voice")
                else:
                    speaker_summary.append("Human")
                
                # Fraud status
                if speaker_data.get("classification") == "Fraud":
                    risk = speaker_data.get("risk_level", "MEDIUM")
                    speaker_summary.append(f"Fraud ({risk})")
                    if speaker_data.get("detected_patterns"):
                        speaker_summary.append(f"{len(speaker_data['detected_patterns'])} keywords")
                else:
                    speaker_summary.append("Safe")
                
                # Language
                if speaker_data.get("language_name"):
                    speaker_summary.append(speaker_data["language_name"])
                
                # Duration
                if "duration_seconds" in speaker_data:
                    duration = speaker_data["duration_seconds"]
                    speaker_summary.append(f"{duration:.1f}s")
                
                result[speaker_key]["formatted_summary"] = " | ".join(speaker_summary)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversation analysis failed: {str(e)}")    
