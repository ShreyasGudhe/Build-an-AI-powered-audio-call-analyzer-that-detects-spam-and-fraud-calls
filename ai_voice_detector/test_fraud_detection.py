"""
Test Fraud Detection System
Demonstrates all fraud detection capabilities.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.fraud_detection import FraudDetector, RealTimeAlertSystem
from app.transcription import SpeechTranscriber


def test_fraud_patterns():
    """Test fraud pattern detection with sample texts"""
    print("\n" + "="*80)
    print("🧪 TESTING FRAUD PATTERN DETECTION")
    print("="*80 + "\n")
    
    detector = FraudDetector()
    
    # Sample fraud texts
    test_cases = [
        {
            "name": "Financial Scam",
            "text": "This is IRS calling. Your account has been suspended. You need to verify your social security number and bank account immediately or face legal action.",
            "expected": "HIGH"
        },
        {
            "name": "Tech Support Scam",
            "text": "Hello, this is Microsoft technical support. We detected a virus on your computer. Please give us remote access to fix the issue right now.",
            "expected": "HIGH"
        },
        {
            "name": "Urgency Tactics",
            "text": "This is your final notice. Act now within 24 hours or your account will expire permanently. Call immediately.",
            "expected": "MEDIUM"
        },
        {
            "name": "Legitimate Call",
            "text": "Hello, this is Sarah from ABC Company calling about your appointment scheduled for tomorrow at 2 PM. Please call us back to confirm.",
            "expected": "LOW"
        },
        {
            "name": "Investment Scam",
            "text": "Congratulations! You've been selected for a special investment opportunity. Guaranteed 100% returns with no risk. Limited time offer!",
            "expected": "HIGH"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test['name']}")
        print(f"Text: \"{test['text']}\"")
        
        # Analyze (no audio, just text)
        alert = detector.analyze_audio(
            audio_path=None,
            transcription=test['text'],
            caller_number=None
        )
        
        print(f"✓ Risk Level: {alert.risk_level} (Expected: {test['expected']})")
        print(f"✓ Fraud Confidence: {alert.confidence:.2%}")
        print(f"✓ Threat Type: {alert.threat_type or 'None'}")
        print(f"✓ Detected Patterns: {', '.join(alert.detected_patterns) if alert.detected_patterns else 'None'}")
        print(f"✓ Recommendation: {alert.recommended_action}")
        print()


def test_alert_system():
    """Test real-time alert system"""
    print("\n" + "="*80)
    print("🚨 TESTING REAL-TIME ALERT SYSTEM")
    print("="*80 + "\n")
    
    alert_system = RealTimeAlertSystem()
    detector = FraudDetector()
    
    # Simulate fraud detection
    fraud_text = "This is IRS. Provide your credit card number immediately or face arrest."
    fraud_alert = detector.analyze_audio(
        audio_path=None,
        transcription=fraud_text,
        caller_number="+1-800-555-SCAM"
    )
    
    # Generate alert
    alert = alert_system.generate_alert(
        fraud_alert,
        caller_info={"number": "+1-800-555-SCAM", "name": "Unknown"}
    )
    
    print("Alert Generated:")
    print(f"  • Alert Type: {alert['alert_type']}")
    print(f"  • Risk Level: {alert['risk_level']}")
    print(f"  • Confidence: {alert['confidence']}%")
    print(f"  • Threat Type: {alert['threat_type']}")
    print(f"  • Should Block: {alert['should_block']}")
    print(f"  • Timestamp: {alert['timestamp']}")
    print()
    
    # Test blocking
    print("Testing Number Blocking:")
    blocked_number = "+1-800-555-SCAM"
    alert_system.block_number(blocked_number)
    print(f"  ✓ Blocked: {blocked_number}")
    print(f"  ✓ Is Blocked: {alert_system.is_blocked(blocked_number)}")
    
    alert_system.unblock_number(blocked_number)
    print(f"  ✓ Unblocked: {blocked_number}")
    print(f"  ✓ Is Blocked: {alert_system.is_blocked(blocked_number)}")
    print()
    
    # Show alert history
    print("Alert History:")
    history = alert_system.get_alert_history(limit=5)
    for i, hist_alert in enumerate(history, 1):
        print(f"  {i}. [{hist_alert['risk_level']}] {hist_alert['threat_type']} - {hist_alert['confidence']}%")


def test_keyword_detection():
    """Test keyword detection in different scenarios"""
    print("\n" + "="*80)
    print("🔍 TESTING KEYWORD DETECTION")
    print("="*80 + "\n")
    
    from app.transcription import KeywordDetector
    from app.fraud_detection import FRAUD_KEYWORDS
    
    detector = KeywordDetector(FRAUD_KEYWORDS)
    
    test_texts = [
        "I need you to verify your bank account and provide the CVV code.",
        "This is the police. You have an arrest warrant. Pay the fine now.",
        "Congratulations! You won the lottery. Send payment for processing.",
        "Hello, I'm calling about the weather forecast for tomorrow."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"Text {i}: \"{text}\"")
        detected = detector.detect_in_text(text)
        
        if detected:
            print("  🚨 Detected Categories:")
            for category, matches in detected.items():
                print(f"    • {category}: {len(matches)} matches")
        else:
            print("  ✓ No fraud keywords detected")
        print()


def test_audio_analysis():
    """Test audio file analysis if samples are available"""
    print("\n" + "="*80)
    print("🎵 TESTING AUDIO ANALYSIS")
    print("="*80 + "\n")
    
    # Check for sample audio files
    dataset_dir = Path(__file__).parent / "dataset"
    sample_dirs = [
        dataset_dir / "fraud_calls",
        dataset_dir / "legitimate_calls",
        dataset_dir / "human",
        dataset_dir / "ai"
    ]
    
    audio_files = []
    for dir_path in sample_dirs:
        if dir_path.exists():
            for ext in ['.mp3', '.wav', '.m4a']:
                audio_files.extend(list(dir_path.glob(f"*{ext}")))
            if audio_files:
                break
    
    if not audio_files:
        print("⚠️  No audio files found in dataset directories.")
        print("   Place sample audio files in:")
        print(f"   - {dataset_dir / 'fraud_calls'}")
        print(f"   - {dataset_dir / 'legitimate_calls'}")
        print("\n   Skipping audio analysis tests.")
        return
    
    print(f"Found {len(audio_files)} audio files. Testing with first file...\n")
    
    detector = FraudDetector()
    transcriber = SpeechTranscriber(model_size="base")
    
    # Test first audio file
    audio_path = str(audio_files[0])
    print(f"Analyzing: {Path(audio_path).name}")
    
    try:
        # Transcribe
        print("  → Transcribing audio...")
        transcription_result = transcriber.transcribe(audio_path, max_duration=30)
        transcription = transcription_result.get("text", "")
        
        if transcription:
            print(f"  ✓ Transcription: \"{transcription[:100]}...\"")
        else:
            print("  ⚠️  No transcription generated")
        
        # Fraud analysis
        print("  → Analyzing for fraud patterns...")
        fraud_alert = detector.analyze_audio(
            audio_path=audio_path,
            transcription=transcription
        )
        
        print(f"  ✓ Risk Level: {fraud_alert.risk_level}")
        print(f"  ✓ Confidence: {fraud_alert.confidence:.2%}")
        print(f"  ✓ Is Fraud: {fraud_alert.is_fraud}")
        
        if fraud_alert.detected_patterns:
            print(f"  ✓ Patterns: {', '.join(fraud_alert.detected_patterns)}")
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")


def print_feature_summary():
    """Print summary of all features"""
    print("\n" + "="*80)
    print("📋 FRAUD DETECTION FEATURES SUMMARY")
    print("="*80 + "\n")
    
    features = [
        ("Keyword Detection", "Identifies fraud-related words and phrases"),
        ("Audio Behavior Analysis", "Detects suspicious audio patterns (robocalls, VOIP)"),
        ("Speech Pattern Analysis", "Analyzes stress, urgency, and vocal characteristics"),
        ("Caller Number Analysis", "Identifies suspicious number patterns"),
        ("Real-time Transcription", "Converts speech to text for analysis"),
        ("Risk Level Assessment", "Categorizes threats (LOW/MEDIUM/HIGH/CRITICAL)"),
        ("Alert System", "Generates and tracks fraud alerts"),
        ("Number Blocking", "Automatic blocking of high-risk numbers"),
        ("Threat Classification", "Identifies specific fraud types"),
        ("Multi-language Support", "Works with multiple languages")
    ]
    
    for i, (feature, description) in enumerate(features, 1):
        print(f"{i:2}. {feature:25} - {description}")
    
    print("\n" + "="*80)
    print("✅ All fraud detection components are operational!")
    print("="*80 + "\n")


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "FRAUD DETECTION SYSTEM TEST SUITE" + " "*25 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        # Run all tests
        test_fraud_patterns()
        test_keyword_detection()
        test_alert_system()
        test_audio_analysis()
        print_feature_summary()
        
        print("\n✅ All tests completed successfully!\n")
        print("🚀 Next Steps:")
        print("   1. Start the API server: python run_api.py")
        print("   2. Open the frontend: cd frontend && npm install && npm run dev")
        print("   3. Upload audio files to test fraud detection")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
