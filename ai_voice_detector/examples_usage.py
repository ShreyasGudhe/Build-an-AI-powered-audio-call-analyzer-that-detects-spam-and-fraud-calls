"""
Example Usage Script
Demonstrates how to use the fraud detection system programmatically.
"""
import base64
import json
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from app.fraud_detection import FraudDetector, RealTimeAlertSystem
from app.transcription import SpeechTranscriber, KeywordDetector
from app.fraud_detection import FRAUD_KEYWORDS


def example_1_text_only_analysis():
    """Example 1: Analyze text for fraud patterns (no audio required)"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Text-Only Fraud Detection")
    print("="*80 + "\n")
    
    # Sample suspicious text
    text = """
    Hello, this is the IRS calling. Your social security number has been 
    suspended due to suspicious activity. You need to verify your account 
    information immediately by providing your credit card number and CVV. 
    If you don't act within the next hour, you will face legal action and arrest.
    """
    
    print(f"Analyzing text: {text.strip()}\n")
    
    # Create detector
    detector = FraudDetector()
    
    # Analyze
    result = detector.analyze_audio(
        audio_path=None,
        transcription=text,
        caller_number=None
    )
    
    # Display results
    print(f"🚨 FRAUD DETECTED: {result.is_fraud}")
    print(f"📊 Risk Level: {result.risk_level}")
    print(f"💯 Confidence: {result.confidence:.1%}")
    print(f"⚠️  Threat Type: {result.threat_type}")
    print(f"\n📋 Detected Patterns:")
    for pattern in result.detected_patterns:
        print(f"   • {pattern}")
    print(f"\n💡 Recommendation: {result.recommended_action}")


def example_2_keyword_detection():
    """Example 2: Real-time keyword detection"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Real-Time Keyword Detection")
    print("="*80 + "\n")
    
    detector = KeywordDetector(FRAUD_KEYWORDS)
    
    # Stream of text segments (simulating real-time transcription)
    segments = [
        {"text": "Hello, how are you today?", "start": 0, "end": 2},
        {"text": "I'm calling from your bank about your credit card.", "start": 2, "end": 5},
        {"text": "We need you to verify your account by providing the CVV number.", "start": 5, "end": 9},
        {"text": "This is urgent, you must act now or your account will be suspended.", "start": 9, "end": 13},
    ]
    
    print("Analyzing text segments in real-time:\n")
    
    for segment in segments:
        detected = detector.detect_in_text(segment["text"])
        
        print(f"[{segment['start']}-{segment['end']}s] \"{segment['text']}\"")
        
        if detected:
            print("   🚨 ALERT!")
            for category, matches in detected.items():
                print(f"      Category: {category}")
                print(f"      Matches: {len(matches)}")
        else:
            print("   ✓ Clean")
        print()


def example_3_audio_file_analysis():
    """Example 3: Complete audio file analysis"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Audio File Analysis")
    print("="*80 + "\n")
    
    # Check for sample audio
    dataset_dir = Path(__file__).parent / "dataset"
    audio_dirs = [
        dataset_dir / "fraud_calls",
        dataset_dir / "legitimate_calls",
        dataset_dir / "human"
    ]
    
    audio_file = None
    for dir_path in audio_dirs:
        if dir_path.exists():
            files = list(dir_path.glob("*.mp3")) + list(dir_path.glob("*.wav"))
            if files:
                audio_file = files[0]
                break
    
    if not audio_file:
        print("⚠️  No audio files found. Skipping this example.")
        print("   Add audio files to dataset/ directories to test this feature.\n")
        return
    
    print(f"Analyzing audio file: {audio_file.name}\n")
    
    # Initialize components
    detector = FraudDetector()
    transcriber = SpeechTranscriber(model_size="base")
    alert_system = RealTimeAlertSystem()
    
    try:
        # Step 1: Transcribe
        print("Step 1: Transcribing audio...")
        transcription_result = transcriber.transcribe(str(audio_file))
        transcription = transcription_result["text"]
        language = transcription_result["language"]
        
        print(f"✓ Language: {language}")
        print(f"✓ Transcription: {transcription[:200]}...")
        
        # Step 2: Fraud Analysis
        print("\nStep 2: Analyzing for fraud patterns...")
        fraud_alert = detector.analyze_audio(
            audio_path=str(audio_file),
            transcription=transcription,
            caller_number="+1-555-1234"
        )
        
        print(f"✓ Risk Level: {fraud_alert.risk_level}")
        print(f"✓ Confidence: {fraud_alert.confidence:.1%}")
        print(f"✓ Is Fraud: {fraud_alert.is_fraud}")
        
        # Step 3: Generate Alert
        print("\nStep 3: Generating alert...")
        alert = alert_system.generate_alert(
            fraud_alert,
            caller_info={"number": "+1-555-1234", "language": language}
        )
        
        print(f"✓ Alert Type: {alert['alert_type']}")
        print(f"✓ Should Block: {alert['should_block']}")
        print(f"✓ Timestamp: {alert['timestamp']}")
        
        # Display full results
        print("\n" + "-"*80)
        print("COMPLETE ANALYSIS RESULTS")
        print("-"*80)
        print(json.dumps(alert, indent=2))
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def example_4_alert_system():
    """Example 4: Using the alert system"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Alert System Management")
    print("="*80 + "\n")
    
    alert_system = RealTimeAlertSystem()
    detector = FraudDetector()
    
    # Simulate multiple fraud detections
    test_calls = [
        {
            "text": "IRS calling about your tax refund. Provide SSN.",
            "number": "+1-800-555-0001",
            "name": "High Risk Call"
        },
        {
            "text": "Hello, this is customer service calling.",
            "number": "+1-555-0002",
            "name": "Low Risk Call"
        },
        {
            "text": "Your computer has a virus. Give us remote access now!",
            "number": "+1-800-555-0003",
            "name": "Tech Scam"
        }
    ]
    
    print("Simulating multiple call analyses...\n")
    
    for i, call in enumerate(test_calls, 1):
        print(f"Call {i}: {call['name']}")
        
        # Analyze
        fraud_alert = detector.analyze_audio(
            audio_path=None,
            transcription=call["text"],
            caller_number=call["number"]
        )
        
        # Generate alert
        alert = alert_system.generate_alert(
            fraud_alert,
            caller_info={"number": call["number"], "name": call["name"]}
        )
        
        print(f"   Risk: {alert['risk_level']} ({alert['confidence']}%)")
        print(f"   Block: {alert['should_block']}")
        print()
    
    # Display alert history
    print("-"*80)
    print("Alert History:")
    print("-"*80)
    
    history = alert_system.get_alert_history()
    for alert in history:
        print(f"[{alert['risk_level']:8}] {alert['threat_type']:25} - {alert['confidence']}%")
    
    # Display blocked numbers
    print("\n" + "-"*80)
    print("Blocked Numbers:")
    print("-"*80)
    
    for number in alert_system.blocked_numbers:
        print(f"   🚫 {number}")


def example_5_custom_patterns():
    """Example 5: Adding custom fraud patterns"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Fraud Patterns")
    print("="*80 + "\n")
    
    # Create detector with custom patterns
    custom_patterns = {
        **FRAUD_KEYWORDS,  # Include default patterns
        "custom_category": [
            r'\b(special offer|limited time|act fast)\b',
            r'\b(exclusive deal|one time opportunity)\b',
        ]
    }
    
    detector = FraudDetector()
    detector.fraud_keywords = custom_patterns
    
    # Test with custom patterns
    text = "Exclusive deal! Limited time offer. Act fast to get this special opportunity!"
    
    print(f"Text: {text}\n")
    
    result = detector.analyze_audio(
        audio_path=None,
        transcription=text,
        caller_number=None
    )
    
    print(f"Risk Level: {result.risk_level}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Detected Patterns: {', '.join(result.detected_patterns)}")
    print("\n✓ Custom patterns successfully integrated!")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*25 + "FRAUD DETECTION EXAMPLES" + " "*30 + "║")
    print("╚" + "="*78 + "╝")
    
    examples = [
        ("Text-Only Analysis", example_1_text_only_analysis),
        ("Keyword Detection", example_2_keyword_detection),
        ("Audio File Analysis", example_3_audio_file_analysis),
        ("Alert System", example_4_alert_system),
        ("Custom Patterns", example_5_custom_patterns),
    ]
    
    for name, func in examples:
        try:
            func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error in {name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ All examples completed!")
    print("="*80)
    print("\n💡 Integration Tips:")
    print("   • Use these patterns in your own applications")
    print("   • Customize fraud patterns for your specific needs")
    print("   • Combine with phone system for real-time protection")
    print("   • Build dashboards using the alert system")
    print("   • Export alerts for compliance and reporting")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
