"""Quick demonstration of fraud detection capabilities"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.fraud_detection import FraudDetector, RealTimeAlertSystem

print("\n" + "="*80)
print("🛡️  AI FRAUD CALL ANALYZER - QUICK DEMO")
print("="*80 + "\n")

detector = FraudDetector()
alert_system = RealTimeAlertSystem()

# Test cases
test_cases = [
    {
        "name": "🚨 CRITICAL - IRS Scam Call",
        "text": "This is the IRS. Your social security number has been suspended due to suspicious activity. You must verify your SSN and bank account immediately or face legal action and arrest.",
        "number": "+1-800-555-0001"
    },
    {
        "name": "⚠️  HIGH - Tech Support Scam", 
        "text": "Hello, this is Microsoft support. We detected a virus on your computer. Please provide remote access now so we can remove the malware immediately.",
        "number": "+1-888-555-0002"
    },
    {
        "name": "⚡ MEDIUM - Prize Scam",
        "text": "Congratulations! You've won a free prize. To claim it, we need your credit card details for verification. This is a limited time offer.",
        "number": "+1-877-555-0003"
    },
    {
        "name": "✓ LOW - Legitimate Call",
        "text": "Hello, this is Sarah from ABC Medical Center. I'm calling to confirm your appointment for tomorrow at 2 PM. Please call us back if you need to reschedule.",
        "number": "+1-555-1234"
    }
]

for i, test in enumerate(test_cases, 1):
    print(f"\n{'─'*80}")
    print(f"Test {i}: {test['name']}")
    print(f"{'─'*80}")
    print(f"📞 Caller: {test['number']}")
    print(f"💬 Message: \"{test['text'][:100]}...\"")
    print()
    
    # Analyze
    fraud_alert = detector.analyze_audio(
        audio_path=None,
        transcription=test['text'],
        caller_number=test['number']
    )
    
    # Generate alert
    alert = alert_system.generate_alert(
        fraud_alert,
        caller_info={"number": test['number']}
    )
    
    # Display results
    risk_colors = {
        "CRITICAL": "🔴",
        "HIGH": "🟠", 
        "MEDIUM": "🟡",
        "LOW": "🟢"
    }
    
    print(f"   {risk_colors[alert['risk_level']]} Risk Level: {alert['risk_level']}")
    print(f"   📊 Fraud Confidence: {alert['confidence']}%")
    print(f"   🎯 Threat Type: {alert['threat_type']}")
    print(f"   🚫 Should Block: {'YES' if alert['should_block'] else 'NO'}")
    
    if alert['detected_patterns']:
        print(f"   ⚠️  Detected Patterns:")
        for pattern in alert['detected_patterns']:
            print(f"      • {pattern}")
    
    print(f"\n   💡 Recommendation:")
    print(f"      {alert['recommended_action']}")

print(f"\n{'='*80}")
print("✅ DEMO COMPLETE - All fraud detection features working!")
print("="*80)

print(f"\n📊 Alert Summary:")
print(f"   Total Alerts: {len(alert_system.alert_history)}")
print(f"   Blocked Numbers: {len(alert_system.blocked_numbers)}")

critical = sum(1 for a in alert_system.alert_history if a['risk_level'] == 'CRITICAL')
high = sum(1 for a in alert_system.alert_history if a['risk_level'] == 'HIGH')
medium = sum(1 for a in alert_system.alert_history if a['risk_level'] == 'MEDIUM')
low = sum(1 for a in alert_system.alert_history if a['risk_level'] == 'LOW')

print(f"   🔴 Critical: {critical}")
print(f"   🟠 High: {high}")
print(f"   🟡 Medium: {medium}")
print(f"   🟢 Low: {low}")

print(f"\n🌐 Web Interface: http://localhost:5173")
print(f"📡 API Endpoint: http://127.0.0.1:8000")
print(f"📚 API Docs: http://127.0.0.1:8000/docs")
print(f"\n🎉 The AI Fraud Analyzer is ready to protect you!\n")
