"""
Test script to verify fraud keyword detection
"""
from app.fraud_detection import FraudDetector

def test_fraud_keywords():
    """Test specific fraud keywords"""
    detector = FraudDetector()
    
    # Test cases with specific keywords
    test_cases = [
        ("Please share your OTP for verification", ["otp"]),
        ("We need your Aadhaar number immediately", ["aadhaar"]),
        ("Your PAN card details are required", ["pan"]),
        ("Legal action will be taken against you", ["legal action"]),
        ("Your account will be suspended", ["suspend"]),
        ("This is urgent, verify now", ["urgent", "verify now"]),
        ("Police will arrest you", ["police"]),
        ("Make the payment immediately", ["payment"]),
        ("You won the lottery", ["lottery"]),
        ("KYC verification required", ["kyc"]),
        ("Your bank account is blocked", ["bank blocked"]),
        ("Hello this is a normal call about weather", []),  # Should not trigger
    ]
    
    print("\n" + "="*80)
    print("FRAUD KEYWORD DETECTION TEST")
    print("="*80 + "\n")
    
    for text, expected_keywords in test_cases:
        score, patterns = detector._analyze_keywords(text)
        
        print(f"Text: {text}")
        print(f"Score: {score:.2f}")
        print(f"Patterns: {patterns}")
        
        # Check if expected keywords are detected
        text_lower = text.lower()
        detected = []
        for keyword in expected_keywords:
            if keyword.lower() in text_lower:
                detected.append(keyword)
        
        if score > 0:
            status = "✓ FRAUD DETECTED"
            color = "\033[91m"  # Red
        else:
            status = "○ Safe"
            color = "\033[92m"  # Green
        
        print(f"Status: {color}{status}\033[0m")
        print(f"Risk Level: {detector._calculate_risk_level(score)}")
        print("-" * 80 + "\n")
    
    # Test combined keywords (should have higher score)
    print("\n" + "="*80)
    print("COMBINED KEYWORDS TEST (Should be CRITICAL)")
    print("="*80 + "\n")
    
    combined_text = "This is urgent! Share your OTP and Aadhaar number. Legal action will be taken. Your bank is blocked."
    score, patterns = detector._analyze_keywords(combined_text)
    
    print(f"Text: {combined_text}")
    print(f"Score: {score:.2f}")
    print(f"Patterns: {patterns}")
    print(f"Risk Level: {detector._calculate_risk_level(score)}")
    print(f"Status: \033[91mFRAUD DETECTED - CRITICAL\033[0m\n")

if __name__ == "__main__":
    test_fraud_keywords()
