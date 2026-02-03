"""
Test Strict Keyword Detection - Only specific keywords trigger fraud
Tests: If keyword present = FRAUD, If keyword absent = NOT FRAUD
"""
from app.fraud_detection import FraudDetector

def test_strict_keywords():
    detector = FraudDetector()
    
    print("=" * 80)
    print("STRICT KEYWORD FRAUD DETECTION TEST")
    print("=" * 80)
    
    # Test cases with the 12 specific keywords
    fraud_test_cases = [
        ("Please share your OTP with me", ["otp"]),
        ("Send your Aadhaar number immediately", ["aadhaar"]),
        ("Provide PAN card details", ["pan"]),
        ("Legal action will be taken against you", ["legal action"]),
        ("Your account will be suspended", ["suspended"]),
        ("This is urgent, act now", ["urgent"]),
        ("Police will come to arrest you", ["police"]),
        ("You need to verify now", ["verify now"]),
        ("Make the payment today", ["payment"]),
        ("You have won the lottery", ["lottery"]),
        ("Complete your KYC verification", ["kyc"]),
        ("Your bank account is blocked", ["bank blocked"]),
        ("Share OTP for Aadhaar verification and PAN update", ["otp", "aadhaar", "pan"]),
        ("Police action, urgent payment needed, verify now!", ["police", "urgent", "payment", "verify now"]),
    ]
    
    # Test cases WITHOUT the keywords (should NOT be fraud)
    normal_test_cases = [
        "Hello, how are you doing today?",
        "The weather is nice today",
        "I'm calling about the meeting tomorrow",
        "Can we schedule a call for next week?",
        "Thank you for your help yesterday",
        "I just wanted to check in and say hello",
        "The project deadline is approaching",
        "Let's discuss the proposal",
        "Have you completed the report?",
        "I'll send you the documents by email",
    ]
    
    print("\n" + "=" * 80)
    print("TEST 1: FRAUD KEYWORD DETECTION (Should detect as FRAUD)")
    print("=" * 80)
    
    fraud_pass = 0
    fraud_total = len(fraud_test_cases)
    
    for i, (text, expected_keywords) in enumerate(fraud_test_cases, 1):
        score, patterns = detector._analyze_keywords(text)
        is_fraud = score > 0
        
        print(f"\n[Test {i}] Text: \"{text}\"")
        print(f"  Expected Keywords: {', '.join(expected_keywords)}")
        print(f"  Fraud Score: {score:.2f}")
        print(f"  Patterns: {patterns}")
        print(f"  Status: ", end="")
        
        if is_fraud and score >= 0.85:
            print("✅ PASS - Correctly detected as FRAUD")
            fraud_pass += 1
        else:
            print(f"❌ FAIL - Should be FRAUD (score should be >= 0.85, got {score:.2f})")
    
    print("\n" + "=" * 80)
    print("TEST 2: NORMAL CALLS (Should NOT detect as fraud)")
    print("=" * 80)
    
    normal_pass = 0
    normal_total = len(normal_test_cases)
    
    for i, text in enumerate(normal_test_cases, 1):
        score, patterns = detector._analyze_keywords(text)
        is_fraud = score > 0
        
        print(f"\n[Test {i}] Text: \"{text}\"")
        print(f"  Fraud Score: {score:.2f}")
        print(f"  Patterns: {patterns}")
        print(f"  Status: ", end="")
        
        if not is_fraud and score == 0.0:
            print("✅ PASS - Correctly identified as NORMAL")
            normal_pass += 1
        else:
            print(f"❌ FAIL - Should be NORMAL (score should be 0.0, got {score:.2f})")
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Fraud Detection Tests: {fraud_pass}/{fraud_total} PASSED")
    print(f"Normal Call Tests: {normal_pass}/{normal_total} PASSED")
    print(f"Overall: {fraud_pass + normal_pass}/{fraud_total + normal_total} PASSED")
    
    if fraud_pass == fraud_total and normal_pass == normal_total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\n✅ IF KEYWORD IN AUDIO → FRAUD CALL")
        print("✅ IF NO KEYWORD IN AUDIO → NOT FRAUD")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    print("=" * 80)

if __name__ == "__main__":
    test_strict_keywords()
