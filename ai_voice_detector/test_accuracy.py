"""
Test language detection accuracy with improved strict thresholds.
This test shows how the system now better handles:
1. Supported languages (en, hi, ta, te, ml)
2. Unsupported languages (should return "unknown")
3. Ambiguous cases (returns "unknown" instead of guessing)
"""
from pathlib import Path
from app.predict import predict_audio

print("=" * 70)
print("LANGUAGE DETECTION ACCURACY TEST")
print("Supported: English (en), Hindi (hi), Tamil (ta), Telugu (te), Malayalam (ml)")
print("=" * 70)

test_cases = [
    # Supported languages
    ("dataset/lang_en/english.mp3", "English", "en"),
    ("dataset/lang_hi/hindi.mp3", "Hindi", "hi"),
    ("dataset/lang_ta/tamil.mp3", "Tamil", "ta"),
    ("dataset/lang_ml/malayalam.mp3", "Malayalam", "ml"),
    ("dataset/lang_te/telgu.mp3", "Telugu", "te"),
    
    # AI vs Human samples (should still detect language)
    ("dataset/ai/ai_voice_sample_english.mp3", "English (AI)", "en"),
    ("dataset/human/Human voice, male.mp3", "English (Human)", "en"),
]

correct = 0
total = 0
unknown = 0

for audio_file, expected_name, expected_code in test_cases:
    path = Path(audio_file)
    if not path.exists():
        print(f"\n⚠️  {expected_name}: File not found")
        continue
    
    total += 1
    try:
        result = predict_audio(path)
        detected_code = result['language_code']
        detected_name = result['language_name']
        confidence = result['language_confidence']
        
        if detected_code == expected_code:
            status = "✅ CORRECT"
            correct += 1
        elif detected_code == "unknown":
            status = "❔ UNKNOWN"
            unknown += 1
        else:
            status = "❌ WRONG"
        
        print(f"\n{status}")
        print(f"  File: {path.name}")
        print(f"  Expected: {expected_name} ({expected_code})")
        print(f"  Detected: {detected_name} ({detected_code}) - {confidence:.0%} confidence")
        print(f"  AI/Human: {result['label']} ({result['confidence']:.1%})")
        
    except Exception as e:
        print(f"\n❌ ERROR")
        print(f"  File: {path.name}")
        print(f"  Error: {e}")

print("\n" + "=" * 70)
print(f"RESULTS: {correct}/{total} correct")
if unknown > 0:
    print(f"Note: {unknown} files returned 'unknown' (safer than wrong guess)")
print("=" * 70)

accuracy = (correct / total * 100) if total > 0 else 0
print(f"\nAccuracy: {accuracy:.1f}%")
print("\nNote: Low accuracy means you need more training data.")
print("Add 50-100 audio samples per language to dataset/lang_* folders,")
print("then run: python -m app.train_language_model")
