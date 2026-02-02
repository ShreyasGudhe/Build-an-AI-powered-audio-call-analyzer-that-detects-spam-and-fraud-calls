from pathlib import Path
from app.predict import detect_language, resolve_language_name, _predict_language_via_classifier
from app.predict import _ensure_language_model
from faster_whisper.transcribe import decode_audio

# Test with different language samples
test_files = [
    ("dataset/lang_en/english.mp3", "English"),
    ("dataset/lang_hi/hindi.mp3", "Hindi"),
    ("dataset/lang_ta/tamil.mp3", "Tamil"),
    ("dataset/lang_ml/malayalam.mp3", "Malayalam"),
    ("dataset/lang_te/telgu.mp3", "Telugu"),
]

print("=" * 80)
print("DETAILED LANGUAGE DETECTION TEST")
print("=" * 80)

for test_file, expected_lang in test_files:
    path = Path(test_file)
    if not path.exists():
        print(f"\n❌ Skipping: {expected_lang} - File not found")
        continue
    
    print(f"\n{'=' * 80}")
    print(f"Testing: {expected_lang} ({path.name})")
    print(f"{'=' * 80}")
    
    try:
        # Test classifier alone
        clf_result = _predict_language_via_classifier(path)
        if clf_result:
            print(f"  📊 Classifier: {resolve_language_name(clf_result[0])} ({clf_result[0]}) - {clf_result[1]:.1%}")
        else:
            print(f"  📊 Classifier: Not available")
        
        # Test Whisper detect_language
        try:
            model = _ensure_language_model()
            audio = decode_audio(path, sampling_rate=16000)
            detected = model.detect_language(audio[:16000*10])  # First 10 seconds
            if len(detected) == 2:
                code, prob = detected
            else:
                code, prob, _ = detected
            print(f"  🎤 Whisper:    {resolve_language_name(code)} ({code}) - {prob:.1%}")
        except Exception as e:
            print(f"  🎤 Whisper:    Error - {e}")
        
        # Final combined detection
        lang_code, confidence = detect_language(path)
        lang_name = resolve_language_name(lang_code)
        
        correct = "✅" if lang_code in expected_lang.lower() or expected_lang.lower() in lang_name.lower() else "❌"
        print(f"\n  {correct} FINAL: {lang_name} ({lang_code}) - {confidence:.1%}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "=" * 80)
print("Testing Complete!")
print("=" * 80)
