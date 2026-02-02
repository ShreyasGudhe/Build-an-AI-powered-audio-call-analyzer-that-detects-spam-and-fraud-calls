from pathlib import Path
from app.predict import predict_audio

# Test with different language samples
test_files = [
    "dataset/lang_en/english.mp3",
    "dataset/lang_hi/hindi.mp3",
    "dataset/lang_ta/tamil.mp3",
    "dataset/lang_ml/malayalam.mp3",
    "dataset/lang_te/telgu.mp3"
]

print("=" * 70)
print("Testing Improved Language Detection")
print("=" * 70)

for test_file in test_files:
    path = Path(test_file)
    if path.exists():
        print(f"\nTesting: {path.name}")
        try:
            result = predict_audio(path)
            print(f"  AI/Human: {result['label']} ({result['confidence']:.1%} confidence)")
            print(f"  Language: {result['language_name']} ({result['language_code']})")
            print(f"  Language Confidence: {result['language_confidence']:.0%}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print(f"\nSkipping: {path.name} (file not found)")

print("\n" + "=" * 70)
print("Testing Complete!")
print("=" * 70)
