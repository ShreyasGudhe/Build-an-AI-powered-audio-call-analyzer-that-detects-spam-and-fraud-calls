from pathlib import Path
from app.predict import detect_language, resolve_language_name

test_files = [
    ("dataset/lang_en/english.mp3", "English"),
    ("dataset/lang_hi/hindi.mp3", "Hindi"),
    ("dataset/lang_ta/tamil.mp3", "Tamil"),
    ("dataset/lang_ml/malayalam.mp3", "Malayalam"),
    ("dataset/lang_te/telgu.mp3", "Telugu"),
]

print("\nLanguage Detection Results:")
print("-" * 60)

for test_file, expected in test_files:
    path = Path(test_file)
    if path.exists():
        try:
            code, conf = detect_language(path)
            name = resolve_language_name(code)
            status = "✓" if code != "unknown" else "✗"
            print(f"{status} {expected:12} -> {name:12} ({code:3}) {conf:.0%}")
        except Exception as e:
            print(f"✗ {expected:12} -> Error: {e}")
    else:
        print(f"✗ {expected:12} -> File not found")

print("-" * 60)
