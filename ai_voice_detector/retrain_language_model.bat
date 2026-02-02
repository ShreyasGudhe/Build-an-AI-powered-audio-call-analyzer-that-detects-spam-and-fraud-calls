@echo off
echo ========================================
echo Retraining Language Detection Model
echo ========================================
echo.
echo This will retrain the language classifier using improved features.
echo Make sure you have audio files in dataset/lang_* folders.
echo.
echo Supported languages: en, hi, ta, te, ml
echo.
pause

cd /d "%~dp0"
python -m app.train_language_model

echo.
echo ========================================
echo Training Complete!
echo ========================================
pause
