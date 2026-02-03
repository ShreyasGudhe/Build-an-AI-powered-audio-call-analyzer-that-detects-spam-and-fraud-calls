@echo off
echo ========================================
echo Starting Fraud Analyzer API Server
echo ========================================
echo.
echo API will be available at: http://127.0.0.1:8000
echo.
echo Endpoints:
echo   - POST /analyze-call         (Comprehensive analysis)
echo   - POST /detect-fraud         (Fraud detection only)
echo   - POST /detect-voice         (Voice classification)
echo   - POST /transcribe           (Audio transcription)
echo   - GET  /alert-history        (View alert history)
echo   - POST /block-number         (Block/unblock numbers)
echo.
echo Press Ctrl+C to stop the server
echo.

python run_api.py

pause
