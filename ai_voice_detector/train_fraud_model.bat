@echo off
echo ========================================
echo Training Fraud Detection Model
echo ========================================
echo.

REM Check if dataset directories exist
if not exist "dataset\fraud_calls" (
    echo ERROR: dataset\fraud_calls directory not found!
    echo Please create the directory and add fraud call samples.
    pause
    exit /b 1
)

if not exist "dataset\legitimate_calls" (
    echo ERROR: dataset\legitimate_calls directory not found!
    echo Please create the directory and add legitimate call samples.
    pause
    exit /b 1
)

REM Count files
echo Checking dataset...
dir /b /a-d "dataset\fraud_calls\*.*" 2>nul | find /c /v "" > temp_count.txt
set /p FRAUD_COUNT=<temp_count.txt

dir /b /a-d "dataset\legitimate_calls\*.*" 2>nul | find /c /v "" > temp_count.txt
set /p LEGIT_COUNT=<temp_count.txt
del temp_count.txt

echo Fraud calls: %FRAUD_COUNT% files
echo Legitimate calls: %LEGIT_COUNT% files
echo.

if %FRAUD_COUNT% LSS 10 (
    echo WARNING: Less than 10 fraud call samples found!
    echo Recommend at least 50 samples per category for good accuracy.
)

if %LEGIT_COUNT% LSS 10 (
    echo WARNING: Less than 10 legitimate call samples found!
    echo Recommend at least 50 samples per category for good accuracy.
)

echo.
echo Starting training...
echo.

python -m app.train_fraud_model dataset\fraud_calls dataset\legitimate_calls models\fraud_model.pkl

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo Model saved to: models\fraud_model.pkl
echo.
pause
