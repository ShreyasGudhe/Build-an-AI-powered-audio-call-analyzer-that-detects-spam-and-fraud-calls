@echo off
cls
echo.
echo ========================================================================
echo   _____ ____     _____              _      _                _                   
echo  / ____|___ \   / ____|            ^| ^|    / \   _ __   __ _^| ^|_   _ _______ _ __ 
echo ^| ^|  __  __) ^| ^| ^|     __ _  __ _ ^| ^|   / _ \ ^| '_ \ / _` ^| ^| ^| ^| ^|_  / _ \ '__^|
echo ^| ^| ^|_ ^|/ __/  ^| ^|___ / _` ^|/ _` ^|^| ^|  / ___ \^| ^| ^| ^| (_^| ^| ^| ^|_^| ^|/ /  __/ ^|   
echo ^| ^|__^|^|_____^|  \_____\__,_^|\__,_^|^|_^| /_/   \_\_^| ^|_^|\__,_^|^|_^|\__, ^|/___\___^|_^|   
echo  \_____^|                                                  ^|___/            
echo.
echo                    AI-Powered Fraud Call Analyzer
echo                         v1.0 - Fraud Detection Suite
echo ========================================================================
echo.
echo Welcome! This tool helps you detect fraud and spam calls using AI.
echo.
echo What would you like to do?
echo.
echo   1. Start the Fraud Analyzer (API + Frontend)
echo   2. Run Tests to verify everything works
echo   3. Train Custom Fraud Detection Model
echo   4. View Documentation
echo   5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto start_app
if "%choice%"=="2" goto run_tests
if "%choice%"=="3" goto train_model
if "%choice%"=="4" goto view_docs
if "%choice%"=="5" goto exit_app
goto invalid_choice

:start_app
cls
echo ========================================================================
echo Starting AI Fraud Analyzer...
echo ========================================================================
echo.
echo Step 1: Starting API Server...
echo.
start "Fraud Analyzer API" cmd /k "cd /d %~dp0 && python run_api.py"
timeout /t 3 /nobreak >nul

echo Step 2: Installing Frontend Dependencies (if needed)...
cd frontend
if not exist "node_modules\" (
    echo Installing npm packages...
    call npm install
)

echo.
echo Step 3: Starting Frontend...
echo.
start "Fraud Analyzer Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

timeout /t 2 /nobreak >nul
echo.
echo ========================================================================
echo SUCCESS! Fraud Analyzer is now running!
echo ========================================================================
echo.
echo API Server:  http://127.0.0.1:8000
echo Frontend:    http://localhost:5173
echo API Docs:    http://127.0.0.1:8000/docs
echo.
echo Opening browser...
timeout /t 2 /nobreak >nul
start http://localhost:5173
echo.
echo Press any key to return to menu (servers will keep running)...
pause >nul
goto menu

:run_tests
cls
echo ========================================================================
echo Running Fraud Detection Tests...
echo ========================================================================
echo.
python test_fraud_detection.py
echo.
echo ========================================================================
echo Tests Complete!
echo ========================================================================
echo.
echo Press any key to return to menu...
pause >nul
goto menu

:train_model
cls
echo ========================================================================
echo Train Custom Fraud Detection Model
echo ========================================================================
echo.
echo To train a custom model, you need:
echo   - Audio files in dataset/fraud_calls/
echo   - Audio files in dataset/legitimate_calls/
echo   - At least 50 files per category (200+ recommended)
echo.
echo Do you have the training data ready?
echo.
set /p train_confirm="Continue with training? (Y/N): "
if /i "%train_confirm%"=="Y" (
    echo.
    echo Starting training...
    call train_fraud_model.bat
) else (
    echo.
    echo Training cancelled.
    echo.
    echo See dataset/DATASET_GUIDE.md for instructions on preparing training data.
)
echo.
echo Press any key to return to menu...
pause >nul
goto menu

:view_docs
cls
echo ========================================================================
echo Documentation
echo ========================================================================
echo.
echo Available documentation files:
echo.
echo   1. QUICK_START.md              - Get started in 5 minutes
echo   2. FRAUD_DETECTION_GUIDE.md    - Complete user guide
echo   3. dataset/DATASET_GUIDE.md    - Training data guide
echo   4. PROJECT_COMPLETE.md         - Project overview
echo   5. README.md                   - Main readme
echo.
set /p doc_choice="Which would you like to open? (1-5, or 0 to go back): "

if "%doc_choice%"=="1" start QUICK_START.md
if "%doc_choice%"=="2" start FRAUD_DETECTION_GUIDE.md
if "%doc_choice%"=="3" start dataset\DATASET_GUIDE.md
if "%doc_choice%"=="4" start PROJECT_COMPLETE.md
if "%doc_choice%"=="5" start README.md
if "%doc_choice%"=="0" goto menu

timeout /t 1 /nobreak >nul
goto view_docs

:invalid_choice
echo.
echo Invalid choice. Please enter 1-5.
timeout /t 2 /nobreak >nul
goto menu

:menu
cls
echo.
echo ========================================================================
echo                    AI-Powered Fraud Call Analyzer
echo ========================================================================
echo.
echo What would you like to do?
echo.
echo   1. Start the Fraud Analyzer (API + Frontend)
echo   2. Run Tests to verify everything works
echo   3. Train Custom Fraud Detection Model
echo   4. View Documentation
echo   5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto start_app
if "%choice%"=="2" goto run_tests
if "%choice%"=="3" goto train_model
if "%choice%"=="4" goto view_docs
if "%choice%"=="5" goto exit_app
goto invalid_choice

:exit_app
echo.
echo Thank you for using AI Fraud Analyzer!
echo Stay safe from fraud calls! 
echo.
timeout /t 2 /nobreak >nul
exit
