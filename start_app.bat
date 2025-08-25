@echo off
echo 🎓 Starting Placement Prediction Web App...
echo =========================================

REM Check if Python is available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3 to continue.
    pause
    exit /b
)

REM Check dependencies
echo 📦 Checking dependencies...
python -c "import flask, pandas, numpy, sklearn, joblib, requests" >nul 2>nul
if %errorlevel% neq 0 (
    echo ⚠  Some dependencies are missing. Installing required packages...
    pip install Flask Flask-CORS pandas numpy scikit-learn joblib requests
)

REM Start the application in background
start "" python app.py

REM Wait a few seconds for Flask to start
timeout /t 5 /nobreak >nul

REM Open in default browser
start http://localhost:5000

echo 🚀 Flask application started!
echo 📱 App is running at: http://localhost:5000
echo 📊 Dashboard: http://localhost:5000/dashboard
echo ⏹  Press Ctrl+C in the terminal to stop the application
pause