@echo off
echo ==============================
echo Plant Disease Detection System
echo ==============================

IF NOT EXIST venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate

pip show streamlit >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Installing requirements...
    pip install -r requirements.txt
) ELSE (
    echo Requirements already installed.
)

echo Starting Streamlit App...
streamlit run app.py

pause
