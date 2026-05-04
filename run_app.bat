@echo off
echo Starting Concept Drift Detection Platform...
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate" (
    echo Virtual environment not found. Please create it first:
    echo   python -m venv venv
    echo   venv\Scripts\pip install -r requirements.txt
    exit /b 1
)

REM Activate virtual environment and run Streamlit
call venv\Scripts\activate
streamlit run app.py

pause
