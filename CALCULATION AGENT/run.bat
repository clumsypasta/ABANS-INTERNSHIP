@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Starting the Calculation Agent system...
echo (Using API key from .env file if available)
python main.py --streamlit

pause
