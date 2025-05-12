@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Starting Streamlit app...
streamlit run agents/frontend/app.py

pause
