#!/bin/bash
echo "Activating virtual environment..."
source venv/Scripts/activate

echo "Starting Streamlit app..."
streamlit run agents/frontend/app.py
