"""
Streamlit Cloud entry point for the Trade Position Calculator application.
This file imports and runs the main Streamlit app.
"""

import os
import sys
import json
import base64
import io
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Add the current directory to the path so we can import from the agents package
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Load environment variables from .env file if it exists
env_path = Path('.') / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Configure Google Generative AI with API key from Streamlit secrets or environment variable
try:
    # Try to get API key from Streamlit secrets
    if 'GOOGLE_API_KEY' in st.secrets:
        google_api_key = st.secrets['GOOGLE_API_KEY']
    # Fall back to environment variable
    else:
        google_api_key = os.environ.get('GOOGLE_API_KEY')

    if google_api_key and google_api_key != 'your_api_key_here':
        genai.configure(api_key=google_api_key)
except Exception as e:
    # If there's an error, we'll handle it in the AI Assistant tab
    pass

# Import the main app code
from agents.frontend.app import *

# Run the app directly
if __name__ == "__main__":
    # The app code will run automatically since we've imported everything from app.py
    pass
