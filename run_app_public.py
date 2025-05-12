"""
Launcher script to run the Streamlit app with a public URL using ngrok.
"""

import os
import subprocess
import threading
from pyngrok import ngrok

# Set the port
port = 8081

# Start ngrok tunnel
public_url = ngrok.connect(port)
print(f"\n\n🌐 Public URL: {public_url}\n\n")

# Run the Streamlit app
streamlit_cmd = f"python -m streamlit run \"CALCULATION AGENT/agents/frontend/app.py\" --server.port {port}"
subprocess.run(streamlit_cmd, shell=True)
