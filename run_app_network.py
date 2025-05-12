"""
Launcher script to run the Streamlit app with network sharing enabled.
"""

import subprocess

# Run the Streamlit app with network sharing enabled
streamlit_cmd = "python -m streamlit run \"CALCULATION AGENT/agents/frontend/app.py\" --server.port 8081 --server.enableCORS false --server.enableXsrfProtection false --server.headless true"
subprocess.run(streamlit_cmd, shell=True)
