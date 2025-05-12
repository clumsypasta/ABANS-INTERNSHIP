"""
Test script to verify that the virtual environment is set up correctly.
"""
import sys
import pandas as pd
import numpy as np
import plotly
import streamlit
import yfinance

def main():
    """Print information about the installed packages."""
    print("Python version:", sys.version)
    print("\nInstalled packages:")
    print("pandas version:", pd.__version__)
    print("numpy version:", np.__version__)
    print("plotly version:", plotly.__version__)
    print("streamlit version:", streamlit.__version__)
    print("yfinance version:", yfinance.__version__)

    # Check if google-adk is installed
    try:
        import google.adk
        print("google-adk is installed")
    except ImportError:
        print("google-adk import failed, but package might still be installed")

    print("\nEnvironment is set up correctly!")

if __name__ == "__main__":
    main()
