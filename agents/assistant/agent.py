from typing import Dict, List, Any
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools.function_tool import FunctionTool
import google.generativeai as genai

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

def analyze_data(trades_json: str = None, positions_json: str = None, question: str = None) -> Dict[str, Any]:
    """
    Analyze trade and position data to answer user questions.

    Args:
        trades_json: JSON string containing a list of trades (optional)
        positions_json: JSON string containing a list of positions (optional)
        question: The user's question about the data

    Returns:
        Dictionary containing the analysis and answer
    """
    try:
        # Prepare context for the AI
        context = {}

        # Parse trades if provided
        if trades_json:
            trades_data = json.loads(trades_json)
            context["trades"] = trades_data
            context["trade_count"] = len(trades_data)

        # Parse positions if provided
        if positions_json:
            positions_data = json.loads(positions_json)
            context["positions"] = positions_data
            context["position_count"] = len(positions_data)

        # If no data is provided, return an error
        if not trades_json and not positions_json:
            return {
                "status": "error",
                "error_message": "No data provided for analysis. Please upload and process trade data first."
            }

        # If no question is provided, generate general insights
        if not question:
            return {
                "status": "success",
                "answer": "I can analyze your trade and position data. Ask me specific questions about your data, such as 'What are my largest positions?' or 'How many trades do I have for GOLD?'",
                "context": context
            }

        # Return the context along with the question for the AI to process
        return {
            "status": "success",
            "question": question,
            "context": context
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error analyzing data: {str(e)}"
        }

# Function to create the AI Assistant Agent with API key
def create_assistant_agent(api_key=None):
    """
    Create the AI Assistant Agent with the provided API key.

    Args:
        api_key: Google API key for Gemini (optional, can use environment variable)

    Returns:
        The configured AI Assistant Agent
    """
    # Use provided API key or get from environment
    google_api_key = api_key or os.environ.get("GOOGLE_API_KEY")

    if not google_api_key:
        raise ValueError(
            "Google API key is required. Either pass it as an argument or "
            "set the GOOGLE_API_KEY environment variable."
        )

    # Configure the Google Generative AI library with the API key
    genai.configure(api_key=google_api_key)

    # Create and return the agent
    return Agent(
        model="gemini-2.0-flash",
        name="assistant_agent",
        description="AI assistant that analyzes trade data and answers questions",
        instruction="""
        You are an AI Assistant that helps users understand their trade data and positions.

        When a user asks a question, you'll receive:
        1. The user's question
        2. Context containing trade data and/or position data

        Your job is to:
        - Analyze the provided data to answer the user's question
        - Provide clear, concise explanations
        - Offer insights about their trading activity and positions
        - Help them understand FIFO and LIFO calculations

        If the data doesn't contain information to answer the question, politely explain what information is missing.

        Always be helpful, accurate, and educational about trading concepts.
        """,
        tools=[analyze_data]
    )

# Create a placeholder for the agent - will be properly initialized in server.py
assistant_agent = None
