import base64
import io
import json
from typing import Dict, List, Any, Optional

from google.adk.agents import Agent
from google.adk.tools.function_tool import FunctionTool

from common.models.trade import Trade
from common.utils.trade_utils import parse_excel_file, parse_csv_file


def parse_file(file_bytes: str, mime_type: str) -> Dict[str, Any]:
    """
    Parse a file containing trade data and convert it to a uniform JSON format.
    
    Args:
        file_bytes: Base64 encoded file content
        mime_type: MIME type of the file (e.g., 'application/vnd.ms-excel', 'text/csv')
        
    Returns:
        Dictionary containing parsed trades
    """
    # Decode base64 file content
    decoded_bytes = base64.b64decode(file_bytes)
    
    # Parse based on file type
    if 'excel' in mime_type or 'spreadsheet' in mime_type:
        trades = parse_excel_file(decoded_bytes)
    elif 'csv' in mime_type:
        trades = parse_csv_file(decoded_bytes)
    else:
        return {
            "status": "error",
            "error_message": f"Unsupported file type: {mime_type}. Please upload an Excel or CSV file."
        }
    
    # Convert trades to dictionaries
    trade_dicts = [trade.to_dict() for trade in trades]
    
    return {
        "status": "success",
        "trades": trade_dicts,
        "count": len(trade_dicts)
    }


# Create the Data Agent
data_agent = Agent(
    model="gemini-2.0-flash",
    name="data_agent",
    description="Parses CSV/XLS(X) files into a uniform JSON trade list",
    instruction="""
    You are a Data Agent that parses trade data files.
    Your job is to extract trade information from CSV or Excel files and convert it to a uniform JSON format.
    Use the 'parse_file' tool to process the file and return the parsed trades.
    """,
    tools=[parse_file]
)
