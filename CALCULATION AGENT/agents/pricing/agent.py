import yfinance as yf
from typing import Dict, Any, Optional

from google.adk.agents import Agent
from google.adk.tools.function_tool import FunctionTool


def get_price(symbol: str, exchange: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch the current price for a given symbol.
    
    Args:
        symbol: The symbol to fetch the price for (e.g., 'GOLD', 'SILVER')
        exchange: Optional exchange code (e.g., 'MCX')
        
    Returns:
        Dictionary containing the price information
    """
    try:
        # Map commodity symbols to Yahoo Finance tickers
        symbol_map = {
            'GOLD': 'GC=F',      # Gold Futures
            'SILVER': 'SI=F',     # Silver Futures
            'SILVER Mini': 'SI=F', # Silver Futures (mini)
            'COPPER': 'HG=F',     # Copper Futures
            'CRUDE OIL': 'CL=F',  # Crude Oil Futures
            'NATURAL GAS': 'NG=F' # Natural Gas Futures
        }
        
        # Get the Yahoo Finance ticker
        ticker = symbol_map.get(symbol)
        if not ticker:
            # Try to use the symbol directly if not in the map
            ticker = symbol
        
        # Fetch the price
        data = yf.Ticker(ticker)
        price = data.history(period="1d")['Close'].iloc[-1]
        
        return {
            "status": "success",
            "symbol": symbol,
            "exchange": exchange,
            "price": price,
            "currency": "USD"  # Default currency for Yahoo Finance
        }
    
    except Exception as e:
        return {
            "status": "error",
            "symbol": symbol,
            "exchange": exchange,
            "error_message": f"Error fetching price: {str(e)}"
        }


# Create the Pricing Agent
pricing_agent = Agent(
    model="gemini-2.0-flash",
    name="pricing_agent",
    description="Fetches real-time prices for commodities and other financial instruments",
    instruction="""
    You are a Pricing Agent that fetches real-time prices.
    Your job is to retrieve current market prices for various commodities and financial instruments.
    Use the 'get_price' tool to fetch prices for a given symbol.
    """,
    tools=[get_price]
)
