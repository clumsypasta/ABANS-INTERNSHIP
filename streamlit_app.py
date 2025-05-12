"""
Streamlit Cloud entry point for the Trade Position Calculator application.
This is a standalone version that includes all the necessary code.
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
from collections import defaultdict, deque
import numpy as np
import re

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

# Set page config
st.set_page_config(
    page_title="Trade Position Calculator",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Trade Position Calculator")
st.markdown("""
This application calculates FIFO and LIFO weighted average prices for open positions based on trade data.
Upload your trade data file (Excel or CSV) to get started.
""")

# Helper functions for AI Assistant
def generate_ai_response(prompt, data_context):
    """
    Generate a response using Google's Gemini AI model.

    Args:
        prompt: The user's question
        data_context: Dictionary containing trade and position data

    Returns:
        AI-generated response as a string
    """
    # Format the data context for the AI
    context_str = format_data_for_ai(data_context)

    # Create a system prompt with instructions
    system_prompt = """
    You are an AI Assistant specialized in analyzing trade data and positions.

    Your capabilities:
    - Analyze trade data and positions to answer user questions
    - Explain financial concepts like FIFO and LIFO
    - Provide insights about trading patterns and positions
    - Calculate statistics and metrics from the data
    - Access and analyze the raw data from uploaded files

    The data context provided to you includes:
    1. Raw Data Information - The original data from the uploaded file
    2. Position Summary - Statistics about calculated positions
    3. Position Details - Information about each position
    4. P&L Details - Profit and loss calculations
    5. Trade Summary - Information about all trades

    When responding:
    - Be concise but thorough
    - Use markdown formatting for better readability
    - Include relevant numbers and calculations
    - Explain your reasoning
    - If asked about specific contracts, focus on those
    - If asked about the raw data, refer to the Raw Data Information section
    - If asked about specific sheets or data elements, look for them in the raw data
    - If the data doesn't contain information to answer the question, explain what's missing

    IMPORTANT: When asked about specific data in the uploaded file, always check the Raw Data Information section first. This contains the complete data from the file, including all rows and columns.
    """

    # Create the user prompt with the question and data context
    user_prompt = f"""
    Question: {prompt}

    Here's the data I have available:
    {context_str}

    Please analyze this data to answer my question. Provide clear explanations and insights.
    """

    # Configure the model for faster responses
    generation_config = {
        "temperature": 0.1,  # Lower temperature for more focused responses
        "top_p": 0.85,       # Slightly lower top_p for faster generation
        "top_k": 40,         # Lower top_k for faster responses
        "max_output_tokens": 1024,  # Shorter responses
    }

    # Create the model with the flash version for faster responses
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",  # Use the flash model for speed
        generation_config=generation_config
    )

    # Generate the response with a timeout
    try:
        # Set a timeout for the response generation
        response = model.generate_content([
            {"role": "user", "parts": [system_prompt]},
            {"role": "model", "parts": ["I understand. I'll help analyze trade data and positions based on the context provided."]},
            {"role": "user", "parts": [user_prompt]}
        ])
    except Exception as e:
        # If there's a timeout or other error, return a simplified response
        return f"I'm having trouble generating a detailed response right now. Here's what I know about your data:\n\n" + \
               f"- File: {data_context['raw_data']['file_name']}\n" + \
               f"- Columns: {', '.join(data_context['raw_data']['columns'])}\n" + \
               f"- Rows: {data_context['raw_data']['row_count']}\n\n" + \
               f"Error details: {str(e)}"

    return response.text

def format_data_for_ai(data_context):
    """
    Format the data context into a string for the AI.

    Args:
        data_context: Dictionary containing trade and position data

    Returns:
        Formatted string with data context
    """
    context_parts = []

    # Add raw data information first (optimized)
    if "raw_data" in data_context:
        raw_data = data_context["raw_data"]
        context_parts.append(f"## Raw Data Information")
        context_parts.append(f"File: {raw_data['file_name']} ({raw_data['file_type']})")
        context_parts.append(f"Total rows: {raw_data['row_count']}")

        if raw_data['columns']:
            context_parts.append(f"Columns: {', '.join(raw_data['columns'])}")

        # Add sample of raw data (more concise format)
        if raw_data['data_sample']:
            context_parts.append("\n### Raw Data Sample (first rows):")
            # Only show first 5 rows in a more compact format
            for i, row in enumerate(raw_data['data_sample'][:5]):
                compact_row = {k: v for k, v in row.items() if v is not None and v != ''}
                context_parts.append(f"Row {i+1}: {compact_row}")

    # Add position stats (more concise)
    if "position_stats" in data_context:
        stats = data_context["position_stats"]
        context_parts.append(f"\n## Position Summary")
        context_parts.append(f"Positions: {stats['num_positions']} across {stats['total_contracts']} contracts")
        context_parts.append(f"Total quantity: {stats['total_quantity']}, Avg FIFO: {stats['avg_fifo_price']:.2f}, Avg LIFO: {stats['avg_lifo_price']:.2f}")
        context_parts.append(f"Contracts: {', '.join(stats['contracts_list'])}")

    # Add P&L stats if available (more concise)
    if "pnl_stats" in data_context:
        pnl_stats = data_context["pnl_stats"]
        context_parts.append(f"\n## P&L Summary: FIFO P&L: {pnl_stats['total_fifo_pnl']:.2f}, LIFO P&L: {pnl_stats['total_lifo_pnl']:.2f}")

    # Add position details (more concise)
    if "positions" in data_context and data_context["positions"]:
        context_parts.append("\n## Position Details:")
        # Create a more compact representation
        positions_table = []
        for pos in data_context["positions"]:
            positions_table.append(f"{pos['contract']}: Qty={pos['open_qty']}, FIFO={pos['fifo_wap']:.2f}, LIFO={pos['lifo_wap']:.2f}, Expiry={pos['expiry']}, Client={pos['client_code']}")
        context_parts.append("\n".join(positions_table))

    # Add trade data sample (more concise)
    if "trades_sample" in data_context and data_context["trades_sample"]:
        sample_count = len(data_context["trades_sample"])
        total_count = data_context["raw_data"]["row_count"] if "raw_data" in data_context else sample_count

        context_parts.append(f"\n## Trade Data Sample ({sample_count} of {total_count} rows):")
        # Only include key fields for each trade
        for i, trade in enumerate(data_context["trades_sample"][:10]):  # Limit to 10 trades
            key_fields = {k: trade.get(k) for k in ['date', 'contract', 'side', 'quantity', 'price', 'client_code'] if k in trade}
            context_parts.append(f"Trade {i+1}: {key_fields}")

    # Add data rows if available
    if "raw_data" in data_context and "data_rows" in data_context["raw_data"] and data_context["raw_data"]["data_rows"]:
        context_parts.append(f"\n## Complete Data Rows:")
        # Include the first 10 rows in a compact format
        for i, row in enumerate(data_context["raw_data"]["data_rows"][:10]):
            compact_row = {k: v for k, v in row.items() if v is not None and v != ''}
            context_parts.append(f"Row {i+1}: {compact_row}")

        if len(data_context["raw_data"]["data_rows"]) > 10:
            context_parts.append(f"... and {len(data_context['raw_data']['data_rows']) - 10} more rows available.")

    return "\n".join(context_parts)

def generate_quick_data_response(prompt, data_context):
    """
    Generate quick responses for simple data-related questions without using the AI API.
    This provides immediate responses for common data questions.

    Args:
        prompt: The user's question
        data_context: Dictionary containing all data context

    Returns:
        Quick response as a string or None if the question requires more complex analysis
    """
    prompt_lower = prompt.lower()

    # Check if raw data exists
    if "raw_data" not in data_context:
        return None

    raw_data = data_context["raw_data"]

    # Questions about what data is available
    if any(q in prompt_lower for q in ["what data can you see", "what data do you have", "what data is available",
                                      "what data can you access", "what data is in the file", "what data is uploaded"]):
        response = f"""
        # Data Available

        I can see the following data:

        - **File**: {raw_data['file_name']} ({raw_data['file_type']})
        - **Rows**: {raw_data['row_count']} rows of data
        - **Columns**: {', '.join(raw_data['columns'])}

        The data contains trade information including dates, commodities, buy/sell quantities and prices,
        exchange information, expiry dates, client codes, and strategy information.

        I can answer questions about this data, analyze positions, and provide insights about your trades.
        """
        return response

    # Questions about columns
    if any(q in prompt_lower for q in ["what columns", "what fields", "column names", "field names"]):
        response = f"""
        # Columns in the Data

        The data contains the following columns:

        {', '.join(raw_data['columns'])}
        """
        return response

    # Questions about specific column values
    for col in raw_data['columns']:
        if col.lower() in prompt_lower and "values" in prompt_lower:
            # Extract unique values from the sample data
            values = set()
            for row in raw_data['data_sample'] + raw_data.get('data_rows', []):
                if col in row and row[col] is not None:
                    values.add(str(row[col]))

            values_list = list(values)
            if len(values_list) > 20:
                values_list = values_list[:20]
                values_str = ", ".join(values_list) + f", ... (and {len(values) - 20} more unique values)"
            else:
                values_str = ", ".join(values_list)

            response = f"""
            # Values in the '{col}' Column

            Based on the data sample, the '{col}' column contains these values:

            {values_str}
            """
            return response

    # Questions about row count
    if any(q in prompt_lower for q in ["how many rows", "row count", "number of rows", "data size"]):
        response = f"""
        # Data Size

        The uploaded file contains **{raw_data['row_count']} rows** of data.
        """
        return response

    # Questions about specific rows
    if "first row" in prompt_lower or "row 1" in prompt_lower:
        if raw_data['data_sample']:
            row = raw_data['data_sample'][0]
            row_str = "\n".join([f"- **{k}**: {v}" for k, v in row.items() if v is not None])
            response = f"""
            # First Row of Data

            The first row contains:

            {row_str}
            """
            return response

    # Return None if no quick response is available
    return None

def generate_simple_response(prompt, positions_df):
    """
    Generate a simple response without using the AI API.
    This is a fallback when the API key is not configured.

    Args:
        prompt: The user's question
        positions_df: DataFrame containing position data

    Returns:
        Simple response as a string
    """
    # Get basic stats about the positions
    num_positions = len(positions_df)
    total_contracts = positions_df['contract'].nunique()
    contracts_list = positions_df['contract'].unique().tolist()

    # Find largest position
    if not positions_df.empty:
        largest_position = positions_df.loc[positions_df['open_qty'].idxmax()]
        largest_contract = largest_position['contract']
        largest_qty = largest_position['open_qty']
    else:
        largest_contract = "None"
        largest_qty = 0

    # Generate response based on the question
    prompt_lower = prompt.lower()

    if "largest" in prompt_lower or "biggest" in prompt_lower:
        if num_positions > 0:
            return f"Based on your data, your largest position is in **{largest_contract}** with an open quantity of **{largest_qty}**."
        else:
            return "You currently have no open positions in your data."

    elif any(contract.lower() in prompt_lower for contract in contracts_list):
        # Find which contract they're asking about
        for contract in contracts_list:
            if contract.lower() in prompt_lower:
                contract_data = positions_df[positions_df['contract'] == contract].iloc[0]
                return f"""
                You have a position in **{contract}** with:
                - Open quantity: **{contract_data['open_qty']}**
                - FIFO weighted average price: **${contract_data['fifo_wap']}**
                - LIFO weighted average price: **${contract_data['lifo_wap']}**
                - Expiry date: **{contract_data['expiry']}**
                """

    elif "fifo" in prompt_lower or "lifo" in prompt_lower:
        if num_positions > 0:
            # Create a table of FIFO vs LIFO prices
            table_rows = []
            for _, row in positions_df.iterrows():
                table_rows.append(f"| {row['contract']} | ${row['fifo_wap']} | ${row['lifo_wap']} |")

            table = "\n".join(table_rows)

            return f"""
            **FIFO** (First-In-First-Out) and **LIFO** (Last-In-First-Out) are methods for calculating the weighted average price of your positions:

            - **FIFO**: Assumes that the oldest trades are closed first
            - **LIFO**: Assumes that the newest trades are closed first

            Your positions show different weighted average prices using these methods:

            | Contract | FIFO WAP | LIFO WAP |
            |----------|----------|----------|
            {table}
            """
        else:
            return "You currently have no open positions to calculate FIFO or LIFO prices."

    else:
        if num_positions > 0:
            positions_summary = []
            for _, row in positions_df.iterrows():
                positions_summary.append(f"**{row['contract']}** ({row['open_qty']} units)")

            positions_text = ", ".join(positions_summary)

            return f"""
            I've analyzed your trade data. You have {num_positions} open positions: {positions_text}.

            What specific information would you like to know about your positions?

            You can ask about:
            - Details about a specific contract
            - Your largest position
            - FIFO vs LIFO pricing
            - Expiry dates
            """
        else:
            return "You currently have no open positions in your data. Please upload trade data and calculate positions first."

# Initialize session state variables
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "df" not in st.session_state:
    st.session_state.df = None
if "positions_df" not in st.session_state:
    st.session_state.positions_df = None
if "prices_df" not in st.session_state:
    st.session_state.prices_df = None
if "pnl_df" not in st.session_state:
    st.session_state.pnl_df = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Data Upload & Positions",
    "📊 Visualizations",
    "💰 Market Prices & P&L",
    "🤖 AI Assistant"
])

# Tab 1: Data Upload & Positions
with tab1:
    # File upload section
    uploaded_file = st.file_uploader("Upload trade data file", type=["xlsx", "csv"])

    if uploaded_file is not None:
        # Store the uploaded file in session state
        st.session_state.uploaded_file = uploaded_file

        # Display file info
        st.info(f"File uploaded: {uploaded_file.name} ({uploaded_file.type})")

        # Read file content
        file_bytes = uploaded_file.read()

        # Encode file content as base64
        encoded_file = base64.b64encode(file_bytes).decode()

        # Call Data Agent to parse file
        st.subheader("Parsing trade data...")

        # This would be replaced with actual A2A call to Data Agent
        # For now, we'll simulate the parsing
        if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            df = pd.read_csv(io.BytesIO(file_bytes))

        # Validate the dataframe has the required columns
        required_columns = ['Date', 'Commodity', 'Buy', 'Buy Average', 'Sell', 'Sell Average',
                           'Exchange', 'Expiry', 'Client Code', 'Strategy', 'Code']

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"The uploaded file is missing the following required columns: {', '.join(missing_columns)}")
            st.info("Please ensure your file has all the required columns: " +
                   ", ".join(required_columns) + " (Remarks and Tagging are optional)")
            # Don't store the invalid dataframe
            st.stop()  # Use st.stop() instead of return

        # Store the dataframe in session state
        st.session_state.df = df

        # Display sample of parsed data
        st.dataframe(df.head())

        # Show data types to help with debugging
        st.expander("Data Types Information").write(df.dtypes)

    # Process trades button
    if st.session_state.df is not None:
        if st.button("Calculate Positions"):
            st.subheader("Calculating positions...")

            # Convert the uploaded data to the format expected by the compute agent
            df = st.session_state.df
            trades = []

            for _, row in df.iterrows():
                # Skip rows without buy or sell data
                if pd.isna(row.get('Buy', None)) and pd.isna(row.get('Sell', None)):
                    continue

                # Determine if it's a buy or sell trade
                if pd.notna(row.get('Buy', None)):
                    side = 'Buy'
                    quantity = row['Buy']
                    price = row['Buy Average']
                else:
                    side = 'Sell'
                    quantity = row['Sell']
                    price = row['Sell Average']

                # Create trade dictionary with safe type conversions
                try:
                    # Handle date and expiry
                    date_val = row['Date'].isoformat() if isinstance(row['Date'], datetime) else row['Date']
                    expiry_val = row['Expiry'].isoformat() if isinstance(row['Expiry'], datetime) else row['Expiry']

                    # Handle numeric values with safe conversion
                    try:
                        quantity_val = float(quantity)
                    except (ValueError, TypeError):
                        st.warning(f"Invalid quantity value: {quantity}. Using 0.")
                        quantity_val = 0.0

                    try:
                        price_val = float(price)
                    except (ValueError, TypeError):
                        st.warning(f"Invalid price value: {price}. Using 0.")
                        price_val = 0.0

                    # Keep client_code as string to handle alphanumeric values
                    client_code_val = str(row['Client Code'])

                    trade = {
                        'date': date_val,
                        'contract': row['Commodity'],
                        'side': side,
                        'quantity': quantity_val,
                        'price': price_val,
                        'expiry': expiry_val,
                        'exchange': row['Exchange'],
                        'client_code': client_code_val,
                        'strategy': row['Strategy'],
                        'code': row['Code'],
                        'remarks': row['Remarks'] if 'Remarks' in row and pd.notna(row['Remarks']) else None,
                        'tagging': row['Tagging'] if 'Tagging' in row and pd.notna(row['Tagging']) else None
                    }
                except Exception as e:
                    st.error(f"Error processing row: {e}")
                    st.write(f"Problematic row: {row}")
                    continue

                trades.append(trade)

            # Convert trades to JSON
            trades_json = json.dumps(trades)

            # Implement the compute_positions function directly in the frontend
            # This avoids the dependency on google.adk
            try:
                # Parse trades from JSON
                trades_data = json.loads(trades_json)

                # Get current date for checking contract expiry
                current_date = datetime.now()
                st.info(f"Current date for filtering: {current_date.strftime('%Y-%m-%d')}")

                # Sort trades by date
                trades_data.sort(key=lambda x: x['date'])

                # Group trades by contract and other identifiers
                grouped_trades = defaultdict(list)
                for trade in trades_data:
                    # Create a unique key for each contract group
                    # All values in the key should be treated as strings for consistency
                    key = (
                        str(trade['contract']),
                        str(trade['exchange']),
                        str(trade['expiry']),
                        str(trade['client_code']),  # Now client_code is already a string
                        str(trade['strategy']),
                        str(trade['code']),
                        str(trade.get('tagging', ''))  # Handle None values
                    )
                    grouped_trades[key].append(trade)

                # Process each group of trades
                positions = []
                for key, contract_trades in grouped_trades.items():
                    contract, exchange, expiry_str, client_code, strategy, code, tagging = key

                    # Convert expiry to datetime if it's a string with error handling
                    try:
                        if isinstance(expiry_str, str):
                            # Try to parse the expiry date string
                            try:
                                # First try ISO format
                                expiry = datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
                            except ValueError:
                                # If that fails, try common date formats
                                for date_format in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y']:
                                    try:
                                        expiry = datetime.strptime(expiry_str, date_format)
                                        break
                                    except ValueError:
                                        continue
                                else:
                                    # If all formats fail, use a far future date and warn the user
                                    st.warning(f"Could not parse expiry date: {expiry_str}. Using a default future date.")
                                    expiry = datetime.now().replace(year=datetime.now().year + 10)
                        else:
                            expiry = expiry_str
                    except Exception as e:
                        st.error(f"Error processing expiry date: {e}")
                        # Use a far future date as fallback
                        expiry = datetime.now().replace(year=datetime.now().year + 10)

                    # Skip expired contracts - only process active contracts
                    if expiry <= current_date:
                        st.write(f"Skipping expired contract: {contract} (Expiry: {expiry.strftime('%Y-%m-%d')})")
                        continue
                    else:
                        st.write(f"Processing active contract: {contract} (Expiry: {expiry.strftime('%Y-%m-%d')})")

                    # Initialize deques for FIFO and LIFO
                    fifo_deque = deque()
                    lifo_deque = deque()

                    # Process trades in chronological order
                    for trade in contract_trades:
                        if trade['side'] == 'Buy':
                            # Add buy trade to both deques
                            lot = {"quantity": trade['quantity'], "price": trade['price']}
                            fifo_deque.append(lot)
                            lifo_deque.append(lot)
                        elif trade['side'] == 'Sell':
                            # Process sell trade using FIFO
                            remaining_sell_qty = trade['quantity']
                            while remaining_sell_qty > 0 and fifo_deque:
                                lot = fifo_deque[0]
                                if lot["quantity"] <= remaining_sell_qty:
                                    # Use entire lot
                                    remaining_sell_qty -= lot["quantity"]
                                    fifo_deque.popleft()
                                else:
                                    # Use partial lot
                                    lot["quantity"] -= remaining_sell_qty
                                    remaining_sell_qty = 0

                            # Process sell trade using LIFO
                            remaining_sell_qty = trade['quantity']
                            while remaining_sell_qty > 0 and lifo_deque:
                                lot = lifo_deque[-1]
                                if lot["quantity"] <= remaining_sell_qty:
                                    # Use entire lot
                                    remaining_sell_qty -= lot["quantity"]
                                    lifo_deque.pop()
                                else:
                                    # Use partial lot
                                    lot["quantity"] -= remaining_sell_qty
                                    remaining_sell_qty = 0

                    # Calculate open quantity and weighted average prices
                    fifo_open_qty = sum(lot["quantity"] for lot in fifo_deque)
                    lifo_open_qty = sum(lot["quantity"] for lot in lifo_deque)

                    # Skip if no open position
                    if fifo_open_qty < 0.001:
                        continue

                    # Calculate weighted average prices
                    def calculate_weighted_average(lots):
                        if not lots:
                            return 0.0
                        total_quantity = sum(lot["quantity"] for lot in lots)
                        if total_quantity == 0:
                            return 0.0
                        weighted_sum = sum(lot["quantity"] * lot["price"] for lot in lots)
                        return weighted_sum / total_quantity

                    fifo_wap = calculate_weighted_average(list(fifo_deque))
                    lifo_wap = calculate_weighted_average(list(lifo_deque))

                    # Create position with safe type handling
                    try:
                        position = {
                            "contract": str(contract),
                            "open_qty": float(fifo_open_qty),
                            "fifo_wap": float(fifo_wap),
                            "lifo_wap": float(lifo_wap),
                            "exchange": str(exchange),
                            "expiry": expiry.isoformat() if isinstance(expiry, datetime) else str(expiry),
                            "client_code": str(client_code),  # Keep as string to handle alphanumeric values
                            "strategy": str(strategy),
                            "code": str(code),
                            "tagging": str(tagging) if tagging is not None else None
                        }
                        positions.append(position)
                    except Exception as e:
                        st.error(f"Error creating position: {e}")
                        st.write(f"Problematic position data: contract={contract}, client_code={client_code}")

                # Success message
                st.success(f"Successfully calculated {len(positions)} positions.")

                # Convert to DataFrame
                positions_df = pd.DataFrame(positions)

                # Format expiry date if needed
                if 'expiry' in positions_df.columns and not positions_df.empty:
                    positions_df['expiry'] = pd.to_datetime(positions_df['expiry']).dt.strftime('%Y-%m-%d')

                # Format prices
                if 'fifo_wap' in positions_df.columns and not positions_df.empty:
                    positions_df['fifo_wap'] = positions_df['fifo_wap'].round(2)
                if 'lifo_wap' in positions_df.columns and not positions_df.empty:
                    positions_df['lifo_wap'] = positions_df['lifo_wap'].round(2)

            except Exception as e:
                st.error(f"Error calculating positions: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()  # Stop execution if there's an error

            # Store positions in session state
            st.session_state.positions_df = positions_df

            # Display table
            st.subheader("Open Positions")
            st.dataframe(positions_df)

            # Save positions to CSV
            if len(positions_df) > 0:
                output_dir = 'OUTPUTS'
                import os
                os.makedirs(output_dir, exist_ok=True)

                output_file = os.path.join(output_dir, f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}_export.csv")
                positions_df.to_csv(output_file)

                st.download_button(
                    label="Download Positions as CSV",
                    data=positions_df.to_csv().encode('utf-8'),
                    file_name=f"positions_{datetime.now().strftime('%Y-%m-%d')}.csv",
                    mime="text/csv"
                )

                st.info(f"Positions saved to {output_file}")

            # Success message
            st.success("Positions calculated successfully! Check the Visualizations tab for charts.")

# Tab 2: Visualizations
with tab2:
    if st.session_state.positions_df is not None:
        st.header("Position Visualizations")

        # This would be replaced with actual A2A call to Visualization Agent
        # For now, we'll create simple charts

        positions_df = st.session_state.positions_df

        # Create two columns for charts
        col1, col2 = st.columns(2)

        with col1:
            # Bar chart of open quantities
            fig_qty = go.Figure(data=[
                go.Bar(
                    x=positions_df['contract'],
                    y=positions_df['open_qty'],
                    marker_color='blue'
                )
            ])
            fig_qty.update_layout(
                title="Open Quantities by Contract",
                xaxis_title="Contract",
                yaxis_title="Quantity",
                height=400
            )
            st.plotly_chart(fig_qty, use_container_width=True)

        with col2:
            # Bar chart comparing FIFO and LIFO WAP
            fig_wap = go.Figure(data=[
                go.Bar(
                    name='FIFO WAP',
                    x=positions_df['contract'],
                    y=positions_df['fifo_wap'],
                    marker_color='blue'
                ),
                go.Bar(
                    name='LIFO WAP',
                    x=positions_df['contract'],
                    y=positions_df['lifo_wap'],
                    marker_color='green'
                )
            ])
            fig_wap.update_layout(
                title="FIFO vs LIFO Weighted Average Prices",
                xaxis_title="Contract",
                yaxis_title="Price",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_wap, use_container_width=True)

        # Add a pie chart showing position distribution
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=positions_df['contract'],
                values=positions_df['open_qty'],
                hole=.3
            )
        ])
        fig_pie.update_layout(
            title="Position Distribution by Contract",
            height=500
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    else:
        st.info("Upload a file and calculate positions to view visualizations.")

# Tab 3: Market Prices & P&L
with tab3:
    if st.session_state.positions_df is not None:
        st.header("Market Prices & P&L Analysis")

        # Option to fetch current prices
        if st.button("Fetch Current Market Prices"):
            st.info("Fetching current prices...")

            # Get unique contracts from positions
            contracts = st.session_state.positions_df['contract'].unique().tolist()

            # Create input form for entering market prices
            st.subheader("Enter Current Market Prices")

            # Create a form for entering prices
            with st.form("market_prices_form"):
                prices = []

                for contract in contracts:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        price = st.number_input(f"{contract} Price", min_value=0.0, step=0.01, format="%.2f")
                    with col2:
                        currency = st.selectbox(f"{contract} Currency", options=["USD", "EUR", "GBP", "INR"], index=0, key=f"currency_{contract}")

                    prices.append({
                        "symbol": contract,
                        "price": price,
                        "currency": currency
                    })

                submit_button = st.form_submit_button("Submit Prices")

                if submit_button:
                    # Display prices
                    prices_df = pd.DataFrame(prices)
                    st.session_state.prices_df = prices_df

                    st.subheader("Current Market Prices")
                    st.dataframe(prices_df)

            # Calculate unrealized P&L
            positions = st.session_state.positions_df.to_dict('records')

            # Merge positions and prices
            pnl_data = []
            for pos in positions:
                for price in prices:
                    if pos['contract'] == price['symbol']:
                        fifo_pnl = (price['price'] - pos['fifo_wap']) * pos['open_qty']
                        lifo_pnl = (price['price'] - pos['lifo_wap']) * pos['open_qty']

                        pnl_data.append({
                            "contract": pos['contract'],
                            "current_price": price['price'],
                            "fifo_pnl": fifo_pnl,
                            "lifo_pnl": lifo_pnl
                        })

            # Convert to DataFrame
            pnl_df = pd.DataFrame(pnl_data)

            # Format P&L values
            pnl_df['fifo_pnl'] = pnl_df['fifo_pnl'].round(2)
            pnl_df['lifo_pnl'] = pnl_df['lifo_pnl'].round(2)

            # Store in session state
            st.session_state.pnl_df = pnl_df

        # Display P&L data if available
        if st.session_state.pnl_df is not None:
            st.subheader("Unrealized P&L")
            st.dataframe(st.session_state.pnl_df)

            pnl_df = st.session_state.pnl_df

            # Bar chart of P&L
            fig_pnl = go.Figure(data=[
                go.Bar(
                    name='FIFO P&L',
                    x=pnl_df['contract'],
                    y=pnl_df['fifo_pnl'],
                    marker_color='blue'
                ),
                go.Bar(
                    name='LIFO P&L',
                    x=pnl_df['contract'],
                    y=pnl_df['lifo_pnl'],
                    marker_color='green'
                )
            ])
            fig_pnl.update_layout(
                title="Unrealized P&L by Contract",
                xaxis_title="Contract",
                yaxis_title="P&L",
                barmode='group'
            )
            st.plotly_chart(fig_pnl)

            # Add a waterfall chart for P&L contribution
            fig_waterfall = go.Figure(go.Waterfall(
                name="FIFO P&L",
                orientation="v",
                measure=["relative"] * len(pnl_df) + ["total"],
                x=list(pnl_df['contract']) + ["Total"],
                y=list(pnl_df['fifo_pnl']) + [pnl_df['fifo_pnl'].sum()],
                connector={"line":{"color":"rgb(63, 63, 63)"}},
            ))

            fig_waterfall.update_layout(
                title="P&L Contribution by Contract (FIFO)",
                showlegend=False
            )
            st.plotly_chart(fig_waterfall)
    else:
        st.info("Upload a file and calculate positions to access market prices and P&L analysis.")

# Tab 4: AI Assistant
with tab4:
    if st.session_state.positions_df is not None:
        st.header("🤖 AI Assistant")
        st.markdown("""
        Ask questions about your trade data and positions. The AI assistant can help you understand your data and provide insights.
        """)

        # Check if API key is configured
        api_key_configured = False
        try:
            if 'GOOGLE_API_KEY' in st.secrets and st.secrets['GOOGLE_API_KEY'] != 'your_api_key_here':
                api_key_configured = True
            elif os.environ.get('GOOGLE_API_KEY') and os.environ.get('GOOGLE_API_KEY') != 'your_api_key_here':
                api_key_configured = True
        except Exception:
            pass

        if not api_key_configured:
            st.warning("""
            **Google API Key Not Configured**

            To use the AI Assistant, you need to configure a Google API key for Gemini.

            You can do this by:
            1. Creating a `.streamlit/secrets.toml` file with your API key
            2. Setting the GOOGLE_API_KEY environment variable

            Without an API key, the AI Assistant will use a simplified mode with limited capabilities.
            """)
            use_advanced_ai = False
        else:
            use_advanced_ai = True

        # Initialize session state for data context
        if "data_context" not in st.session_state:
            st.session_state.data_context = {}

        # Prepare data context for AI
        positions_df = st.session_state.positions_df
        trades_df = st.session_state.df if st.session_state.df is not None else None

        # Create a more efficient data context
        data_context = {
            # Include the raw dataframe information (optimized)
            "raw_data": {
                "file_name": st.session_state.uploaded_file.name if st.session_state.uploaded_file else "No file uploaded",
                "file_type": st.session_state.uploaded_file.type if st.session_state.uploaded_file else "Unknown",
                "columns": trades_df.columns.tolist() if trades_df is not None else [],
                "row_count": len(trades_df) if trades_df is not None else 0,
                "data_sample": trades_df.head(10).to_dict('records') if trades_df is not None else [],
                "data_summary": trades_df.describe().to_dict() if trades_df is not None else {},
                # Only include first 50 rows of full data to avoid overwhelming the AI
                "data_rows": trades_df.head(50).to_dict('records') if trades_df is not None else []
            },

            # Include processed position data (all positions since there are usually fewer)
            "positions": positions_df.to_dict('records') if positions_df is not None else [],

            # Include a sample of original trade data
            "trades_sample": trades_df.head(20).to_dict('records') if trades_df is not None else [],

            # Include statistical summaries
            "position_stats": {
                "num_positions": len(positions_df) if positions_df is not None else 0,
                "total_contracts": positions_df['contract'].nunique() if positions_df is not None else 0,
                "contracts_list": positions_df['contract'].unique().tolist() if positions_df is not None else [],
                "total_quantity": positions_df['open_qty'].sum() if positions_df is not None else 0,
                "avg_fifo_price": positions_df['fifo_wap'].mean() if positions_df is not None else 0,
                "avg_lifo_price": positions_df['lifo_wap'].mean() if positions_df is not None else 0
            }
        }

        # If we have P&L data, add it to the context
        if st.session_state.pnl_df is not None:
            data_context["pnl"] = st.session_state.pnl_df.to_dict('records')
            data_context["pnl_stats"] = {
                "total_fifo_pnl": st.session_state.pnl_df['fifo_pnl'].sum(),
                "total_lifo_pnl": st.session_state.pnl_df['lifo_pnl'].sum()
            }

        # Update session state with the comprehensive data context
        st.session_state.data_context = data_context

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your trade data..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Check if it's a simple question about the data that we can answer directly
                    quick_response = generate_quick_data_response(prompt, st.session_state.data_context)
                    if quick_response:
                        response = quick_response
                    elif use_advanced_ai:
                        try:
                            # Use Gemini AI for advanced response
                            response = generate_ai_response(prompt, st.session_state.data_context)
                        except Exception as e:
                            st.error(f"Error generating AI response: {str(e)}")
                            # Fall back to simple response
                            response = generate_simple_response(prompt, positions_df)
                    else:
                        # Use simple response generation
                        response = generate_simple_response(prompt, positions_df)

                    st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("Upload a file and calculate positions to use the AI Assistant.")

# Add sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application calculates FIFO and LIFO weighted average prices for open positions based on trade data.

    Features:
    1. Parse trade data files (Excel or CSV)
    2. Net buy/sell trades per contract
    3. Compute FIFO and LIFO weighted average prices
    4. Fetch current market prices (manual entry)
    5. Generate visualizations
    6. Provide AI-powered insights and answers
    """)

    # Add information about the AI Assistant
    st.header("AI Assistant")
    st.markdown("""
    The AI Assistant tab uses Google's Gemini AI to analyze your trade data and answer questions.

    Example questions you can ask:
    - "What are my largest positions?"
    - "Explain the difference between FIFO and LIFO"
    - "How many trades do I have for GOLD?"
    - "What's my average purchase price for SILVER?"
    - "Which position has the biggest unrealized P&L?"
    - "What's the total value of my portfolio?"
    - "When do my positions expire?"

    For the AI Assistant to work, you need to configure a Google API key.
    """)

    st.header("Instructions")
    st.markdown("""
    1. Upload a trade data file (Excel or CSV)
    2. Click "Calculate Positions" to process the data
    3. Optionally, click "Fetch Current Prices" to get market prices and P&L
    4. Use the AI Assistant to ask questions about your data

    The file should contain columns for:
    - Date (in any standard date format)
    - Commodity/Contract
    - Buy/Sell quantities (numeric values)
    - Buy/Sell prices (numeric values)
    - Exchange
    - Expiry (in any standard date format)
    - Client Code (can be alphanumeric)
    - Strategy
    - Code
    - Remarks (optional)
    - Tagging (optional)

    Note: The application now supports alphanumeric client codes and provides better error handling for various data formats.
    """)

    # Add a link to the example data
    st.download_button(
        label="Download Example Data",
        data=open("example_data/sample_trades.csv", "rb").read(),
        file_name="sample_trades.csv",
        mime="text/csv"
    )
