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
                    # Generate response based on actual data
                    positions_df = st.session_state.positions_df

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
                    if "largest" in prompt.lower() or "biggest" in prompt.lower():
                        if num_positions > 0:
                            response = f"Based on your data, your largest position is in **{largest_contract}** with an open quantity of **{largest_qty}**."
                        else:
                            response = "You currently have no open positions in your data."

                    elif any(contract.lower() in prompt.lower() for contract in contracts_list):
                        # Find which contract they're asking about
                        for contract in contracts_list:
                            if contract.lower() in prompt.lower():
                                contract_data = positions_df[positions_df['contract'] == contract].iloc[0]
                                response = f"""
                                You have a position in **{contract}** with:
                                - Open quantity: **{contract_data['open_qty']}**
                                - FIFO weighted average price: **${contract_data['fifo_wap']}**
                                - LIFO weighted average price: **${contract_data['lifo_wap']}**
                                - Expiry date: **{contract_data['expiry']}**
                                """
                                break

                    elif "fifo" in prompt.lower() or "lifo" in prompt.lower():
                        if num_positions > 0:
                            # Create a table of FIFO vs LIFO prices
                            table_rows = []
                            for _, row in positions_df.iterrows():
                                table_rows.append(f"| {row['contract']} | ${row['fifo_wap']} | ${row['lifo_wap']} |")

                            table = "\n".join(table_rows)

                            response = f"""
                            **FIFO** (First-In-First-Out) and **LIFO** (Last-In-First-Out) are methods for calculating the weighted average price of your positions:

                            - **FIFO**: Assumes that the oldest trades are closed first
                            - **LIFO**: Assumes that the newest trades are closed first

                            Your positions show different weighted average prices using these methods:

                            | Contract | FIFO WAP | LIFO WAP |
                            |----------|----------|----------|
                            {table}
                            """
                        else:
                            response = "You currently have no open positions to calculate FIFO or LIFO prices."

                    else:
                        if num_positions > 0:
                            positions_summary = []
                            for _, row in positions_df.iterrows():
                                positions_summary.append(f"**{row['contract']}** ({row['open_qty']} units)")

                            positions_text = ", ".join(positions_summary)

                            response = f"""
                            I've analyzed your trade data. You have {num_positions} open positions: {positions_text}.

                            What specific information would you like to know about your positions?

                            You can ask about:
                            - Details about a specific contract
                            - Your largest position
                            - FIFO vs LIFO pricing
                            - Expiry dates
                            """
                        else:
                            response = "You currently have no open positions in your data. Please upload trade data and calculate positions first."

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
