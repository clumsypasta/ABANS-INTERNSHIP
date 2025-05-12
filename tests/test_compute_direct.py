"""
Test script to directly implement the position calculation logic.
"""

import pandas as pd
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import os

# Define simple classes for the demo
class Lot:
    """Represents a lot of a specific contract with quantity and price."""
    def __init__(self, quantity: float, price: float):
        self.quantity = quantity
        self.price = price

def calculate_weighted_average(lots):
    """Calculate the weighted average price of a list of lots."""
    if not lots:
        return 0.0

    total_quantity = sum(lot.quantity for lot in lots)
    if total_quantity == 0:
        return 0.0

    weighted_sum = sum(lot.quantity * lot.price for lot in lots)
    return weighted_sum / total_quantity

def compute_positions(trades_data):
    """
    Compute positions from a list of trades using FIFO and LIFO methods.
    Only considers active contracts (contracts that haven't expired yet).

    Args:
        trades_data: List of trade dictionaries

    Returns:
        Dictionary containing computed positions
    """
    try:
        # Get current date for checking contract expiry
        current_date = datetime.now()
        print(f"Current date for filtering: {current_date.strftime('%Y-%m-%d')}")

        # Group trades by contract and other identifiers
        grouped_trades = defaultdict(list)
        for trade in trades_data:
            # Convert string dates to datetime objects if needed
            if isinstance(trade['date'], str):
                trade['date'] = datetime.fromisoformat(trade['date'].replace('Z', '+00:00'))
            if isinstance(trade['expiry'], str):
                trade['expiry'] = datetime.fromisoformat(trade['expiry'].replace('Z', '+00:00'))

            # Create a unique key for each contract group
            key = (
                trade['contract'],
                trade['exchange'],
                trade['expiry'].isoformat() if isinstance(trade['expiry'], datetime) else trade['expiry'],
                trade['client_code'],
                trade['strategy'],
                trade['code'],
                trade.get('tagging')
            )
            grouped_trades[key].append(trade)

        # Process each group of trades
        positions = []
        for key, contract_trades in grouped_trades.items():
            contract, exchange, expiry_str, client_code, strategy, code, tagging = key

            # Convert expiry back to datetime if needed
            expiry = datetime.fromisoformat(expiry_str.replace('Z', '+00:00')) if isinstance(expiry_str, str) else expiry_str

            # Skip expired contracts - only process active contracts
            if expiry <= current_date:
                print(f"Skipping expired contract: {contract} (Expiry: {expiry.strftime('%Y-%m-%d')})")
                continue
            else:
                print(f"Processing active contract: {contract} (Expiry: {expiry.strftime('%Y-%m-%d')})")

            # Sort trades by date
            contract_trades.sort(key=lambda x: x['date'])

            # Initialize deques for FIFO and LIFO
            fifo_deque = deque()
            lifo_deque = deque()

            # Process trades in chronological order
            for trade in contract_trades:
                if trade['side'] == 'Buy':
                    # Add buy trade to both deques
                    lot = Lot(quantity=trade['quantity'], price=trade['price'])
                    fifo_deque.append(lot)
                    lifo_deque.append(lot)
                elif trade['side'] == 'Sell':
                    # Process sell trade using FIFO
                    remaining_sell_qty = trade['quantity']
                    while remaining_sell_qty > 0 and fifo_deque:
                        lot = fifo_deque[0]
                        if lot.quantity <= remaining_sell_qty:
                            # Use entire lot
                            remaining_sell_qty -= lot.quantity
                            fifo_deque.popleft()
                        else:
                            # Use partial lot
                            lot.quantity -= remaining_sell_qty
                            remaining_sell_qty = 0

                    # Process sell trade using LIFO
                    remaining_sell_qty = trade['quantity']
                    while remaining_sell_qty > 0 and lifo_deque:
                        lot = lifo_deque[-1]
                        if lot.quantity <= remaining_sell_qty:
                            # Use entire lot
                            remaining_sell_qty -= lot.quantity
                            lifo_deque.pop()
                        else:
                            # Use partial lot
                            lot.quantity -= remaining_sell_qty
                            remaining_sell_qty = 0

            # Calculate open quantity and weighted average prices
            fifo_open_qty = sum(lot.quantity for lot in fifo_deque)
            lifo_open_qty = sum(lot.quantity for lot in lifo_deque)

            # Skip if no open position
            if fifo_open_qty < 0.001:
                continue

            # Calculate weighted average prices
            fifo_wap = calculate_weighted_average(list(fifo_deque))
            lifo_wap = calculate_weighted_average(list(lifo_deque))

            # Create position
            position = {
                'contract': contract,
                'open_qty': fifo_open_qty,
                'fifo_wap': fifo_wap,
                'lifo_wap': lifo_wap,
                'exchange': exchange,
                'expiry': expiry.isoformat() if isinstance(expiry, datetime) else expiry,
                'client_code': client_code,
                'strategy': strategy,
                'code': code,
                'tagging': tagging
            }
            positions.append(position)

        return {
            "status": "success",
            "positions": positions,
            "count": len(positions)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error_message": f"Error computing positions: {str(e)}"
        }

def convert_excel_to_trades(excel_path):
    """Convert Excel data to trade objects."""
    # Read the Excel file
    df = pd.read_excel(excel_path)

    # Convert to list of trade dictionaries
    trades = []

    for _, row in df.iterrows():
        # Skip rows without buy or sell data
        if pd.isna(row['Buy']) and pd.isna(row['Sell']):
            continue

        # Determine if it's a buy or sell trade
        if pd.notna(row['Buy']):
            side = 'Buy'
            quantity = row['Buy']
            price = row['Buy Average']
        else:
            side = 'Sell'
            quantity = row['Sell']
            price = row['Sell Average']

        # Create trade dictionary
        trade = {
            'date': row['Date'],
            'contract': row['Commodity'],
            'side': side,
            'quantity': float(quantity),
            'price': float(price),
            'expiry': row['Expiry'],
            'exchange': row['Exchange'],
            'client_code': int(row['Client Code']),
            'strategy': row['Strategy'],
            'code': row['Code'],
            'remarks': row['Remarks'] if pd.notna(row['Remarks']) else None,
            'tagging': row['Tagging'] if pd.notna(row['Tagging']) else None
        }

        trades.append(trade)

    return trades

def main():
    """Process the sample data through the compute agent."""
    # Path to the sample data
    excel_path = 'CALCULATION AGENT/DATA/Sample Data.xlsx'

    # Convert Excel data to trades
    trades = convert_excel_to_trades(excel_path)

    # Print some info about the trades
    print(f"Loaded {len(trades)} trades from {excel_path}")
    print("\nSample of trades:")
    for i, trade in enumerate(trades[:5], 1):
        print(f"Trade {i}: {trade['contract']} - {trade['side']} {trade['quantity']} @ {trade['price']} - Expiry: {trade['expiry'].strftime('%Y-%m-%d')}")

    # Print current date for reference
    current_date = datetime.now()
    print(f"\nCurrent date: {current_date.strftime('%Y-%m-%d')}")

    # Check which trades should be filtered based on expiry
    print("\nExpiry status of trades:")
    for contract in set(trade['contract'] for trade in trades):
        expiry = next(trade['expiry'] for trade in trades if trade['contract'] == contract)
        status = "ACTIVE" if expiry > current_date else "EXPIRED"
        print(f"{contract}: Expiry {expiry.strftime('%Y-%m-%d')} - Status: {status}")

    # Process trades through compute_positions
    print("\nProcessing trades through compute_positions...")
    result = compute_positions(trades)

    # Print the result
    if result["status"] == "success":
        positions = result["positions"]
        print(f"\nComputed {len(positions)} positions:")

        for i, position in enumerate(positions, 1):
            expiry = datetime.fromisoformat(position['expiry'].replace('Z', '+00:00')) if isinstance(position['expiry'], str) else position['expiry']
            print(f"Position {i}: {position['contract']} - Qty: {position['open_qty']} - FIFO WAP: {position['fifo_wap']} - LIFO WAP: {position['lifo_wap']} - Expiry: {expiry.strftime('%Y-%m-%d')}")

        # Save the result to a CSV file in the tests/outputs directory
        output_dir = 'tests/outputs'
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}_test_export.csv")

        # Convert to DataFrame and save
        positions_df = pd.DataFrame(positions)
        positions_df.to_csv(output_file)

        print(f"\nSaved positions to {output_file}")
    else:
        print(f"Error: {result['error_message']}")

if __name__ == "__main__":
    main()
