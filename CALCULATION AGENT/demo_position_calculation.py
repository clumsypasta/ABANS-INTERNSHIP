"""
Demo script to demonstrate the position calculation logic.
This script shows how the compute_positions function filters out expired contracts.
"""

import json
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the compute_positions function
from agents.compute.agent import compute_positions
from common.models.trade import Trade

def create_sample_trades():
    """Create sample trade data with both active and expired contracts."""
    # Current date for reference
    current_date = datetime.now()
    
    # Create dates for trades and expiries
    trade_date = current_date - timedelta(days=30)  # 30 days ago
    
    # Create expiry dates - some in the past, some in the future
    expired_expiry = current_date - timedelta(days=5)  # 5 days ago (expired)
    active_expiry = current_date + timedelta(days=60)  # 60 days in future (active)
    
    # Create sample trades
    trades = [
        # Active contract - GOLD
        {
            "date": trade_date.isoformat(),
            "contract": "GOLD",
            "side": "Buy",
            "quantity": 100.0,
            "price": 1850.25,
            "expiry": active_expiry.isoformat(),
            "exchange": "MCX",
            "client_code": 1,
            "strategy": "RATIO",
            "code": "RATIO",
            "tagging": "GC/SI"
        },
        # Expired contract - SILVER
        {
            "date": trade_date.isoformat(),
            "contract": "SILVER",
            "side": "Buy",
            "quantity": 500.0,
            "price": 24.35,
            "expiry": expired_expiry.isoformat(),
            "exchange": "MCX",
            "client_code": 1,
            "strategy": "RATIO",
            "code": "RATIO",
            "tagging": "GC/SI"
        },
        # Another active contract - COPPER
        {
            "date": trade_date.isoformat(),
            "contract": "COPPER",
            "side": "Buy",
            "quantity": 200.0,
            "price": 4.25,
            "expiry": active_expiry.isoformat(),
            "exchange": "MCX",
            "client_code": 1,
            "strategy": "RATIO",
            "code": "RATIO",
            "tagging": "CU"
        }
    ]
    
    return trades

def run_demo():
    """Run the demo to show position calculation with active contracts only."""
    print("=" * 80)
    print("POSITION CALCULATION DEMO")
    print("=" * 80)
    print("\nThis demo shows how the position calculation logic filters out expired contracts.")
    
    # Create sample trades
    trades = create_sample_trades()
    
    # Display all trades
    print("\n1. ALL TRADES (INCLUDING EXPIRED CONTRACTS):")
    print("-" * 80)
    for i, trade in enumerate(trades, 1):
        expiry_date = datetime.fromisoformat(trade['expiry'])
        is_active = expiry_date > datetime.now()
        status = "ACTIVE" if is_active else "EXPIRED"
        print(f"Trade {i}: {trade['contract']} - {trade['quantity']} @ {trade['price']} - Expiry: {expiry_date.strftime('%Y-%m-%d')} - Status: {status}")
    
    # Convert trades to JSON
    trades_json = json.dumps(trades)
    
    # Call compute_positions function
    result = compute_positions(trades_json)
    
    # Display result
    print("\n2. COMPUTED POSITIONS (ONLY ACTIVE CONTRACTS):")
    print("-" * 80)
    
    if result["status"] == "success":
        positions = result["positions"]
        print(f"Found {len(positions)} active positions:")
        
        for i, position in enumerate(positions, 1):
            expiry_date = datetime.fromisoformat(position['expiry'].replace('Z', '+00:00'))
            print(f"Position {i}: {position['contract']} - Qty: {position['open_qty']} - FIFO WAP: {position['fifo_wap']} - LIFO WAP: {position['lifo_wap']} - Expiry: {expiry_date.strftime('%Y-%m-%d')}")
        
        # Check which contracts were filtered out
        position_contracts = [p['contract'] for p in positions]
        all_contracts = [t['contract'] for t in trades]
        filtered_contracts = [c for c in all_contracts if c not in position_contracts]
        
        if filtered_contracts:
            print("\nThe following contracts were filtered out because they are expired:")
            for contract in filtered_contracts:
                print(f"- {contract}")
    else:
        print(f"Error: {result['error_message']}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_demo()
