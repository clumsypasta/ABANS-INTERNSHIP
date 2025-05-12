"""
Standalone demo script to demonstrate the position calculation logic.
This script shows how the position calculation filters out expired contracts.
"""

import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Any, Tuple

# Define simple classes for the demo
class Lot:
    """Represents a lot of a specific contract with quantity and price."""
    def __init__(self, quantity: float, price: float):
        self.quantity = quantity
        self.price = price

class Trade:
    """Represents a single trade with buy or sell information."""
    def __init__(self, date, contract, side, quantity, price, expiry, exchange, client_code, strategy, code, tagging=None):
        self.date = date
        self.contract = contract
        self.side = side
        self.quantity = quantity
        self.price = price
        self.expiry = expiry
        self.exchange = exchange
        self.client_code = client_code
        self.strategy = strategy
        self.code = code
        self.tagging = tagging

class Position:
    """Represents a position with FIFO and LIFO weighted average prices."""
    def __init__(self, contract, open_qty, fifo_wap, lifo_wap, exchange, expiry, client_code, strategy, code, tagging=None):
        self.contract = contract
        self.open_qty = open_qty
        self.fifo_wap = fifo_wap
        self.lifo_wap = lifo_wap
        self.exchange = exchange
        self.expiry = expiry
        self.client_code = client_code
        self.strategy = strategy
        self.code = code
        self.tagging = tagging
    
    def to_dict(self):
        """Convert Position object to a dictionary."""
        return {
            'contract': self.contract,
            'open_qty': self.open_qty,
            'fifo_wap': self.fifo_wap,
            'lifo_wap': self.lifo_wap,
            'exchange': self.exchange,
            'expiry': self.expiry.isoformat() if isinstance(self.expiry, datetime) else self.expiry,
            'client_code': self.client_code,
            'strategy': self.strategy,
            'code': self.code,
            'tagging': self.tagging
        }

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
        
        # Convert to Trade objects
        trades = []
        for trade_dict in trades_data:
            # Convert string dates to datetime objects if needed
            if isinstance(trade_dict['date'], str):
                trade_dict['date'] = datetime.fromisoformat(trade_dict['date'].replace('Z', '+00:00'))
            if isinstance(trade_dict['expiry'], str):
                trade_dict['expiry'] = datetime.fromisoformat(trade_dict['expiry'].replace('Z', '+00:00'))
            
            trade = Trade(
                date=trade_dict['date'],
                contract=trade_dict['contract'],
                side=trade_dict['side'],
                quantity=trade_dict['quantity'],
                price=trade_dict['price'],
                expiry=trade_dict['expiry'],
                exchange=trade_dict['exchange'],
                client_code=trade_dict['client_code'],
                strategy=trade_dict['strategy'],
                code=trade_dict['code'],
                tagging=trade_dict.get('tagging')
            )
            trades.append(trade)
        
        # Sort trades by date
        trades.sort(key=lambda x: x.date)
        
        # Group trades by contract and other identifiers
        grouped_trades = defaultdict(list)
        for trade in trades:
            # Create a unique key for each contract group
            key = (
                trade.contract, 
                trade.exchange, 
                trade.expiry.isoformat() if isinstance(trade.expiry, datetime) else trade.expiry,
                trade.client_code,
                trade.strategy,
                trade.code,
                trade.tagging
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
            
            # Initialize deques for FIFO and LIFO
            fifo_deque = deque()
            lifo_deque = deque()
            
            # Process trades in chronological order
            for trade in contract_trades:
                if trade.side == 'Buy':
                    # Add buy trade to both deques
                    lot = Lot(quantity=trade.quantity, price=trade.price)
                    fifo_deque.append(lot)
                    lifo_deque.append(lot)
                elif trade.side == 'Sell':
                    # Process sell trade using FIFO
                    remaining_sell_qty = trade.quantity
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
                    remaining_sell_qty = trade.quantity
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
            position = Position(
                contract=contract,
                open_qty=fifo_open_qty,
                fifo_wap=fifo_wap,
                lifo_wap=lifo_wap,
                exchange=exchange,
                expiry=expiry,
                client_code=client_code,
                strategy=strategy,
                code=code,
                tagging=tagging
            )
            positions.append(position)
        
        # Convert positions to dictionaries
        position_dicts = [position.to_dict() for position in positions]
        
        return {
            "status": "success",
            "positions": position_dicts,
            "count": len(position_dicts)
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error computing positions: {str(e)}"
        }

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
            "date": trade_date,
            "contract": "GOLD",
            "side": "Buy",
            "quantity": 100.0,
            "price": 1850.25,
            "expiry": active_expiry,
            "exchange": "MCX",
            "client_code": 1,
            "strategy": "RATIO",
            "code": "RATIO",
            "tagging": "GC/SI"
        },
        # Expired contract - SILVER
        {
            "date": trade_date,
            "contract": "SILVER",
            "side": "Buy",
            "quantity": 500.0,
            "price": 24.35,
            "expiry": expired_expiry,
            "exchange": "MCX",
            "client_code": 1,
            "strategy": "RATIO",
            "code": "RATIO",
            "tagging": "GC/SI"
        },
        # Another active contract - COPPER
        {
            "date": trade_date,
            "contract": "COPPER",
            "side": "Buy",
            "quantity": 200.0,
            "price": 4.25,
            "expiry": active_expiry,
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
        expiry_date = trade['expiry']
        is_active = expiry_date > datetime.now()
        status = "ACTIVE" if is_active else "EXPIRED"
        print(f"Trade {i}: {trade['contract']} - {trade['quantity']} @ {trade['price']} - Expiry: {expiry_date.strftime('%Y-%m-%d')} - Status: {status}")
    
    # Call compute_positions function
    print("\n2. COMPUTING POSITIONS (ONLY ACTIVE CONTRACTS):")
    print("-" * 80)
    result = compute_positions(trades)
    
    # Display result
    if result["status"] == "success":
        positions = result["positions"]
        print(f"\nFound {len(positions)} active positions:")
        
        for i, position in enumerate(positions, 1):
            expiry_date = datetime.fromisoformat(position['expiry'].replace('Z', '+00:00')) if isinstance(position['expiry'], str) else position['expiry']
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
