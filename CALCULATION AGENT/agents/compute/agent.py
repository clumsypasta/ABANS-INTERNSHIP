from collections import defaultdict, deque
from typing import Dict, List, Any, Tuple
from datetime import datetime
import json

from google.adk.agents import Agent
from google.adk.tools.function_tool import FunctionTool

from common.models.trade import Trade, Position, Lot
from common.utils.trade_utils import calculate_weighted_average


def compute_positions(trades_json: str) -> Dict[str, Any]:
    """
    Compute positions from a list of trades using FIFO and LIFO methods.
    Only considers active contracts (contracts that haven't expired yet).

    Args:
        trades_json: JSON string containing a list of trades

    Returns:
        Dictionary containing computed positions
    """
    try:
        # Parse trades from JSON
        trades_data = json.loads(trades_json)

        # Get current date for checking contract expiry
        current_date = datetime.now()

        # Convert to Trade objects
        trades = []
        for trade_dict in trades_data:
            # Convert string dates to datetime objects
            if isinstance(trade_dict['date'], str):
                trade_dict['date'] = datetime.fromisoformat(trade_dict['date'].replace('Z', '+00:00'))
            if isinstance(trade_dict['expiry'], str):
                trade_dict['expiry'] = datetime.fromisoformat(trade_dict['expiry'].replace('Z', '+00:00'))

            trade = Trade.from_dict(trade_dict)
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

            # Convert expiry back to datetime
            expiry = datetime.fromisoformat(expiry_str.replace('Z', '+00:00')) if isinstance(expiry_str, str) else expiry_str

            # Skip expired contracts - only process active contracts
            if expiry <= current_date:
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

            # Ensure both methods result in the same open quantity
            if abs(fifo_open_qty - lifo_open_qty) > 0.001:
                return {
                    "status": "error",
                    "error_message": f"FIFO and LIFO open quantities don't match for {contract}: {fifo_open_qty} vs {lifo_open_qty}"
                }

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


# Create the Compute Agent
compute_agent = Agent(
    model="gemini-2.0-flash",
    name="compute_agent",
    description="Computes FIFO and LIFO weighted average prices for active open positions",
    instruction="""
    You are a Compute Agent that processes trade data.
    Your job is to net buy/sell trades per contract and compute FIFO and LIFO weighted average prices for open positions.
    Only consider active contracts (contracts that haven't expired yet) when calculating positions.
    Use the 'compute_positions' tool to process the trades and return the computed positions.
    """,
    tools=[compute_positions]
)
