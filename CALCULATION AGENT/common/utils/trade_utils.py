from collections import deque
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from common.models.trade import Trade, Position, Lot


def parse_excel_file(file_bytes: bytes) -> List[Trade]:
    """Parse an Excel file and convert it to a list of Trade objects."""
    df = pd.read_excel(file_bytes)
    trades = []
    
    # Process each row
    for _, row in df.iterrows():
        # Check if it's a buy or sell trade
        if pd.notna(row['Buy']):
            side = 'Buy'
            quantity = row['Buy']
            price = row['Buy Average']
        elif pd.notna(row['Sell']):
            side = 'Sell'
            quantity = row['Sell']
            price = row['Sell Average']
        else:
            # Skip rows without buy or sell data
            continue
        
        trade = Trade(
            date=row['Date'],
            contract=row['Commodity'],
            side=side,
            quantity=quantity,
            price=price,
            expiry=row['Expiry'],
            exchange=row['Exchange'],
            client_code=row['Client Code'],
            strategy=row['Strategy'],
            code=row['Code'],
            remarks=row['Remarks'] if pd.notna(row['Remarks']) else None,
            tagging=row['Tagging'] if pd.notna(row['Tagging']) else None
        )
        trades.append(trade)
    
    return trades


def parse_csv_file(file_bytes: bytes) -> List[Trade]:
    """Parse a CSV file and convert it to a list of Trade objects."""
    df = pd.read_csv(file_bytes)
    return parse_dataframe(df)


def parse_dataframe(df: pd.DataFrame) -> List[Trade]:
    """Parse a pandas DataFrame and convert it to a list of Trade objects."""
    trades = []
    
    # Process each row
    for _, row in df.iterrows():
        # Check if it's a buy or sell trade
        if pd.notna(row['Buy']):
            side = 'Buy'
            quantity = row['Buy']
            price = row['Buy Average']
        elif pd.notna(row['Sell']):
            side = 'Sell'
            quantity = row['Sell']
            price = row['Sell Average']
        else:
            # Skip rows without buy or sell data
            continue
        
        trade = Trade(
            date=row['Date'],
            contract=row['Commodity'],
            side=side,
            quantity=quantity,
            price=price,
            expiry=row['Expiry'],
            exchange=row['Exchange'],
            client_code=row['Client Code'],
            strategy=row['Strategy'],
            code=row['Code'],
            remarks=row['Remarks'] if pd.notna(row['Remarks']) else None,
            tagging=row['Tagging'] if pd.notna(row['Tagging']) else None
        )
        trades.append(trade)
    
    return trades


def calculate_weighted_average(lots: List[Lot]) -> float:
    """Calculate the weighted average price of a list of lots."""
    if not lots:
        return 0.0
    
    total_quantity = sum(lot.quantity for lot in lots)
    if total_quantity == 0:
        return 0.0
    
    weighted_sum = sum(lot.quantity * lot.price for lot in lots)
    return weighted_sum / total_quantity
