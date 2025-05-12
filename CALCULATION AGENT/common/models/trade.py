from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, Union


@dataclass
class Trade:
    """Represents a single trade with buy or sell information."""
    date: datetime
    contract: str
    side: str  # "Buy" or "Sell"
    quantity: float
    price: float
    expiry: datetime
    exchange: str
    client_code: int
    strategy: str
    code: str
    remarks: Optional[str] = None
    tagging: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create a Trade object from a dictionary."""
        return cls(
            date=data['date'],
            contract=data['contract'],
            side=data['side'],
            quantity=data['quantity'],
            price=data['price'],
            expiry=data['expiry'],
            exchange=data['exchange'],
            client_code=data['client_code'],
            strategy=data['strategy'],
            code=data['code'],
            remarks=data.get('remarks'),
            tagging=data.get('tagging')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Trade object to a dictionary."""
        return {
            'date': self.date.isoformat() if isinstance(self.date, datetime) else self.date,
            'contract': self.contract,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'expiry': self.expiry.isoformat() if isinstance(self.expiry, datetime) else self.expiry,
            'exchange': self.exchange,
            'client_code': self.client_code,
            'strategy': self.strategy,
            'code': self.code,
            'remarks': self.remarks,
            'tagging': self.tagging
        }


@dataclass
class Position:
    """Represents a position with FIFO and LIFO weighted average prices."""
    contract: str
    open_qty: float
    fifo_wap: float
    lifo_wap: float
    exchange: str
    expiry: datetime
    client_code: int
    strategy: str
    code: str
    tagging: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
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


@dataclass
class Lot:
    """Represents a lot of a specific contract with quantity and price."""
    quantity: float
    price: float
