"""
Real Trading Interface
Abstract base classes for real and paper trading implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from decimal import Decimal
import datetime as dt

from .trading_strategy import Order, Position, OrderType, OrderSide


class AccountType(Enum):
    """Account types"""
    PAPER = "paper"
    REAL = "real"
    DEMO = "demo"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Trade:
    """Executed trade information"""
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: pd.Timestamp
    order_id: str
    trade_id: str
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of the trade"""
        return self.quantity * self.price
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'order_id': self.order_id,
            'trade_id': self.trade_id,
            'commission': self.commission,
            'notional_value': self.notional_value,
            'metadata': self.metadata
        }


@dataclass
class OrderExecution:
    """Order execution result"""
    order_id: str
    status: OrderStatus
    filled_quantity: int = 0
    remaining_quantity: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    timestamp: Optional[pd.Timestamp] = None
    error_message: Optional[str] = None
    trades: List[Trade] = field(default_factory=list)
    
    @property
    def is_complete(self) -> bool:
        """Check if order is completely filled"""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_partial(self) -> bool:
        """Check if order is partially filled"""
        return self.status == OrderStatus.PARTIALLY_FILLED
    
    @property
    def total_commission(self) -> float:
        """Calculate total commission from all trades"""
        return sum(trade.commission for trade in self.trades)


@dataclass
class AccountBalance:
    """Account balance information"""
    cash: float
    buying_power: float
    portfolio_value: float
    day_trade_buying_power: float
    unsettled_funds: float
    timestamp: pd.Timestamp
    
    @property
    def total_equity(self) -> float:
        """Calculate total equity"""
        return self.cash + self.portfolio_value


@dataclass
class PortfolioSummary:
    """Portfolio summary information"""
    total_equity: float
    cash_balance: float
    positions_value: float
    day_pnl: float
    total_pnl: float
    positions: List[Position]
    timestamp: pd.Timestamp
    
    @property
    def position_count(self) -> int:
        """Number of positions"""
        return len([pos for pos in self.positions if not pos.is_flat])
    
    @property
    def long_positions(self) -> List[Position]:
        """Get long positions"""
        return [pos for pos in self.positions if pos.is_long]
    
    @property
    def short_positions(self) -> List[Position]:
        """Get short positions"""
        return [pos for pos in self.positions if pos.is_short]


class TradingAccountInterface(ABC):
    """
    Abstract interface for trading accounts (paper and real)
    """
    
    def __init__(self, account_type: AccountType):
        self.account_type = account_type
        self._is_connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to trading platform"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from trading platform"""
        pass
    
    @abstractmethod
    async def get_account_balance(self) -> AccountBalance:
        """Get current account balance"""
        pass
    
    @abstractmethod
    async def get_portfolio_summary(self) -> PortfolioSummary:
        """Get portfolio summary"""
        pass
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        pass
    
    @abstractmethod
    async def get_all_positions(self) -> List[Position]:
        """Get all positions"""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> OrderExecution:
        """Place trading order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderExecution:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_trade_history(
        self, 
        symbol: Optional[str] = None,
        start_date: Optional[dt.datetime] = None,
        end_date: Optional[dt.datetime] = None
    ) -> List[Trade]:
        """Get trade history"""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for symbol"""
        pass
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to trading platform"""
        return self._is_connected
    
    @property
    def is_paper_account(self) -> bool:
        """Check if this is a paper trading account"""
        return self.account_type == AccountType.PAPER


class TradingBrokerInterface(ABC):
    """
    Abstract interface for trading brokers (Robinhood, TD Ameritrade, etc.)
    """
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """Authenticate with broker"""
        pass
    
    @abstractmethod
    async def get_account(self, account_id: str) -> TradingAccountInterface:
        """Get trading account by ID"""
        pass
    
    @abstractmethod
    async def get_available_symbols(self) -> List[str]:
        """Get list of tradeable symbols"""
        pass
    
    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get symbol information"""
        pass


class RiskManager(ABC):
    """Abstract risk manager for trading operations"""
    
    @abstractmethod
    def validate_order(self, order: Order, account: TradingAccountInterface) -> Tuple[bool, str]:
        """Validate order against risk parameters"""
        pass
    
    @abstractmethod
    def check_position_size(self, symbol: str, quantity: int, account: TradingAccountInterface) -> bool:
        """Check if position size is within limits"""
        pass
    
    @abstractmethod
    def calculate_position_size(
        self, 
        symbol: str, 
        signal_strength: float, 
        account: TradingAccountInterface
    ) -> int:
        """Calculate appropriate position size"""
        pass