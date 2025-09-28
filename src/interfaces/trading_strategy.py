"""
Trading Strategy Interface
Abstract base class for trading strategies in the system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import datetime as dt

from .signal_generator import TradingSignal, SignalType


class OrderType(Enum):
    """Enumeration of order types"""
    MARKET = "market"
    LIMIT = "limit" 
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Enumeration of order sides"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """
    Standardized order structure
    """
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None  # For limit/stop orders
    stop_price: Optional[float] = None  # For stop orders
    timestamp: Optional[pd.Timestamp] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary format"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'price': self.price,
            'stop_price': self.stop_price,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata or {}
        }


@dataclass
class Position:
    """
    Current position information
    """
    symbol: str
    quantity: int
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: pd.Timestamp
    
    @property
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat"""
        return self.quantity == 0


@dataclass
class StrategyPerformance:
    """
    Strategy performance metrics
    """
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert performance to dictionary format"""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss
        }


class TradingStrategy(ABC):
    """
    Abstract base class for trading strategies
    
    Implements Strategy Pattern for different trading approaches
    Follows Single Responsibility Principle for strategy logic
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize trading strategy
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        self.name = name
        self.config = config or {}
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.performance: Optional[StrategyPerformance] = None
    
    @abstractmethod
    def generate_orders(self, signals: List[TradingSignal], 
                       market_data: pd.DataFrame,
                       positions: Dict[str, Position]) -> List[Order]:
        """
        Generate orders based on trading signals
        
        Args:
            signals: List of trading signals
            market_data: Current market data
            positions: Current positions
            
        Returns:
            List of orders to execute
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal, 
                               current_price: float,
                               account_value: float,
                               current_position: Optional[Position] = None) -> int:
        """
        Calculate position size for a trade
        
        Args:
            signal: Trading signal
            current_price: Current market price
            account_value: Total account value
            current_position: Existing position (if any)
            
        Returns:
            Position size in shares
        """
        pass
    
    @abstractmethod
    def validate_order(self, order: Order, 
                      market_data: pd.DataFrame,
                      positions: Dict[str, Position],
                      account_value: float) -> bool:
        """
        Validate an order before execution
        
        Args:
            order: Order to validate
            market_data: Current market data
            positions: Current positions
            account_value: Account value
            
        Returns:
            True if order is valid, False otherwise
        """
        pass
    
    def get_name(self) -> str:
        """Get strategy name"""
        return self.name
    
    def get_config(self) -> Dict[str, Any]:
        """Get strategy configuration"""
        return self.config.copy()
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update strategy configuration"""
        self.config.update(config)
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        return self.positions.copy()
    
    def get_performance(self) -> Optional[StrategyPerformance]:
        """Get strategy performance metrics"""
        return self.performance