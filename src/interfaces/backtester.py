"""
Backtester Interface
Abstract base class for backtesting engines in the trading system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
import pandas as pd
import datetime as dt
from dataclasses import dataclass

from .trading_strategy import TradingStrategy, Order, Position, StrategyPerformance


@dataclass
class BacktestConfig:
    """
    Backtesting configuration
    """
    start_date: dt.datetime
    end_date: dt.datetime
    initial_capital: float
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    benchmark: str = 'SPY'  # Benchmark symbol
    risk_free_rate: float = 0.02  # Annual risk-free rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'commission': self.commission,
            'slippage': self.slippage,
            'market_impact': self.market_impact,
            'benchmark': self.benchmark,
            'risk_free_rate': self.risk_free_rate
        }


@dataclass
class Trade:
    """
    Completed trade record
    """
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    return_pct: float
    duration_days: int
    side: str  # 'long' or 'short'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary"""
        return {
            'symbol': self.symbol,
            'entry_date': self.entry_date.isoformat(),
            'exit_date': self.exit_date.isoformat(),
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'pnl': self.pnl,
            'return_pct': self.return_pct,
            'duration_days': self.duration_days,
            'side': self.side
        }


@dataclass
class BacktestResults:
    """
    Complete backtesting results
    """
    strategy_name: str
    config: BacktestConfig
    performance: StrategyPerformance
    trades: List[Trade]
    portfolio_values: pd.Series
    positions_history: pd.DataFrame
    orders_history: List[Order]
    benchmark_performance: Optional[StrategyPerformance] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary"""
        return {
            'strategy_name': self.strategy_name,
            'config': self.config.to_dict(),
            'performance': self.performance.to_dict(),
            'trades': [trade.to_dict() for trade in self.trades],
            'benchmark_performance': self.benchmark_performance.to_dict() if self.benchmark_performance else None
        }


class Backtester(ABC):
    """
    Abstract base class for backtesting engines
    
    Implements Strategy Pattern for different backtesting approaches
    Follows Single Responsibility Principle for backtesting logic
    """
    
    def __init__(self, name: str, config: BacktestConfig):
        """
        Initialize backtester
        
        Args:
            name: Backtester name
            config: Backtesting configuration
        """
        self.name = name
        self.config = config
        self.results: Optional[BacktestResults] = None
    
    @abstractmethod
    def run_backtest(self, strategy: TradingStrategy, 
                    market_data: Dict[str, pd.DataFrame]) -> BacktestResults:
        """
        Run backtest for a trading strategy
        
        Args:
            strategy: Trading strategy to backtest
            market_data: Historical market data for all symbols
            
        Returns:
            Backtesting results
        """
        pass
    
    @abstractmethod
    def simulate_order_execution(self, order: Order, 
                               market_data: pd.DataFrame,
                               timestamp: pd.Timestamp) -> Dict[str, Any]:
        """
        Simulate order execution with realistic constraints
        
        Args:
            order: Order to execute
            market_data: Market data at execution time
            timestamp: Execution timestamp
            
        Returns:
            Execution details (fill_price, fill_quantity, etc.)
        """
        pass
    
    @abstractmethod
    def calculate_portfolio_value(self, positions: Dict[str, Position],
                                cash: float,
                                market_data: Dict[str, pd.DataFrame],
                                timestamp: pd.Timestamp) -> float:
        """
        Calculate total portfolio value
        
        Args:
            positions: Current positions
            cash: Available cash
            market_data: Current market data
            timestamp: Calculation timestamp
            
        Returns:
            Total portfolio value
        """
        pass
    
    @abstractmethod
    def calculate_performance_metrics(self, portfolio_values: pd.Series,
                                    benchmark_data: Optional[pd.Series] = None) -> StrategyPerformance:
        """
        Calculate performance metrics from portfolio values
        
        Args:
            portfolio_values: Time series of portfolio values
            benchmark_data: Benchmark price data (optional)
            
        Returns:
            Performance metrics
        """
        pass
    
    def get_results(self) -> Optional[BacktestResults]:
        """Get the latest backtest results"""
        return self.results
    
    def get_name(self) -> str:
        """Get backtester name"""
        return self.name
    
    def get_config(self) -> BacktestConfig:
        """Get backtesting configuration"""
        return self.config