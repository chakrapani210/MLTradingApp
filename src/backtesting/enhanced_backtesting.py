"""
Enhanced Backtesting Implementation
Implements comprehensive backtesting with portfolio management, risk metrics, and performance analytics
"""

import datetime as dt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

from ..interfaces.backtester import Backtester, BacktestResult
from ..interfaces.trading_strategy import TradingStrategy, TradingSignal
from ..interfaces.risk_manager import RiskManager
from ..interfaces.data_provider import DataProvider

warnings.filterwarnings('ignore')


class OrderType(Enum):
    """Order types for backtesting"""
    MARKET = "market"
    LIMIT = "limit" 
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Trading order representation"""
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    executed: bool = False
    execution_price: Optional[float] = None
    execution_timestamp: Optional[pd.Timestamp] = None
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Portfolio position representation"""
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_price: float = 0.0


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Returns
    total_return: float
    annualized_return: float
    benchmark_return: float
    alpha: float
    
    # Risk Metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    value_at_risk: float
    
    # Trading Metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade_duration: float
    
    # Additional Metrics
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    beta: float


class Portfolio:
    """Portfolio management for backtesting"""
    
    def __init__(self, initial_cash: float = 100000, commission_rate: float = 0.001, 
                 slippage_rate: float = 0.0005):
        """
        Initialize Portfolio
        
        Args:
            initial_cash: Starting cash amount
            commission_rate: Commission rate per trade (e.g., 0.001 = 0.1%)
            slippage_rate: Slippage rate per trade
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.daily_returns: List[float] = []
        
        print(f"[PORTFOLIO] Initialized with ${initial_cash:,.0f}")
        print(f"            Commission Rate: {commission_rate:.3%}")
        print(f"            Slippage Rate: {slippage_rate:.3%}")
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.last_price = current_prices[symbol]
                position.market_value = position.quantity * current_prices[symbol]
                position.unrealized_pnl = position.market_value - (position.quantity * position.avg_cost)
                total_value += position.market_value
        
        return total_value
    
    def place_order(self, order: Order) -> bool:
        """Place a trading order"""
        self.orders.append(order)
        return True
    
    def execute_order(self, order: Order, execution_price: float, 
                     execution_timestamp: pd.Timestamp) -> bool:
        """Execute a trading order"""
        if order.executed:
            return False
        
        # Calculate commission and slippage
        trade_value = order.quantity * execution_price
        commission = trade_value * self.commission_rate
        slippage = trade_value * self.slippage_rate
        
        if order.side == OrderSide.BUY:
            total_cost = trade_value + commission + slippage
            
            if self.cash < total_cost:
                print(f"[ORDER] Insufficient funds for {order.symbol} order")
                return False
            
            # Execute buy order
            self.cash -= total_cost
            
            if order.symbol in self.positions:
                # Update existing position
                pos = self.positions[order.symbol]
                total_quantity = pos.quantity + order.quantity
                total_cost_basis = (pos.quantity * pos.avg_cost) + (order.quantity * execution_price)
                pos.avg_cost = total_cost_basis / total_quantity
                pos.quantity = total_quantity
            else:
                # Create new position
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_cost=execution_price
                )
            
        else:  # SELL
            if order.symbol not in self.positions or self.positions[order.symbol].quantity < order.quantity:
                print(f"[ORDER] Insufficient shares for {order.symbol} sell order")
                return False
            
            # Execute sell order
            total_proceeds = trade_value - commission - slippage
            self.cash += total_proceeds
            
            pos = self.positions[order.symbol]
            realized_pnl = (execution_price - pos.avg_cost) * order.quantity
            pos.realized_pnl += realized_pnl
            pos.quantity -= order.quantity
            
            # Remove position if quantity is zero
            if pos.quantity == 0:
                del self.positions[order.symbol]
        
        # Mark order as executed
        order.executed = True
        order.execution_price = execution_price
        order.execution_timestamp = execution_timestamp
        order.commission = commission
        order.slippage = slippage
        
        # Record trade
        trade = {
            'timestamp': execution_timestamp,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': execution_price,
            'value': trade_value,
            'commission': commission,
            'slippage': slippage,
            'pnl': realized_pnl if order.side == OrderSide.SELL else 0
        }
        self.trades.append(trade)
        
        print(f"[TRADE] {order.side.value.upper()} {order.quantity} {order.symbol} @ ${execution_price:.2f}")
        
        return True
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol"""
        return self.positions.get(symbol)
    
    def get_positions_summary(self) -> Dict[str, Any]:
        """Get summary of all positions"""
        return {
            'cash': self.cash,
            'positions': {symbol: {
                'quantity': pos.quantity,
                'avg_cost': pos.avg_cost,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl
            } for symbol, pos in self.positions.items()},
            'total_trades': len(self.trades)
        }


class EnhancedBacktester(Backtester):
    """
    Enhanced Backtesting Engine
    
    Features:
    - Comprehensive order management
    - Realistic execution modeling
    - Advanced performance metrics
    - Risk management integration
    - Multiple asset support
    - Custom commission and slippage models
    """
    
    def __init__(self, data_provider: DataProvider, initial_cash: float = 100000,
                 commission_rate: float = 0.001, slippage_rate: float = 0.0005,
                 risk_manager: Optional[RiskManager] = None):
        """
        Initialize Enhanced Backtester
        
        Args:
            data_provider: Data provider for historical data
            initial_cash: Starting portfolio value
            commission_rate: Commission rate per trade
            slippage_rate: Slippage rate per trade
            risk_manager: Optional risk manager
        """
        self.data_provider = data_provider
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.risk_manager = risk_manager
        
        # Initialize portfolio
        self.portfolio = Portfolio(initial_cash, commission_rate, slippage_rate)
        
        # Backtesting state
        self.current_timestamp = None
        self.current_prices = {}
        self.benchmark_data = None
        
        print(f"[BACKTEST] Enhanced Backtester initialized")
        print(f"           Initial Cash: ${initial_cash:,.0f}")
        print(f"           Risk Manager: {'Enabled' if risk_manager else 'Disabled'}")
    
    def run_backtest(self, strategy: TradingStrategy, symbols: List[str], 
                    start_date: dt.datetime, end_date: dt.datetime,
                    benchmark_symbol: str = 'SPY', rebalance_frequency: str = 'daily') -> BacktestResult:
        """
        Run comprehensive backtest
        
        Args:
            strategy: Trading strategy to backtest
            symbols: List of symbols to trade
            start_date: Backtest start date
            end_date: Backtest end date
            benchmark_symbol: Benchmark symbol for comparison
            rebalance_frequency: Rebalancing frequency ('daily', 'weekly', 'monthly')
            
        Returns:
            BacktestResult with comprehensive metrics and analysis
        """
        print(f"[BACKTEST] Running backtest for {len(symbols)} symbols")
        print(f"           Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"           Strategy: {type(strategy).__name__}")
        
        # Get historical data for all symbols + benchmark
        all_symbols = symbols + [benchmark_symbol]
        historical_data = self.data_provider.get_historical_data(
            symbols=all_symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        # Store benchmark data
        self.benchmark_data = historical_data.get(benchmark_symbol)
        
        # Get trading dates
        trading_dates = self._get_trading_dates(historical_data, rebalance_frequency)
        
        # Initialize tracking variables
        equity_curve = []
        signal_history = []
        trade_history = []
        
        # Run backtest day by day
        for current_date in trading_dates:
            self.current_timestamp = current_date
            
            # Update current prices
            self.current_prices = {}
            for symbol in symbols:
                if symbol in historical_data and current_date in historical_data[symbol].index:
                    price_idx = historical_data[symbol].index.get_loc(current_date)
                    self.current_prices[symbol] = historical_data[symbol].iloc[price_idx].values[0]
            
            # Generate signals for each symbol
            for symbol in symbols:
                if symbol in historical_data and current_date in historical_data[symbol].index:
                    # Get historical data up to current date
                    symbol_data = historical_data[symbol]
                    historical_slice = symbol_data[symbol_data.index <= current_date]
                    
                    if len(historical_slice) > 20:  # Ensure enough history
                        # Generate signal
                        signal = strategy.generate_signal(historical_slice, current_date)
                        
                        if signal.direction != 0:  # Non-hold signal
                            signal_history.append({
                                'date': current_date,
                                'symbol': symbol,
                                'signal': signal.direction,
                                'confidence': signal.confidence,
                                'signal_type': signal.signal_type
                            })
                            
                            # Execute trade
                            trade_result = self._execute_strategy_signal(symbol, signal, historical_slice)
                            if trade_result:
                                trade_history.append(trade_result)
            
            # Calculate portfolio value
            portfolio_value = self.portfolio.get_portfolio_value(self.current_prices)
            equity_curve.append((current_date, portfolio_value))
            
            # Calculate daily return
            if len(equity_curve) > 1:
                prev_value = equity_curve[-2][1]
                daily_return = (portfolio_value - prev_value) / prev_value
                self.portfolio.daily_returns.append(daily_return)
        
        # Store equity curve
        self.portfolio.equity_curve = equity_curve
        
        # Calculate comprehensive performance metrics
        performance_metrics = self._calculate_performance_metrics(
            equity_curve, self.benchmark_data, start_date, end_date
        )
        
        # Create backtest result
        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_value=self.initial_cash,
            final_value=equity_curve[-1][1] if equity_curve else self.initial_cash,
            total_return=performance_metrics.total_return,
            sharpe_ratio=performance_metrics.sharpe_ratio,
            max_drawdown=performance_metrics.max_drawdown,
            trades=self.portfolio.trades,
            performance_metrics=performance_metrics,
            additional_metrics={
                'equity_curve': equity_curve,
                'signal_history': signal_history,
                'trade_history': trade_history,
                'positions_summary': self.portfolio.get_positions_summary(),
                'symbols_traded': symbols,
                'benchmark_symbol': benchmark_symbol
            }
        )
        
        # Print summary
        self._print_backtest_summary(result)
        
        return result
    
    def _execute_strategy_signal(self, symbol: str, signal: TradingSignal, 
                               data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Execute trading signal through strategy"""
        try:
            # Get current position
            current_position = self.portfolio.get_position(symbol)
            current_quantity = current_position.quantity if current_position else 0
            
            # Determine order size (simplified - strategy would handle this)
            current_price = self.current_prices.get(symbol, 0)
            if current_price == 0:
                return None
            
            # Simple position sizing based on portfolio value
            portfolio_value = self.portfolio.get_portfolio_value(self.current_prices)
            target_allocation = 0.1 * signal.confidence  # Max 10% per position
            target_value = portfolio_value * target_allocation
            target_quantity = int(target_value / current_price)
            
            # Determine trade quantity
            if signal.direction > 0:  # BUY signal
                if current_quantity >= target_quantity:
                    return None  # Already at target
                trade_quantity = target_quantity - current_quantity
                side = OrderSide.BUY
            else:  # SELL signal
                if current_quantity <= 0:
                    return None  # No position to sell
                trade_quantity = current_quantity  # Sell entire position
                side = OrderSide.SELL
            
            if trade_quantity <= 0:
                return None
            
            # Apply risk management
            if self.risk_manager:
                risk_check = self.risk_manager.validate_trade(
                    symbol=symbol,
                    quantity=trade_quantity,
                    price=current_price,
                    portfolio_value=portfolio_value
                )
                if not risk_check.approved:
                    return {
                        'date': self.current_timestamp,
                        'symbol': symbol,
                        'action': 'REJECTED',
                        'reason': risk_check.rejection_reason,
                        'signal_type': signal.signal_type
                    }
            
            # Create and execute order
            order = Order(
                timestamp=self.current_timestamp,
                symbol=symbol,
                side=side,
                quantity=trade_quantity,
                order_type=OrderType.MARKET,
                price=current_price,
                metadata={
                    'signal_type': signal.signal_type,
                    'confidence': signal.confidence,
                    'signal_metadata': signal.metadata
                }
            )
            
            # Execute immediately (market order)
            if self.portfolio.execute_order(order, current_price, self.current_timestamp):
                return {
                    'date': self.current_timestamp,
                    'symbol': symbol,
                    'action': side.value.upper(),
                    'quantity': trade_quantity,
                    'price': current_price,
                    'signal_type': signal.signal_type,
                    'confidence': signal.confidence
                }
        
        except Exception as e:
            print(f"[ERROR] Failed to execute signal for {symbol}: {e}")
            return None
    
    def _get_trading_dates(self, historical_data: Dict[str, pd.DataFrame], 
                          frequency: str) -> List[pd.Timestamp]:
        """Get trading dates based on frequency"""
        # Get all available dates
        all_dates = set()
        for symbol_data in historical_data.values():
            if symbol_data is not None:
                all_dates.update(symbol_data.index)
        
        if not all_dates:
            return []
        
        sorted_dates = sorted(list(all_dates))
        
        if frequency == 'daily':
            return sorted_dates
        elif frequency == 'weekly':
            # Return every 5th trading day (approximate week)
            return sorted_dates[::5]
        elif frequency == 'monthly':
            # Return every 21st trading day (approximate month)
            return sorted_dates[::21]
        else:
            return sorted_dates
    
    def _calculate_performance_metrics(self, equity_curve: List[Tuple[pd.Timestamp, float]], 
                                     benchmark_data: Optional[pd.DataFrame],
                                     start_date: dt.datetime, end_date: dt.datetime) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not equity_curve or len(equity_curve) < 2:
            return self._create_empty_metrics()
        
        # Convert equity curve to series
        dates = [point[0] for point in equity_curve]
        values = [point[1] for point in equity_curve]
        equity_series = pd.Series(values, index=dates)
        
        # Calculate returns
        returns = equity_series.pct_change().dropna()
        
        # Basic metrics
        total_return = (values[-1] - values[0]) / values[0]
        periods_per_year = 252  # Trading days
        annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        running_max = equity_series.expanding().max()
        drawdowns = (equity_series - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(periods_per_year) if len(negative_returns) > 0 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # VaR (95% confidence)
        value_at_risk = returns.quantile(0.05) if len(returns) > 0 else 0
        
        # Trading metrics
        trades = self.portfolio.trades
        total_trades = len(trades)
        
        if total_trades > 0:
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
            
            win_rate = len(winning_trades) / total_trades
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Benchmark comparison
        benchmark_return = 0
        alpha = 0
        beta = 0
        information_ratio = 0
        treynor_ratio = 0
        
        if benchmark_data is not None and len(benchmark_data) > 1:
            # Align benchmark data with equity curve dates
            benchmark_aligned = benchmark_data.reindex(dates, method='ffill').dropna()
            if len(benchmark_aligned) > 1:
                benchmark_returns = benchmark_aligned.pct_change().dropna()
                benchmark_return = (benchmark_aligned.iloc[-1].values[0] - benchmark_aligned.iloc[0].values[0]) / benchmark_aligned.iloc[0].values[0]
                
                # Alpha and Beta
                if len(returns) > 0 and len(benchmark_returns) > 0:
                    aligned_returns = returns.align(benchmark_returns, join='inner')
                    if len(aligned_returns[0]) > 10:
                        covariance = np.cov(aligned_returns[0], aligned_returns[1])[0, 1]
                        benchmark_variance = np.var(aligned_returns[1])
                        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                        alpha = annualized_return - (risk_free_rate + beta * (benchmark_returns.mean() * periods_per_year - risk_free_rate))
                        
                        # Information ratio
                        active_returns = aligned_returns[0] - aligned_returns[1]
                        tracking_error = active_returns.std() * np.sqrt(periods_per_year)
                        information_ratio = (annualized_return - benchmark_returns.mean() * periods_per_year) / tracking_error if tracking_error > 0 else 0
                        
                        # Treynor ratio
                        treynor_ratio = (annualized_return - risk_free_rate) / beta if beta > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            benchmark_return=benchmark_return,
            alpha=alpha,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            value_at_risk=value_at_risk,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_duration=0,  # Would need to calculate based on entry/exit pairs
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            beta=beta
        )
    
    def _create_empty_metrics(self) -> PerformanceMetrics:
        """Create empty performance metrics"""
        return PerformanceMetrics(
            total_return=0, annualized_return=0, benchmark_return=0, alpha=0,
            volatility=0, sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
            value_at_risk=0, total_trades=0, win_rate=0, profit_factor=0,
            avg_win=0, avg_loss=0, avg_trade_duration=0, calmar_ratio=0,
            information_ratio=0, treynor_ratio=0, beta=0
        )
    
    def _print_backtest_summary(self, result: BacktestResult):
        """Print comprehensive backtest summary"""
        print("\n" + "="*60)
        print("[BACKTEST] COMPREHENSIVE BACKTEST RESULTS")
        print("="*60)
        
        # Performance Summary
        print(f"[PERFORMANCE] SUMMARY")
        print(f"   Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
        print(f"   Initial Value: ${result.initial_value:,.0f}")
        print(f"   Final Value: ${result.final_value:,.0f}")
        print(f"   Total Return: {result.total_return:.1%}")
        print(f"   Annualized Return: {result.performance_metrics.annualized_return:.1%}")
        print(f"   Benchmark Return: {result.performance_metrics.benchmark_return:.1%}")
        print(f"   Alpha: {result.performance_metrics.alpha:.1%}")
        
        # Risk Metrics
        print(f"\n[RISK] METRICS")
        print(f"   Volatility: {result.performance_metrics.volatility:.1%}")
        print(f"   Sharpe Ratio: {result.performance_metrics.sharpe_ratio:.3f}")
        print(f"   Sortino Ratio: {result.performance_metrics.sortino_ratio:.3f}")
        print(f"   Max Drawdown: {result.performance_metrics.max_drawdown:.1%}")
        print(f"   VaR (95%): {result.performance_metrics.value_at_risk:.1%}")
        print(f"   Beta: {result.performance_metrics.beta:.3f}")
        
        # Trading Metrics
        print(f"\n[TRADING] METRICS")
        print(f"   Total Trades: {result.performance_metrics.total_trades}")
        print(f"   Win Rate: {result.performance_metrics.win_rate:.1%}")
        print(f"   Profit Factor: {result.performance_metrics.profit_factor:.3f}")
        print(f"   Avg Win: {result.performance_metrics.avg_win:.3f}")
        print(f"   Avg Loss: {result.performance_metrics.avg_loss:.3f}")
        
        # Additional Ratios
        print(f"\n[RATIOS] ADDITIONAL")
        print(f"   Calmar Ratio: {result.performance_metrics.calmar_ratio:.3f}")
        print(f"   Information Ratio: {result.performance_metrics.information_ratio:.3f}")
        print(f"   Treynor Ratio: {result.performance_metrics.treynor_ratio:.3f}")
        
        print("="*60)
        print("[SUCCESS] Backtest completed successfully!")
        print("="*60)