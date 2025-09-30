"""
TradingView-Style Chart Generator
=================================
Creates comprehensive trading charts with candlesticks, indicators, signals, orders, and portfolio tracking.

This module provides strongly-typed classes and functions for generating professional trading charts
with proper type safety and error handling.

Typical usage example:
    chart_generator = TradingViewChartGenerator(results_dir="charts")
    chart_path = chart_generator.create_comprehensive_chart(
        symbol="AAPL",
        period="6mo",
        real_signals=trading_signals,
        backtest_results=results
    )
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from typing import (
    Dict, List, Optional, Tuple, Any, Union, Protocol, TypeVar, 
    Literal, ClassVar, Final, Callable, Iterator
)
import os
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Import the main TradingSignal from the interfaces module
try:
    from interfaces.signal_generator import TradingSignal, SignalType as MainSignalType
except ImportError:
    try:
        from src.interfaces.signal_generator import TradingSignal, SignalType as MainSignalType
    except ImportError:
        # Fallback - define minimal classes for testing
        from enum import Enum
        class MainSignalType(Enum):
            BUY = 1
            SELL = -1
            HOLD = 0
        
        @dataclass
        class TradingSignal:
            """Fallback TradingSignal implementation with proper typing."""
            symbol: str
            signal_type: 'MainSignalType'
            confidence: float
            source: str
            timestamp: pd.Timestamp
            strength: float = 0.0
            metadata: Optional[Dict[str, Any]] = None
            
            def __post_init__(self) -> None:
                """Validate signal parameters after initialization."""
                if not 0.0 <= self.confidence <= 1.0:
                    raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
                if self.metadata is None:
                    self.metadata = {}


class SignalType(Enum):
    """Enumeration for trading signal types with typed values."""
    BUY: int = 1
    SELL: int = -1
    HOLD: int = 0
    
    def __str__(self) -> str:
        return self.name
    
    @classmethod
    def from_value(cls, value: Union[int, str]) -> 'SignalType':
        """Create SignalType from various input formats."""
        if isinstance(value, int):
            for signal_type in cls:
                if signal_type.value == value:
                    return signal_type
        elif isinstance(value, str):
            value_upper = value.upper()
            for signal_type in cls:
                if signal_type.name == value_upper:
                    return signal_type
        raise ValueError(f"Invalid signal type: {value}")


class OrderType(Enum):
    """Enumeration for order types with string values."""
    MARKET: str = "market"
    LIMIT: str = "limit"
    STOP: str = "stop"
    STOP_LIMIT: str = "stop_limit"
    
    def __str__(self) -> str:
        return self.value


# Type aliases for better readability
OrderSide = Literal['buy', 'sell']
OrderStatus = Literal['filled', 'pending', 'cancelled', 'rejected']


@dataclass(frozen=True)  # Immutable for data integrity
class Order:
    """Represents a trading order with comprehensive type safety."""
    timestamp: datetime
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: float
    price: float
    size_usd: float
    status: OrderStatus = "filled"
    reason: str = ""
    
    def __post_init__(self) -> None:
        """Validate order parameters."""
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {self.quantity}")
        if self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")
        if self.size_usd <= 0:
            raise ValueError(f"Size USD must be positive, got {self.size_usd}")
        if not self.symbol.strip():
            raise ValueError("Symbol cannot be empty")
    
    @property
    def is_buy(self) -> bool:
        """Check if this is a buy order."""
        return self.side == 'buy'
    
    @property
    def is_filled(self) -> bool:
        """Check if this order is filled."""
        return self.status == 'filled'


@dataclass(frozen=True)  # Immutable for data integrity
class PortfolioSnapshot:
    """Represents a portfolio snapshot at a specific point in time."""
    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    daily_pnl: float
    total_pnl: float
    daily_return: float
    total_return: float
    
    def __post_init__(self) -> None:
        """Validate portfolio snapshot values."""
        if self.total_value < 0:
            raise ValueError(f"Total value cannot be negative, got {self.total_value}")
        if self.cash < 0:
            raise ValueError(f"Cash cannot be negative, got {self.cash}")
        # Allow negative PnL values as they represent losses
    
    @property
    def allocation_ratio(self) -> float:
        """Calculate the ratio of positions to total value."""
        return self.positions_value / self.total_value if self.total_value > 0 else 0.0


class TechnicalIndicators:
    """Calculate various technical indicators for charting with proper type safety."""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average.
        
        Args:
            data: Price series data
            window: Period for moving average calculation
            
        Returns:
            Series containing SMA values
            
        Raises:
            ValueError: If window is not positive or data is empty
        """
        if window <= 0:
            raise ValueError(f"Window must be positive, got {window}")
        if len(data) == 0:
            raise ValueError("Data series cannot be empty")
        return data.rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average.
        
        Args:
            data: Price series data
            window: Period for EMA calculation
            
        Returns:
            Series containing EMA values
            
        Raises:
            ValueError: If window is not positive or data is empty
        """
        if window <= 0:
            raise ValueError(f"Window must be positive, got {window}")
        if len(data) == 0:
            raise ValueError("Data series cannot be empty")
        return data.ewm(span=window, min_periods=1).mean()
    
    @staticmethod
    def bollinger_bands(
        data: pd.Series, 
        window: int = 20, 
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands.
        
        Args:
            data: Price series data
            window: Period for moving average and standard deviation
            std_dev: Number of standard deviations for band calculation
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
            
        Raises:
            ValueError: If parameters are invalid
        """
        if window <= 0:
            raise ValueError(f"Window must be positive, got {window}")
        if std_dev <= 0:
            raise ValueError(f"Standard deviation multiplier must be positive, got {std_dev}")
        if len(data) == 0:
            raise ValueError("Data series cannot be empty")
            
        sma = data.rolling(window=window, min_periods=1).mean()
        std = data.rolling(window=window, min_periods=1).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index.
        
        Args:
            data: Price series data
            window: Period for RSI calculation
            
        Returns:
            Series containing RSI values (0-100)
            
        Raises:
            ValueError: If parameters are invalid
        """
        if window <= 0:
            raise ValueError(f"Window must be positive, got {window}")
        if len(data) == 0:
            raise ValueError("Data series cannot be empty")
            
        delta = data.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # Use rolling mean with min_periods to handle edge cases
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        # Avoid division by zero using numpy operations
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
            rsi_values = 100 - (100 / (1 + rs))
            
        return pd.Series(rsi_values, index=data.index, name='RSI')
    
    @staticmethod
    def macd(
        data: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price series data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
            
        Raises:
            ValueError: If parameters are invalid
        """
        if fast <= 0 or slow <= 0 or signal <= 0:
            raise ValueError("All periods must be positive")
        if fast >= slow:
            raise ValueError(f"Fast period ({fast}) must be less than slow period ({slow})")
        if len(data) == 0:
            raise ValueError("Data series cannot be empty")
            
        exp1 = data.ewm(span=fast, min_periods=1).mean()
        exp2 = data.ewm(span=slow, min_periods=1).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, min_periods=1).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def volume_profile(
        prices: pd.Series, 
        volumes: pd.Series, 
        bins: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Volume Profile.
        
        Args:
            prices: Price series data
            volumes: Volume series data
            bins: Number of price bins for profile calculation
            
        Returns:
            Tuple of (price_levels, volume_profile)
            
        Raises:
            ValueError: If parameters are invalid
        """
        if bins <= 0:
            raise ValueError(f"Bins must be positive, got {bins}")
        if len(prices) != len(volumes):
            raise ValueError(f"Prices and volumes must have same length: {len(prices)} vs {len(volumes)}")
        if len(prices) == 0:
            return np.array([]), np.array([])
        
        price_min, price_max = float(prices.min()), float(prices.max())
        
        # Handle edge case where all prices are the same
        if np.isclose(price_min, price_max):
            price_levels = np.array([price_min])
            volume_profile = np.array([float(volumes.sum())])
            return price_levels, volume_profile
        
        price_bins = np.linspace(price_min, price_max, bins + 1)
        volume_profile = np.zeros(bins, dtype=float)
        
        for i in range(len(prices)):
            bin_idx = np.digitize(float(prices.iloc[i]), price_bins) - 1
            bin_idx = max(0, min(bins - 1, bin_idx))
            volume_profile[bin_idx] += float(volumes.iloc[i])
        
        price_levels = (price_bins[:-1] + price_bins[1:]) / 2
        return price_levels, volume_profile


# Type aliases for chart generation
ChartPeriod = Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
ColorScheme = Dict[str, str]
SignalData = Union[TradingSignal, Dict[str, Any]]
BacktestResults = Dict[str, Any]


class TradingViewChartGenerator:
    """
    Generate comprehensive TradingView-style charts with all trading information.
    
    This class provides methods to create professional trading charts with proper
    type safety, error handling, and performance optimizations.
    """
    
    # Class constants for better maintainability
    DEFAULT_RESULTS_DIR: Final[str] = "results"
    DEFAULT_CHART_HEIGHT: Final[int] = 1200
    DEFAULT_CHART_WIDTH: Final[int] = 1600
    
    # TradingView-inspired color scheme
    DEFAULT_COLORS: Final[ColorScheme] = {
        'background': '#131722',
        'grid': '#363c4e',
        'text': '#d1d4dc',
        'green': '#26a69a',  # Bullish candles
        'red': '#ef5350',    # Bearish candles
        'blue': '#2196f3',   # Indicators
        'orange': '#ff9800', # Signals
        'purple': '#9c27b0', # Orders
        'yellow': '#ffeb3b'  # Highlights
    }
    
    def __init__(self, results_dir: Optional[str] = None) -> None:
        """Initialize the chart generator.
        
        Args:
            results_dir: Directory to save chart files. Defaults to 'results'
            
        Raises:
            OSError: If directory cannot be created
        """
        self.results_dir: str = results_dir or self.DEFAULT_RESULTS_DIR
        
        try:
            os.makedirs(self.results_dir, exist_ok=True)
        except OSError as e:
            raise OSError(f"Cannot create results directory '{self.results_dir}': {e}") from e
        
        # Use immutable color scheme
        self.colors: ColorScheme = self.DEFAULT_COLORS.copy()
    
    def get_market_data(self, symbol: str, period: ChartPeriod = "1y") -> pd.DataFrame:
        """Fetch market data using yfinance with proper error handling.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'GOOGL')
            period: Time period for data retrieval
            
        Returns:
            DataFrame with OHLCV data, empty if fetch fails
            
        Raises:
            ValueError: If symbol is empty or invalid
        """
        if not symbol or not symbol.strip():
            raise ValueError("Symbol cannot be empty")
            
        symbol = symbol.strip().upper()
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            # Validate that we got meaningful data
            if data.empty:
                print(f"Warning: No data returned for symbol {symbol}")
                return pd.DataFrame()
                
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                print(f"Warning: Missing required columns for {symbol}: {missing_columns}")
                return pd.DataFrame()
                
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def _convert_real_signals_to_chart_format(
        self, 
        real_signals: List[SignalData], 
        data: pd.DataFrame
    ) -> List[TradingSignal]:
        """Convert real trading signals to chart format using the main TradingSignal structure.
        
        Args:
            real_signals: List of trading signals to convert
            data: Market data DataFrame for price lookup
            
        Returns:
            List of chart-compatible TradingSignal objects
            
        Raises:
            ValueError: If data DataFrame is invalid
        """
        if data.empty:
            raise ValueError("Market data DataFrame cannot be empty")
            
        if 'Close' not in data.columns:
            raise ValueError("Market data must contain 'Close' column")
            
        chart_signals: List[TradingSignal] = []
        
        for signal in real_signals:
            try:
                # Skip HOLD signals for cleaner charts
                signal_type = getattr(signal, 'signal_type', None)
                if signal_type == MainSignalType.HOLD:
                    continue
                
                # Extract signal timestamp safely
                if hasattr(signal, 'timestamp'):
                    signal_date = signal.timestamp
                elif isinstance(signal, dict) and 'timestamp' in signal:
                    signal_date = signal['timestamp']
                else:
                    print(f"Warning: Signal missing timestamp, skipping: {signal}")
                    continue
                
                # Find the price at the signal timestamp
                try:
                    closest_idx = data.index.get_indexer([signal_date], method='nearest')[0]
                    if closest_idx >= 0 and closest_idx < len(data):
                        price = float(data['Close'].iloc[closest_idx])
                    else:
                        print(f"Warning: Could not find price for signal at {signal_date}")
                        continue
                except Exception as e:
                    print(f"Warning: Error finding price for signal: {e}")
                    continue
                
                # Extract signal attributes safely
                if hasattr(signal, 'confidence'):
                    confidence = float(signal.confidence)
                    strength = float(getattr(signal, 'strength', 0.0))
                    source = str(signal.source)
                    metadata = getattr(signal, 'metadata', {}) or {}
                elif isinstance(signal, dict):
                    confidence = float(signal.get('confidence', 0.5))
                    strength = float(signal.get('strength', 0.0))
                    source = str(signal.get('source', 'Unknown'))
                    metadata = signal.get('metadata', {}) or {}
                else:
                    print(f"Warning: Invalid signal format, skipping: {signal}")
                    continue
                
                # Create chart-compatible metadata
                chart_metadata = metadata.copy()
                chart_metadata.update({
                    'chart_price': price,
                    'chart_reason': f"Confidence: {confidence:.3f}, Strength: {strength:.3f}",
                    'chart_display_type': signal_type.name.lower() if signal_type else 'unknown'
                })
                
                # Create chart signal (this might fail if TradingSignal class structure differs)
                try:
                    chart_signal = TradingSignal(
                        symbol=getattr(signal, 'symbol', 'UNKNOWN'),
                        timestamp=signal_date,
                        signal_type=signal_type,
                        confidence=confidence,
                        strength=strength,
                        source=source,
                        metadata=chart_metadata
                    )
                    chart_signals.append(chart_signal)
                except Exception as e:
                    print(f"Warning: Failed to create chart signal: {e}")
                    continue
                
            except Exception as e:
                print(f"Warning: Failed to convert signal {signal}: {e}")
                continue
        
        print(f"   Converted {len(chart_signals)} real signals for chart display")
        return chart_signals
    
    def create_comprehensive_chart(
        self, 
        symbol: str, 
        period: ChartPeriod = "6mo", 
        real_signals: Optional[List[SignalData]] = None,
        backtest_results: Optional[BacktestResults] = None
    ) -> str:
        """
        Create a comprehensive TradingView-style chart with all components.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'GOOGL')
            period: Time period for chart data
            real_signals: Optional list of real trading signals to display
            backtest_results: Optional backtest results with performance data
        
        Returns:
            Path to the saved chart file
            
        Raises:
            ValueError: If symbol is invalid or no data available
            OSError: If chart file cannot be saved
        """
        # Validate inputs
        if not symbol or not symbol.strip():
            raise ValueError("Symbol cannot be empty")
            
        symbol = symbol.strip().upper()
        
        # Get market data
        print(f"📊 Fetching market data for {symbol}...")
        data = self.get_market_data(symbol, period)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol} with period {period}")
        
        # Calculate technical indicators
        print("🔧 Calculating technical indicators...")
        sma_20 = TechnicalIndicators.sma(data['Close'], 20)
        sma_50 = TechnicalIndicators.sma(data['Close'], 50)
        ema_12 = TechnicalIndicators.ema(data['Close'], 12)
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(data['Close'])
        rsi = TechnicalIndicators.rsi(data['Close'])
        macd_line, signal_line, histogram = TechnicalIndicators.macd(data['Close'])
        
        # Generate trading data with proper type checking
        print("🎯 Generating trading signals...")
        signals: List[TradingSignal] = []
        if real_signals is not None:
            if not isinstance(real_signals, list):
                raise ValueError("real_signals must be a list")
            print(f"   Using {len(real_signals)} real trading signals")
            signals = self._convert_real_signals_to_chart_format(real_signals, data)
        else:
            print("   No real signals provided - chart will show price data only")
        
        # Use real backtest data if available
        orders: List[Order] = []
        portfolio_snapshots: List[PortfolioSnapshot] = []
        
        if backtest_results is not None:
            if not isinstance(backtest_results, dict):
                raise ValueError("backtest_results must be a dictionary")
            print("   Using real backtest results for orders and portfolio")
            orders = self._extract_orders_from_backtest(backtest_results)
            portfolio_snapshots = self._extract_portfolio_from_backtest(backtest_results, data)
        else:
            print("   No backtest results provided - chart will show basic portfolio tracking")
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            column_widths=[0.8, 0.2],
            specs=[
                [{"secondary_y": True}, {"rowspan": 4}],  # Main price chart + Volume profile
                [{"secondary_y": False}, None],           # RSI
                [{"secondary_y": False}, None],           # MACD  
                [{"secondary_y": False}, None]            # Portfolio
            ],
            subplot_titles=[
                f'{symbol} - Price & Volume', 'Volume Profile',
                'RSI (14)', '',
                'MACD', '',
                'Portfolio Value', ''
            ],
            vertical_spacing=0.05,
            horizontal_spacing=0.05
        )
        
        # 1. MAIN PRICE CHART (Candlesticks)
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=symbol,
                increasing_line_color=self.colors['green'],
                decreasing_line_color=self.colors['red'],
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 2. MOVING AVERAGES
        fig.add_trace(
            go.Scatter(
                x=data.index, y=sma_20, 
                name='SMA 20', 
                line=dict(color=self.colors['blue'], width=1),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index, y=sma_50, 
                name='SMA 50', 
                line=dict(color=self.colors['orange'], width=1),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index, y=ema_12, 
                name='EMA 12', 
                line=dict(color=self.colors['purple'], width=1, dash='dot'),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # 3. BOLLINGER BANDS
        fig.add_trace(
            go.Scatter(
                x=data.index, y=bb_upper,
                name='BB Upper',
                line=dict(color=self.colors['yellow'], width=1),
                opacity=0.3
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index, y=bb_lower,
                name='BB Lower',
                line=dict(color=self.colors['yellow'], width=1),
                fill='tonexty',
                fillcolor='rgba(255, 235, 59, 0.1)',
                opacity=0.3
            ),
            row=1, col=1
        )
        
        # 4. VOLUME (Secondary Y-axis)
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=self.colors['blue'],
                opacity=0.3,
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # 5. TRADING SIGNALS with proper type checking
        buy_signals: List[Union[TradingSignal, Dict[str, Any]]] = []
        sell_signals: List[Union[TradingSignal, Dict[str, Any]]] = []
        
        for s in signals:
            try:
                # Handle TradingSignal objects (preferred)
                if hasattr(s, 'signal_type') and hasattr(s, 'timestamp'):
                    signal_type = s.signal_type
                    if signal_type == MainSignalType.BUY:
                        buy_signals.append(s)
                    elif signal_type == MainSignalType.SELL:
                        sell_signals.append(s)
                    # Skip HOLD signals for cleaner charts
                    elif signal_type == MainSignalType.HOLD:
                        continue
                # Handle dictionary format (legacy support)
                elif isinstance(s, dict):
                    signal_type = s.get('signal_type', '')
                    # Support multiple formats for signal types
                    if (signal_type == MainSignalType.BUY or 
                        'BUY' in str(signal_type).upper() or 
                        signal_type == 1):
                        buy_signals.append(s)
                    elif (signal_type == MainSignalType.SELL or 
                          'SELL' in str(signal_type).upper() or 
                          signal_type == -1):
                        sell_signals.append(s)
                else:
                    print(f"Warning: Unknown signal format: {type(s)} - {s}")
                    continue
            except Exception as e:
                print(f"Warning: Failed to process signal {s}: {e}")
                continue
        
        # Plot buy signals with proper type checking
        if buy_signals:
            buy_timestamps: List[pd.Timestamp] = []
            buy_prices: List[float] = []
            buy_texts: List[str] = []
            
            for s in buy_signals:
                try:
                    # Extract data from TradingSignal objects
                    if hasattr(s, 'timestamp') and hasattr(s, 'confidence'):
                        timestamp = s.timestamp
                        if not isinstance(timestamp, pd.Timestamp):
                            timestamp = pd.to_datetime(timestamp)
                        
                        price = float(s.metadata.get('chart_price', 0.0)) if s.metadata else 0.0
                        confidence = float(s.confidence)
                        reason = str(s.metadata.get('chart_reason', '')) if s.metadata else ''
                        
                        buy_timestamps.append(timestamp)
                        buy_prices.append(price)
                        buy_texts.append(f"Buy: {confidence:.1%}<br>{reason}")
                        
                    # Handle dictionary format
                    elif isinstance(s, dict):
                        timestamp_raw = s.get('timestamp', pd.Timestamp.now())
                        timestamp = pd.to_datetime(timestamp_raw) if not isinstance(timestamp_raw, pd.Timestamp) else timestamp_raw
                        
                        price = float(s.get('price', s.get('chart_price', 0.0)))
                        confidence = float(s.get('confidence', 0.5))
                        reason = str(s.get('reason', s.get('chart_reason', '')))
                        
                        buy_timestamps.append(timestamp)
                        buy_prices.append(price)
                        buy_texts.append(f"Buy: {confidence:.1%}<br>{reason}")
                    else:
                        print(f"Warning: Unexpected buy signal format: {s}")
                        continue
                        
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"Warning: Failed to extract buy signal data: {e}")
                    continue
            
            # Only add trace if we have valid data
            if buy_timestamps and buy_prices:
                fig.add_trace(
                    go.Scatter(
                        x=buy_timestamps,
                        y=buy_prices,
                        mode='markers',
                        name='Buy Signals',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color=self.colors['green'],
                            line=dict(width=2, color='white')
                        ),
                        text=buy_texts,
                        textposition="top center",
                        hovertemplate='<b>%{text}</b><br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Plot sell signals with proper type checking
        if sell_signals:
            sell_timestamps: List[pd.Timestamp] = []
            sell_prices: List[float] = []
            sell_texts: List[str] = []
            
            for s in sell_signals:
                try:
                    # Extract data from TradingSignal objects
                    if hasattr(s, 'timestamp') and hasattr(s, 'confidence'):
                        timestamp = s.timestamp
                        if not isinstance(timestamp, pd.Timestamp):
                            timestamp = pd.to_datetime(timestamp)
                        
                        price = float(s.metadata.get('chart_price', 0.0)) if s.metadata else 0.0
                        confidence = float(s.confidence)
                        reason = str(s.metadata.get('chart_reason', '')) if s.metadata else ''
                        
                        sell_timestamps.append(timestamp)
                        sell_prices.append(price)
                        sell_texts.append(f"Sell: {confidence:.1%}<br>{reason}")
                        
                    # Handle dictionary format
                    elif isinstance(s, dict):
                        timestamp_raw = s.get('timestamp', pd.Timestamp.now())
                        timestamp = pd.to_datetime(timestamp_raw) if not isinstance(timestamp_raw, pd.Timestamp) else timestamp_raw
                        
                        price = float(s.get('price', s.get('chart_price', 0.0)))
                        confidence = float(s.get('confidence', 0.5))
                        reason = str(s.get('reason', s.get('chart_reason', '')))
                        
                        sell_timestamps.append(timestamp)
                        sell_prices.append(price)
                        sell_texts.append(f"Sell: {confidence:.1%}<br>{reason}")
                    else:
                        print(f"Warning: Unexpected sell signal format: {s}")
                        continue
                        
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"Warning: Failed to extract sell signal data: {e}")
                    continue
            
            # Only add trace if we have valid data
            if sell_timestamps and sell_prices:
                fig.add_trace(
                    go.Scatter(
                        x=sell_timestamps,
                        y=sell_prices,
                        mode='markers',
                        name='Sell Signals',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color=self.colors['red'],
                            line=dict(width=2, color='white')
                        ),
                        text=sell_texts,
                        textposition="bottom center",
                        hovertemplate='<b>%{text}</b><br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # 6. ORDERS with proper type checking and validation
        buy_orders: List[Order] = []
        sell_orders: List[Order] = []
        
        for order in orders:
            try:
                if not isinstance(order, Order):
                    print(f"Warning: Expected Order object, got {type(order)}")
                    continue
                    
                # Validate order has required attributes
                if not hasattr(order, 'side') or not hasattr(order, 'timestamp') or not hasattr(order, 'price'):
                    print(f"Warning: Order missing required attributes: {order}")
                    continue
                
                side = str(order.side).upper()
                if side == 'BUY':
                    buy_orders.append(order)
                elif side == 'SELL':
                    sell_orders.append(order)
                else:
                    print(f"Warning: Unknown order side '{side}' for order {order.order_id}")
                    continue
                    
            except Exception as e:
                print(f"Warning: Failed to process order: {e}")
                continue
        
        # Plot buy orders with proper validation
        if buy_orders:
            try:
                # Extract order data with validation
                buy_timestamps = []
                buy_prices = []
                buy_sizes = []
                buy_hover_texts = []
                buy_display_texts = []
                
                for order in buy_orders:
                    try:
                        # Validate numeric fields
                        price = float(order.price)
                        quantity = float(order.quantity)
                        size_usd = price * quantity
                        
                        # Calculate marker size (scale based on USD value)
                        marker_size = min(25, max(12, size_usd / 3000))
                        
                        # Get order reason if available
                        reason = getattr(order, 'reason', 'N/A')
                        if not reason or reason == 'N/A':
                            reason = f"Market order @ ${price:.2f}"
                        
                        buy_timestamps.append(order.timestamp)
                        buy_prices.append(price)
                        buy_sizes.append(marker_size)
                        buy_display_texts.append(f"BUY<br>${size_usd:,.0f}")
                        buy_hover_texts.append(
                            f"Buy Order: ${size_usd:,.0f}<br>"
                            f"{quantity:.1f} shares @ ${price:.2f}<br>"
                            f"{reason}"
                        )
                        
                    except (ValueError, TypeError, AttributeError) as e:
                        print(f"Warning: Failed to process buy order {order.order_id}: {e}")
                        continue
                
                # Only add trace if we have valid orders
                if buy_timestamps:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_timestamps,
                            y=buy_prices,
                            mode='markers+text',
                            name='💰 Buy Orders',
                            marker=dict(
                                symbol='triangle-up',
                                size=buy_sizes,
                                color=self.colors['green'],
                                opacity=0.9,
                                line=dict(width=2, color='white')
                            ),
                            text=buy_display_texts,
                            textposition="top center",
                            textfont=dict(size=8, color='white'),
                            hovertext=buy_hover_texts,
                            hoverinfo='text'
                        ),
                        row=1, col=1
                    )
                    
            except Exception as e:
                print(f"Warning: Failed to plot buy orders: {e}")
        
        # Plot sell orders with proper validation
        if sell_orders:
            try:
                # Extract order data with validation
                sell_timestamps = []
                sell_prices = []
                sell_sizes = []
                sell_hover_texts = []
                sell_display_texts = []
                
                for order in sell_orders:
                    try:
                        # Validate numeric fields
                        price = float(order.price)
                        quantity = float(order.quantity)
                        size_usd = price * quantity
                        
                        # Calculate marker size (scale based on USD value)
                        marker_size = min(25, max(12, size_usd / 3000))
                        
                        # Get order reason if available
                        reason = getattr(order, 'reason', 'N/A')
                        if not reason or reason == 'N/A':
                            reason = f"Market order @ ${price:.2f}"
                        
                        sell_timestamps.append(order.timestamp)
                        sell_prices.append(price)
                        sell_sizes.append(marker_size)
                        sell_display_texts.append(f"SELL<br>${size_usd:,.0f}")
                        sell_hover_texts.append(
                            f"Sell Order: ${size_usd:,.0f}<br>"
                            f"{quantity:.1f} shares @ ${price:.2f}<br>"
                            f"{reason}"
                        )
                        
                    except (ValueError, TypeError, AttributeError) as e:
                        print(f"Warning: Failed to process sell order {order.order_id}: {e}")
                        continue
                
                # Only add trace if we have valid orders
                if sell_timestamps:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_timestamps,
                            y=sell_prices,
                            mode='markers+text',
                            name='💸 Sell Orders',
                            marker=dict(
                                symbol='triangle-down',
                                size=sell_sizes,
                                color=self.colors['red'],
                                opacity=0.9,
                                line=dict(width=2, color='white')
                            ),
                            text=sell_display_texts,
                            textposition="bottom center",
                            textfont=dict(size=8, color='white'),
                            hovertext=sell_hover_texts,
                            hoverinfo='text'
                        ),
                        row=1, col=1
                    )
                    
            except Exception as e:
                print(f"Warning: Failed to plot sell orders: {e}")
        
        # 7. VOLUME PROFILE (Right side)
        price_levels, volume_profile = TechnicalIndicators.volume_profile(data['Close'], data['Volume'])
        fig.add_trace(
            go.Scatter(
                x=volume_profile,
                y=price_levels,
                mode='lines',
                name='Volume Profile',
                fill='tozeroy',
                line=dict(color=self.colors['blue'], width=2),
                fillcolor='rgba(33, 150, 243, 0.3)'
            ),
            row=1, col=2
        )
        
        # 8. RSI INDICATOR
        fig.add_trace(
            go.Scatter(
                x=data.index, y=rsi,
                name='RSI',
                line=dict(color=self.colors['purple'], width=2)
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color=self.colors['red'], opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=self.colors['green'], opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color=self.colors['text'], opacity=0.3, row=2, col=1)
        
        # 9. MACD INDICATOR
        fig.add_trace(
            go.Scatter(
                x=data.index, y=macd_line,
                name='MACD',
                line=dict(color=self.colors['blue'], width=2)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index, y=signal_line,
                name='Signal',
                line=dict(color=self.colors['orange'], width=2)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=data.index, y=histogram,
                name='Histogram',
                marker_color=self.colors['green'],
                opacity=0.6
            ),
            row=3, col=1
        )
        
        # 10. PORTFOLIO VALUE
        if portfolio_snapshots:
            portfolio_dates = [p.timestamp for p in portfolio_snapshots]
            portfolio_values = [p.total_value for p in portfolio_snapshots]
            
            fig.add_trace(
                go.Scatter(
                    x=portfolio_dates,
                    y=portfolio_values,
                    name='Portfolio Value',
                    line=dict(color=self.colors['yellow'], width=3),
                    fill='tonexty'
                ),
                row=4, col=1
            )
            
            # Add portfolio annotations
            final_value = portfolio_values[-1]
            initial_value = portfolio_values[0]
            total_return = (final_value / initial_value - 1) * 100
            
            fig.add_annotation(
                x=portfolio_dates[-1],
                y=final_value,
                text=f"${final_value:,.0f}<br>({total_return:+.1f}%)",
                showarrow=True,
                arrowhead=2,
                arrowcolor=self.colors['yellow'],
                bgcolor=self.colors['background'],
                bordercolor=self.colors['yellow'],
                borderwidth=1,
                row=4, col=1
            )
        
        # UPDATE LAYOUT
        fig.update_layout(
            title=dict(
                text=f'🚀 {symbol} Trading Analysis - TradingView Style Dashboard',
                x=0.5,
                y=0.98,  # Move title higher to create space
                font=dict(size=20, color=self.colors['text'])
            ),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.95,  # Position legend below title
                xanchor="center",
                x=0.5,
                bgcolor='rgba(19, 23, 34, 0.9)',
                bordercolor=self.colors['grid'],
                borderwidth=1,
                font=dict(size=10)  # Smaller font for compact legend
            ),
            height=1200,
            width=1600,
            margin=dict(t=120, b=60, l=60, r=60)  # Add top margin for title/legend
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=self.colors['grid'],
            showline=True,
            linewidth=1,
            linecolor=self.colors['grid']
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=self.colors['grid'],
            showline=True,
            linewidth=1,
            linecolor=self.colors['grid']
        )
        
        # Hide volume y-axis labels on secondary axis
        fig.update_yaxes(secondary_y=True, showticklabels=False, row=1, col=1)
        
        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trading_chart_{symbol}_{timestamp}.html"
        filepath = os.path.join(self.results_dir, filename)
        
        fig.write_html(filepath)
        print(f"📈 Chart saved: {filepath}")
        
        # Also save as PNG if kaleido is available
        # Temporarily disabled PNG generation due to hanging issues
        # try:
        #     png_filepath = filepath.replace('.html', '.png')
        #     fig.write_image(png_filepath, width=1600, height=1200)
        #     print(f"📸 PNG chart saved: {png_filepath}")
        # except Exception as e:
        #     print(f"⚠️ Could not save PNG (install kaleido for PNG export): {e}")
        print("📸 PNG generation disabled to prevent hanging")
        
        return filepath

    def _extract_orders_from_backtest(self, backtest_results: BacktestResults) -> List[Order]:
        """Extract order data from backtest results.
        
        Args:
            backtest_results: Backtest results containing trade history
            
        Returns:
            List of Order objects extracted from backtest
            
        Raises:
            ValueError: If backtest results format is invalid
        """
        orders: List[Order] = []
        
        if not isinstance(backtest_results, dict):
            raise ValueError("backtest_results must be a dictionary")
            
        if 'trades' not in backtest_results:
            print("Warning: No trades found in backtest results")
            return orders
            
        trades = backtest_results['trades']
        if not isinstance(trades, list):
            print("Warning: trades is not a list in backtest results")
            return orders
            
        for i, trade in enumerate(trades):
            try:
                if not isinstance(trade, dict):
                    print(f"Warning: Invalid trade format at index {i}: {trade}")
                    continue
                    
                # Extract and validate trade data
                symbol = str(trade.get('symbol', 'UNKNOWN')).upper().strip()
                if not symbol or symbol == 'UNKNOWN':
                    print(f"Warning: Invalid symbol in trade {i}")
                    continue
                
                # Parse timestamp
                timestamp_raw = trade.get('entry_date', trade.get('timestamp'))
                if timestamp_raw is None:
                    timestamp = pd.Timestamp.now()
                    print(f"Warning: Missing timestamp for trade {i}, using current time")
                else:
                    try:
                        timestamp = pd.to_datetime(timestamp_raw)
                    except Exception as e:
                        print(f"Warning: Invalid timestamp for trade {i}: {e}")
                        timestamp = pd.Timestamp.now()
                
                # Validate numeric fields
                try:
                    quantity = float(trade.get('quantity', 0))
                    price = float(trade.get('entry_price', trade.get('price', 0)))
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid numeric values in trade {i}: {e}")
                    continue
                    
                if quantity <= 0 or price <= 0:
                    print(f"Warning: Invalid quantity ({quantity}) or price ({price}) in trade {i}")
                    continue
                
                # Validate action/side
                action = str(trade.get('action', 'BUY')).upper().strip()
                if action not in ['BUY', 'SELL']:
                    print(f"Warning: Invalid action '{action}' in trade {i}, defaulting to BUY")
                    action = 'BUY'
                
                # Create Order object
                order = Order(
                    order_id=f"backtest_order_{i}",
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side=action,  # type: ignore
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    status='FILLED'  # Backtest trades are assumed filled
                )
                orders.append(order)
                
            except Exception as e:
                print(f"Warning: Failed to process trade {i}: {e}")
                continue
                
        print(f"   Extracted {len(orders)} valid orders from {len(trades)} trades")
        return orders
    
    def _extract_portfolio_from_backtest(
        self, 
        backtest_results: BacktestResults, 
        data: pd.DataFrame
    ) -> List[PortfolioSnapshot]:
        """Extract portfolio snapshots from backtest results.
        
        Args:
            backtest_results: Backtest results containing portfolio history
            data: Market data DataFrame for date reference
            
        Returns:
            List of PortfolioSnapshot objects
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(backtest_results, dict):
            raise ValueError("backtest_results must be a dictionary")
            
        if data.empty:
            raise ValueError("Market data DataFrame cannot be empty")
            
        portfolio_snapshots: List[PortfolioSnapshot] = []
        
        # Try to extract detailed portfolio history
        if 'portfolio_history' in backtest_results:
            portfolio_history = backtest_results['portfolio_history']
            if isinstance(portfolio_history, list):
                for i, snapshot_data in enumerate(portfolio_history):
                    try:
                        if not isinstance(snapshot_data, dict):
                            print(f"Warning: Invalid portfolio snapshot format at index {i}")
                            continue
                            
                        # Parse timestamp
                        date_raw = snapshot_data.get('date', snapshot_data.get('timestamp'))
                        if date_raw is None:
                            print(f"Warning: Missing date in portfolio snapshot {i}")
                            continue
                            
                        try:
                            timestamp = pd.to_datetime(date_raw)
                        except Exception as e:
                            print(f"Warning: Invalid date in portfolio snapshot {i}: {e}")
                            continue
                        
                        # Extract and validate portfolio values
                        try:
                            total_value = float(snapshot_data.get('value', snapshot_data.get('total_value', 0)))
                            cash = float(snapshot_data.get('cash', 0))
                            positions_value = float(snapshot_data.get('positions', snapshot_data.get('positions_value', 0)))
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Invalid numeric values in portfolio snapshot {i}: {e}")
                            continue
                            
                        if total_value < 0:
                            print(f"Warning: Negative portfolio value in snapshot {i}: {total_value}")
                            continue
                        
                        # Ensure consistency (total = cash + positions)
                        if abs(total_value - (cash + positions_value)) > 0.01 and total_value > 0:
                            # Adjust positions_value to maintain consistency
                            positions_value = max(0, total_value - cash)
                        
                        snapshot = PortfolioSnapshot(
                            timestamp=timestamp,
                            total_value=total_value,
                            cash=cash,
                            positions_value=positions_value
                        )
                        portfolio_snapshots.append(snapshot)
                        
                    except Exception as e:
                        print(f"Warning: Failed to process portfolio snapshot {i}: {e}")
                        continue
        
        # Generate synthetic portfolio progression if no detailed history
        if not portfolio_snapshots:
            print("   No detailed portfolio history found, generating synthetic progression")
            
            try:
                # Extract performance metrics
                performance = backtest_results.get('performance', {})
                initial_value = float(backtest_results.get('initial_capital', 100000))
                
                # Calculate final value from total return
                total_return = float(performance.get('total_return', 0.0))
                final_value = initial_value * (1 + total_return)
                
                # Use recent market data for timeline
                recent_data = data.tail(min(30, len(data)))  # Last 30 days or available data
                if recent_data.empty:
                    recent_data = data  # Use all available data if less than 30 days
                
                # Create linear progression of portfolio value
                for i, date in enumerate(recent_data.index):
                    try:
                        progress = i / max(1, len(recent_data) - 1) if len(recent_data) > 1 else 1.0
                        current_value = initial_value + (final_value - initial_value) * progress
                        
                        # Dynamic cash allocation (start high, decrease over time)
                        cash_ratio = 0.2 - 0.1 * progress  # Start at 20%, end at 10%
                        cash_ratio = max(0.05, min(0.2, cash_ratio))  # Clamp between 5% and 20%
                        
                        cash = current_value * cash_ratio
                        positions_value = current_value - cash
                        
                        snapshot = PortfolioSnapshot(
                            timestamp=date,
                            total_value=current_value,
                            cash=cash,
                            positions_value=positions_value
                        )
                        portfolio_snapshots.append(snapshot)
                        
                    except Exception as e:
                        print(f"Warning: Failed to create synthetic portfolio snapshot for date {date}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Warning: Failed to generate synthetic portfolio progression: {e}")
                # Create minimal fallback
                if not data.empty:
                    fallback_value = 100000.0
                    fallback_snapshot = PortfolioSnapshot(
                        timestamp=data.index[-1],
                        total_value=fallback_value,
                        cash=fallback_value * 0.1,
                        positions_value=fallback_value * 0.9
                    )
                    portfolio_snapshots.append(fallback_snapshot)
        
        print(f"   Extracted {len(portfolio_snapshots)} portfolio snapshots")
        return portfolio_snapshots


if __name__ == "__main__":
    # Example usage: python tradingview_charts.py
    print("TradingView Chart Generator")
    print("Import this module to use chart generation functionality")
    print("Example: chart_generator = TradingViewChartGenerator()")
    print("         chart_path = chart_generator.create_comprehensive_chart('AAPL')")