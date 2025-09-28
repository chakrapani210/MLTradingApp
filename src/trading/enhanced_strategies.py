"""
Enhanced Trading Strategies Implementation
Implements all features from enhanced_strategy.py in the new modular architecture
"""

import datetime as dt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
from dataclasses import dataclass
from enum import Enum

from ..interfaces.trading_strategy import TradingStrategy, TradingSignal
from ..interfaces.signal_generator import SignalGenerator
from ..interfaces.model_manager import ModelManagerInterface
from ..interfaces.risk_manager import RiskManager
from ..data.providers import YFinanceProvider
from ..signals.technical import RSISignalGenerator, MACDSignalGenerator, BollingerBandsSignalGenerator

warnings.filterwarnings('ignore')


class OrderSizingStrategy(Enum):
    """Order sizing strategies"""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity"


@dataclass
class OrderSizingConfig:
    """Configuration for order sizing strategies"""
    strategy: OrderSizingStrategy = OrderSizingStrategy.PERCENTAGE
    
    # Fixed sizing
    fixed_shares: int = 100
    
    # Percentage sizing
    portfolio_pct: float = 0.1
    min_shares: int = 10
    max_shares: int = 1000
    
    # Volatility adjusted
    volatility_window: int = 20
    volatility_target: float = 0.02
    adjustment_factor: float = 0.5
    base_shares: int = 100
    
    # Kelly criterion
    win_rate: float = 0.55
    avg_win: float = 0.05
    avg_loss: float = 0.03
    kelly_fraction: float = 0.25
    base_portfolio_pct: float = 0.1
    
    # Risk parity
    lookback_days: int = 60
    portfolio_risk_budget: float = 0.1
    
    # Market conditions
    market_conditions_enabled: bool = True
    bull_market_multiplier: float = 1.2
    bear_market_multiplier: float = 0.8
    
    # Position limits
    max_position_pct: float = 0.25


class AutoOrderSizeManager:
    """
    Intelligent Order Size Management System
    
    Supports multiple sizing strategies:
    - Fixed: Always use the same number of shares
    - Percentage: Size based on portfolio percentage
    - Volatility Adjusted: Adjust size based on stock volatility
    - Kelly Criterion: Optimal sizing based on win rate and risk/reward
    - Risk Parity: Size based on risk contribution to portfolio
    """
    
    def __init__(self, config: OrderSizingConfig, starting_portfolio_value: float = 100000):
        """Initialize the order size manager"""
        self.config = config
        self.current_portfolio_value = starting_portfolio_value
        self.current_positions = {}
        
        print(f"[ORDER_SIZE] Auto Order Size Manager initialized")
        print(f"             Strategy: {config.strategy.value}")
        print(f"             Starting Value: ${starting_portfolio_value:,.0f}")
    
    def calculate_order_size(self, symbol: str, signal: TradingSignal, 
                           stock_data: pd.DataFrame) -> int:
        """
        Calculate optimal order size based on configured strategy
        
        Args:
            symbol: Trading symbol
            signal: Trading signal with direction and confidence
            stock_data: Historical stock data for calculations
            
        Returns:
            Number of shares to trade (positive for buy, negative for sell)
        """
        if signal.direction == 0:  # HOLD signal
            return 0
            
        current_price = float(stock_data.iloc[-1])
        
        # Calculate base size using selected strategy
        if self.config.strategy == OrderSizingStrategy.FIXED:
            base_size = self._calculate_fixed_size()
        elif self.config.strategy == OrderSizingStrategy.PERCENTAGE:
            base_size = self._calculate_percentage_size(current_price)
        elif self.config.strategy == OrderSizingStrategy.VOLATILITY_ADJUSTED:
            base_size = self._calculate_volatility_adjusted_size(stock_data, current_price)
        elif self.config.strategy == OrderSizingStrategy.KELLY_CRITERION:
            base_size = self._calculate_kelly_size(current_price, signal.confidence)
        elif self.config.strategy == OrderSizingStrategy.RISK_PARITY:
            base_size = self._calculate_risk_parity_size(stock_data, current_price)
        else:
            base_size = self._calculate_fixed_size()
        
        # Apply market condition adjustments
        adjusted_size = self._apply_market_conditions_adjustment(base_size, stock_data)
        
        # Apply position limits
        final_size = self._apply_position_limits(adjusted_size, symbol, current_price, signal.direction)
        
        # Apply confidence scaling
        confidence_scaled_size = int(final_size * signal.confidence)
        
        # Ensure minimum viable size
        if abs(confidence_scaled_size) < 1:
            confidence_scaled_size = 1 if signal.direction > 0 else -1
            
        return int(confidence_scaled_size * signal.direction)
    
    def _calculate_fixed_size(self) -> int:
        """Calculate fixed order size"""
        return self.config.fixed_shares
    
    def _calculate_percentage_size(self, current_price: float) -> int:
        """Calculate percentage-based order size"""
        target_value = self.current_portfolio_value * self.config.portfolio_pct
        shares = int(target_value / current_price)
        
        # Apply min/max constraints
        shares = max(self.config.min_shares, min(self.config.max_shares, shares))
        return shares
    
    def _calculate_volatility_adjusted_size(self, stock_data: pd.DataFrame, current_price: float) -> int:
        """Calculate volatility-adjusted order size"""
        window = min(self.config.volatility_window, len(stock_data))
        recent_data = stock_data.tail(window)
        daily_returns = recent_data.pct_change().dropna()
        current_volatility = daily_returns.std() if len(daily_returns) > 0 else 0.02
        
        # Adjust base size based on volatility
        if current_volatility > 0:
            volatility_ratio = self.config.volatility_target / current_volatility
            adjusted_size = self.config.base_shares * (volatility_ratio ** self.config.adjustment_factor)
        else:
            adjusted_size = self.config.base_shares
        
        # Apply constraints
        adjusted_size = max(self.config.min_shares, min(self.config.max_shares, int(adjusted_size)))
        return adjusted_size
    
    def _calculate_kelly_size(self, current_price: float, confidence: float) -> int:
        """Calculate Kelly Criterion optimal size"""
        # Kelly formula: f = (bp - q) / b
        win_rate = self.config.win_rate
        avg_win = self.config.avg_win
        avg_loss = self.config.avg_loss
        
        # Calculate Kelly fraction
        if avg_loss > 0:
            kelly_fraction = (avg_win * win_rate - avg_loss * (1 - win_rate)) / avg_win
        else:
            kelly_fraction = 0
        
        # Apply conservative fraction and confidence scaling
        conservative_kelly = kelly_fraction * self.config.kelly_fraction * confidence
        conservative_kelly = max(0, min(1, conservative_kelly))
        
        # Calculate position size
        base_allocation = self.current_portfolio_value * self.config.base_portfolio_pct
        kelly_allocation = base_allocation * (1 + conservative_kelly)
        shares = int(kelly_allocation / current_price)
        
        # Apply constraints
        shares = max(self.config.min_shares, min(self.config.max_shares, shares))
        return shares
    
    def _calculate_risk_parity_size(self, stock_data: pd.DataFrame, current_price: float) -> int:
        """Calculate risk parity based size"""
        lookback = min(self.config.lookback_days, len(stock_data))
        recent_data = stock_data.tail(lookback)
        daily_returns = recent_data.pct_change().dropna()
        stock_volatility = daily_returns.std() if len(daily_returns) > 0 else 0.02
        
        # Risk parity sizing: allocate based on inverse volatility
        risk_budget = self.current_portfolio_value * self.config.portfolio_risk_budget
        
        if stock_volatility > 0:
            position_value = risk_budget / stock_volatility
            shares = int(position_value / current_price)
        else:
            shares = self.config.min_shares
        
        # Apply constraints
        shares = max(self.config.min_shares, min(self.config.max_shares, shares))
        return shares
    
    def _apply_market_conditions_adjustment(self, base_size: int, stock_data: pd.DataFrame) -> int:
        """Apply market condition based adjustments"""
        if not self.config.market_conditions_enabled:
            return base_size
        
        # Simple market regime detection
        if len(stock_data) >= 20:
            recent_returns = stock_data.pct_change().tail(20).mean()
            
            if recent_returns > 0.01:  # Bull market
                return int(base_size * self.config.bull_market_multiplier)
            elif recent_returns < -0.01:  # Bear market
                return int(base_size * self.config.bear_market_multiplier)
        
        return base_size
    
    def _apply_position_limits(self, size: int, symbol: str, current_price: float, signal: int) -> int:
        """Apply position and risk limits"""
        max_position_value = self.current_portfolio_value * self.config.max_position_pct
        max_shares = int(max_position_value / current_price)
        
        # Apply position limit
        limited_size = min(abs(size), max_shares)
        return limited_size
    
    def update_position(self, symbol: str, shares: int, price: float):
        """Update current position tracking"""
        if symbol not in self.current_positions:
            self.current_positions[symbol] = 0
        
        self.current_positions[symbol] += shares
        self.current_portfolio_value -= shares * price  # Update portfolio value
    
    def get_current_position(self, symbol: str) -> int:
        """Get current position for symbol"""
        return self.current_positions.get(symbol, 0)


class GoldenCrossSignalGenerator(SignalGenerator):
    """Golden Cross/Death Cross signal generator"""
    
    def __init__(self, short_window: int = 20, long_window: int = 50, 
                 strength_threshold: float = 0.6, confirmation_days: int = 3):
        """
        Initialize Golden Cross signal generator
        
        Args:
            short_window: Short-term moving average window
            long_window: Long-term moving average window
            strength_threshold: Minimum strength for signal generation
            confirmation_days: Days to confirm pattern
        """
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.strength_threshold = strength_threshold
        self.confirmation_days = confirmation_days
        self.name = f"GoldenCross_{short_window}_{long_window}"
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate golden cross/death cross signals"""
        if len(data) < self.long_window:
            return []
        
        # Calculate moving averages
        sma_short = data.rolling(window=self.short_window).mean()
        sma_long = data.rolling(window=self.long_window).mean()
        
        signals = []
        
        for i in range(self.long_window, len(data)):
            current_date = data.index[i]
            
            # Check for golden cross (short MA crosses above long MA)
            if (sma_short.iloc[i] > sma_long.iloc[i] and 
                sma_short.iloc[i-1] <= sma_long.iloc[i-1]):
                
                # Calculate signal strength based on separation and momentum
                separation = (sma_short.iloc[i] - sma_long.iloc[i]) / sma_long.iloc[i]
                momentum = (sma_short.iloc[i] - sma_short.iloc[i-5]) / sma_short.iloc[i-5] if i >= 5 else 0
                strength = min(1.0, abs(separation) * 10 + abs(momentum) * 5)
                
                if strength >= self.strength_threshold:
                    signal = TradingSignal(
                        timestamp=current_date,
                        symbol=data.name if hasattr(data, 'name') else 'UNKNOWN',
                        direction=1,
                        confidence=strength,
                        signal_type='GOLDEN_CROSS',
                        metadata={
                            'sma_short': float(sma_short.iloc[i]),
                            'sma_long': float(sma_long.iloc[i]),
                            'separation': float(separation),
                            'momentum': float(momentum)
                        }
                    )
                    signals.append(signal)
            
            # Check for death cross (short MA crosses below long MA)
            elif (sma_short.iloc[i] < sma_long.iloc[i] and 
                  sma_short.iloc[i-1] >= sma_long.iloc[i-1]):
                
                separation = (sma_long.iloc[i] - sma_short.iloc[i]) / sma_long.iloc[i]
                momentum = (sma_short.iloc[i-5] - sma_short.iloc[i]) / sma_short.iloc[i] if i >= 5 else 0
                strength = min(1.0, abs(separation) * 10 + abs(momentum) * 5)
                
                if strength >= self.strength_threshold:
                    signal = TradingSignal(
                        timestamp=current_date,
                        symbol=data.name if hasattr(data, 'name') else 'UNKNOWN',
                        direction=-1,
                        confidence=strength,
                        signal_type='DEATH_CROSS',
                        metadata={
                            'sma_short': float(sma_short.iloc[i]),
                            'sma_long': float(sma_long.iloc[i]),
                            'separation': float(separation),
                            'momentum': float(momentum)
                        }
                    )
                    signals.append(signal)
        
        return signals


class ShortTermPatternSignalGenerator(SignalGenerator):
    """Short-term pattern signal generator combining RSI, Bollinger Bands, and Candlestick patterns"""
    
    def __init__(self, rsi_period: int = 14, bb_period: int = 20, bb_std: float = 2.0, enable_candlestick: bool = True):
        """
        Initialize short-term pattern signal generator
        
        Args:
            rsi_period: RSI calculation period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation
            enable_candlestick: Enable candlestick pattern recognition
        """
        super().__init__()
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.enable_candlestick = enable_candlestick
        self.name = f"ShortTermPattern_RSI{rsi_period}_BB{bb_period}_Candle{enable_candlestick}"
        
        # Initialize sub-generators
        self.rsi_generator = RSISignalGenerator(period=rsi_period)
        self.bb_generator = BollingerBandsSignalGenerator(period=bb_period, std_dev=bb_std)
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate combined short-term pattern signals including candlestick patterns"""
        if len(data) < max(self.rsi_period, self.bb_period):
            return []
        
        # Get signals from individual generators
        rsi_signals = self.rsi_generator.generate_signals(data)
        bb_signals = self.bb_generator.generate_signals(data)
        
        # Get candlestick pattern signals if enabled
        candlestick_signals = []
        if self.enable_candlestick:
            candlestick_signals = self._detect_candlestick_patterns(data)
        
        # Combine all signals
        combined_signals = self._combine_all_signals(data, rsi_signals, bb_signals, candlestick_signals)
        
        return combined_signals
        
        # Combine signals by timestamp
        combined_signals = []
        signal_map = {}
        
        # Map RSI signals
        for signal in rsi_signals:
            signal_map[signal.timestamp] = {'rsi': signal}
        
        # Map BB signals
        for signal in bb_signals:
            if signal.timestamp in signal_map:
                signal_map[signal.timestamp]['bb'] = signal
            else:
                signal_map[signal.timestamp] = {'bb': signal}
        
        # Generate combined signals
        for timestamp, signals_dict in signal_map.items():
            rsi_signal = signals_dict.get('rsi')
            bb_signal = signals_dict.get('bb')
            
            # Combine signal directions and confidences
            combined_direction = 0
            combined_confidence = 0.0
            pattern_count = 0
            
            if rsi_signal:
                combined_direction += rsi_signal.direction
                combined_confidence += rsi_signal.confidence * 0.6  # RSI weight
                pattern_count += 1
            
            if bb_signal:
                combined_direction += bb_signal.direction
                combined_confidence += bb_signal.confidence * 0.4  # BB weight
                pattern_count += 1
            
            if pattern_count > 0:
                # Normalize direction
                final_direction = 1 if combined_direction > 0 else (-1 if combined_direction < 0 else 0)
                final_confidence = min(1.0, combined_confidence)
                
                # Only create signal if confidence is meaningful
                if final_confidence >= 0.3:
                    signal = TradingSignal(
                        timestamp=timestamp,
                        symbol=data.name if hasattr(data, 'name') else 'UNKNOWN',
                        direction=final_direction,
                        confidence=final_confidence,
                        signal_type='SHORT_TERM_PATTERN',
                        metadata={
                            'pattern_count': pattern_count,
                            'rsi_contribution': rsi_signal.confidence if rsi_signal else 0,
                            'bb_contribution': bb_signal.confidence if bb_signal else 0,
                            'combined_strength': final_confidence
                        }
                    )
                    combined_signals.append(signal)
        
        return combined_signals
    
    def _detect_candlestick_patterns(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Detect candlestick patterns using TA-Lib (matches original enhanced_strategy.py)"""
        if not self.enable_candlestick:
            return []
            
        try:
            import talib
        except ImportError:
            print("[WARNING] TA-Lib not available for candlestick patterns")
            return []
        
        signals = []
        
        # Prepare OHLC data
        if len(data.columns) >= 4:
            high = data.iloc[:, 1].values  # High
            low = data.iloc[:, 2].values   # Low
            close = data.iloc[:, 3].values # Close
            open_prices = data.iloc[:, 0].values  # Open or use close shifted
        else:
            # Approximate OHLC from close prices
            close = data.iloc[:, 0].values
            high = close.copy()
            low = close.copy()
            open_prices = np.roll(close, 1)  # Previous close as open
            open_prices[0] = close[0]
        
        try:
            # Key reversal patterns (matching indicators.py)
            patterns = {
                'doji': talib.CDLDOJI(open_prices, high, low, close),
                'hammer': talib.CDLHAMMER(open_prices, high, low, close),
                'hanging_man': talib.CDLHANGINGMAN(open_prices, high, low, close),
                'engulfing': talib.CDLENGULFING(open_prices, high, low, close),
                'morning_star': talib.CDLMORNINGSTAR(open_prices, high, low, close),
                'evening_star': talib.CDLEVENINGSTAR(open_prices, high, low, close),
                'shooting_star': talib.CDLSHOOTINGSTAR(open_prices, high, low, close)
            }
            
            # Generate signals from patterns
            for i in range(1, len(close)):
                timestamp = data.index[i]
                bullish_strength = 0.0
                bearish_strength = 0.0
                detected_patterns = []
                
                # Check each pattern
                for pattern_name, pattern_values in patterns.items():
                    if pattern_values[i] > 0:  # Bullish pattern
                        if pattern_name in ['hammer', 'morning_star']:
                            bullish_strength += 0.3
                        elif pattern_name == 'engulfing' and pattern_values[i] > 0:
                            bullish_strength += 0.4
                        detected_patterns.append(f"bullish_{pattern_name}")
                    elif pattern_values[i] < 0:  # Bearish pattern
                        if pattern_name in ['hanging_man', 'evening_star', 'shooting_star']:
                            bearish_strength += 0.3
                        elif pattern_name == 'engulfing' and pattern_values[i] < 0:
                            bearish_strength += 0.4
                        detected_patterns.append(f"bearish_{pattern_name}")
                    elif pattern_values[i] != 0:  # Neutral patterns like doji
                        if pattern_name == 'doji':
                            detected_patterns.append('doji')
                
                # Create signal if pattern strength is significant
                if bullish_strength >= 0.3:
                    signal = TradingSignal(
                        timestamp=timestamp,
                        symbol=data.name if hasattr(data, 'name') else 'UNKNOWN',
                        direction=1,
                        confidence=min(1.0, bullish_strength),
                        signal_type='CANDLESTICK_BULLISH',
                        metadata={
                            'patterns': detected_patterns,
                            'strength': bullish_strength,
                            'close_price': float(close[i])
                        }
                    )
                    signals.append(signal)
                elif bearish_strength >= 0.3:
                    signal = TradingSignal(
                        timestamp=timestamp,
                        symbol=data.name if hasattr(data, 'name') else 'UNKNOWN',
                        direction=-1,
                        confidence=min(1.0, bearish_strength),
                        signal_type='CANDLESTICK_BEARISH',
                        metadata={
                            'patterns': detected_patterns,
                            'strength': bearish_strength,
                            'close_price': float(close[i])
                        }
                    )
                    signals.append(signal)
        
        except Exception as e:
            print(f"[WARNING] Candlestick pattern detection failed: {e}")
        
        return signals
    
    def _combine_all_signals(self, data: pd.DataFrame, rsi_signals: List[TradingSignal], 
                            bb_signals: List[TradingSignal], candlestick_signals: List[TradingSignal]) -> List[TradingSignal]:
        """Combine RSI, Bollinger Band, and Candlestick signals"""
        combined_signals = []
        
        # Create timestamp-based signal mapping
        signal_map = {}
        
        # Map RSI signals
        for signal in rsi_signals:
            signal_map[signal.timestamp] = {'rsi': signal}
        
        # Map BB signals
        for signal in bb_signals:
            if signal.timestamp in signal_map:
                signal_map[signal.timestamp]['bb'] = signal
            else:
                signal_map[signal.timestamp] = {'bb': signal}
        
        # Map Candlestick signals
        for signal in candlestick_signals:
            if signal.timestamp in signal_map:
                signal_map[signal.timestamp]['candlestick'] = signal
            else:
                signal_map[signal.timestamp] = {'candlestick': signal}
        
        # Generate combined signals
        for timestamp, signals_dict in signal_map.items():
            rsi_signal = signals_dict.get('rsi')
            bb_signal = signals_dict.get('bb')
            candlestick_signal = signals_dict.get('candlestick')
            
            # Combine signal directions and confidences with weights
            combined_direction = 0
            combined_confidence = 0.0
            pattern_count = 0
            
            if rsi_signal:
                combined_direction += rsi_signal.direction
                combined_confidence += rsi_signal.confidence * 0.4  # RSI weight (reduced)
                pattern_count += 1
            
            if bb_signal:
                combined_direction += bb_signal.direction
                combined_confidence += bb_signal.confidence * 0.3  # BB weight (reduced)
                pattern_count += 1
            
            if candlestick_signal:
                combined_direction += candlestick_signal.direction
                combined_confidence += candlestick_signal.confidence * 0.3  # Candlestick weight
                pattern_count += 1
            
            if pattern_count > 0:
                # Normalize direction
                final_direction = 1 if combined_direction > 0 else (-1 if combined_direction < 0 else 0)
                final_confidence = min(1.0, combined_confidence)
                
                # Only create signal if confidence is meaningful
                if final_confidence >= 0.3:
                    signal = TradingSignal(
                        timestamp=timestamp,
                        symbol=data.name if hasattr(data, 'name') else 'UNKNOWN',
                        direction=final_direction,
                        confidence=final_confidence,
                        signal_type='SHORT_TERM_PATTERN',
                        metadata={
                            'pattern_count': pattern_count,
                            'rsi_contribution': rsi_signal.confidence if rsi_signal else 0,
                            'bb_contribution': bb_signal.confidence if bb_signal else 0,
                            'candlestick_contribution': candlestick_signal.confidence if candlestick_signal else 0,
                            'combined_strength': final_confidence
                        }
                    )
                    combined_signals.append(signal)
        
        return combined_signals


class EnhancedMLTradingStrategy(TradingStrategy):
    """
    Enhanced ML Trading Strategy with all features from enhanced_strategy.py
    
    Features:
    - Intelligent order sizing with multiple strategies
    - Golden Cross/Death Cross analysis
    - Short-term pattern detection
    - Market context analysis
    - ML model management
    - Comprehensive performance analytics
    """
    
    def __init__(self, 
                 symbol: str,
                 data_provider: YFinanceProvider,
                 model_manager: ModelManagerInterface,
                 order_sizing_config: OrderSizingConfig = None,
                 golden_cross_config: Dict = None,
                 short_term_config: Dict = None,
                 starting_portfolio_value: float = 100000):
        """
        Initialize Enhanced ML Trading Strategy
        
        Args:
            symbol: Trading symbol
            data_provider: Data provider for market data
            model_manager: ML model manager
            order_sizing_config: Order sizing configuration
            golden_cross_config: Golden Cross configuration
            short_term_config: Short-term pattern configuration
            starting_portfolio_value: Starting portfolio value
        """
        super().__init__()
        
        self.symbol = symbol
        self.data_provider = data_provider
        self.model_manager = model_manager
        self.starting_portfolio_value = starting_portfolio_value
        
        # Initialize order sizing
        self.order_sizing_config = order_sizing_config or OrderSizingConfig()
        self.order_size_manager = AutoOrderSizeManager(
            config=self.order_sizing_config,
            starting_portfolio_value=starting_portfolio_value
        )
        
        # Initialize signal generators
        self._initialize_signal_generators(golden_cross_config, short_term_config)
        
        # Performance tracking
        self.performance_metrics = {}
        self.trade_history = []
        
        print(f"[INIT] Enhanced ML Trading Strategy initialized for {symbol}")
        print(f"       Order Sizing: {self.order_sizing_config.strategy.value}")
        print(f"       Starting Value: ${starting_portfolio_value:,.0f}")
    
    def _initialize_signal_generators(self, golden_cross_config: Dict = None, 
                                    short_term_config: Dict = None):
        """Initialize signal generators"""
        self.signal_generators = []
        
        # Golden Cross generator
        if golden_cross_config is None:
            golden_cross_config = {
                'enabled': True,
                'short_window': 20,
                'long_window': 50,
                'strength_threshold': 0.6,
                'confirmation_days': 3
            }
        
        if golden_cross_config.get('enabled', True):
            self.golden_cross_generator = GoldenCrossSignalGenerator(
                short_window=golden_cross_config.get('short_window', 20),
                long_window=golden_cross_config.get('long_window', 50),
                strength_threshold=golden_cross_config.get('strength_threshold', 0.6),
                confirmation_days=golden_cross_config.get('confirmation_days', 3)
            )
            self.signal_generators.append(self.golden_cross_generator)
        
        # Short-term pattern generator
        if short_term_config is None:
            short_term_config = {
                'enabled': True,
                'rsi_period': 14,
                'bb_period': 20,
                'bb_std': 2.0
            }
        
        if short_term_config.get('enabled', True):
            self.short_term_generator = ShortTermPatternSignalGenerator(
                rsi_period=short_term_config.get('rsi_period', 14),
                bb_period=short_term_config.get('bb_period', 20),
                bb_std=short_term_config.get('bb_std', 2.0)
            )
            self.signal_generators.append(self.short_term_generator)
        
        # Add individual technical generators
        self.signal_generators.extend([
            RSISignalGenerator(period=14),
            MACDSignalGenerator(fast_period=12, slow_period=26, signal_period=9),
            BollingerBandsSignalGenerator(period=20, std_dev=2.0)
        ])
    
    def generate_signal(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> TradingSignal:
        """Generate trading signal using all available generators and ML model"""
        # Get signals from all generators
        all_signals = []
        for generator in self.signal_generators:
            try:
                signals = generator.generate_signals(data)
                # Get the most recent signal
                if signals:
                    latest_signal = max(signals, key=lambda s: s.timestamp)
                    if latest_signal.timestamp <= timestamp:
                        all_signals.append(latest_signal)
            except Exception as e:
                print(f"[WARNING] Signal generator {generator.name} failed: {e}")
        
        # Combine signals intelligently
        final_signal = self._combine_signals(all_signals, data, timestamp)
        
        # Apply ML model prediction if available
        if self.model_manager and hasattr(self.model_manager, 'predict'):
            try:
                ml_prediction = self._get_ml_prediction(data, timestamp)
                if ml_prediction:
                    final_signal = self._enhance_signal_with_ml(final_signal, ml_prediction)
            except Exception as e:
                print(f"[WARNING] ML prediction failed: {e}")
        
        return final_signal
    
    def _combine_signals(self, signals: List[TradingSignal], data: pd.DataFrame, 
                        timestamp: pd.Timestamp) -> TradingSignal:
        """Combine multiple signals intelligently"""
        if not signals:
            return TradingSignal(
                timestamp=timestamp,
                symbol=self.symbol,
                direction=0,
                confidence=0.0,
                signal_type='COMBINED_NO_SIGNAL',
                metadata={'reason': 'No signals generated'}
            )
        
        # Weight different signal types
        signal_weights = {
            'GOLDEN_CROSS': 0.3,
            'DEATH_CROSS': 0.3,
            'SHORT_TERM_PATTERN': 0.2,
            'RSI': 0.1,
            'MACD': 0.1,
            'BOLLINGER_BANDS': 0.1
        }
        
        # Calculate weighted average
        total_direction = 0.0
        total_confidence = 0.0
        signal_contributions = {}
        
        for signal in signals:
            weight = signal_weights.get(signal.signal_type, 0.1)
            weighted_contribution = signal.direction * signal.confidence * weight
            total_direction += weighted_contribution
            total_confidence += signal.confidence * weight
            
            signal_contributions[signal.signal_type] = {
                'direction': signal.direction,
                'confidence': signal.confidence,
                'weight': weight,
                'contribution': weighted_contribution
            }
        
        # Determine final direction
        final_direction = 1 if total_direction > 0.1 else (-1 if total_direction < -0.1 else 0)
        final_confidence = min(1.0, abs(total_confidence))
        
        return TradingSignal(
            timestamp=timestamp,
            symbol=self.symbol,
            direction=final_direction,
            confidence=final_confidence,
            signal_type='COMBINED_ENHANCED',
            metadata={
                'signal_count': len(signals),
                'contributions': signal_contributions,
                'total_weighted_direction': total_direction,
                'combination_method': 'weighted_average'
            }
        )
    
    def _get_ml_prediction(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> Optional[TradingSignal]:
        """Get ML model prediction if available"""
        # This would interface with the ML model manager
        # For now, return None - implement based on model_manager interface
        return None
    
    def _enhance_signal_with_ml(self, base_signal: TradingSignal, 
                               ml_signal: TradingSignal) -> TradingSignal:
        """Enhance base signal with ML prediction"""
        # Combine base signal with ML prediction
        combined_direction = (base_signal.direction * 0.4 + ml_signal.direction * 0.6)
        final_direction = 1 if combined_direction > 0.2 else (-1 if combined_direction < -0.2 else 0)
        
        combined_confidence = (base_signal.confidence * 0.4 + ml_signal.confidence * 0.6)
        
        return TradingSignal(
            timestamp=base_signal.timestamp,
            symbol=self.symbol,
            direction=final_direction,
            confidence=combined_confidence,
            signal_type='ML_ENHANCED',
            metadata={
                'base_signal': base_signal.metadata,
                'ml_signal': ml_signal.metadata,
                'enhancement_method': 'ml_weighted_combination'
            }
        )
    
    def execute_trade(self, signal: TradingSignal, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute trade with intelligent order sizing"""
        if signal.direction == 0:
            return {'executed': False, 'reason': 'HOLD signal'}
        
        # Calculate order size
        order_size = self.order_size_manager.calculate_order_size(
            symbol=self.symbol,
            signal=signal,
            stock_data=data
        )
        
        if abs(order_size) < 1:
            return {'executed': False, 'reason': 'Order size too small'}
        
        # Get current price
        current_price = float(data.iloc[-1])
        
        # Create trade record
        trade = {
            'timestamp': signal.timestamp,
            'symbol': self.symbol,
            'direction': 'BUY' if signal.direction > 0 else 'SELL',
            'shares': abs(order_size),
            'price': current_price,
            'value': abs(order_size) * current_price,
            'signal_type': signal.signal_type,
            'confidence': signal.confidence,
            'order_sizing_strategy': self.order_sizing_config.strategy.value,
            'metadata': signal.metadata
        }
        
        # Update position tracking
        self.order_size_manager.update_position(self.symbol, order_size, current_price)
        self.trade_history.append(trade)
        
        print(f"[TRADE] {trade['direction']} {trade['shares']} shares of {self.symbol} at ${current_price:.2f}")
        print(f"        Signal: {signal.signal_type} (confidence: {signal.confidence:.2f})")
        print(f"        Order Size Strategy: {self.order_sizing_config.strategy.value}")
        
        return {
            'executed': True,
            'trade': trade,
            'new_position': self.order_size_manager.get_current_position(self.symbol)
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.trade_history:
            return {'total_trades': 0, 'performance': 'No trades executed'}
        
        # Calculate basic metrics
        total_trades = len(self.trade_history)
        buy_trades = len([t for t in self.trade_history if t['direction'] == 'BUY'])
        sell_trades = len([t for t in self.trade_history if t['direction'] == 'SELL'])
        
        # Calculate P&L (simplified)
        total_invested = sum(t['value'] for t in self.trade_history if t['direction'] == 'BUY')
        total_received = sum(t['value'] for t in self.trade_history if t['direction'] == 'SELL')
        unrealized_pnl = total_received - total_invested
        
        # Order sizing analysis
        order_sizes = [t['shares'] for t in self.trade_history]
        avg_order_size = np.mean(order_sizes) if order_sizes else 0
        order_size_std = np.std(order_sizes) if len(order_sizes) > 1 else 0
        
        return {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_invested': total_invested,
            'total_received': total_received,
            'unrealized_pnl': unrealized_pnl,
            'avg_order_size': avg_order_size,
            'order_size_std': order_size_std,
            'order_sizing_strategy': self.order_sizing_config.strategy.value,
            'current_position': self.order_size_manager.get_current_position(self.symbol),
            'trade_history': self.trade_history[-10:]  # Last 10 trades
        }