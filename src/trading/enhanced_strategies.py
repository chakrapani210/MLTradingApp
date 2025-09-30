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

from src.interfaces.trading_strategy import TradingStrategy, TradingSignal, Position, Order, OrderType
from src.interfaces.signal_generator import SignalGenerator, SignalType
from src.interfaces.model_manager import ModelManagerInterface
from src.interfaces.risk_manager import RiskManager
from src.data.providers import YFinanceProvider
from src.signals.technical import (
    RSISignalGenerator, MACDSignalGenerator, BollingerBandsSignalGenerator,
    SMACrossoverSignalGenerator, EMASignalGenerator, VolumeAnalysisSignalGenerator
)

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
        super().__init__(name=f"GoldenCross_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
        self.strength_threshold = strength_threshold
        self.confirmation_days = confirmation_days
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """Generate golden cross/death cross signals"""
        if len(data) < self.long_window:
            return []
        
        # Extract close price column for MA calculation
        close_data = data.iloc[:, 0]  # First column is typically close price
        
        # Calculate moving averages
        sma_short = close_data.rolling(window=self.short_window).mean()
        sma_long = close_data.rolling(window=self.long_window).mean()
        
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
                        symbol=symbol,
                        timestamp=current_date,
                        signal_type=SignalType.BUY,
                        confidence=strength,
                        strength=strength,
                        source='GOLDEN_CROSS',
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
                        symbol=symbol,
                        timestamp=current_date,
                        signal_type=SignalType.SELL,
                        confidence=strength,
                        strength=strength,
                        source='DEATH_CROSS',
                        metadata={
                            'sma_short': float(sma_short.iloc[i]),
                            'sma_long': float(sma_long.iloc[i]),
                            'separation': float(separation),
                            'momentum': float(momentum)
                        }
                    )
                    signals.append(signal)
        
        return signals
    
    def get_required_columns(self) -> List[str]:
        """Get required DataFrame columns for golden cross analysis"""
        return ['close']  # Only need close prices for moving averages
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that input data has required structure"""
        required_columns = self.get_required_columns()
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check minimum data length
        if len(data) < self.long_window:
            return False
            
        # Check for valid numeric data
        try:
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    return False
            return True
        except Exception:
            return False


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
        super().__init__(name=f"ShortTermPattern_RSI{rsi_period}_BB{bb_period}_Candle{enable_candlestick}")
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.enable_candlestick = enable_candlestick
        
        # Initialize sub-generators
        self.rsi_generator = RSISignalGenerator(config={'period': rsi_period})
        self.bb_generator = BollingerBandsSignalGenerator(config={'period': bb_period, 'std_dev': bb_std})
    
    def _signal_to_direction(self, signal: TradingSignal) -> int:
        """Convert TradingSignal to direction for compatibility"""
        if signal.signal_type == SignalType.BUY:
            return 1
        elif signal.signal_type == SignalType.SELL:
            return -1
        else:
            return 0
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """Generate combined short-term pattern signals including candlestick patterns"""
        if len(data) < max(self.rsi_period, self.bb_period):
            return []
        
        # Get signals from individual generators
        rsi_signals = self.rsi_generator.generate_signals(data, symbol)
        bb_signals = self.bb_generator.generate_signals(data, symbol)
        
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
                combined_direction += self._signal_to_direction(rsi_signal)
                combined_confidence += rsi_signal.confidence * 0.6  # RSI weight
                pattern_count += 1
            
            if bb_signal:
                combined_direction += self._signal_to_direction(bb_signal)
                combined_confidence += bb_signal.confidence * 0.4  # BB weight
                pattern_count += 1
            
            if pattern_count > 0:
                # Normalize direction
                final_direction = 1 if combined_direction > 0 else (-1 if combined_direction < 0 else 0)
                final_confidence = min(1.0, combined_confidence)
                
                # Only create signal if confidence is meaningful
                if final_confidence >= 0.3:
                    # Convert direction to signal type
                    if final_direction > 0:
                        signal_type = SignalType.BUY
                    elif final_direction < 0:
                        signal_type = SignalType.SELL
                    else:
                        signal_type = SignalType.HOLD
                    
                    signal = TradingSignal(
                        timestamp=timestamp,
                        symbol=data.name if hasattr(data, 'name') else 'UNKNOWN',
                        signal_type=signal_type,
                        confidence=final_confidence,
                        strength=abs(final_direction),
                        source='SHORT_TERM_PATTERN',
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
                        signal_type=SignalType.BUY,
                        confidence=min(1.0, bullish_strength),
                        strength=bullish_strength,
                        source='CANDLESTICK_BULLISH',
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
                        signal_type=SignalType.SELL,
                        confidence=min(1.0, bearish_strength),
                        strength=bearish_strength,
                        source='CANDLESTICK_BEARISH',
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
                combined_direction += self._signal_to_direction(rsi_signal)
                combined_confidence += rsi_signal.confidence * 0.4  # RSI weight (reduced)
                pattern_count += 1
            
            if bb_signal:
                combined_direction += self._signal_to_direction(bb_signal)
                combined_confidence += bb_signal.confidence * 0.3  # BB weight (reduced)
                pattern_count += 1
            
            if candlestick_signal:
                combined_direction += self._signal_to_direction(candlestick_signal)
                combined_confidence += candlestick_signal.confidence * 0.3  # Candlestick weight
                pattern_count += 1
            
            if pattern_count > 0:
                # Normalize direction
                final_direction = 1 if combined_direction > 0 else (-1 if combined_direction < 0 else 0)
                final_confidence = min(1.0, combined_confidence)
                
                # Only create signal if confidence is meaningful
                if final_confidence >= 0.3:
                    # Convert direction to signal type
                    if final_direction > 0:
                        signal_type = SignalType.BUY
                    elif final_direction < 0:
                        signal_type = SignalType.SELL
                    else:
                        signal_type = SignalType.HOLD
                    
                    signal = TradingSignal(
                        timestamp=timestamp,
                        symbol=data.name if hasattr(data, 'name') else 'UNKNOWN',
                        signal_type=signal_type,
                        confidence=final_confidence,
                        strength=abs(final_direction),
                        source='SHORT_TERM_PATTERN',
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
    
    def get_required_columns(self) -> List[str]:
        """Get required DataFrame columns for short-term pattern analysis"""
        return ['open', 'high', 'low', 'close']  # Need OHLC for RSI, BB, and candlestick patterns
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that input data has required structure"""
        required_columns = self.get_required_columns()
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check minimum data length
        min_length = max(self.rsi_period, self.bb_period)
        if len(data) < min_length:
            return False
            
        # Check for valid numeric data
        try:
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    return False
            return True
        except Exception:
            return False


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
        super().__init__(
            name=f"EnhancedMLTradingStrategy_{symbol}",
            config={
                'symbol': symbol,
                'order_sizing_strategy': order_sizing_config.strategy.value if order_sizing_config else 'percentage',
                'starting_portfolio_value': starting_portfolio_value,
                'golden_cross_config': golden_cross_config,
                'short_term_config': short_term_config
            }
        )
        
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
        
        # Add individual technical generators with configuration
        technical_config = self.config.get('technical_generators', {})
        
        # RSI Generator
        rsi_config = technical_config.get('rsi', {'period': 14})
        if rsi_config.get('enabled', True):
            self.signal_generators.append(RSISignalGenerator(config=rsi_config))
        
        # MACD Generator  
        macd_config = technical_config.get('macd', {
            'fast_period': 12, 'slow_period': 26, 'signal_period': 9
        })
        if macd_config.get('enabled', True):
            self.signal_generators.append(MACDSignalGenerator(config=macd_config))
        
        # Bollinger Bands Generator
        bb_config = technical_config.get('bollinger_bands', {'period': 20, 'std_dev': 2.0})
        if bb_config.get('enabled', True):
            self.signal_generators.append(BollingerBandsSignalGenerator(config=bb_config))
        
        # SMA Crossover Generator
        sma_config = technical_config.get('sma_crossover', {
            'short_period': 20, 'long_period': 50, 
            'signal_strength_threshold': 0.5,
            'confirmation_periods': 2
        })
        if sma_config.get('enabled', True):
            self.signal_generators.append(SMACrossoverSignalGenerator(config=sma_config))
        
        # EMA Generator
        ema_config = technical_config.get('ema', {
            'periods': [12, 26], 
            'crossover_pairs': [(12, 26)],
            'slope_threshold': 0.001,
            'min_confidence': 0.3
        })
        if ema_config.get('enabled', True):
            self.signal_generators.append(EMASignalGenerator(config=ema_config))
        
        # Volume Analysis Generator
        volume_config = technical_config.get('volume_analysis', {
            'volume_surge_threshold': 2.0,
            'volume_sma_period': 20,
            'obv_period': 10,
            'price_volume_confirmation': True,
            'min_confidence': 0.4
        })
        if volume_config.get('enabled', True):
            self.signal_generators.append(VolumeAnalysisSignalGenerator(config=volume_config))
        
        print(f"[INIT] Initialized {len(self.signal_generators)} signal generators:")
        for generator in self.signal_generators:
            print(f"       - {generator.name}")
    
    def _signal_to_direction(self, signal: TradingSignal) -> int:
        """
        Convert new TradingSignal format to direction for compatibility
        
        Returns:
            1 for BUY, -1 for SELL, 0 for HOLD
        """
        if signal.signal_type == SignalType.BUY:
            return 1
        elif signal.signal_type == SignalType.SELL:
            return -1
        else:
            return 0
    
    def generate_signal(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> TradingSignal:
        """Generate trading signal using all available generators and ML model"""
        # Get signals from all generators
        all_signals = []
        for generator in self.signal_generators:
            try:
                signals = generator.generate_signals(data, self.symbol)
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
                symbol=self.symbol,
                timestamp=timestamp,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                strength=0.0,
                source='COMBINED_NO_SIGNAL',
                metadata={'reason': 'No signals generated'}
            )
        
        # Enhanced weight system for all signal types
        signal_weights = {
            # Golden Cross/Death Cross patterns
            'GOLDEN_CROSS': 0.20,
            'DEATH_CROSS': 0.20,
            'SMA_GOLDEN_CROSS': 0.18,
            'SMA_DEATH_CROSS': 0.18,
            
            # EMA signals
            'EMA_BULLISH_CROSS': 0.15,
            'EMA_BEARISH_CROSS': 0.15,
            
            # Volume signals (important confirmation)
            'VOLUME_BREAKOUT': 0.25,
            'OBV_BULLISH_DIVERGENCE': 0.15,
            'OBV_BEARISH_DIVERGENCE': 0.15,
            
            # Short-term patterns
            'SHORT_TERM_PATTERN': 0.12,
            
            # Individual technical indicators
            'RSI_OVERSOLD': 0.08,
            'RSI_OVERBOUGHT': 0.08,
            'MACD_BULLISH': 0.10,
            'MACD_BEARISH': 0.10,
            'BOLLINGER_UPPER': 0.08,
            'BOLLINGER_LOWER': 0.08,
            
            # Default weight for unknown signal types
            'DEFAULT': 0.05
        }
        
        # Calculate weighted average
        total_direction = 0.0
        total_confidence = 0.0
        signal_contributions = {}
        
        for signal in signals:
            weight = signal_weights.get(signal.signal_type, signal_weights['DEFAULT'])
            weighted_contribution = self._signal_to_direction(signal) * signal.confidence * weight
            total_direction += weighted_contribution
            total_confidence += signal.confidence * weight
            
            signal_contributions[signal.signal_type] = {
                'direction': self._signal_to_direction(signal),
                'confidence': signal.confidence,
                'weight': weight,
                'contribution': weighted_contribution
            }
        
        # Determine final direction
        final_direction = 1 if total_direction > 0.1 else (-1 if total_direction < -0.1 else 0)
        final_confidence = min(1.0, abs(total_confidence))
        
        # Convert direction to signal type
        if final_direction > 0:
            signal_type = SignalType.BUY
        elif final_direction < 0:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        return TradingSignal(
            symbol=self.symbol,
            timestamp=timestamp,
            signal_type=signal_type,
            confidence=final_confidence,
            strength=abs(total_direction),
            source='COMBINED_ENHANCED',
            metadata={
                'signal_count': len(signals),
                'contributions': signal_contributions,
                'total_weighted_direction': total_direction,
                'combination_method': 'weighted_average'
            }
        )
    
    def _get_ml_prediction(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> Optional[TradingSignal]:
        """Get ML model prediction if available"""
        try:
            # Check if model exists for this symbol
            if not self.model_manager.model_exists(self.symbol):
                return None
            
            # Prepare features for ML prediction
            features = self._prepare_ml_features(data)
            if features is None:
                return None
            
            # Get ML prediction
            prediction_result = self.model_manager.predict(
                symbol=self.symbol,
                features=features.reshape(1, -1)
            )
            
            if not prediction_result.predictions:
                return None
            
            prediction = prediction_result.predictions[0]
            confidence = prediction_result.confidence
            
            # Only generate signal if prediction is confident enough
            if abs(prediction) > 0 and confidence >= 0.6:
                return TradingSignal(
                    symbol=self.symbol,
                    timestamp=timestamp,
                    signal_type=SignalType.BUY if prediction > 0 else SignalType.SELL,
                    confidence=confidence,
                    strength=abs(float(prediction)),
                    source='ML_PREDICTION',
                    metadata={
                        'model_version': prediction_result.model_version,
                        'feature_importance': prediction_result.feature_importance,
                        'prediction_confidence': confidence,
                        'model_type': 'ML',
                        'prediction_metadata': prediction_result.prediction_metadata,
                        'prediction_value': prediction
                    }
                )
            
            return None
            
        except Exception as e:
            print(f"[ML_PREDICTION] Error getting ML prediction for {self.symbol}: {e}")
            return None
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for ML model prediction"""
        try:
            if len(data) < 20:  # Need minimum data for feature calculation
                return None
            
            # Use the last 20 periods for feature calculation
            recent_data = data.tail(20)
            # Extract scalar values from pandas Series
            close_col = 'Close' if 'Close' in data.columns else data.columns[0]
            current_price = recent_data[close_col].iloc[-1]
            
            features = []
            
            # Price-based features
            price_change_1d = (recent_data[close_col].iloc[-1] - recent_data[close_col].iloc[-2]) / recent_data[close_col].iloc[-2] if len(recent_data) >= 2 else 0
            price_change_5d = (recent_data[close_col].iloc[-1] - recent_data[close_col].iloc[-6]) / recent_data[close_col].iloc[-6] if len(recent_data) >= 6 else 0
            
            # Moving averages
            sma_5 = recent_data[close_col].tail(5).mean() if len(recent_data) >= 5 else current_price
            sma_10 = recent_data[close_col].tail(10).mean() if len(recent_data) >= 10 else current_price
            
            # Volatility
            volatility = recent_data[close_col].tail(10).std() if len(recent_data) >= 10 else 0.02
            
            # RSI calculation
            returns = recent_data[close_col].pct_change().dropna()
            if len(returns) >= 14:
                up_moves = returns.where(returns > 0, 0)
                down_moves = returns.where(returns < 0, 0).abs()
                avg_up = up_moves.tail(14).mean()
                avg_down = down_moves.tail(14).mean()
                rs = avg_up / avg_down if avg_down != 0 else 100
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50
            
            # Price position within range
            price_min = recent_data[close_col].min()
            price_max = recent_data[close_col].max()
            price_position = (current_price - price_min) / (price_max - price_min) if price_max != price_min else 0.5
            
            # Assemble feature vector
            features = [
                price_change_1d,
                price_change_5d, 
                (sma_5 - current_price) / current_price,
                (sma_10 - current_price) / current_price,
                volatility,
                rsi / 100.0,  # Normalize RSI
                price_position
            ]
            
            # Ensure no NaN values
            features = [0.0 if pd.isna(f) else float(f) for f in features]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"[ML_FEATURES] Error preparing features for {self.symbol}: {e}")
            return None
    
    def _enhance_signal_with_ml(self, base_signal: TradingSignal, 
                               ml_signal: TradingSignal) -> TradingSignal:
        """Enhance base signal with ML prediction"""
        # Convert signal types to numeric values for combination
        base_strength = base_signal.strength if base_signal.signal_type == SignalType.BUY else -base_signal.strength
        ml_strength = ml_signal.strength if ml_signal.signal_type == SignalType.BUY else -ml_signal.strength
        
        # Combine signals (60% ML, 40% technical)
        combined_strength = (base_strength * 0.4 + ml_strength * 0.6)
        combined_confidence = (base_signal.confidence * 0.4 + ml_signal.confidence * 0.6)
        
        # Determine final signal type
        if combined_strength > 0.2:
            final_signal_type = SignalType.BUY
            final_strength = abs(combined_strength)
        elif combined_strength < -0.2:
            final_signal_type = SignalType.SELL
            final_strength = abs(combined_strength)
        else:
            final_signal_type = SignalType.HOLD
            final_strength = 0.0
        
        return TradingSignal(
            symbol=self.symbol,
            timestamp=base_signal.timestamp,
            signal_type=final_signal_type,
            confidence=combined_confidence,
            strength=final_strength,
            source='ML_ENHANCED',
            metadata={
                'base_signal': base_signal.metadata,
                'ml_signal': ml_signal.metadata,
                'enhancement_method': 'ml_weighted_combination',
                'combination_weights': {'technical': 0.4, 'ml': 0.6}
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
    
    # Abstract method implementations required by TradingStrategy interface
    def generate_orders(self, signals: List[TradingSignal], 
                       market_data: pd.DataFrame,
                       positions: Dict[str, Position]) -> List[Order]:
        """Generate orders based on trading signals"""
        orders = []
        
        for signal in signals:
            if signal.direction == 0:  # HOLD signal
                continue
                
            # Get current price
            current_price = float(market_data.iloc[-1]['close'])
            
            # Get current position for this symbol
            current_position = positions.get(signal.symbol)
            
            # Calculate position size
            account_value = self.starting_portfolio_value  # Simplified
            position_size = self.calculate_position_size(
                signal, current_price, account_value, current_position
            )
            
            if abs(position_size) >= 1:  # Only create order if size is meaningful
                order = Order(
                    symbol=signal.symbol,
                    order_type=OrderType.MARKET,
                    side='BUY' if signal.direction > 0 else 'SELL',
                    quantity=abs(position_size),
                    price=current_price,
                    timestamp=signal.timestamp,
                    metadata={
                        'signal_type': signal.signal_type,
                        'confidence': signal.confidence,
                        'order_sizing_strategy': self.order_sizing_config.strategy.value
                    }
                )
                orders.append(order)
        
        return orders
    
    def calculate_position_size(self, signal: TradingSignal, 
                               current_price: float,
                               account_value: float,
                               current_position: Optional[Position] = None) -> int:
        """Calculate position size for a trade"""
        # Use the order size manager for intelligent sizing
        return self.order_size_manager.calculate_order_size(
            symbol=signal.symbol,
            signal=signal,
            stock_data=pd.DataFrame({'close': [current_price]})  # Simplified data
        )
    
    def validate_order(self, order, 
                      market_data: pd.DataFrame,
                      positions: Dict[str, Position],
                      account_value: float) -> bool:
        """Validate an order before execution"""
        # Basic validation checks
        
        # Check if we have valid price data
        if market_data is None or market_data.empty:
            return False
            
        # Check order quantity is positive
        if not hasattr(order, 'quantity') or order.quantity <= 0:
            return False
            
        # Check if we have enough account value for buy orders
        if hasattr(order, 'side') and order.side == 'BUY':
            order_value = order.quantity * order.price
            if order_value > account_value * 0.95:  # Don't use more than 95% of account
                return False
        
        # Check if we have enough shares for sell orders
        if hasattr(order, 'side') and order.side == 'SELL':
            current_position = positions.get(order.symbol)
            if current_position is None or current_position.quantity < order.quantity:
                return False
        
        return True