"""
Technical Signal Generators
Concrete implementations of technical analysis based signal generators
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import talib
import logging

from ..interfaces.signal_generator import SignalGenerator, TradingSignal, SignalType

logger = logging.getLogger(__name__)


class RSISignalGenerator(SignalGenerator):
    """
    RSI-based signal generator
    
    Generates BUY signals when RSI < oversold_threshold (default 30)
    Generates SELL signals when RSI > overbought_threshold (default 70)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize RSI signal generator
        
        Args:
            config: Configuration with 'period', 'oversold_threshold', 'overbought_threshold'
        """
        super().__init__("RSI Signal Generator", config)
        self.period = self.config.get('period', 14)
        self.oversold_threshold = self.config.get('oversold_threshold', 30)
        self.overbought_threshold = self.config.get('overbought_threshold', 70)
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """
        Generate RSI-based trading signals
        
        Args:
            data: Market data with OHLCV columns
            symbol: Trading symbol
            
        Returns:
            List of TradingSignal objects
        """
        try:
            if not self.validate_data(data):
                return []
            
            signals = []
            
            # Calculate RSI
            rsi = talib.RSI(data['close'].values, timeperiod=self.period)
            
            for i in range(len(data)):
                if pd.isna(rsi[i]):
                    continue
                
                timestamp = data.index[i]
                rsi_value = rsi[i]
                
                # Generate signals based on RSI thresholds
                if rsi_value < self.oversold_threshold:
                    # Oversold - BUY signal
                    confidence = (self.oversold_threshold - rsi_value) / self.oversold_threshold
                    signals.append(TradingSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.BUY,
                        confidence=min(confidence, 1.0),
                        strength=confidence,
                        source="RSI",
                        metadata={
                            'rsi_value': rsi_value,
                            'threshold': self.oversold_threshold,
                            'condition': 'oversold'
                        }
                    ))
                
                elif rsi_value > self.overbought_threshold:
                    # Overbought - SELL signal
                    confidence = (rsi_value - self.overbought_threshold) / (100 - self.overbought_threshold)
                    signals.append(TradingSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.SELL,
                        confidence=min(confidence, 1.0),
                        strength=confidence,
                        source="RSI",
                        metadata={
                            'rsi_value': rsi_value,
                            'threshold': self.overbought_threshold,
                            'condition': 'overbought'
                        }
                    ))
            
            logger.info(f"Generated {len(signals)} RSI signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating RSI signals for {symbol}: {str(e)}")
            return []
    
    def get_required_columns(self) -> List[str]:
        """Get required DataFrame columns"""
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if data is valid, False otherwise
        """
        if data.empty:
            logger.warning("Empty data provided to RSI generator")
            return False
        
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns for RSI: {missing_cols}")
            return False
        
        if len(data) < self.period + 5:  # Need enough data for RSI calculation
            logger.warning(f"Insufficient data for RSI calculation (need {self.period + 5}, got {len(data)})")
            return False
        
        return True


class MACDSignalGenerator(SignalGenerator):
    """
    MACD-based signal generator
    
    Generates BUY signals on bullish MACD crossover
    Generates SELL signals on bearish MACD crossover
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize MACD signal generator
        
        Args:
            config: Configuration with 'fast_period', 'slow_period', 'signal_period'
        """
        super().__init__("MACD Signal Generator", config)
        self.fast_period = self.config.get('fast_period', 12)
        self.slow_period = self.config.get('slow_period', 26)
        self.signal_period = self.config.get('signal_period', 9)
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """
        Generate MACD-based trading signals
        
        Args:
            data: Market data with OHLCV columns
            symbol: Trading symbol
            
        Returns:
            List of TradingSignal objects
        """
        try:
            if not self.validate_data(data):
                return []
            
            signals = []
            
            # Calculate MACD
            macd, macd_signal, macd_histogram = talib.MACD(
                data['close'].values,
                fastperiod=self.fast_period,
                slowperiod=self.slow_period,
                signalperiod=self.signal_period
            )
            
            # Detect crossovers
            for i in range(1, len(data)):
                if pd.isna(macd[i]) or pd.isna(macd_signal[i]) or pd.isna(macd[i-1]) or pd.isna(macd_signal[i-1]):
                    continue
                
                timestamp = data.index[i]
                
                # Bullish crossover: MACD crosses above signal line
                if macd[i-1] <= macd_signal[i-1] and macd[i] > macd_signal[i]:
                    confidence = min(abs(macd[i] - macd_signal[i]) / abs(macd_signal[i]), 1.0) if macd_signal[i] != 0 else 0.5
                    signals.append(TradingSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.BUY,
                        confidence=confidence,
                        strength=confidence,
                        source="MACD",
                        metadata={
                            'macd_value': macd[i],
                            'signal_value': macd_signal[i],
                            'histogram': macd_histogram[i],
                            'crossover_type': 'bullish'
                        }
                    ))
                
                # Bearish crossover: MACD crosses below signal line
                elif macd[i-1] >= macd_signal[i-1] and macd[i] < macd_signal[i]:
                    confidence = min(abs(macd[i] - macd_signal[i]) / abs(macd_signal[i]), 1.0) if macd_signal[i] != 0 else 0.5
                    signals.append(TradingSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.SELL,
                        confidence=confidence,
                        strength=confidence,
                        source="MACD",
                        metadata={
                            'macd_value': macd[i],
                            'signal_value': macd_signal[i],
                            'histogram': macd_histogram[i],
                            'crossover_type': 'bearish'
                        }
                    ))
            
            logger.info(f"Generated {len(signals)} MACD signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating MACD signals for {symbol}: {str(e)}")
            return []
    
    def get_required_columns(self) -> List[str]:
        """Get required DataFrame columns"""
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if data is valid, False otherwise
        """
        if data.empty:
            logger.warning("Empty data provided to MACD generator")
            return False
        
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns for MACD: {missing_cols}")
            return False
        
        min_periods = self.slow_period + self.signal_period + 5
        if len(data) < min_periods:
            logger.warning(f"Insufficient data for MACD calculation (need {min_periods}, got {len(data)})")
            return False
        
        return True


class BollingerBandsSignalGenerator(SignalGenerator):
    """
    Bollinger Bands signal generator
    
    Generates BUY signals when price touches lower band
    Generates SELL signals when price touches upper band
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Bollinger Bands signal generator
        
        Args:
            config: Configuration with 'period', 'std_dev'
        """
        super().__init__("Bollinger Bands Signal Generator", config)
        self.period = self.config.get('period', 20)
        self.std_dev = self.config.get('std_dev', 2.0)
        self.touch_threshold = self.config.get('touch_threshold', 0.01)  # 1% from band
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """
        Generate Bollinger Bands-based trading signals
        
        Args:
            data: Market data with OHLCV columns
            symbol: Trading symbol
            
        Returns:
            List of TradingSignal objects
        """
        try:
            if not self.validate_data(data):
                return []
            
            signals = []
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                data['close'].values,
                timeperiod=self.period,
                nbdevup=self.std_dev,
                nbdevdn=self.std_dev
            )
            
            for i in range(len(data)):
                if pd.isna(bb_upper[i]) or pd.isna(bb_lower[i]) or pd.isna(bb_middle[i]):
                    continue
                
                timestamp = data.index[i]
                close_price = data['close'].iloc[i]
                
                # Check if price is near lower band (oversold)
                lower_distance = (close_price - bb_lower[i]) / bb_lower[i]
                if lower_distance <= self.touch_threshold:
                    confidence = 1.0 - (lower_distance / self.touch_threshold)
                    signals.append(TradingSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.BUY,
                        confidence=confidence,
                        strength=confidence,
                        source="Bollinger Bands",
                        metadata={
                            'close_price': close_price,
                            'lower_band': bb_lower[i],
                            'middle_band': bb_middle[i],
                            'upper_band': bb_upper[i],
                            'band_width': (bb_upper[i] - bb_lower[i]) / bb_middle[i],
                            'position': 'near_lower'
                        }
                    ))
                
                # Check if price is near upper band (overbought)
                upper_distance = (bb_upper[i] - close_price) / bb_upper[i]
                if upper_distance <= self.touch_threshold:
                    confidence = 1.0 - (upper_distance / self.touch_threshold)
                    signals.append(TradingSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.SELL,
                        confidence=confidence,
                        strength=confidence,
                        source="Bollinger Bands",
                        metadata={
                            'close_price': close_price,
                            'lower_band': bb_lower[i],
                            'middle_band': bb_middle[i],
                            'upper_band': bb_upper[i],
                            'band_width': (bb_upper[i] - bb_lower[i]) / bb_middle[i],
                            'position': 'near_upper'
                        }
                    ))
            
            logger.info(f"Generated {len(signals)} Bollinger Bands signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating Bollinger Bands signals for {symbol}: {str(e)}")
            return []
    
    def get_required_columns(self) -> List[str]:
        """Get required DataFrame columns"""
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data
        
        Args:
            data: Input DataFrame
            
        Returns:
            True if data is valid, False otherwise
        """
        if data.empty:
            logger.warning("Empty data provided to Bollinger Bands generator")
            return False
        
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns for Bollinger Bands: {missing_cols}")
            return False
        
        if len(data) < self.period + 5:
            logger.warning(f"Insufficient data for Bollinger Bands calculation (need {self.period + 5}, got {len(data)})")
            return False
        
        return True