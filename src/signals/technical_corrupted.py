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
from ...config_manager import get_config_manager

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
                                        signals.append(TradingSignal(
                                timestamp=timestamp,
                                symbol=symbol,
                                signal_type=SignalType.BUY,
                                confidence=confidence,
                                source='VOLUME_BUY',: Configuration with 'period', 'overs                            signals.append(TradingSignal(
                                timestamp=timestamp,
                                symbol=symbol,
                                signal_type=SignalType.SELL,
                                confidence=confidence,
                                source='VOLUME_SELL',reshold', 'overbought_threshold'
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
        
        if len(data) < self.period + 2:  # Reduced buffer for more reliable operation
            logger.warning(f"Insufficient data for RSI calculation (need {self.period + 2}, got {len(data)})")
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
        
        min_periods = self.slow_period + self.signal_period + 2  # Reduced buffer
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
        
        if len(data) < self.period + 2:  # Reduced buffer for more reliable operation
            logger.warning(f"Insufficient data for Bollinger Bands calculation (need {self.period + 2}, got {len(data)})")
            return False
        
        return True


class SMACrossoverSignalGenerator(SignalGenerator):
    """
    SMA Crossover Signal Generator with configurable periods
    
    Generates signals when shorter SMA crosses above/below longer SMA
    Supports multiple timeframe analysis and trend confirmation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize SMA Crossover signal generator
        
        Args:
            config: Configuration with 'short_period', 'long_period', 'signal_strength_threshold'
        """
        super().__init__("SMA Crossover Signal Generator", config)
        self.short_period = self.config.get('short_period', 20)
        self.long_period = self.config.get('long_period', 50)
        self.signal_strength_threshold = self.config.get('signal_strength_threshold', 0.5)
        self.confirmation_periods = self.config.get('confirmation_periods', 2)
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """Generate SMA crossover signals"""
        logger.debug(f"[SMA_GENERATE] Starting signal generation for {symbol} with {len(data)} data points")
        try:
            if not self.validate_data(data):
                return []
            
            close_prices = data['close'].values
            signals = []
            
            # Calculate SMAs
            sma_short = talib.SMA(close_prices, timeperiod=self.short_period)
            sma_long = talib.SMA(close_prices, timeperiod=self.long_period)
            
            for i in range(self.long_period + self.confirmation_periods, len(data)):
                # Ensure we don't go out of bounds
                if i >= len(sma_short) or i >= len(sma_long) or i >= len(data):
                    logger.warning(f"[SMA_BOUNDS] Out of bounds at index {i}: sma_short={len(sma_short)}, sma_long={len(sma_long)}, data={len(data)}")
                    break
                
                # Skip if NaN values
                if pd.isna(sma_short[i]) or pd.isna(sma_long[i]) or pd.isna(sma_short[i-1]) or pd.isna(sma_long[i-1]):
                    logger.debug(f"[SMA_NAN] Skipping index {i} due to NaN values: sma_short[{i}]={sma_short[i]}, sma_long[{i}]={sma_long[i]}")
                    continue
                    
                timestamp = data.index[i]
                
                # Check for crossover
                short_above = sma_short[i] > sma_long[i]
                short_above_prev = sma_short[i-1] > sma_long[i-1]
                
                # Bullish crossover (Golden Cross)
                if short_above and not short_above_prev:
                    # Calculate signal strength
                    separation = abs(sma_short[i] - sma_long[i]) / sma_long[i]
                    momentum = (sma_short[i] - sma_short[i-5]) / sma_short[i-5] if i >= 5 else 0
                    strength = min(1.0, separation * 10 + abs(momentum) * 2)
                    
                    if strength >= self.signal_strength_threshold:
                        signals.append(TradingSignal(
                            timestamp=timestamp,
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            confidence=strength,
                            strength=strength,
                            source='SMA_GOLDEN_CROSS',
                            metadata={
                                'sma_short': float(sma_short[i]),
                                'sma_long': float(sma_long[i]),
                                'separation_pct': float(separation * 100),
                                'momentum': float(momentum),
                                'short_period': self.short_period,
                                'long_period': self.long_period
                            }
                        ))
                
                # Bearish crossover (Death Cross)
                elif not short_above and short_above_prev:
                    separation = abs(sma_long[i] - sma_short[i]) / sma_long[i]
                    momentum = abs((sma_short[i-5] - sma_short[i]) / sma_short[i]) if i >= 5 else 0
                    strength = min(1.0, separation * 10 + momentum * 2)
                    
                    if strength >= self.signal_strength_threshold:
                        signals.append(TradingSignal(
                            timestamp=timestamp,
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            confidence=strength,
                            strength=strength,
                            source='SMA_DEATH_CROSS',
                            metadata={
                                'sma_short': float(sma_short[i]),
                                'sma_long': float(sma_long[i]),
                                'separation_pct': float(separation * 100),
                                'momentum': float(momentum),
                                'short_period': self.short_period,
                                'long_period': self.long_period
                            }
                        ))
            
            return signals
            
        except Exception as e:
            logger.error(f"[SMA_ERROR] Error generating SMA crossover signals for {symbol}: {e}")
            logger.error(f"[SMA_ERROR] Data shape: {data.shape if hasattr(data, 'shape') else 'unknown'}")
            logger.error(f"[SMA_ERROR] Periods: short={self.short_period}, long={self.long_period}")
            import traceback
            logger.error(f"[SMA_ERROR] Traceback: {traceback.format_exc()}")
            return []
    
    def get_required_columns(self) -> List[str]:
        """Get required DataFrame columns"""
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        required_columns = self.get_required_columns()
        if not all(col in data.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            logger.error(f"[SMA_VALIDATION] Missing required columns: {missing_cols}. Available: {list(data.columns)}")
            return False
        
        if len(data) < self.long_period + 5:  # Reduced buffer for more practical operation
            logger.warning(f"[SMA_VALIDATION] Insufficient data for SMA crossover calculation (need {self.long_period + 5}, got {len(data)})")
            return False
        
        logger.debug(f"[SMA_VALIDATION] Data validation passed: {len(data)} rows, columns: {list(data.columns)}")
        return True


class EMASignalGenerator(SignalGenerator):
    """
    EMA Signal Generator with configurable periods and crossover analysis
    
    Supports multiple EMA periods and generates signals based on:
    - EMA crossovers
    - Price vs EMA relationship
    - EMA slope analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize EMA signal generator
        
        Args:
            config: Configuration with 'periods', 'crossover_pairs', 'slope_threshold'
        """
        super().__init__("EMA Signal Generator", config)
        self.periods = self.config.get('periods', [12, 26])
        self.crossover_pairs = self.config.get('crossover_pairs', [(12, 26)])
        self.slope_threshold = self.config.get('slope_threshold', 0.001)
        self.min_confidence = self.config.get('min_confidence', 0.3)
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """Generate EMA-based signals"""
        logger.debug(f"[EMA_GENERATE] Starting signal generation for {symbol} with {len(data)} data points, periods: {self.periods}")
        try:
            if not self.validate_data(data):
                return []
            
            close_prices = data['close'].values
            signals = []
            
            # Calculate EMAs for all periods
            emas = {}
            for period in self.periods:
                emas[period] = talib.EMA(close_prices, timeperiod=period)
            
            max_period = max(self.periods)
            
            for i in range(max_period + 5, len(data)):
                # Ensure we don't go out of bounds
                if i >= len(close_prices) or i >= len(data):
                    logger.warning(f"[EMA_BOUNDS] Out of bounds at index {i}: close_prices={len(close_prices)}, data={len(data)}")
                    break
                
                timestamp = data.index[i]
                current_price = close_prices[i]
                
                # Generate crossover signals
                for short_period, long_period in self.crossover_pairs:
                    if short_period in emas and long_period in emas:
                        short_ema = emas[short_period]
                        long_ema = emas[long_period]
                        
                        # Skip if out of bounds or NaN values
                        if (i >= len(short_ema) or i >= len(long_ema) or 
                            pd.isna(short_ema[i]) or pd.isna(long_ema[i]) or 
                            pd.isna(short_ema[i-1]) or pd.isna(long_ema[i-1])):
                            logger.debug(f"[EMA_SKIP] Index {i}, pair {short_period}/{long_period}: bounds check failed or NaN values detected")
                            continue
                        
                        # Check for crossover
                        short_above = short_ema[i] > long_ema[i]
                        short_above_prev = short_ema[i-1] > long_ema[i-1]
                        
                        if short_above and not short_above_prev:
                            # Bullish crossover
                            separation = (short_ema[i] - long_ema[i]) / long_ema[i]
                            ema_slope = (short_ema[i] - short_ema[i-3]) / short_ema[i-3]
                            confidence = min(1.0, abs(separation) * 15 + abs(ema_slope) * 5)
                            
                            if confidence >= self.min_confidence:
                                signals.append(TradingSignal(
                                    timestamp=timestamp,
                                    symbol=symbol,
                                    signal_type=SignalType.BUY,
                                    confidence=confidence,
                                    strength=confidence,
                                    source='EMA_BULLISH_CROSS',
                                    metadata={
                                        'ema_short': float(short_ema[i]),
                                        'ema_long': float(long_ema[i]),
                                        'short_period': short_period,
                                        'long_period': long_period,
                                        'separation_pct': float(separation * 100),
                                        'ema_slope': float(ema_slope)
                                    }
                                ))
                        
                        elif not short_above and short_above_prev:
                            # Bearish crossover
                            separation = (long_ema[i] - short_ema[i]) / long_ema[i]
                            ema_slope = abs((short_ema[i-3] - short_ema[i]) / short_ema[i])
                            confidence = min(1.0, separation * 15 + ema_slope * 5)
                            
                            if confidence >= self.min_confidence:
                                signals.append(TradingSignal(
                                    timestamp=timestamp,
                                    symbol=symbol,
                                    signal_type=SignalType.SELL,
                                    confidence=confidence,
                                    strength=confidence,
                                    source='EMA_BEARISH_CROSS',
                                    metadata={
                                        'ema_short': float(short_ema[i]),
                                        'ema_long': float(long_ema[i]),
                                        'short_period': short_period,
                                        'long_period': long_period,
                                        'separation_pct': float(separation * 100),
                                        'ema_slope': float(ema_slope)
                                    }
                                ))
            
            return signals
            
        except Exception as e:
            logger.error(f"[EMA_ERROR] Error generating EMA signals for {symbol}: {e}")
            logger.error(f"[EMA_ERROR] Data shape: {data.shape if hasattr(data, 'shape') else 'unknown'}")
            logger.error(f"[EMA_ERROR] Periods: {self.periods}, crossover_pairs: {self.crossover_pairs}")
            import traceback
            logger.error(f"[EMA_ERROR] Traceback: {traceback.format_exc()}")
            return []
    
    def get_required_columns(self) -> List[str]:
        """Get required DataFrame columns"""
        return ['close']
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        required_columns = self.get_required_columns()
        if not all(col in data.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            logger.error(f"[EMA_VALIDATION] Missing required columns: {missing_cols}. Available: {list(data.columns)}")
            return False
        
        max_period = max(self.periods) if self.periods else 26
        if len(data) < max_period + 5:  # Reduced buffer for more practical operation
            logger.warning(f"[EMA_VALIDATION] Insufficient data for EMA calculation (need {max_period + 5}, got {len(data)})")
            return False
        
        logger.debug(f"[EMA_VALIDATION] Data validation passed: {len(data)} rows, max_period: {max_period}")
        return True


class VolumeAnalysisSignalGenerator(SignalGenerator):
    """
    Volume Analysis Signal Generator
    
    Generates signals based on:
    - Volume surges (unusual volume activity)
    - Volume breakouts (price + volume confirmation)  
    - On Balance Volume trends
    - Volume profile analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Volume Analysis signal generator
        
        Args:
            config: Configuration with volume analysis parameters
        """
        super().__init__("Volume Analysis Signal Generator", config)
        self.volume_surge_threshold = self.config.get('volume_surge_threshold', 2.0)  # 2x average volume
        self.volume_sma_period = self.config.get('volume_sma_period', 20)
        self.obv_period = self.config.get('obv_period', 10)
        self.price_volume_confirmation = self.config.get('price_volume_confirmation', True)
        self.min_confidence = self.config.get('min_confidence', 0.4)
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """Generate volume-based signals"""
        logger.debug(f"[VOL_GENERATE] Starting signal generation for {symbol} with {len(data)} data points")
        try:
            if not self.validate_data(data):
                return []
            
            close_prices = data['close'].values
            volume = data['volume'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            signals = []
            
            # Calculate volume indicators
            volume_sma = talib.SMA(volume.astype(float), timeperiod=self.volume_sma_period)
            obv = talib.OBV(close_prices, volume.astype(float))
            
            for i in range(self.volume_sma_period + 5, len(data)):
                # Ensure we don't go out of bounds - comprehensive check
                if (i >= len(volume_sma) or i >= len(obv) or i >= len(close_prices) or 
                    i >= len(volume) or i >= len(data) or i >= len(data.index)):
                    logger.warning(f"[VOL_BOUNDS] Out of bounds at index {i}: volume_sma={len(volume_sma)}, obv={len(obv)}, close={len(close_prices)}, volume={len(volume)}, data={len(data)}, index_len={len(data.index)}")
                    break
                
                # Additional safety check for previous index access
                if i <= 0:
                    logger.debug(f"[VOL_BOUNDS] Skipping index {i}: cannot access previous values")
                    continue
                
                # Skip if NaN values
                if pd.isna(volume_sma[i]) or pd.isna(obv[i]):
                    logger.debug(f"[VOL_NAN] Skipping index {i}: volume_sma[{i}]={volume_sma[i]}, obv[{i}]={obv[i]}")
                    continue
                    
                timestamp = data.index[i]
                current_volume = volume[i]
                avg_volume = volume_sma[i]
                current_price = close_prices[i]
                prev_price = close_prices[i-1]
                
                # Volume surge detection
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                if volume_ratio >= self.volume_surge_threshold:
                    # Determine direction based on price movement
                    price_change = (current_price - prev_price) / prev_price
                    
                    # Volume breakout signal
                    if abs(price_change) > 0.02:  # Significant price movement with high volume
                        direction = 1 if price_change > 0 else -1
                        
                        # Calculate confidence based on volume ratio and price movement
                        confidence = min(1.0, 
                                       (volume_ratio / self.volume_surge_threshold) * 0.3 + 
                                       abs(price_change) * 10 * 0.4 +
                                       0.3)  # Base confidence
                        
                        if confidence >= self.min_confidence:
                            signals.append(TradingSignal(
                                timestamp=timestamp,
                                symbol=symbol,
                                signal_type=SignalType.BUY if direction == 1 else SignalType.SELL,
                                confidence=confidence,
                                strength=confidence,
                                source='VOLUME_BREAKOUT',
                                metadata={
                                    'volume_ratio': float(volume_ratio),
                                    'avg_volume': float(avg_volume),
                                    'current_volume': float(current_volume),
                                    'price_change_pct': float(price_change * 100),
                                    'surge_threshold': self.volume_surge_threshold
                                }
                            ))
                
                # OBV trend analysis
                if i >= self.obv_period + 5:
                    obv_trend = (obv[i] - obv[i-self.obv_period]) / abs(obv[i-self.obv_period]) if obv[i-self.obv_period] != 0 else 0
                    price_trend = (current_price - close_prices[i-self.obv_period]) / close_prices[i-self.obv_period]
                    
                    # OBV divergence detection
                    if (obv_trend > 0.05 and price_trend < -0.02):  # Bullish divergence
                        confidence = min(1.0, abs(obv_trend) * 2 + abs(price_trend) * 5)
                        
                        if confidence >= self.min_confidence:
                            signals.append(TradingSignal(
                                timestamp=timestamp,
                                symbol=symbol,
                                signal_type=SignalType.BUY,
                                confidence=confidence,
                                strength=confidence,
                                source='OBV_BULLISH_DIVERGENCE',
                                metadata={
                                    'obv_trend': float(obv_trend),
                                    'price_trend': float(price_trend),
                                    'obv_current': float(obv[i]),
                                    'divergence_type': 'bullish'
                                }
                            ))
                    
                    elif (obv_trend < -0.05 and price_trend > 0.02):  # Bearish divergence
                        confidence = min(1.0, abs(obv_trend) * 2 + abs(price_trend) * 5)
                        
                        if confidence >= self.min_confidence:
                            signals.append(TradingSignal(
                                timestamp=timestamp,
                                symbol=symbol,
                                signal_type=SignalType.SELL,
                                confidence=confidence,
                                strength=confidence,
                                source='OBV_BEARISH_DIVERGENCE',
                                metadata={
                                    'obv_trend': float(obv_trend),
                                    'price_trend': float(price_trend),
                                    'obv_current': float(obv[i]),
                                    'divergence_type': 'bearish'
                                }
                            ))
            
            return signals
            
        except Exception as e:
            logger.error(f"[VOL_ERROR] Error generating volume signals for {symbol}: {e}")
            logger.error(f"[VOL_ERROR] Data shape: {data.shape if hasattr(data, 'shape') else 'unknown'}")
            logger.error(f"[VOL_ERROR] Volume SMA period: {self.volume_sma_period}")
            import traceback
            logger.error(f"[VOL_ERROR] Traceback: {traceback.format_exc()}")
            return []
    
    def get_required_columns(self) -> List[str]:
        """Get required DataFrame columns"""
        return ['open', 'high', 'low', 'close', 'volume']
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        required_columns = self.get_required_columns()
        if not all(col in data.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            logger.error(f"[VOL_VALIDATION] Missing required columns: {missing_cols}. Available: {list(data.columns)}")
            return False
        
        if len(data) < self.volume_sma_period + self.obv_period + 5:  # Reduced buffer for more practical operation
            logger.warning(f"[VOL_VALIDATION] Insufficient data for volume analysis (need {self.volume_sma_period + self.obv_period + 5}, got {len(data)})")
            return False
        
        # Check for valid volume data
        if data['volume'].isna().any() or (data['volume'] < 0).any():
            logger.warning("[VOL_VALIDATION] Invalid volume data detected")
            return False
        
        logger.debug(f"[VOL_VALIDATION] Data validation passed: {len(data)} rows, volume_sma_period: {self.volume_sma_period}")
        return True