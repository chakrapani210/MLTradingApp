"""
Technical Signal Generators - Clean Version with Centralized Configuration
Concrete implementations of technical analysis based signal generators
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import talib
import logging
import sys
import os

# Add project root to path for config_manager import
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ..interfaces.signal_generator import SignalGenerator, TradingSignal, SignalType
from config_manager import get_config_manager

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
        
        # Get centralized configuration
        config_manager = get_config_manager()
        rsi_config = config_manager.get_signal_generator_config('rsi')
        
        # Use centralized config with local overrides
        self.period = self.config.get('period', rsi_config.get('period', 14))
        self.oversold_threshold = self.config.get('oversold_threshold', rsi_config.get('oversold_threshold', 30))
        self.overbought_threshold = self.config.get('overbought_threshold', rsi_config.get('overbought_threshold', 70))
    
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
                        confidence=confidence,
                        source='RSI_OVERSOLD',
                        metadata={'rsi_value': rsi_value, 'threshold': self.oversold_threshold}
                    ))
                elif rsi_value > self.overbought_threshold:
                    # Overbought - SELL signal
                    confidence = (rsi_value - self.overbought_threshold) / (100 - self.overbought_threshold)
                    signals.append(TradingSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.SELL,
                        confidence=confidence,
                        source='RSI_OVERBOUGHT',
                        metadata={'rsi_value': rsi_value, 'threshold': self.overbought_threshold}
                    ))
            
            logger.debug(f"[RSI_SIGNALS] Generated {len(signals)} signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"[RSI_ERROR] Error generating signals for {symbol}: {e}")
            return []
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data is sufficient for RSI calculation"""
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
        
        # Get centralized configuration
        config_manager = get_config_manager()
        macd_config = config_manager.get_signal_generator_config('macd')
        
        # Use centralized config with local overrides
        self.fast_period = self.config.get('fast_period', macd_config.get('fast_period', 12))
        self.slow_period = self.config.get('slow_period', macd_config.get('slow_period', 26))
        self.signal_period = self.config.get('signal_period', macd_config.get('signal_period', 9))
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """Generate MACD-based signals"""
        try:
            if not self.validate_data(data):
                return []
            
            signals = []
            
            # Calculate MACD
            macd, macdsignal, macdhist = talib.MACD(
                data['close'].values,
                fastperiod=self.fast_period,
                slowperiod=self.slow_period,
                signalperiod=self.signal_period
            )
            
            for i in range(1, len(data)):
                if pd.isna(macd[i]) or pd.isna(macdsignal[i]):
                    continue
                
                timestamp = data.index[i]
                prev_macd = macd[i-1]
                prev_signal = macdsignal[i-1]
                curr_macd = macd[i]
                curr_signal = macdsignal[i]
                
                # Check for crossovers
                if prev_macd <= prev_signal and curr_macd > curr_signal:
                    # Bullish crossover - BUY signal
                    confidence = min(0.9, abs(curr_macd - curr_signal) / abs(curr_macd) if curr_macd != 0 else 0.5)
                    signals.append(TradingSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.BUY,
                        confidence=confidence,
                        source='MACD_BULLISH_CROSSOVER',
                        metadata={'macd': curr_macd, 'signal': curr_signal, 'histogram': macdhist[i]}
                    ))
                elif prev_macd >= prev_signal and curr_macd < curr_signal:
                    # Bearish crossover - SELL signal
                    confidence = min(0.9, abs(curr_macd - curr_signal) / abs(curr_macd) if curr_macd != 0 else 0.5)
                    signals.append(TradingSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.SELL,
                        confidence=confidence,
                        source='MACD_BEARISH_CROSSOVER',
                        metadata={'macd': curr_macd, 'signal': curr_signal, 'histogram': macdhist[i]}
                    ))
            
            logger.debug(f"[MACD_SIGNALS] Generated {len(signals)} signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"[MACD_ERROR] Error generating signals for {symbol}: {e}")
            return []
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data for MACD calculation"""
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
        
        # Get centralized configuration
        config_manager = get_config_manager()
        bb_config = config_manager.get_signal_generator_config('bollinger_bands')
        
        # Use centralized config with local overrides
        self.period = self.config.get('period', bb_config.get('period', 20))
        self.std_dev = self.config.get('std_dev', bb_config.get('std_dev', 2))
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """Generate Bollinger Bands signals"""
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
                if pd.isna(bb_upper[i]) or pd.isna(bb_lower[i]):
                    continue
                
                timestamp = data.index[i]
                close_price = data['close'].iloc[i]
                upper_band = bb_upper[i]
                lower_band = bb_lower[i]
                middle_band = bb_middle[i]
                
                # Generate signals based on band touches
                if close_price <= lower_band:
                    # Price at or below lower band - BUY signal
                    band_distance = abs(close_price - lower_band) / abs(middle_band - lower_band) if middle_band != lower_band else 0
                    confidence = min(0.9, 0.7 + band_distance * 0.2)
                    signals.append(TradingSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.BUY,
                        confidence=confidence,
                        source='BB_LOWER_TOUCH',
                        metadata={
                            'price': close_price,
                            'upper_band': upper_band,
                            'middle_band': middle_band,
                            'lower_band': lower_band
                        }
                    ))
                elif close_price >= upper_band:
                    # Price at or above upper band - SELL signal
                    band_distance = abs(close_price - upper_band) / abs(middle_band - upper_band) if middle_band != upper_band else 0
                    confidence = min(0.9, 0.7 + band_distance * 0.2)
                    signals.append(TradingSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.SELL,
                        confidence=confidence,
                        source='BB_UPPER_TOUCH',
                        metadata={
                            'price': close_price,
                            'upper_band': upper_band,
                            'middle_band': middle_band,
                            'lower_band': lower_band
                        }
                    ))
            
            logger.debug(f"[BB_SIGNALS] Generated {len(signals)} signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"[BB_ERROR] Error generating signals for {symbol}: {e}")
            return []
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data for Bollinger Bands calculation"""
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
    SMA Crossover signal generator
    
    Generates BUY signals on bullish SMA crossover (short > long)
    Generates SELL signals on bearish SMA crossover (short < long)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize SMA Crossover signal generator
        
        Args:
            config: Configuration with 'short_period', 'long_period', 'confirmation_periods'
        """
        super().__init__("SMA Crossover Signal Generator", config)
        
        # Get centralized configuration
        config_manager = get_config_manager()
        sma_config = config_manager.get_signal_generator_config('sma_crossover')
        
        # Use centralized config with local overrides
        self.short_period = self.config.get('short_period', sma_config.get('short_period', 20))
        self.long_period = self.config.get('long_period', sma_config.get('long_period', 50))
        self.confirmation_periods = self.config.get('confirmation_periods', sma_config.get('confirmation_periods', 2))
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """Generate SMA crossover signals"""
        try:
            if not self.validate_data(data):
                return []
            
            signals = []
            close_prices = data['close'].values
            
            # Calculate SMAs
            sma_short = talib.SMA(close_prices, timeperiod=self.short_period)
            sma_long = talib.SMA(close_prices, timeperiod=self.long_period)
            
            for i in range(self.long_period + self.confirmation_periods, len(data)):
                if pd.isna(sma_short[i]) or pd.isna(sma_long[i]):
                    continue
                
                timestamp = data.index[i]
                
                # Check for crossovers with confirmation
                bullish_cross = all(sma_short[i-j] > sma_long[i-j] for j in range(self.confirmation_periods))
                bearish_cross = all(sma_short[i-j] < sma_long[i-j] for j in range(self.confirmation_periods))
                
                prev_bullish = all(sma_short[i-j-1] > sma_long[i-j-1] for j in range(self.confirmation_periods))
                prev_bearish = all(sma_short[i-j-1] < sma_long[i-j-1] for j in range(self.confirmation_periods))
                
                if bullish_cross and not prev_bullish:
                    # Bullish crossover - BUY signal
                    spread_ratio = (sma_short[i] - sma_long[i]) / sma_long[i] if sma_long[i] != 0 else 0
                    confidence = min(0.9, 0.6 + abs(spread_ratio) * 10)
                    signals.append(TradingSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.BUY,
                        confidence=confidence,
                        source='SMA_BULLISH_CROSSOVER',
                        metadata={
                            'short_sma': sma_short[i],
                            'long_sma': sma_long[i],
                            'short_period': self.short_period,
                            'long_period': self.long_period
                        }
                    ))
                elif bearish_cross and not prev_bearish:
                    # Bearish crossover - SELL signal
                    spread_ratio = (sma_long[i] - sma_short[i]) / sma_long[i] if sma_long[i] != 0 else 0
                    confidence = min(0.9, 0.6 + abs(spread_ratio) * 10)
                    signals.append(TradingSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.SELL,
                        confidence=confidence,
                        source='SMA_BEARISH_CROSSOVER',
                        metadata={
                            'short_sma': sma_short[i],
                            'long_sma': sma_long[i],
                            'short_period': self.short_period,
                            'long_period': self.long_period
                        }
                    ))
            
            logger.debug(f"[SMA_SIGNALS] Generated {len(signals)} signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"[SMA_ERROR] Error generating signals for {symbol}: {e}")
            logger.error(f"[SMA_ERROR] Periods: short={self.short_period}, long={self.long_period}")
            return []
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data for SMA crossover calculation"""
        if data.empty:
            logger.warning("Empty data provided to SMA crossover generator")
            return False
        
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns for SMA crossover: {missing_cols}")
            return False
        
        if len(data) < self.long_period + 5:  # Reduced buffer for more practical operation
            logger.warning(f"[SMA_VALIDATION] Insufficient data for SMA crossover calculation (need {self.long_period + 5}, got {len(data)})")
            return False
        
        return True


class EMASignalGenerator(SignalGenerator):
    """
    EMA-based signal generator
    
    Generates signals based on EMA crossovers and trends
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize EMA signal generator
        
        Args:
            config: Configuration with 'periods', 'crossover_pairs', 'slope_threshold'
        """
        super().__init__("EMA Signal Generator", config)
        
        # Get centralized configuration
        config_manager = get_config_manager()
        ema_config = config_manager.get_signal_generator_config('ema')
        
        # Use centralized config with local overrides
        self.periods = self.config.get('periods', ema_config.get('periods', [12, 26]))
        self.crossover_pairs = self.config.get('crossover_pairs', ema_config.get('crossover_pairs', [[12, 26]]))
        self.slope_threshold = self.config.get('slope_threshold', ema_config.get('slope_threshold', 0.001))
        self.min_confidence = self.config.get('min_confidence', ema_config.get('min_confidence', 0.3))
    
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
            
            # Generate crossover signals
            if len(emas) >= 2:
                for short_period, long_period in self.crossover_pairs:
                    if short_period in emas and long_period in emas:
                        short_ema = emas[short_period]
                        long_ema = emas[long_period]
                        
                        for i in range(max(short_period, long_period) + 1, len(data)):
                            if i >= len(short_ema) or i >= len(long_ema) or pd.isna(short_ema[i]) or pd.isna(long_ema[i]):
                                logger.debug(f"[EMA_SKIP] Index {i}, pair {short_period}/{long_period}: bounds check failed or NaN values detected")
                                continue
                            
                            timestamp = data.index[i]
                            
                            # Check for crossovers
                            if i > 0:
                                prev_short = short_ema[i-1]
                                prev_long = long_ema[i-1]
                                curr_short = short_ema[i]
                                curr_long = long_ema[i]
                                
                                # Bullish crossover
                                if prev_short <= prev_long and curr_short > curr_long:
                                    spread_ratio = (curr_short - curr_long) / curr_long if curr_long != 0 else 0
                                    confidence = max(self.min_confidence, min(0.9, 0.6 + abs(spread_ratio) * 10))
                                    signals.append(TradingSignal(
                                        symbol=symbol,
                                        timestamp=timestamp,
                                        signal_type=SignalType.BUY,
                                        confidence=confidence,
                                        source='EMA_BULLISH_CROSSOVER',
                                        metadata={
                                            'short_ema': curr_short,
                                            'long_ema': curr_long,
                                            'short_period': short_period,
                                            'long_period': long_period,
                                            'spread_ratio': spread_ratio
                                        }
                                    ))
                                # Bearish crossover
                                elif prev_short >= prev_long and curr_short < curr_long:
                                    spread_ratio = (curr_long - curr_short) / curr_long if curr_long != 0 else 0
                                    confidence = max(self.min_confidence, min(0.9, 0.6 + abs(spread_ratio) * 10))
                                    signals.append(TradingSignal(
                                        symbol=symbol,
                                        timestamp=timestamp,
                                        signal_type=SignalType.SELL,
                                        confidence=confidence,
                                        source='EMA_BEARISH_CROSSOVER',
                                        metadata={
                                            'short_ema': curr_short,
                                            'long_ema': curr_long,
                                            'short_period': short_period,
                                            'long_period': long_period,
                                            'spread_ratio': spread_ratio
                                        }
                                    ))
            
            logger.debug(f"[EMA_SIGNALS] Generated {len(signals)} signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"[EMA_ERROR] Error generating signals for {symbol}: {e}")
            logger.error(f"[EMA_ERROR] Periods: {self.periods}, crossover_pairs: {self.crossover_pairs}")
            return []
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data for EMA calculation"""
        if data.empty:
            logger.warning("Empty data provided to EMA generator")
            return False
        
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns for EMA: {missing_cols}")
            return False
        
        max_period = max(self.periods) if self.periods else 26
        if len(data) < max_period + 5:  # Reduced buffer for more practical operation
            logger.warning(f"[EMA_VALIDATION] Insufficient data for EMA calculation (need {max_period + 5}, got {len(data)})")
            return False
        
        return True


class VolumeAnalysisSignalGenerator(SignalGenerator):
    """
    Volume analysis signal generator
    
    Generates signals based on volume patterns and price-volume relationships
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Volume Analysis signal generator
        
        Args:
            config: Configuration with volume analysis parameters
        """
        super().__init__("Volume Analysis Signal Generator", config)
        
        # Get centralized configuration
        config_manager = get_config_manager()
        volume_config = config_manager.get_signal_generator_config('volume')
        
        # Use centralized config with local overrides
        self.volume_sma_period = self.config.get('volume_sma_period', volume_config.get('volume_sma_period', 20))
        self.obv_period = self.config.get('obv_period', volume_config.get('obv_period', 10))
        self.volume_spike_threshold = self.config.get('volume_spike_threshold', volume_config.get('volume_spike_threshold', 2.0))
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """Generate volume-based signals"""
        try:
            if not self.validate_data(data):
                return []
            
            signals = []
            
            # Calculate volume indicators
            volume_sma = talib.SMA(data['volume'].values, timeperiod=self.volume_sma_period)
            obv = talib.OBV(data['close'].values, data['volume'].values)
            
            for i in range(self.volume_sma_period, len(data)):
                if pd.isna(volume_sma[i]) or pd.isna(obv[i]):
                    continue
                
                timestamp = data.index[i]
                current_volume = data['volume'].iloc[i]
                avg_volume = volume_sma[i]
                price_change = data['close'].iloc[i] - data['close'].iloc[i-1] if i > 0 else 0
                
                # Volume spike with price increase
                if current_volume > avg_volume * self.volume_spike_threshold and price_change > 0:
                    volume_ratio = current_volume / avg_volume
                    confidence = min(0.9, 0.5 + (volume_ratio - self.volume_spike_threshold) * 0.1)
                    signals.append(TradingSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.BUY,
                        confidence=confidence,
                        source='VOLUME_BUY',
                        metadata={
                            'current_volume': current_volume,
                            'avg_volume': avg_volume,
                            'volume_ratio': volume_ratio,
                            'price_change': price_change,
                            'obv': obv[i]
                        }
                    ))
                # Volume spike with price decrease
                elif current_volume > avg_volume * self.volume_spike_threshold and price_change < 0:
                    volume_ratio = current_volume / avg_volume
                    confidence = min(0.9, 0.5 + (volume_ratio - self.volume_spike_threshold) * 0.1)
                    signals.append(TradingSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.SELL,
                        confidence=confidence,
                        source='VOLUME_SELL',
                        metadata={
                            'current_volume': current_volume,
                            'avg_volume': avg_volume,
                            'volume_ratio': volume_ratio,
                            'price_change': price_change,
                            'obv': obv[i]
                        }
                    ))
            
            logger.debug(f"[VOLUME_SIGNALS] Generated {len(signals)} signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"[VOLUME_ERROR] Error generating signals for {symbol}: {e}")
            return []
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data for volume analysis"""
        required_columns = ['close', 'volume']
        
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