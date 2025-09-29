"""
Unit tests for src/signals modules
Tests technical signal generators and signal processing
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import datetime as dt
import numpy as np
import talib
from typing import List

# Import signal modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.signals.technical import RSISignalGenerator, MACDSignalGenerator, BollingerBandsSignalGenerator
from src.interfaces.signal_generator import SignalGenerator, TradingSignal, SignalType


class TestRSISignalGenerator(unittest.TestCase):
    """Test RSI-based signal generation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = RSISignalGenerator()
        # Create sample data that will produce clear RSI signals
        self.sample_data = pd.DataFrame({
            'open': [100.0] * 50,
            'high': [102.0] * 25 + [98.0] * 25,  # High then low prices
            'low': [98.0] * 25 + [96.0] * 25,
            'close': [101.0] * 10 + [95.0] * 15 + [105.0] * 15 + [100.0] * 10,  # Clear trend patterns
            'volume': [1000000] * 50
        }, index=pd.date_range('2023-01-01', periods=50))
        
    def test_initialization_default_config(self):
        """Test RSI generator initialization with default config"""
        generator = RSISignalGenerator()
        
        self.assertEqual(generator.period, 14)
        self.assertEqual(generator.oversold_threshold, 30)
        self.assertEqual(generator.overbought_threshold, 70)
        self.assertEqual(generator.name, "RSI Signal Generator")
        self.assertTrue(generator.enabled)
        
    def test_initialization_custom_config(self):
        """Test RSI generator initialization with custom config"""
        config = {
            'period': 21,
            'oversold_threshold': 25,
            'overbought_threshold': 75,
            'enabled': False
        }
        generator = RSISignalGenerator(config)
        
        self.assertEqual(generator.period, 21)
        self.assertEqual(generator.oversold_threshold, 25)
        self.assertEqual(generator.overbought_threshold, 75)
        self.assertFalse(generator.enabled)
        
    def test_generate_signals_basic(self):
        """Test basic RSI signal generation"""
        signals = self.generator.generate_signals(self.sample_data, "TEST")
        
        self.assertIsInstance(signals, list)
        for signal in signals:
            self.assertIsInstance(signal, TradingSignal)
            self.assertEqual(signal.symbol, "TEST")
            self.assertEqual(signal.source, "RSI")
            self.assertIn(signal.signal_type, [SignalType.BUY, SignalType.SELL])
            self.assertGreaterEqual(signal.confidence, 0.0)
            self.assertLessEqual(signal.confidence, 1.0)
            
    def test_generate_signals_oversold_condition(self):
        """Test RSI signal generation in oversold condition"""
        # Create data that will definitely produce oversold RSI
        decreasing_prices = pd.DataFrame({
            'open': [100.0 - i for i in range(30)],
            'high': [101.0 - i for i in range(30)],
            'low': [99.0 - i for i in range(30)],
            'close': [100.0 - i*2 for i in range(30)],  # Strong downtrend
            'volume': [1000000] * 30
        }, index=pd.date_range('2023-01-01', periods=30))
        
        signals = self.generator.generate_signals(decreasing_prices, "OVERSOLD")
        
        # Should have some BUY signals due to oversold condition
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        self.assertGreater(len(buy_signals), 0)
        
        # Check signal metadata
        for signal in buy_signals:
            self.assertIn('rsi_value', signal.metadata)
            self.assertIn('condition', signal.metadata)
            self.assertEqual(signal.metadata['condition'], 'oversold')
            self.assertLess(signal.metadata['rsi_value'], self.generator.oversold_threshold)
            
    def test_generate_signals_overbought_condition(self):
        """Test RSI signal generation in overbought condition"""
        # Create data that will definitely produce overbought RSI
        increasing_prices = pd.DataFrame({
            'open': [100.0 + i for i in range(30)],
            'high': [101.0 + i for i in range(30)],
            'low': [99.0 + i for i in range(30)],
            'close': [100.0 + i*2 for i in range(30)],  # Strong uptrend
            'volume': [1000000] * 30
        }, index=pd.date_range('2023-01-01', periods=30))
        
        signals = self.generator.generate_signals(increasing_prices, "OVERBOUGHT")
        
        # Should have some SELL signals due to overbought condition
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        self.assertGreater(len(sell_signals), 0)
        
        # Check signal metadata
        for signal in sell_signals:
            self.assertIn('rsi_value', signal.metadata)
            self.assertIn('condition', signal.metadata)
            self.assertEqual(signal.metadata['condition'], 'overbought')
            self.assertGreater(signal.metadata['rsi_value'], self.generator.overbought_threshold)
            
    def test_get_required_columns(self):
        """Test required columns specification"""
        required = self.generator.get_required_columns()
        
        self.assertIsInstance(required, list)
        self.assertIn('close', required)
        
    def test_validate_data_valid(self):
        """Test data validation with valid data"""
        valid_data = pd.DataFrame({
            'close': [100, 101, 102, 103]
        })
        
        result = self.generator.validate_data(valid_data)
        self.assertTrue(result)
        
    def test_validate_data_invalid(self):
        """Test data validation with invalid data"""
        # Missing required column
        invalid_data = pd.DataFrame({
            'open': [100, 101, 102, 103]
        })
        
        result = self.generator.validate_data(invalid_data)
        self.assertFalse(result)
        
    def test_generate_signals_empty_data(self):
        """Test signal generation with empty data"""
        empty_data = pd.DataFrame()
        
        signals = self.generator.generate_signals(empty_data, "EMPTY")
        
        self.assertEqual(len(signals), 0)
        
    def test_generate_signals_insufficient_data(self):
        """Test signal generation with insufficient data for RSI calculation"""
        minimal_data = pd.DataFrame({
            'close': [100, 101]  # Only 2 points, need more for RSI
        }, index=pd.date_range('2023-01-01', periods=2))
        
        signals = self.generator.generate_signals(minimal_data, "MINIMAL")
        
        # Should handle gracefully, probably return empty signals
        self.assertIsInstance(signals, list)


class TestMACDSignalGenerator(unittest.TestCase):
    """Test MACD-based signal generation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = MACDSignalGenerator()
        # Create data with clear trend changes for MACD signals
        prices = []
        for i in range(50):
            if i < 20:
                prices.append(100 + i * 0.5)  # Uptrend
            elif i < 30:
                prices.append(110 - (i-20) * 0.3)  # Downtrend
            else:
                prices.append(107 + (i-30) * 0.4)  # Uptrend again
                
        self.sample_data = pd.DataFrame({
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000000] * 50
        }, index=pd.date_range('2023-01-01', periods=50))
        
    def test_initialization_default_config(self):
        """Test MACD generator initialization with default config"""
        generator = MACDSignalGenerator()
        
        self.assertEqual(generator.fast_period, 12)
        self.assertEqual(generator.slow_period, 26)
        self.assertEqual(generator.signal_period, 9)
        self.assertEqual(generator.name, "MACD Signal Generator")
        
    def test_initialization_custom_config(self):
        """Test MACD generator initialization with custom config"""
        config = {
            'fast_period': 8,
            'slow_period': 21,
            'signal_period': 5
        }
        generator = MACDSignalGenerator(config)
        
        self.assertEqual(generator.fast_period, 8)
        self.assertEqual(generator.slow_period, 21)
        self.assertEqual(generator.signal_period, 5)
        
    def test_generate_signals_basic(self):
        """Test basic MACD signal generation"""
        signals = self.generator.generate_signals(self.sample_data, "TEST")
        
        self.assertIsInstance(signals, list)
        for signal in signals:
            self.assertIsInstance(signal, TradingSignal)
            self.assertEqual(signal.symbol, "TEST")
            self.assertEqual(signal.source, "MACD")
            self.assertIn(signal.signal_type, [SignalType.BUY, SignalType.SELL])
            
    def test_generate_signals_bullish_crossover(self):
        """Test MACD bullish crossover signal detection"""
        # Create data that should produce bullish crossover
        trend_change_data = pd.DataFrame({
            'close': [100 - i for i in range(30)] + [70 + i*0.5 for i in range(30)]  # Down then up
        }, index=pd.date_range('2023-01-01', periods=60))
        
        signals = self.generator.generate_signals(trend_change_data, "BULLISH")
        
        # Should detect some crossover signals
        self.assertGreater(len(signals), 0)
        
        # Check for BUY signals (bullish crossovers)
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        for signal in buy_signals:
            self.assertIn('macd_value', signal.metadata)
            self.assertIn('signal_value', signal.metadata)
            self.assertEqual(signal.metadata['condition'], 'bullish_crossover')
            
    def test_get_required_columns(self):
        """Test required columns specification"""
        required = self.generator.get_required_columns()
        
        self.assertIsInstance(required, list)
        self.assertIn('close', required)
        
    def test_validate_data_valid(self):
        """Test data validation with valid data"""
        valid_data = pd.DataFrame({
            'close': [100 + i for i in range(50)]
        })
        
        result = self.generator.validate_data(valid_data)
        self.assertTrue(result)


class TestBollingerBandsSignalGenerator(unittest.TestCase):
    """Test Bollinger Bands-based signal generation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = BollingerBandsSignalGenerator()
        
        # Create data with clear price moves outside bands
        base_prices = [100] * 20  # Stable period
        volatile_prices = [95, 90, 85, 80, 85, 90, 95, 100, 105, 110, 115, 120, 115, 110, 105, 100]
        all_prices = base_prices + volatile_prices
        
        self.sample_data = pd.DataFrame({
            'open': all_prices,
            'high': [p + 1 for p in all_prices],
            'low': [p - 1 for p in all_prices],
            'close': all_prices,
            'volume': [1000000] * len(all_prices)
        }, index=pd.date_range('2023-01-01', periods=len(all_prices)))
        
    def test_initialization_default_config(self):
        """Test Bollinger Bands generator initialization with default config"""
        generator = BollingerBandsSignalGenerator()
        
        self.assertEqual(generator.window, 20)
        self.assertEqual(generator.num_std, 2)
        self.assertEqual(generator.name, "Bollinger Bands Signal Generator")
        
    def test_initialization_custom_config(self):
        """Test Bollinger Bands generator initialization with custom config"""
        config = {
            'window': 14,
            'num_std': 2.5
        }
        generator = BollingerBandsSignalGenerator(config)
        
        self.assertEqual(generator.window, 14)
        self.assertEqual(generator.num_std, 2.5)
        
    def test_generate_signals_basic(self):
        """Test basic Bollinger Bands signal generation"""
        signals = self.generator.generate_signals(self.sample_data, "TEST")
        
        self.assertIsInstance(signals, list)
        for signal in signals:
            self.assertIsInstance(signal, TradingSignal)
            self.assertEqual(signal.symbol, "TEST")
            self.assertEqual(signal.source, "BollingerBands")
            self.assertIn(signal.signal_type, [SignalType.BUY, SignalType.SELL])
            
    def test_generate_signals_band_touches(self):
        """Test signal generation when price touches bands"""
        # Create data with clear band penetration
        extreme_data = pd.DataFrame({
            'close': [100] * 25 + [80, 75, 70, 75, 80] + [100] * 5 + [120, 125, 130, 125, 120] + [100] * 10
        }, index=pd.date_range('2023-01-01', periods=45))
        
        signals = self.generator.generate_signals(extreme_data, "EXTREME")
        
        # Should generate signals when price touches bands
        self.assertGreater(len(signals), 0)
        
        # Check signal metadata
        for signal in signals:
            self.assertIn('price', signal.metadata)
            self.assertIn('bb_upper', signal.metadata)
            self.assertIn('bb_lower', signal.metadata)
            self.assertIn('condition', signal.metadata)
            
    def test_get_required_columns(self):
        """Test required columns specification"""
        required = self.generator.get_required_columns()
        
        self.assertIsInstance(required, list)
        self.assertIn('close', required)


class TestSignalGeneratorBase(unittest.TestCase):
    """Test base SignalGenerator functionality"""
    
    def test_signal_generator_base_interface(self):
        """Test that SignalGenerator provides proper interface"""
        # Test with a concrete implementation
        generator = RSISignalGenerator()
        
        # Test interface methods exist
        self.assertTrue(hasattr(generator, 'generate_signals'))
        self.assertTrue(hasattr(generator, 'get_required_columns'))
        self.assertTrue(hasattr(generator, 'validate_data'))
        
        # Test properties
        self.assertTrue(hasattr(generator, 'name'))
        self.assertTrue(hasattr(generator, 'config'))
        self.assertTrue(hasattr(generator, 'enabled'))
        
    def test_signal_generator_configuration(self):
        """Test signal generator configuration management"""
        config = {
            'enabled': False,
            'custom_param': 'value'
        }
        generator = RSISignalGenerator(config)
        
        self.assertEqual(generator.config['custom_param'], 'value')
        self.assertFalse(generator.enabled)


class TestSignalAggregation(unittest.TestCase):
    """Test signal aggregation and combination"""
    
    def setUp(self):
        """Set up multiple signal generators"""
        self.rsi_generator = RSISignalGenerator()
        self.macd_generator = MACDSignalGenerator()
        self.bb_generator = BollingerBandsSignalGenerator()
        
        # Create comprehensive test data
        self.test_data = pd.DataFrame({
            'open': [100 + i*0.1 for i in range(100)],
            'high': [102 + i*0.1 for i in range(100)],
            'low': [98 + i*0.1 for i in range(100)],
            'close': [101 + i*0.1 + np.sin(i/5)*2 for i in range(100)],  # Trending with oscillation
            'volume': [1000000 + i*10000 for i in range(100)]
        }, index=pd.date_range('2023-01-01', periods=100))
        
    def test_multiple_generator_signals(self):
        """Test combining signals from multiple generators"""
        # Generate signals from all generators
        rsi_signals = self.rsi_generator.generate_signals(self.test_data, "MULTI")
        macd_signals = self.macd_generator.generate_signals(self.test_data, "MULTI")
        bb_signals = self.bb_generator.generate_signals(self.test_data, "MULTI")
        
        # Combine all signals
        all_signals = rsi_signals + macd_signals + bb_signals
        
        # Verify signal diversity
        sources = set(signal.source for signal in all_signals)
        self.assertIn("RSI", sources)
        self.assertIn("MACD", sources)
        self.assertIn("BollingerBands", sources)
        
    def test_signal_consensus(self):
        """Test identifying signal consensus"""
        # Generate signals from all generators
        rsi_signals = self.rsi_generator.generate_signals(self.test_data, "CONSENSUS")
        macd_signals = self.macd_generator.generate_signals(self.test_data, "CONSENSUS")
        bb_signals = self.bb_generator.generate_signals(self.test_data, "CONSENSUS")
        
        all_signals = rsi_signals + macd_signals + bb_signals
        
        # Group signals by timestamp and check for consensus
        signal_groups = {}
        for signal in all_signals:
            timestamp = signal.timestamp.date()
            if timestamp not in signal_groups:
                signal_groups[timestamp] = []
            signal_groups[timestamp].append(signal)
            
        # Check for days with multiple signals (potential consensus)
        consensus_days = [day for day, signals in signal_groups.items() if len(signals) > 1]
        
        # Should have some days with multiple signals
        self.assertGreater(len(consensus_days), 0)
        
    def test_signal_confidence_weighting(self):
        """Test weighting signals by confidence"""
        signals = self.rsi_generator.generate_signals(self.test_data, "WEIGHT")
        
        if signals:
            # Check confidence distribution
            confidences = [signal.confidence for signal in signals]
            avg_confidence = np.mean(confidences)
            
            self.assertGreater(avg_confidence, 0.0)
            self.assertLessEqual(avg_confidence, 1.0)
            
            # High confidence signals should be more actionable
            high_conf_signals = [s for s in signals if s.confidence > 0.7]
            low_conf_signals = [s for s in signals if s.confidence < 0.3]
            
            # High confidence signals should generally have higher strength
            if high_conf_signals and low_conf_signals:
                avg_high_strength = np.mean([s.strength for s in high_conf_signals])
                avg_low_strength = np.mean([s.strength for s in low_conf_signals])
                self.assertGreaterEqual(avg_high_strength, avg_low_strength)


if __name__ == '__main__':
    unittest.main()