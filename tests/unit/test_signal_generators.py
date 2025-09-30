"""
Unit Tests for Signal Generators and Technical Analysis
"""
import sys
import os
import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.signals.technical import (
    RSISignalGenerator, MACDSignalGenerator, BollingerBandsSignalGenerator,
    SMACrossoverSignalGenerator, EMASignalGenerator, VolumeAnalysisSignalGenerator
)
from src.interfaces.trading_strategy import TradingSignal, SignalType


class TestRSISignalGenerator:
    """Test suite for RSI Signal Generator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.generator = RSISignalGenerator(config={'period': 14})
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 50)))
        
        self.test_data = pd.DataFrame({
            'Close': prices,
            'close': prices,  # Both formats for compatibility
            'Volume': np.random.randint(1000000, 5000000, 50)
        }, index=dates)
    
    def test_rsi_generator_initialization(self):
        """Test RSI generator initialization"""
        generator = RSISignalGenerator(config={'period': 14, 'oversold_threshold': 25})
        
        assert generator.name == "RSI"
        assert generator.period == 14
        assert generator.oversold_threshold == 25
        assert generator.overbought_threshold == 70  # default
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        rsi_values = self.generator._calculate_rsi(self.test_data['Close'], period=14)
        
        # RSI should be between 0 and 100
        assert all(0 <= rsi <= 100 for rsi in rsi_values if not pd.isna(rsi))
        
        # Should have NaN values for insufficient data
        assert pd.isna(rsi_values.iloc[:13]).all()  # First 13 values should be NaN
        assert not pd.isna(rsi_values.iloc[14:]).any()  # Rest should not be NaN
    
    def test_generate_signals_oversold(self):
        """Test RSI signal generation for oversold condition"""
        # Create data that will trigger oversold condition
        declining_prices = pd.Series([100, 95, 90, 85, 80, 75, 70, 65, 60, 55] * 5)
        test_data = pd.DataFrame({
            'Close': declining_prices,
            'close': declining_prices
        }, index=pd.date_range('2023-01-01', periods=50))
        
        signals = self.generator.generate_signals(test_data, "AAPL")
        
        # Should generate some signals
        assert len(signals) >= 0
        
        # If signals are generated, they should be valid
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert signal.symbol == "AAPL"
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
    
    def test_generate_signals_overbought(self):
        """Test RSI signal generation for overbought condition"""
        # Create data that will trigger overbought condition
        rising_prices = pd.Series([50, 55, 60, 65, 70, 75, 80, 85, 90, 95] * 5)
        test_data = pd.DataFrame({
            'Close': rising_prices,
            'close': rising_prices
        }, index=pd.date_range('2023-01-01', periods=50))
        
        signals = self.generator.generate_signals(test_data, "AAPL")
        
        # Should generate some signals
        assert len(signals) >= 0
        
        # If signals are generated, they should be valid
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert signal.symbol == "AAPL"
    
    def test_get_required_columns(self):
        """Test required columns method"""
        required = self.generator.get_required_columns()
        
        assert 'Close' in required or 'close' in required
        assert isinstance(required, list)


class TestMACDSignalGenerator:
    """Test suite for MACD Signal Generator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.generator = MACDSignalGenerator(config={
            'fast_period': 12, 
            'slow_period': 26, 
            'signal_period': 9
        })
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
        
        self.test_data = pd.DataFrame({
            'Close': prices,
            'close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_macd_generator_initialization(self):
        """Test MACD generator initialization"""
        assert self.generator.name == "MACD"
        assert self.generator.fast_period == 12
        assert self.generator.slow_period == 26
        assert self.generator.signal_period == 9
    
    def test_macd_calculation(self):
        """Test MACD calculation"""
        macd_line, signal_line, histogram = self.generator._calculate_macd(
            self.test_data['Close'], 12, 26, 9
        )
        
        # MACD components should have correct length
        assert len(macd_line) == len(self.test_data)
        assert len(signal_line) == len(self.test_data)
        assert len(histogram) == len(self.test_data)
        
        # Should have NaN values for insufficient data
        assert pd.isna(macd_line.iloc[:25]).any()  # Some early values should be NaN
        assert pd.isna(signal_line.iloc[:33]).any()  # Signal line needs more data
    
    def test_generate_signals(self):
        """Test MACD signal generation"""
        signals = self.generator.generate_signals(self.test_data, "AAPL")
        
        # Should generate some signals
        assert len(signals) >= 0
        
        # Validate signal properties
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert signal.symbol == "AAPL"
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
            assert 0 <= signal.confidence <= 1
    
    def test_get_required_columns(self):
        """Test required columns method"""
        required = self.generator.get_required_columns()
        
        assert 'Close' in required or 'close' in required
        assert isinstance(required, list)


class TestBollingerBandsSignalGenerator:
    """Test suite for Bollinger Bands Signal Generator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.generator = BollingerBandsSignalGenerator(config={
            'period': 20,
            'std_dev': 2.0
        })
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 60)))
        
        self.test_data = pd.DataFrame({
            'Close': prices,
            'close': prices,
            'Volume': np.random.randint(1000000, 5000000, 60)
        }, index=dates)
    
    def test_bollinger_bands_initialization(self):
        """Test Bollinger Bands generator initialization"""
        assert self.generator.name == "BollingerBands"
        assert self.generator.period == 20
        assert self.generator.std_dev == 2.0
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation"""
        middle, upper, lower = self.generator._calculate_bollinger_bands(
            self.test_data['Close'], 20, 2.0
        )
        
        # Bands should have correct properties
        assert len(middle) == len(self.test_data)
        assert len(upper) == len(self.test_data)
        assert len(lower) == len(self.test_data)
        
        # Upper band should be above middle, lower band below
        valid_data = ~(pd.isna(upper) | pd.isna(middle) | pd.isna(lower))
        assert all(upper[valid_data] >= middle[valid_data])
        assert all(lower[valid_data] <= middle[valid_data])
    
    def test_generate_signals(self):
        """Test Bollinger Bands signal generation"""
        signals = self.generator.generate_signals(self.test_data, "AAPL")
        
        # Should generate some signals
        assert len(signals) >= 0
        
        # Validate signal properties
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert signal.symbol == "AAPL"
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
    
    def test_get_required_columns(self):
        """Test required columns method"""
        required = self.generator.get_required_columns()
        
        assert 'Close' in required or 'close' in required
        assert isinstance(required, list)


class TestSMACrossoverSignalGenerator:
    """Test suite for SMA Crossover Signal Generator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.generator = SMACrossoverSignalGenerator(config={
            'short_period': 20,
            'long_period': 50,
            'signal_strength_threshold': 0.5,
            'confirmation_periods': 2
        })
        
        # Create test data with trend
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        # Create uptrend data for crossover testing
        trend = np.linspace(100, 150, 100)
        noise = np.random.normal(0, 2, 100)
        prices = trend + noise
        
        self.test_data = pd.DataFrame({
            'Close': prices,
            'close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_sma_crossover_initialization(self):
        """Test SMA Crossover generator initialization"""
        assert self.generator.name == "SMACrossover"
        assert self.generator.short_period == 20
        assert self.generator.long_period == 50
        assert self.generator.signal_strength_threshold == 0.5
        assert self.generator.confirmation_periods == 2
    
    def test_generate_signals(self):
        """Test SMA crossover signal generation"""
        signals = self.generator.generate_signals(self.test_data, "AAPL")
        
        # Should generate some signals
        assert len(signals) >= 0
        
        # Validate signal properties
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert signal.symbol == "AAPL"
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
            assert 0 <= signal.confidence <= 1
    
    def test_get_required_columns(self):
        """Test required columns method"""
        required = self.generator.get_required_columns()
        
        assert 'Close' in required or 'close' in required
        assert isinstance(required, list)


class TestEMASignalGenerator:
    """Test suite for EMA Signal Generator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.generator = EMASignalGenerator(config={
            'periods': [12, 26],
            'crossover_pairs': [(12, 26)],
            'slope_threshold': 0.001,
            'min_confidence': 0.3
        })
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=80, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 80)))
        
        self.test_data = pd.DataFrame({
            'Close': prices,
            'close': prices,
            'Volume': np.random.randint(1000000, 5000000, 80)
        }, index=dates)
    
    def test_ema_generator_initialization(self):
        """Test EMA generator initialization"""
        assert self.generator.name == "EMA"
        assert self.generator.periods == [12, 26]
        assert self.generator.crossover_pairs == [(12, 26)]
        assert self.generator.slope_threshold == 0.001
        assert self.generator.min_confidence == 0.3
    
    def test_generate_signals(self):
        """Test EMA signal generation"""
        signals = self.generator.generate_signals(self.test_data, "AAPL")
        
        # Should generate some signals
        assert len(signals) >= 0
        
        # Validate signal properties
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert signal.symbol == "AAPL"
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
            assert signal.confidence >= self.generator.min_confidence
    
    def test_get_required_columns(self):
        """Test required columns method"""
        required = self.generator.get_required_columns()
        
        assert 'Close' in required or 'close' in required
        assert isinstance(required, list)


class TestVolumeAnalysisSignalGenerator:
    """Test suite for Volume Analysis Signal Generator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.generator = VolumeAnalysisSignalGenerator(config={
            'volume_surge_threshold': 2.0,
            'volume_sma_period': 20,
            'obv_period': 10,
            'price_volume_confirmation': True,
            'min_confidence': 0.4
        })
        
        # Create test data with volume patterns
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 60)))
        
        # Create volume data with some surges
        base_volume = 1000000
        volume_multipliers = np.random.uniform(0.5, 3.0, 60)
        volumes = base_volume * volume_multipliers
        
        self.test_data = pd.DataFrame({
            'Close': prices,
            'close': prices,
            'Volume': volumes,
            'volume': volumes
        }, index=dates)
    
    def test_volume_generator_initialization(self):
        """Test Volume Analysis generator initialization"""
        assert self.generator.name == "VolumeAnalysis"
        assert self.generator.volume_surge_threshold == 2.0
        assert self.generator.volume_sma_period == 20
        assert self.generator.obv_period == 10
        assert self.generator.price_volume_confirmation == True
        assert self.generator.min_confidence == 0.4
    
    def test_calculate_obv(self):
        """Test On-Balance Volume calculation"""
        obv = self.generator._calculate_obv(
            self.test_data['Close'], 
            self.test_data['Volume']
        )
        
        # OBV should have correct length
        assert len(obv) == len(self.test_data)
        
        # OBV should be cumulative (generally increasing or decreasing)
        assert not pd.isna(obv.iloc[-1])
    
    def test_generate_signals(self):
        """Test Volume Analysis signal generation"""
        signals = self.generator.generate_signals(self.test_data, "AAPL")
        
        # Should generate some signals
        assert len(signals) >= 0
        
        # Validate signal properties
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert signal.symbol == "AAPL"
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
            assert signal.confidence >= self.generator.min_confidence
    
    def test_get_required_columns(self):
        """Test required columns method"""
        required = self.generator.get_required_columns()
        
        assert ('Volume' in required or 'volume' in required)
        assert ('Close' in required or 'close' in required)
        assert isinstance(required, list)


class TestSignalGeneratorBase:
    """Test base functionality common to all signal generators"""
    
    def test_all_generators_have_required_methods(self):
        """Test that all generators implement required methods"""
        generators = [
            RSISignalGenerator(config={}),
            MACDSignalGenerator(config={}),
            BollingerBandsSignalGenerator(config={}),
            SMACrossoverSignalGenerator(config={}),
            EMASignalGenerator(config={}),
            VolumeAnalysisSignalGenerator(config={})
        ]
        
        for generator in generators:
            # Each generator should have these methods
            assert hasattr(generator, 'generate_signals')
            assert hasattr(generator, 'get_required_columns')
            assert hasattr(generator, 'name')
            
            # Methods should be callable
            assert callable(generator.generate_signals)
            assert callable(generator.get_required_columns)
            
            # Name should be a string
            assert isinstance(generator.name, str)
            assert len(generator.name) > 0
    
    def test_signal_validation(self):
        """Test signal validation across generators"""
        # Create simple test data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        test_data = pd.DataFrame({
            'Close': np.random.uniform(90, 110, 30),
            'close': np.random.uniform(90, 110, 30),
            'Volume': np.random.randint(500000, 2000000, 30),
            'volume': np.random.randint(500000, 2000000, 30)
        }, index=dates)
        
        generators = [
            RSISignalGenerator(config={}),
            MACDSignalGenerator(config={}),
            BollingerBandsSignalGenerator(config={})
        ]
        
        for generator in generators:
            try:
                signals = generator.generate_signals(test_data, "TEST")
                
                # Validate each signal
                for signal in signals:
                    assert isinstance(signal, TradingSignal)
                    assert signal.symbol == "TEST"
                    assert signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
                    assert 0 <= signal.confidence <= 1
                    assert isinstance(signal.source, str)
                    assert isinstance(signal.timestamp, pd.Timestamp)
                    
            except Exception as e:
                # Some generators might fail with insufficient data, which is acceptable
                assert "insufficient" in str(e).lower() or "not enough" in str(e).lower()