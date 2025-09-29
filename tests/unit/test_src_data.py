"""
Unit tests for src/data modules
Tests data providers, analyzers, and preprocessors
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import datetime as dt
import numpy as np
import yfinance as yf
from typing import Dict, Any, List
import warnings

# Import data modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.providers import YFinanceProvider
from src.data.preprocessors import DataPreprocessor, TechnicalIndicatorCalculator
from src.analysis.enhanced_market_analysis import MarketContextAnalyzer


class TestYFinanceProvider(unittest.TestCase):
    """Test YFinanceProvider implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.provider = YFinanceProvider()
        self.start_date = dt.datetime(2023, 1, 1)
        self.end_date = dt.datetime(2023, 12, 31)
        
    @patch('src.data.providers.yf.Ticker')
    def test_get_historical_data_success(self, mock_ticker):
        """Test successful historical data retrieval"""
        # Setup mock
        mock_ticker_instance = Mock()
        expected_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [102.0, 103.0, 104.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [101.0, 102.0, 103.0],
            'Volume': [1000000, 1200000, 1100000],
            'Adj Close': [100.5, 101.5, 102.5]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        mock_ticker_instance.history.return_value = expected_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Execute
        result = self.provider.get_historical_data("AAPL", self.start_date, self.end_date)
        
        # Verify
        self.assertFalse(result.empty)
        self.assertIn('open', result.columns)
        self.assertIn('high', result.columns)
        self.assertIn('low', result.columns)
        self.assertIn('close', result.columns)
        self.assertIn('volume', result.columns)
        self.assertEqual(len(result), 3)
        
        # Check that close uses adjusted close
        np.testing.assert_array_equal(result['close'].values, [100.5, 101.5, 102.5])
        
    @patch('src.data.providers.yf.Ticker')
    def test_get_historical_data_empty_response(self, mock_ticker):
        """Test handling of empty data response"""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        result = self.provider.get_historical_data("INVALID", self.start_date, self.end_date)
        
        self.assertTrue(result.empty)
        
    @patch('src.data.providers.yf.Ticker')
    def test_get_historical_data_exception(self, mock_ticker):
        """Test handling of yfinance exceptions"""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.side_effect = Exception("Network error")
        mock_ticker.return_value = mock_ticker_instance
        
        result = self.provider.get_historical_data("AAPL", self.start_date, self.end_date)
        
        self.assertTrue(result.empty)
        
    @patch('src.data.providers.yf.Ticker')
    def test_get_historical_data_missing_columns(self, mock_ticker):
        """Test handling of missing required columns"""
        mock_ticker_instance = Mock()
        incomplete_data = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [102.0, 103.0]
            # Missing Low, Close, Volume
        }, index=pd.date_range('2023-01-01', periods=2))
        
        mock_ticker_instance.history.return_value = incomplete_data
        mock_ticker.return_value = mock_ticker_instance
        
        result = self.provider.get_historical_data("PARTIAL", self.start_date, self.end_date)
        
        self.assertTrue(result.empty)
        
    def test_get_market_data_multiple_symbols(self):
        """Test getting market data for multiple symbols"""
        symbols = ["AAPL", "GOOGL", "TSLA"]
        
        with patch.object(self.provider, 'get_historical_data') as mock_get_data:
            # Mock return different data for each symbol
            mock_get_data.side_effect = [
                pd.DataFrame({'close': [100, 101]}, index=pd.date_range('2023-01-01', periods=2)),
                pd.DataFrame({'close': [2800, 2850]}, index=pd.date_range('2023-01-01', periods=2)),
                pd.DataFrame({'close': [200, 205]}, index=pd.date_range('2023-01-01', periods=2))
            ]
            
            result = self.provider.get_market_data(symbols, self.start_date, self.end_date)
            
            self.assertEqual(len(result), 3)
            self.assertIn("AAPL", result)
            self.assertIn("GOOGL", result)
            self.assertIn("TSLA", result)
            self.assertEqual(mock_get_data.call_count, 3)
            
    @patch('src.data.providers.yf.Ticker')
    def test_get_current_price_success(self, mock_ticker):
        """Test successful current price retrieval"""
        mock_ticker_instance = Mock()
        mock_info = {'regularMarketPrice': 155.75}
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance
        
        result = self.provider.get_current_price("AAPL")
        
        self.assertEqual(result, 155.75)
        
    @patch('src.data.providers.yf.Ticker')
    def test_get_current_price_fallback_to_history(self, mock_ticker):
        """Test current price fallback to recent history"""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {}  # No price in info
        
        recent_data = pd.DataFrame({
            'Close': [150.0, 152.0, 155.0]
        }, index=pd.date_range('2023-01-01', periods=3))
        mock_ticker_instance.history.return_value = recent_data
        mock_ticker.return_value = mock_ticker_instance
        
        result = self.provider.get_current_price("AAPL")
        
        self.assertEqual(result, 155.0)  # Latest close price
        
    def test_is_available_always_true(self):
        """Test is_available returns True (YFinance is generally available)"""
        result = self.provider.is_available()
        self.assertTrue(result)
        
    @patch('src.data.providers.yf.Ticker')
    def test_validate_symbol_success(self, mock_ticker):
        """Test symbol validation success"""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {'symbol': 'AAPL', 'longName': 'Apple Inc.'}
        mock_ticker.return_value = mock_ticker_instance
        
        result = self.provider.validate_symbol("AAPL")
        
        self.assertTrue(result)
        
    @patch('src.data.providers.yf.Ticker')
    def test_validate_symbol_failure(self, mock_ticker):
        """Test symbol validation failure"""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {}  # Empty info indicates invalid symbol
        mock_ticker.return_value = mock_ticker_instance
        
        result = self.provider.validate_symbol("INVALID123")
        
        self.assertFalse(result)


class TestDataPreprocessor(unittest.TestCase):
    """Test DataPreprocessor functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = DataPreprocessor()
        
    def test_clean_data_basic_cleaning(self):
        """Test basic data cleaning operations"""
        # Create test data with issues
        raw_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 101.0, 101.0],  # Duplicate row
            'high': [102.0, 103.0, 104.0, 103.0, 103.0],
            'low': [99.0, 100.0, 101.0, 100.0, 100.0],
            'close': [101.0, 102.0, 103.0, 102.0, 102.0],
            'volume': [1000000, 1200000, 1100000, 1200000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=5).append(
            pd.date_range('2023-01-04', periods=1)  # Duplicate index
        ))
        
        result = self.preprocessor.clean_data(raw_data, "AAPL")
        
        # Should remove duplicates
        self.assertEqual(len(result), 5)  # Original had duplicate index
        self.assertFalse(result.index.duplicated().any())
        
    def test_clean_data_handle_negative_prices(self):
        """Test handling of negative or zero prices"""
        raw_data = pd.DataFrame({
            'open': [100.0, -1.0, 102.0],  # Negative price
            'high': [102.0, 0.0, 104.0],   # Zero price
            'low': [99.0, -2.0, 101.0],    # Negative price
            'close': [101.0, 0.0, 103.0],  # Zero price
            'volume': [1000000, 1200000, 1100000]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        result = self.preprocessor.clean_data(raw_data, "TEST")
        
        # Should handle negative/zero prices
        self.assertFalse(result.empty)
        # Check that negative values were handled (forward filled or interpolated)
        
    def test_clean_data_missing_required_columns(self):
        """Test handling of missing required columns"""
        incomplete_data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [102.0, 103.0]
            # Missing low, close, volume
        }, index=pd.date_range('2023-01-01', periods=2))
        
        result = self.preprocessor.clean_data(incomplete_data, "INCOMPLETE")
        
        self.assertTrue(result.empty)
        
    def test_clean_data_empty_input(self):
        """Test handling of empty input data"""
        empty_data = pd.DataFrame()
        
        result = self.preprocessor.clean_data(empty_data, "EMPTY")
        
        self.assertTrue(result.empty)
        
    def test_handle_missing_data_forward_fill(self):
        """Test missing data handling with forward fill strategy"""
        preprocessor = DataPreprocessor({'missing_data_strategy': 'forward_fill'})
        
        data_with_nan = pd.DataFrame({
            'open': [100.0, np.nan, 102.0],
            'high': [102.0, np.nan, 104.0],
            'low': [99.0, np.nan, 101.0],
            'close': [101.0, np.nan, 103.0],
            'volume': [1000000, np.nan, 1100000]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        result = preprocessor._handle_missing_data(data_with_nan)
        
        # Should have no NaN values after forward fill
        self.assertFalse(result.isnull().any().any())
        
    def test_validate_ohlc_relationships(self):
        """Test OHLC relationship validation"""
        invalid_ohlc = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [98.0, 100.0],    # High < Open (invalid)
            'low': [102.0, 103.0],    # Low > Open (invalid)
            'close': [101.0, 102.0],
            'volume': [1000000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=2))
        
        result = self.preprocessor._validate_ohlc_relationships(invalid_ohlc)
        
        # High should be >= max(open, close)
        self.assertGreaterEqual(result.iloc[0]['high'], max(result.iloc[0]['open'], result.iloc[0]['close']))
        # Low should be <= min(open, close)
        self.assertLessEqual(result.iloc[0]['low'], min(result.iloc[0]['open'], result.iloc[0]['close']))
        
    def test_remove_outliers(self):
        """Test outlier removal"""
        # Create data with outliers
        normal_prices = [100, 101, 102, 101, 103, 102, 104]
        outlier_prices = normal_prices + [1000]  # Add extreme outlier
        
        data_with_outliers = pd.DataFrame({
            'open': outlier_prices,
            'high': [p + 2 for p in outlier_prices],
            'low': [p - 1 for p in outlier_prices],
            'close': outlier_prices,
            'volume': [1000000] * len(outlier_prices)
        }, index=pd.date_range('2023-01-01', periods=len(outlier_prices)))
        
        result = self.preprocessor._remove_outliers(data_with_outliers)
        
        # Should remove or adjust the extreme outlier
        self.assertLess(result['close'].max(), 500)  # Outlier should be handled


class TestTechnicalIndicatorCalculator(unittest.TestCase):
    """Test TechnicalIndicatorCalculator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = TechnicalIndicatorCalculator()
        self.sample_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0],
            'high': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 106.0, 105.0, 104.0, 103.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 105.0, 104.0, 103.0, 102.0],
            'volume': [1000000, 1200000, 1100000, 1300000, 1250000, 1400000, 1350000, 1150000, 1050000, 1000000]
        }, index=pd.date_range('2023-01-01', periods=10))
        
    def test_calculate_sma(self):
        """Test Simple Moving Average calculation"""
        result = self.calculator.calculate_sma(self.sample_data['close'], window=3)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.sample_data))
        # First 2 values should be NaN due to window=3
        self.assertTrue(pd.isna(result.iloc[0]))
        self.assertTrue(pd.isna(result.iloc[1]))
        # Third value should be mean of first 3 closes: (101+102+103)/3 = 102
        self.assertAlmostEqual(result.iloc[2], 102.0, places=1)
        
    def test_calculate_ema(self):
        """Test Exponential Moving Average calculation"""
        result = self.calculator.calculate_ema(self.sample_data['close'], span=5)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.sample_data))
        # EMA should not have NaN values (unlike SMA)
        self.assertFalse(result.isna().all())
        
    def test_calculate_rsi(self):
        """Test RSI calculation"""
        result = self.calculator.calculate_rsi(self.sample_data['close'], window=5)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.sample_data))
        # RSI should be between 0 and 100
        valid_rsi = result.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())
        
    def test_calculate_macd(self):
        """Test MACD calculation"""
        result = self.calculator.calculate_macd(self.sample_data['close'])
        
        self.assertIsInstance(result, dict)
        self.assertIn('macd', result)
        self.assertIn('signal', result)
        self.assertIn('histogram', result)
        
        for key, series in result.items():
            self.assertIsInstance(series, pd.Series)
            self.assertEqual(len(series), len(self.sample_data))
            
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        result = self.calculator.calculate_bollinger_bands(self.sample_data['close'], window=5)
        
        self.assertIsInstance(result, dict)
        self.assertIn('bb_upper', result)
        self.assertIn('bb_middle', result)
        self.assertIn('bb_lower', result)
        
        for key, series in result.items():
            self.assertIsInstance(series, pd.Series)
            self.assertEqual(len(series), len(self.sample_data))
            
        # Upper should be > middle > lower (where not NaN)
        valid_indices = ~result['bb_upper'].isna()
        self.assertTrue((result['bb_upper'][valid_indices] > result['bb_middle'][valid_indices]).all())
        self.assertTrue((result['bb_middle'][valid_indices] > result['bb_lower'][valid_indices]).all())
        
    def test_calculate_all_indicators(self):
        """Test calculation of all indicators together"""
        result = self.calculator.calculate_all_indicators(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.sample_data))
        
        # Check that key indicators are present
        expected_indicators = ['sma_20', 'ema_12', 'rsi_14', 'macd', 'bb_upper', 'bb_lower']
        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns)
            
    def test_calculate_volume_indicators(self):
        """Test volume-based indicators"""
        # Test volume SMA
        vol_sma = self.calculator.calculate_sma(self.sample_data['volume'], window=3)
        self.assertIsInstance(vol_sma, pd.Series)
        
        # Test volume ratio (current volume / average volume)
        vol_ratio = self.sample_data['volume'] / vol_sma
        self.assertIsInstance(vol_ratio, pd.Series)
        
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        empty_data = pd.DataFrame()
        
        result = self.calculator.calculate_all_indicators(empty_data)
        
        self.assertTrue(result.empty)
        
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data for indicators"""
        minimal_data = pd.DataFrame({
            'close': [100.0, 101.0],  # Only 2 data points
            'volume': [1000000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=2))
        
        result = self.calculator.calculate_all_indicators(minimal_data)
        
        # Should return DataFrame with appropriate NaN handling
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)


class TestMarketDataAnalyzer(unittest.TestCase):
    """Test MarketDataAnalyzer functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = MarketContextAnalyzer()
        
    def test_analyze_price_action(self):
        """Test price action analysis"""
        price_data = pd.Series([100, 102, 104, 103, 105, 107, 106, 104, 102, 100])
        
        result = self.analyzer.analyze_price_action(price_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('trend', result)
        self.assertIn('volatility', result)
        self.assertIn('momentum', result)
        
    def test_analyze_volume_profile(self):
        """Test volume profile analysis"""
        price_data = pd.Series([100, 101, 102, 103, 104, 105, 104, 103, 102, 101])
        volume_data = pd.Series([1000, 1200, 1100, 1300, 1400, 1500, 1350, 1150, 1050, 1000])
        
        result = self.analyzer.analyze_volume_profile(price_data, volume_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('volume_weighted_price', result)
        self.assertIn('high_volume_levels', result)
        
    def test_detect_chart_patterns(self):
        """Test chart pattern detection"""
        # Create data that might form recognizable patterns
        price_data = pd.DataFrame({
            'high': [105, 104, 106, 105, 107, 106, 108, 107, 109, 108],
            'low': [95, 96, 94, 95, 93, 94, 92, 93, 91, 92],
            'close': [100, 101, 100, 102, 101, 103, 102, 104, 103, 105]
        })
        
        result = self.analyzer.detect_chart_patterns(price_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('patterns_detected', result)
        self.assertIn('support_levels', result)
        self.assertIn('resistance_levels', result)


if __name__ == '__main__':
    unittest.main()