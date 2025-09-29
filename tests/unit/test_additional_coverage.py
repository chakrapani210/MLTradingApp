"""
Enhanced Unit Tests for Higher Coverage
======================================
Additional test cases to achieve 90%+ coverage
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from tradingview_charts import (
    TradingViewChartGenerator,
    TechnicalIndicators,
    TradingSignal,
    Order,
    PortfolioSnapshot,
    SignalType,
    OrderType,
    create_sample_charts
)


class TestAdditionalCoverage(unittest.TestCase):
    """Additional tests to improve code coverage"""
    
    def setUp(self):
        """Set up test environment"""
        self.generator = TradingViewChartGenerator(results_dir="test_coverage_results")
        
        # Create sample data with specific characteristics
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        np.random.seed(123)  # Different seed for varied data
        
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(150.0, 160.0, 50),
            'High': np.random.uniform(160.0, 170.0, 50),
            'Low': np.random.uniform(140.0, 150.0, 50),
            'Close': np.random.uniform(150.0, 160.0, 50),
            'Volume': np.random.randint(500000, 2000000, 50).astype(float)
        }, index=dates)
    
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists("test_coverage_results"):
            import shutil
            shutil.rmtree("test_coverage_results", ignore_errors=True)
    
    def test_technical_indicators_edge_values(self):
        """Test technical indicators with edge values"""
        # Test with very small window
        small_data = pd.Series([100.0, 101.0, 99.0])
        
        # SMA with window = 1
        sma = TechnicalIndicators.sma(small_data, window=1)
        self.assertEqual(len(sma), 3)
        self.assertEqual(sma.iloc[0], 100.0)
        
        # EMA with window = 1
        ema = TechnicalIndicators.ema(small_data, window=1)
        self.assertEqual(len(ema), 3)
        
        # Bollinger bands with small window
        upper, middle, lower = TechnicalIndicators.bollinger_bands(small_data, window=2)
        self.assertEqual(len(upper), 3)
    
    def test_rsi_edge_cases(self):
        """Test RSI with edge cases"""
        # All increasing prices (should give RSI near 100)
        increasing_prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        rsi = TechnicalIndicators.rsi(increasing_prices, window=5)
        
        # Check that RSI is calculated (not all NaN)
        self.assertFalse(rsi.isna().all())
        
        # All decreasing prices (should give RSI near 0)
        decreasing_prices = pd.Series([105.0, 104.0, 103.0, 102.0, 101.0, 100.0])
        rsi = TechnicalIndicators.rsi(decreasing_prices, window=5)
        
        # Check that RSI is calculated
        self.assertFalse(rsi.isna().all())
        
        # Constant prices (should handle division by zero)
        constant_prices = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0])
        rsi = TechnicalIndicators.rsi(constant_prices, window=3)
        
        # Should handle constant prices without error
        self.assertEqual(len(rsi), 5)
    
    def test_macd_edge_cases(self):
        """Test MACD with edge cases"""
        # Test with minimal data
        min_data = pd.Series([100.0, 101.0, 102.0])
        macd_line, signal_line, histogram = TechnicalIndicators.macd(min_data, fast=1, slow=2, signal=1)
        
        self.assertEqual(len(macd_line), 3)
        self.assertEqual(len(signal_line), 3)
        self.assertEqual(len(histogram), 3)
        
        # Test with custom parameters
        custom_data = pd.Series(np.random.uniform(100, 110, 30))
        macd_line, signal_line, histogram = TechnicalIndicators.macd(custom_data, fast=5, slow=10, signal=3)
        
        # Verify histogram calculation
        for i in range(len(custom_data)):
            if not (pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i])):
                expected_hist = macd_line.iloc[i] - signal_line.iloc[i]
                self.assertAlmostEqual(histogram.iloc[i], expected_hist, places=6)
    
    def test_volume_profile_various_scenarios(self):
        """Test volume profile with various scenarios"""
        # Test with single price point
        single_price = pd.Series([150.0])
        single_volume = pd.Series([1000.0])
        
        price_levels, volume_profile = TechnicalIndicators.volume_profile(single_price, single_volume, bins=5)
        self.assertEqual(len(price_levels), 1)
        self.assertEqual(volume_profile[0], 1000.0)
        
        # Test with wide price range
        wide_prices = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])
        wide_volumes = pd.Series([1000.0, 2000.0, 1500.0, 3000.0, 2500.0])
        
        price_levels, volume_profile = TechnicalIndicators.volume_profile(wide_prices, wide_volumes, bins=3)
        self.assertEqual(len(price_levels), 3)
        self.assertAlmostEqual(volume_profile.sum(), wide_volumes.sum())
    
    @patch('tradingview_charts.yf.Ticker')
    def test_get_market_data_empty_response(self, mock_ticker):
        """Test market data retrieval with empty response"""
        # Mock yfinance to return empty DataFrame
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        # Should fallback to sample data
        data = self.generator.get_market_data("EMPTY", "1y")
        
        # Should return sample data instead
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
    
    def test_sample_data_generation_various_periods(self):
        """Test sample data generation with various time periods"""
        # Test with very short period
        short_data = self.generator._generate_sample_data("TEST", days=5)
        self.assertEqual(len(short_data), 5)
        
        # Test with longer period
        long_data = self.generator._generate_sample_data("TEST", days=500)
        self.assertEqual(len(long_data), 500)
        
        # Verify OHLC relationships for both
        for data in [short_data, long_data]:
            for i in range(len(data)):
                self.assertGreaterEqual(data['High'].iloc[i], data['Open'].iloc[i])
                self.assertGreaterEqual(data['High'].iloc[i], data['Close'].iloc[i])
                self.assertLessEqual(data['Low'].iloc[i], data['Open'].iloc[i])
                self.assertLessEqual(data['Low'].iloc[i], data['Close'].iloc[i])
    
    def test_signal_generation_edge_cases(self):
        """Test signal generation with edge case data"""
        # Create data that will trigger specific RSI conditions
        volatile_data = pd.DataFrame({
            'Close': [100, 90, 80, 70, 60, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 130, 120, 110, 100],
            'Volume': [1000000] * 19
        }, index=pd.date_range(start='2023-01-01', periods=19, freq='D'))
        
        signals = self.generator.generate_sample_signals(volatile_data, "VOLATILE")
        
        # Should generate both buy and sell signals due to RSI extremes
        signal_types = [s.signal_type for s in signals]
        self.assertTrue(any(st == SignalType.BUY for st in signal_types))
        self.assertTrue(any(st == SignalType.SELL for st in signal_types))
    
    def test_order_generation_with_zero_confidence_signals(self):
        """Test order generation with various confidence levels"""
        signals = [
            TradingSignal(
                timestamp=datetime.now(),
                symbol="TEST",
                signal_type=SignalType.BUY,
                confidence=0.0,  # Very low confidence
                price=100.0,
                source="Test",
                reason="Low confidence signal"
            ),
            TradingSignal(
                timestamp=datetime.now(),
                symbol="TEST",
                signal_type=SignalType.SELL,
                confidence=0.5,  # Exactly at threshold
                price=105.0,
                source="Test",
                reason="Threshold signal"
            ),
            TradingSignal(
                timestamp=datetime.now(),
                symbol="TEST",
                signal_type=SignalType.BUY,
                confidence=1.0,  # Maximum confidence
                price=110.0,
                source="Test",
                reason="High confidence signal"
            )
        ]
        
        orders = self.generator.generate_sample_orders(signals, portfolio_value=50000)
        
        # Should generate orders for signals with confidence >= 0.5
        self.assertEqual(len(orders), 2)  # Only the last two signals
        
        # Verify order properties
        buy_orders = [o for o in orders if o.side == 'buy']
        sell_orders = [o for o in orders if o.side == 'sell']
        
        self.assertEqual(len(buy_orders), 1)
        self.assertEqual(len(sell_orders), 1)
    
    def test_portfolio_generation_with_no_orders(self):
        """Test portfolio generation with no orders"""
        empty_orders = []
        portfolio = self.generator.generate_sample_portfolio(self.sample_data, empty_orders, initial_value=75000)
        
        # Should still generate portfolio snapshots
        self.assertGreater(len(portfolio), 0)
        
        # All snapshots should be based on initial value
        initial_snapshot = portfolio[0]
        self.assertEqual(initial_snapshot.total_value, 75000)
    
    def test_portfolio_generation_edge_cases(self):
        """Test portfolio generation with edge cases"""
        # Test with single data point
        single_point_data = self.sample_data.iloc[:1]
        single_order = [Order(
            timestamp=single_point_data.index[0],
            symbol="TEST",
            order_type=OrderType.LIMIT,
            side="buy",
            quantity=10.0,
            price=150.0,
            size_usd=1500.0
        )]
        
        portfolio = self.generator.generate_sample_portfolio(
            single_point_data, single_order, initial_value=25000
        )
        
        self.assertEqual(len(portfolio), 1)
        self.assertEqual(portfolio[0].total_value, 25000)
    
    @patch('tradingview_charts.yf.Ticker')
    def test_create_chart_with_exception_in_yfinance(self, mock_ticker):
        """Test chart creation when yfinance raises exception"""
        # Mock yfinance to raise a specific exception
        mock_ticker.side_effect = ConnectionError("Network error")
        
        # Should fallback to sample data and complete successfully
        # We won't actually create the chart to avoid file operations in tests
        try:
            # Test the data fetching part
            data = self.generator.get_market_data("ERROR_SYMBOL", "1y")
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0)
        except Exception:
            self.fail("Should handle yfinance exceptions gracefully")
    
    def test_create_sample_charts_with_empty_list(self):
        """Test create_sample_charts with empty symbol list"""
        with patch.object(TradingViewChartGenerator, 'create_comprehensive_chart') as mock_create:
            create_sample_charts([])
            # Should not call create_comprehensive_chart
            mock_create.assert_not_called()
    
    def test_color_scheme_completeness(self):
        """Test that color scheme has all required colors"""
        required_colors = {
            'background', 'grid', 'text', 'green', 'red',
            'blue', 'orange', 'purple', 'yellow'
        }
        
        self.assertEqual(set(self.generator.colors.keys()), required_colors)
        
        # Verify all colors are strings (hex codes)
        for color in self.generator.colors.values():
            self.assertIsInstance(color, str)
            self.assertTrue(color.startswith('#'))
    
    def test_dataclass_default_values(self):
        """Test dataclass default values"""
        # Test TradingSignal with minimal required fields
        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol="TEST",
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=100.0,
            source="TestSource"
        )
        self.assertEqual(signal.reason, "")  # Default value
        
        # Test Order with minimal required fields
        order = Order(
            timestamp=datetime.now(),
            symbol="TEST",
            order_type=OrderType.MARKET,
            side="buy",
            quantity=10.0,
            price=100.0,
            size_usd=1000.0
        )
        self.assertEqual(order.status, "filled")  # Default value
        self.assertEqual(order.reason, "")  # Default value


if __name__ == '__main__':
    # Run the additional coverage tests
    unittest.main(verbosity=2)