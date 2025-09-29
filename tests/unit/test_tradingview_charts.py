"""
Unit Tests for TradingView Charts Module
=======================================
Comprehensive test suite for the TradingView chart generation functionality
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
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


class TestSignalType(unittest.TestCase):
    """Test SignalType enum"""
    
    def test_signal_type_values(self):
        """Test SignalType enum values"""
        self.assertEqual(SignalType.BUY.value, 1)
        self.assertEqual(SignalType.SELL.value, -1)
        self.assertEqual(SignalType.HOLD.value, 0)


class TestOrderType(unittest.TestCase):
    """Test OrderType enum"""
    
    def test_order_type_values(self):
        """Test OrderType enum values"""
        self.assertEqual(OrderType.MARKET.value, "market")
        self.assertEqual(OrderType.LIMIT.value, "limit")
        self.assertEqual(OrderType.STOP.value, "stop")
        self.assertEqual(OrderType.STOP_LIMIT.value, "stop_limit")


class TestTradingSignal(unittest.TestCase):
    """Test TradingSignal dataclass"""
    
    def test_trading_signal_creation(self):
        """Test TradingSignal creation and attributes"""
        timestamp = datetime.now()
        signal = TradingSignal(
            timestamp=timestamp,
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.85,
            price=150.0,
            source="RSI_Strategy",
            reason="RSI oversold"
        )
        
        self.assertEqual(signal.timestamp, timestamp)
        self.assertEqual(signal.symbol, "AAPL")
        self.assertEqual(signal.signal_type, SignalType.BUY)
        self.assertEqual(signal.confidence, 0.85)
        self.assertEqual(signal.price, 150.0)
        self.assertEqual(signal.source, "RSI_Strategy")
        self.assertEqual(signal.reason, "RSI oversold")


class TestOrder(unittest.TestCase):
    """Test Order dataclass"""
    
    def test_order_creation(self):
        """Test Order creation and attributes"""
        timestamp = datetime.now()
        order = Order(
            timestamp=timestamp,
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side="buy",
            quantity=100.0,
            price=150.0,
            size_usd=15000.0,
            status="filled",
            reason="Signal based"
        )
        
        self.assertEqual(order.timestamp, timestamp)
        self.assertEqual(order.symbol, "AAPL")
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.side, "buy")
        self.assertEqual(order.quantity, 100.0)
        self.assertEqual(order.price, 150.0)
        self.assertEqual(order.size_usd, 15000.0)
        self.assertEqual(order.status, "filled")
        self.assertEqual(order.reason, "Signal based")
    
    def test_order_default_values(self):
        """Test Order default values"""
        timestamp = datetime.now()
        order = Order(
            timestamp=timestamp,
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side="buy",
            quantity=100.0,
            price=150.0,
            size_usd=15000.0
        )
        
        self.assertEqual(order.status, "filled")
        self.assertEqual(order.reason, "")


class TestPortfolioSnapshot(unittest.TestCase):
    """Test PortfolioSnapshot dataclass"""
    
    def test_portfolio_snapshot_creation(self):
        """Test PortfolioSnapshot creation and attributes"""
        timestamp = datetime.now()
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            total_value=100000.0,
            cash=20000.0,
            positions_value=80000.0,
            daily_pnl=1000.0,
            total_pnl=5000.0,
            daily_return=0.01,
            total_return=0.05
        )
        
        self.assertEqual(snapshot.timestamp, timestamp)
        self.assertEqual(snapshot.total_value, 100000.0)
        self.assertEqual(snapshot.cash, 20000.0)
        self.assertEqual(snapshot.positions_value, 80000.0)
        self.assertEqual(snapshot.daily_pnl, 1000.0)
        self.assertEqual(snapshot.total_pnl, 5000.0)
        self.assertEqual(snapshot.daily_return, 0.01)
        self.assertEqual(snapshot.total_return, 0.05)


class TestTechnicalIndicators(unittest.TestCase):
    """Test TechnicalIndicators class"""
    
    def setUp(self):
        """Set up test data"""
        self.sample_prices = pd.Series([100, 101, 102, 100, 99, 98, 101, 103, 105, 104])
        self.sample_volumes = pd.Series([1000, 1100, 900, 1200, 800, 1300, 1000, 1100, 900, 1000])
    
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation"""
        sma = TechnicalIndicators.sma(self.sample_prices, window=3)
        
        # Check that NaN values are at the beginning
        self.assertTrue(pd.isna(sma.iloc[0]))
        self.assertTrue(pd.isna(sma.iloc[1]))
        
        # Check calculated values
        expected_sma_2 = (100 + 101 + 102) / 3
        self.assertAlmostEqual(sma.iloc[2], expected_sma_2, places=2)
    
    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation"""
        ema = TechnicalIndicators.ema(self.sample_prices, window=3)
        
        # EMA should have no NaN values (uses expanding window initially)
        self.assertFalse(pd.isna(ema.iloc[0]))
        self.assertEqual(len(ema), len(self.sample_prices))
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(self.sample_prices, window=5, std_dev=2)
        
        # Check lengths
        self.assertEqual(len(upper), len(self.sample_prices))
        self.assertEqual(len(middle), len(self.sample_prices))
        self.assertEqual(len(lower), len(self.sample_prices))
        
        # Check that upper > middle > lower where not NaN
        for i in range(5, len(self.sample_prices)):
            if not (pd.isna(upper.iloc[i]) or pd.isna(middle.iloc[i]) or pd.isna(lower.iloc[i])):
                self.assertGreater(upper.iloc[i], middle.iloc[i])
                self.assertGreater(middle.iloc[i], lower.iloc[i])
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        rsi = TechnicalIndicators.rsi(self.sample_prices, window=5)
        
        # Check length
        self.assertEqual(len(rsi), len(self.sample_prices))
        
        # RSI should be between 0 and 100 (where not NaN)
        for value in rsi.dropna():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 100)
    
    def test_macd_calculation(self):
        """Test MACD calculation"""
        macd_line, signal_line, histogram = TechnicalIndicators.macd(self.sample_prices)
        
        # Check lengths
        self.assertEqual(len(macd_line), len(self.sample_prices))
        self.assertEqual(len(signal_line), len(self.sample_prices))
        self.assertEqual(len(histogram), len(self.sample_prices))
        
        # Histogram should be macd_line - signal_line
        for i in range(len(self.sample_prices)):
            if not (pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i])):
                expected_hist = macd_line.iloc[i] - signal_line.iloc[i]
                self.assertAlmostEqual(histogram.iloc[i], expected_hist, places=6)
    
    def test_volume_profile(self):
        """Test volume profile calculation"""
        price_levels, volume_profile = TechnicalIndicators.volume_profile(
            self.sample_prices, self.sample_volumes, bins=5
        )
        
        # Check lengths
        self.assertEqual(len(price_levels), 5)
        self.assertEqual(len(volume_profile), 5)
        
        # Check that total volume is conserved
        total_original_volume = self.sample_volumes.sum()
        total_profile_volume = volume_profile.sum()
        self.assertAlmostEqual(total_original_volume, total_profile_volume, places=0)


class TestTradingViewChartGenerator(unittest.TestCase):
    """Test TradingViewChartGenerator class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_results_dir = "test_results"
        self.generator = TradingViewChartGenerator(results_dir=self.test_results_dir)
        
        # Sample data for testing with proper data types
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        np.random.seed(42)  # For reproducible tests
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100.0, 110.0, 100),
            'High': np.random.uniform(110.0, 120.0, 100),
            'Low': np.random.uniform(90.0, 100.0, 100),
            'Close': np.random.uniform(100.0, 110.0, 100),
            'Volume': np.random.randint(100000, 1000000, 100).astype(float)
        }, index=dates)
    
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_results_dir):
            import shutil
            shutil.rmtree(self.test_results_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test TradingViewChartGenerator initialization"""
        self.assertEqual(self.generator.results_dir, self.test_results_dir)
        self.assertTrue(os.path.exists(self.test_results_dir))
        
        # Test color scheme
        expected_colors = {
            'background', 'grid', 'text', 'green', 'red', 
            'blue', 'orange', 'purple', 'yellow'
        }
        self.assertEqual(set(self.generator.colors.keys()), expected_colors)
    
    @patch('tradingview_charts.yf.Ticker')
    def test_get_market_data_success(self, mock_ticker):
        """Test successful market data retrieval"""
        # Mock yfinance response
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = self.sample_data
        mock_ticker.return_value = mock_ticker_instance
        
        data = self.generator.get_market_data("AAPL", "1y")
        
        # Verify data
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 100)
        mock_ticker.assert_called_once_with("AAPL")
        mock_ticker_instance.history.assert_called_once_with(period="1y")
    
    @patch('tradingview_charts.yf.Ticker')
    @patch.object(TradingViewChartGenerator, '_generate_sample_data')
    def test_get_market_data_failure(self, mock_sample_data, mock_ticker):
        """Test market data retrieval failure fallback"""
        # Mock yfinance to raise exception
        mock_ticker.side_effect = Exception("API Error")
        mock_sample_data.return_value = self.sample_data
        
        data = self.generator.get_market_data("AAPL", "1y")
        
        # Should fallback to sample data
        mock_sample_data.assert_called_once_with("AAPL")
        self.assertIsInstance(data, pd.DataFrame)
    
    def test_generate_sample_data(self):
        """Test sample data generation"""
        data = self.generator._generate_sample_data("AAPL", days=50)
        
        # Check structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 50)
        
        expected_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        self.assertEqual(set(data.columns), expected_columns)
        
        # Check OHLC relationships
        for i in range(len(data)):
            self.assertGreaterEqual(data['High'].iloc[i], data['Open'].iloc[i])
            self.assertGreaterEqual(data['High'].iloc[i], data['Close'].iloc[i])
            self.assertLessEqual(data['Low'].iloc[i], data['Open'].iloc[i])
            self.assertLessEqual(data['Low'].iloc[i], data['Close'].iloc[i])
    
    def test_generate_sample_signals(self):
        """Test sample signals generation"""
        signals = self.generator.generate_sample_signals(self.sample_data, "AAPL")
        
        # Check signal structure
        self.assertIsInstance(signals, list)
        for signal in signals:
            self.assertIsInstance(signal, TradingSignal)
            self.assertEqual(signal.symbol, "AAPL")
            self.assertIn(signal.signal_type, [SignalType.BUY, SignalType.SELL])
            self.assertGreaterEqual(signal.confidence, 0.5)
            self.assertLessEqual(signal.confidence, 0.9)
            self.assertEqual(signal.source, "RSI_Strategy")
    
    def test_generate_sample_orders(self):
        """Test sample orders generation"""
        # Create sample signals
        signals = [
            TradingSignal(
                timestamp=datetime.now(),
                symbol="AAPL",
                signal_type=SignalType.BUY,
                confidence=0.8,
                price=150.0,
                source="Test",
                reason="Test signal"
            ),
            TradingSignal(
                timestamp=datetime.now(),
                symbol="AAPL",
                signal_type=SignalType.SELL,
                confidence=0.7,
                price=155.0,
                source="Test",
                reason="Test signal"
            )
        ]
        
        orders = self.generator.generate_sample_orders(signals, portfolio_value=100000)
        
        # Check orders structure
        self.assertIsInstance(orders, list)
        self.assertEqual(len(orders), 2)  # Both signals should generate orders
        
        for order in orders:
            self.assertIsInstance(order, Order)
            self.assertEqual(order.symbol, "AAPL")
            self.assertIn(order.side, ["buy", "sell"])
            self.assertEqual(order.order_type, OrderType.MARKET)
            self.assertGreater(order.quantity, 0)
            self.assertGreater(order.size_usd, 0)
    
    def test_generate_sample_orders_low_confidence(self):
        """Test that low confidence signals don't generate orders"""
        signals = [
            TradingSignal(
                timestamp=datetime.now(),
                symbol="AAPL",
                signal_type=SignalType.BUY,
                confidence=0.3,  # Below threshold
                price=150.0,
                source="Test",
                reason="Test signal"
            )
        ]
        
        orders = self.generator.generate_sample_orders(signals)
        self.assertEqual(len(orders), 0)
    
    def test_generate_sample_portfolio(self):
        """Test sample portfolio generation"""
        orders = [
            Order(
                timestamp=datetime.now(),
                symbol="AAPL",
                order_type=OrderType.MARKET,
                side="buy",
                quantity=100.0,
                price=150.0,
                size_usd=15000.0
            )
        ]
        
        portfolio = self.generator.generate_sample_portfolio(
            self.sample_data, orders, initial_value=100000
        )
        
        # Check portfolio structure
        self.assertIsInstance(portfolio, list)
        self.assertGreater(len(portfolio), 0)
        
        for snapshot in portfolio:
            self.assertIsInstance(snapshot, PortfolioSnapshot)
            self.assertGreater(snapshot.total_value, 0)
            self.assertGreaterEqual(snapshot.cash, 0)
            self.assertGreaterEqual(snapshot.positions_value, 0)
    
    @patch('tradingview_charts.yf.Ticker')
    @patch('plotly.graph_objects.Figure.write_html')
    @patch('plotly.graph_objects.Figure.write_image')
    def test_create_comprehensive_chart_success(self, mock_write_image, mock_write_html, mock_ticker):
        """Test successful chart creation"""
        # Mock yfinance
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = self.sample_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock file writing
        mock_write_html.return_value = None
        mock_write_image.return_value = None
        
        filepath = self.generator.create_comprehensive_chart("AAPL", "6mo")
        
        # Check return value
        self.assertIsInstance(filepath, str)
        self.assertIn("trading_chart_AAPL", filepath)
        self.assertTrue(filepath.endswith(".html"))
        
        # Verify mocks were called
        mock_ticker.assert_called_once_with("AAPL")
        mock_write_html.assert_called_once()
        mock_write_image.assert_called_once()
    
    @patch('tradingview_charts.yf.Ticker')
    def test_create_comprehensive_chart_no_data(self, mock_ticker):
        """Test chart creation with no data"""
        # Mock yfinance to return empty DataFrame
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        with self.assertRaises(ValueError) as context:
            self.generator.create_comprehensive_chart("INVALID")
        
        self.assertIn("No data available", str(context.exception))
    
    @patch('tradingview_charts.yf.Ticker')
    @patch('plotly.graph_objects.Figure.write_html')
    @patch('plotly.graph_objects.Figure.write_image')
    def test_create_comprehensive_chart_png_failure(self, mock_write_image, mock_write_html, mock_ticker):
        """Test chart creation with PNG export failure"""
        # Mock yfinance
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = self.sample_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Mock file writing - HTML success, PNG failure
        mock_write_html.return_value = None
        mock_write_image.side_effect = Exception("PNG export failed")
        
        # Should not raise exception, just print warning
        filepath = self.generator.create_comprehensive_chart("AAPL")
        
        self.assertIsInstance(filepath, str)
        mock_write_html.assert_called_once()
        mock_write_image.assert_called_once()


class TestCreateSampleCharts(unittest.TestCase):
    """Test create_sample_charts function"""
    
    @patch.object(TradingViewChartGenerator, 'create_comprehensive_chart')
    def test_create_sample_charts_default(self, mock_create_chart):
        """Test create_sample_charts with default symbols"""
        mock_create_chart.return_value = "test_chart.html"
        
        create_sample_charts()
        
        # Should be called for default symbols
        self.assertEqual(mock_create_chart.call_count, 3)
        
        # Check call arguments
        calls = mock_create_chart.call_args_list
        symbols_called = [call[0][0] for call in calls]
        self.assertEqual(symbols_called, ['AAPL', 'TSLA', 'MSFT'])
    
    @patch.object(TradingViewChartGenerator, 'create_comprehensive_chart')
    def test_create_sample_charts_custom(self, mock_create_chart):
        """Test create_sample_charts with custom symbols"""
        mock_create_chart.return_value = "test_chart.html"
        custom_symbols = ['GOOGL', 'NVDA']
        
        create_sample_charts(custom_symbols)
        
        # Should be called for custom symbols
        self.assertEqual(mock_create_chart.call_count, 2)
        
        # Check call arguments
        calls = mock_create_chart.call_args_list
        symbols_called = [call[0][0] for call in calls]
        self.assertEqual(symbols_called, custom_symbols)
    
    @patch.object(TradingViewChartGenerator, 'create_comprehensive_chart')
    def test_create_sample_charts_exception_handling(self, mock_create_chart):
        """Test create_sample_charts exception handling"""
        mock_create_chart.side_effect = [
            "success_chart.html",
            Exception("Test error"),
            "success_chart2.html"
        ]
        
        # Should not raise exception even if one chart fails
        create_sample_charts(['AAPL', 'INVALID', 'TSLA'])
        
        # All three calls should be attempted
        self.assertEqual(mock_create_chart.call_count, 3)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_results_dir = "integration_test_results"
        self.generator = TradingViewChartGenerator(results_dir=self.test_results_dir)
    
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_results_dir):
            import shutil
            shutil.rmtree(self.test_results_dir, ignore_errors=True)
    
    def test_full_workflow_with_sample_data(self):
        """Test complete workflow using sample data"""
        # Generate sample data
        data = self.generator._generate_sample_data("TEST", days=30)
        
        # Generate signals
        signals = self.generator.generate_sample_signals(data, "TEST")
        
        # Generate orders
        orders = self.generator.generate_sample_orders(signals)
        
        # Generate portfolio
        portfolio = self.generator.generate_sample_portfolio(data, orders)
        
        # Verify workflow
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIsInstance(signals, list)
        self.assertIsInstance(orders, list)
        self.assertIsInstance(portfolio, list)
        
        # Test relationships
        if signals:
            self.assertGreaterEqual(len(orders), 0)  # Orders based on signals
        if orders:
            self.assertGreater(len(portfolio), 0)  # Portfolio affected by orders


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_technical_indicators_empty_data(self):
        """Test technical indicators with empty data"""
        empty_series = pd.Series([], dtype=float)
        
        # Should not crash but return empty results
        sma = TechnicalIndicators.sma(empty_series, 5)
        self.assertEqual(len(sma), 0)
        
        ema = TechnicalIndicators.ema(empty_series, 5)
        self.assertEqual(len(ema), 0)
    
    def test_technical_indicators_insufficient_data(self):
        """Test technical indicators with insufficient data"""
        short_series = pd.Series([100, 101])
        
        # SMA with window larger than data should return NaN
        sma = TechnicalIndicators.sma(short_series, window=5)
        self.assertTrue(pd.isna(sma).all())
        
        # EMA should still work
        ema = TechnicalIndicators.ema(short_series, window=5)
        self.assertFalse(pd.isna(ema).all())
    
    def test_volume_profile_edge_cases(self):
        """Test volume profile with edge cases"""
        # All same price
        same_prices = pd.Series([100.0, 100.0, 100.0])
        volumes = pd.Series([1000, 1100, 900])
        
        price_levels, volume_profile = TechnicalIndicators.volume_profile(same_prices, volumes, bins=3)
        
        # For same prices, should return single price level with total volume
        self.assertEqual(len(price_levels), 1)
        self.assertEqual(len(volume_profile), 1)
        self.assertAlmostEqual(volume_profile[0], volumes.sum())
        self.assertEqual(price_levels[0], 100.0)
    
    def test_chart_generator_invalid_results_dir(self):
        """Test chart generator with invalid results directory"""
        # Test with None as results_dir (should default to "results")
        generator = TradingViewChartGenerator(results_dir=None)
        # Should handle None gracefully by using default
        self.assertEqual(generator.results_dir, "results")
        self.assertTrue(os.path.exists(generator.results_dir))


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestSignalType,
        TestOrderType,
        TestTradingSignal,
        TestOrder,
        TestPortfolioSnapshot,
        TestTechnicalIndicators,
        TestTradingViewChartGenerator,
        TestCreateSampleCharts,
        TestIntegrationScenarios,
        TestEdgeCases
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print coverage summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")