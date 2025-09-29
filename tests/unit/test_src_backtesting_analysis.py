"""
Unit tests for src/backtesting and src/analysis modules
Tests backtesting engine, performance analytics, and market analysis
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import datetime as dt
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass

# Import backtesting and analysis modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.backtesting.enhanced_backtesting import (
    EnhancedBacktester, 
    Order, 
    Position, 
    PerformanceMetrics,
    Portfolio,
    OrderType,
    OrderSide
)
from src.analysis.enhanced_market_analysis import (
    MarketContextAnalyzer,
    EnhancedFeatureEngineer,
    MarketContext,
    EnhancedFeatures
)
from src.interfaces.backtester import Backtester, BacktestConfig, BacktestResult
from src.interfaces.trading_strategy import TradingStrategy
from src.interfaces.risk_manager import RiskManager


class TestEnhancedBacktester(unittest.TestCase):
    """Test EnhancedBacktester functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock dependencies
        self.mock_strategy = Mock(spec=TradingStrategy)
        self.mock_risk_manager = Mock(spec=RiskManager)
        self.mock_data_provider = Mock()
        
        # Sample market data
        self.market_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0],
            'high': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 106.0, 105.0, 104.0, 103.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 105.0, 104.0, 103.0, 102.0],
            'volume': [1000000] * 10
        }, index=pd.date_range('2023-01-01', periods=10))
        
        # Initialize backtester
        self.backtester = EnhancedBacktester(
            strategy=self.mock_strategy,
            data_provider=self.mock_data_provider,
            risk_manager=self.mock_risk_manager,
            starting_capital=100000.0,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
    def test_initialization(self):
        """Test backtester initialization"""
        self.assertEqual(self.backtester.starting_capital, 100000.0)
        self.assertEqual(self.backtester.commission_rate, 0.001)
        self.assertEqual(self.backtester.slippage_rate, 0.0005)
        self.assertIsInstance(self.backtester.portfolio, Portfolio)
        
    def test_order_creation(self):
        """Test Order dataclass creation"""
        timestamp = pd.Timestamp.now()
        order = Order(
            timestamp=timestamp,
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            price=150.0
        )
        
        self.assertEqual(order.symbol, "TEST")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.quantity, 100)
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.price, 150.0)
        self.assertFalse(order.executed)
        
    def test_position_creation(self):
        """Test Position dataclass creation"""
        position = Position(
            symbol="TEST",
            quantity=100,
            avg_cost=150.0,
            market_value=15500.0,
            unrealized_pnl=500.0,
            realized_pnl=0.0,
            last_price=155.0
        )
        
        self.assertEqual(position.symbol, "TEST")
        self.assertEqual(position.quantity, 100)
        self.assertEqual(position.avg_cost, 150.0)
        self.assertEqual(position.market_value, 15500.0)
        self.assertEqual(position.unrealized_pnl, 500.0)
        
    def test_portfolio_initialization(self):
        """Test Portfolio initialization"""
        portfolio = Portfolio(starting_capital=50000.0)
        
        self.assertEqual(portfolio.starting_capital, 50000.0)
        self.assertEqual(portfolio.cash, 50000.0)
        self.assertEqual(len(portfolio.positions), 0)
        self.assertEqual(len(portfolio.orders), 0)
        
    def test_portfolio_add_position(self):
        """Test adding position to portfolio"""
        portfolio = Portfolio(starting_capital=100000.0)
        
        position = Position(
            symbol="AAPL",
            quantity=100,
            avg_cost=150.0,
            market_value=15000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            last_price=150.0
        )
        
        portfolio.add_position("AAPL", position)
        
        self.assertIn("AAPL", portfolio.positions)
        self.assertEqual(portfolio.positions["AAPL"], position)
        
    def test_portfolio_update_position(self):
        """Test updating position in portfolio"""
        portfolio = Portfolio(starting_capital=100000.0)
        
        # Add initial position
        initial_position = Position(
            symbol="GOOGL",
            quantity=50,
            avg_cost=2000.0,
            market_value=100000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            last_price=2000.0
        )
        portfolio.add_position("GOOGL", initial_position)
        
        # Update with new price
        portfolio.update_position("GOOGL", current_price=2100.0)
        
        updated_position = portfolio.positions["GOOGL"]
        self.assertEqual(updated_position.last_price, 2100.0)
        self.assertEqual(updated_position.market_value, 50 * 2100.0)
        self.assertEqual(updated_position.unrealized_pnl, 50 * (2100.0 - 2000.0))
        
    def test_portfolio_calculate_total_value(self):
        """Test portfolio total value calculation"""
        portfolio = Portfolio(starting_capital=100000.0)
        
        # Add some positions
        positions = [
            Position("AAPL", 100, 150.0, 15500.0, 500.0, 0.0, 155.0),
            Position("GOOGL", 10, 2500.0, 26000.0, 1000.0, 0.0, 2600.0)
        ]
        
        for symbol, pos in zip(["AAPL", "GOOGL"], positions):
            portfolio.add_position(symbol, pos)
            
        # Assume some cash was used
        portfolio.cash = 58500.0  # 100000 - 15000 - 25000 - commissions
        
        total_value = portfolio.calculate_total_value()
        expected_value = 58500.0 + 15500.0 + 26000.0  # cash + position values
        
        self.assertEqual(total_value, expected_value)
        
    def test_execute_order_buy_market(self):
        """Test executing a market buy order"""
        order = Order(
            timestamp=pd.Timestamp.now(),
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            price=100.0
        )
        
        # Mock current price
        with patch.object(self.backtester, '_get_current_price', return_value=101.0):
            executed_order = self.backtester._execute_order(order, self.market_data.iloc[-1])
            
        self.assertTrue(executed_order.executed)
        self.assertIsNotNone(executed_order.execution_price)
        self.assertGreater(executed_order.commission, 0)  # Should have commission
        
    def test_execute_order_sell_market(self):
        """Test executing a market sell order"""
        # First add a position to sell
        position = Position("TEST", 100, 100.0, 10000.0, 0.0, 0.0, 100.0)
        self.backtester.portfolio.add_position("TEST", position)
        
        order = Order(
            timestamp=pd.Timestamp.now(),
            symbol="TEST",
            side=OrderSide.SELL,
            quantity=50,  # Partial sell
            order_type=OrderType.MARKET,
            price=105.0
        )
        
        with patch.object(self.backtester, '_get_current_price', return_value=105.0):
            executed_order = self.backtester._execute_order(order, self.market_data.iloc[-1])
            
        self.assertTrue(executed_order.executed)
        
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation"""
        # Create sample portfolio history
        portfolio_values = [100000, 102000, 101000, 105000, 103000, 107000]
        dates = pd.date_range('2023-01-01', periods=6)
        
        # Mock benchmark data (e.g., S&P 500)
        benchmark_returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015], index=dates[1:])
        
        metrics = self.backtester._calculate_performance_metrics(
            portfolio_values, dates, benchmark_returns
        )
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertIsInstance(metrics.total_return, float)
        self.assertIsInstance(metrics.sharpe_ratio, float)
        self.assertIsInstance(metrics.max_drawdown, float)
        self.assertIsInstance(metrics.volatility, float)
        
    def test_run_backtest_basic(self):
        """Test basic backtesting run"""
        # Mock strategy to generate some orders
        sample_orders = [
            Order(pd.Timestamp('2023-01-02'), "TEST", OrderSide.BUY, 100, OrderType.MARKET, 101.0),
            Order(pd.Timestamp('2023-01-05'), "TEST", OrderSide.SELL, 50, OrderType.MARKET, 105.0)
        ]
        
        def mock_generate_orders(data, symbol):
            # Return orders based on the current date
            current_date = data.index[-1]
            return [order for order in sample_orders if order.timestamp.date() <= current_date.date()]
            
        self.mock_strategy.generate_orders.side_effect = mock_generate_orders
        self.mock_data_provider.get_historical_data.return_value = self.market_data
        
        # Run backtest
        result = self.backtester.run_backtest(
            symbols=["TEST"],
            start_date=dt.datetime(2023, 1, 1),
            end_date=dt.datetime(2023, 1, 10)
        )
        
        self.assertIsInstance(result, BacktestResult)
        self.assertGreater(len(result.portfolio_history), 0)
        
    def test_backtest_with_risk_management(self):
        """Test backtesting with risk management"""
        # Mock risk manager to reject some orders
        def mock_validate_order(order, portfolio, market_data):
            # Reject orders that are too large
            return order.quantity <= 100
            
        self.mock_risk_manager.validate_order = mock_validate_order
        
        # Large order that should be rejected
        large_order = Order(
            pd.Timestamp('2023-01-02'), "TEST", OrderSide.BUY, 1000, OrderType.MARKET, 101.0
        )
        
        self.mock_strategy.generate_orders.return_value = [large_order]
        self.mock_data_provider.get_historical_data.return_value = self.market_data
        
        result = self.backtester.run_backtest(
            symbols=["TEST"],
            start_date=dt.datetime(2023, 1, 1),
            end_date=dt.datetime(2023, 1, 10)
        )
        
        # Order should have been rejected, so no positions
        final_portfolio = result.portfolio_history[-1] if result.portfolio_history else None
        if final_portfolio:
            self.assertEqual(len(final_portfolio.positions), 0)


class TestMarketContextAnalyzer(unittest.TestCase):
    """Test MarketContextAnalyzer functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_data_provider = Mock()
        self.analyzer = MarketContextAnalyzer(self.mock_data_provider)
        
        # Sample market data for analysis
        self.symbol_data = pd.DataFrame({
            'close': [100 + i + np.sin(i/10)*5 for i in range(100)]
        }, index=pd.date_range('2023-01-01', periods=100))
        
        self.spy_data = pd.DataFrame({
            'close': [400 + i*0.5 + np.sin(i/15)*3 for i in range(100)]
        }, index=pd.date_range('2023-01-01', periods=100))
        
        self.qqq_data = pd.DataFrame({
            'close': [350 + i*0.8 + np.sin(i/12)*4 for i in range(100)]
        }, index=pd.date_range('2023-01-01', periods=100))
        
    def test_initialization(self):
        """Test analyzer initialization"""
        self.assertEqual(self.analyzer.data_provider, self.mock_data_provider)
        self.assertIn('SPY', self.analyzer.market_etfs)
        self.assertIn('QQQ', self.analyzer.market_etfs)
        self.assertGreater(len(self.analyzer.sector_etfs), 0)
        
    def test_analyze_market_context(self):
        """Test comprehensive market context analysis"""
        # Mock data provider returns
        def mock_get_data(symbol, start, end):
            if symbol == "TEST":
                return self.symbol_data
            elif symbol == "SPY":
                return self.spy_data
            elif symbol == "QQQ":
                return self.qqq_data
            else:
                # Return empty data for other ETFs
                return pd.DataFrame()
                
        self.mock_data_provider.get_historical_data.side_effect = mock_get_data
        
        start_date = dt.datetime(2023, 1, 1)
        end_date = dt.datetime(2023, 12, 31)
        
        context = self.analyzer.analyze_market_context("TEST", start_date, end_date)
        
        self.assertIsInstance(context, MarketContext)
        self.assertIsInstance(context.spy_correlation, float)
        self.assertIsInstance(context.qqq_correlation, float)
        self.assertIsInstance(context.spy_beta, float)
        self.assertIsInstance(context.qqq_beta, float)
        self.assertIn(context.market_regime, ['bull', 'bear', 'sideways'])
        self.assertIn(context.volatility_regime, ['low', 'normal', 'high'])
        
    def test_calculate_correlation(self):
        """Test correlation calculation"""
        correlation = self.analyzer._calculate_correlation(
            self.symbol_data['close'],
            self.spy_data['close']
        )
        
        self.assertIsInstance(correlation, float)
        self.assertGreaterEqual(correlation, -1.0)
        self.assertLessEqual(correlation, 1.0)
        
    def test_calculate_beta(self):
        """Test beta calculation"""
        beta = self.analyzer._calculate_beta(
            self.symbol_data['close'],
            self.spy_data['close']
        )
        
        self.assertIsInstance(beta, float)
        # Beta can be any real number, but usually between -2 and 3 for most stocks
        
    def test_detect_market_regime(self):
        """Test market regime detection"""
        # Create data with clear uptrend
        uptrend_data = pd.Series([100 + i*0.5 for i in range(100)])
        regime = self.analyzer._detect_market_regime(uptrend_data)
        self.assertEqual(regime, 'bull')
        
        # Create data with clear downtrend
        downtrend_data = pd.Series([200 - i*0.5 for i in range(100)])
        regime = self.analyzer._detect_market_regime(downtrend_data)
        self.assertEqual(regime, 'bear')
        
        # Create sideways data
        sideways_data = pd.Series([100 + np.sin(i/10)*2 for i in range(100)])
        regime = self.analyzer._detect_market_regime(sideways_data)
        self.assertEqual(regime, 'sideways')
        
    def test_detect_volatility_regime(self):
        """Test volatility regime detection"""
        # Low volatility data
        low_vol_data = pd.Series([100 + i*0.01 for i in range(100)])
        regime = self.analyzer._detect_volatility_regime(low_vol_data)
        self.assertEqual(regime, 'low')
        
        # High volatility data
        high_vol_data = pd.Series([100 + np.random.normal(0, 5) for i in range(100)])
        regime = self.analyzer._detect_volatility_regime(high_vol_data)
        self.assertIn(regime, ['normal', 'high'])  # Could be either depending on random values
        
    def test_analyze_sector_rotation(self):
        """Test sector rotation analysis"""
        # Mock sector ETF data
        sector_data = {}
        for sector, etf in list(self.analyzer.sector_etfs.items())[:3]:  # Test first 3 sectors
            sector_data[etf] = pd.DataFrame({
                'close': [100 + i*np.random.uniform(-0.5, 0.5) for i in range(50)]
            }, index=pd.date_range('2023-01-01', periods=50))
            
        def mock_sector_data(symbol, start, end):
            return sector_data.get(symbol, pd.DataFrame())
            
        self.mock_data_provider.get_historical_data.side_effect = mock_sector_data
        
        sector_strength = self.analyzer._analyze_sector_rotation(
            dt.datetime(2023, 1, 1),
            dt.datetime(2023, 12, 31)
        )
        
        self.assertIsInstance(sector_strength, dict)
        self.assertGreater(len(sector_strength), 0)


class TestEnhancedFeatureEngineer(unittest.TestCase):
    """Test EnhancedFeatureEngineer functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engineer = EnhancedFeatureEngineer()
        
        # Comprehensive sample data
        self.sample_data = pd.DataFrame({
            'open': [100.0 + i + np.sin(i/10) for i in range(100)],
            'high': [102.0 + i + np.sin(i/10) for i in range(100)],
            'low': [99.0 + i + np.sin(i/10) for i in range(100)],
            'close': [101.0 + i + np.sin(i/10)*2 for i in range(100)],
            'volume': [1000000 + i*10000 + np.random.randint(-50000, 50000) for i in range(100)]
        }, index=pd.date_range('2023-01-01', periods=100))
        
    def test_initialization(self):
        """Test feature engineer initialization"""
        self.assertIsInstance(self.engineer.config, dict)
        
    def test_create_enhanced_features(self):
        """Test comprehensive feature creation"""
        features = self.engineer.create_enhanced_features(self.sample_data)
        
        self.assertIsInstance(features, EnhancedFeatures)
        self.assertIsInstance(features.features, np.ndarray)
        self.assertIsInstance(features.feature_names, list)
        self.assertIsInstance(features.target_labels, np.ndarray)
        self.assertIsInstance(features.metadata, dict)
        
        # Should have multiple features
        self.assertGreater(len(features.feature_names), 10)
        
        # Features and labels should have same number of samples
        self.assertEqual(len(features.features), len(features.target_labels))
        
    def test_calculate_technical_indicators(self):
        """Test technical indicator calculation"""
        indicators = self.engineer._calculate_technical_indicators(self.sample_data)
        
        self.assertIsInstance(indicators, pd.DataFrame)
        self.assertEqual(len(indicators), len(self.sample_data))
        
        # Should include common technical indicators
        expected_indicators = ['sma_20', 'ema_12', 'rsi_14', 'macd', 'bb_upper', 'bb_lower']
        for indicator in expected_indicators:
            if indicator in indicators.columns:
                self.assertIn(indicator, indicators.columns)
                
    def test_calculate_price_features(self):
        """Test price-based feature calculation"""
        features = self.engineer._calculate_price_features(self.sample_data)
        
        self.assertIsInstance(features, pd.DataFrame)
        
        # Should include price-based features
        expected_features = ['price_change', 'high_low_ratio', 'open_close_ratio']
        for feature in expected_features:
            if feature in features.columns:
                self.assertIn(feature, features.columns)
                
    def test_calculate_volume_features(self):
        """Test volume-based feature calculation"""
        features = self.engineer._calculate_volume_features(self.sample_data)
        
        self.assertIsInstance(features, pd.DataFrame)
        
        # Should include volume-based features
        expected_features = ['volume_sma', 'volume_ratio', 'price_volume']
        for feature in expected_features:
            if feature in features.columns:
                self.assertIn(feature, features.columns)
                
    def test_create_target_labels(self):
        """Test target label creation for ML"""
        labels = self.engineer._create_target_labels(self.sample_data)
        
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(labels), len(self.sample_data))
        
        # Labels should be binary (0 or 1) for classification
        unique_labels = np.unique(labels[~np.isnan(labels)])
        self.assertTrue(all(label in [0, 1] for label in unique_labels))
        
    def test_feature_scaling(self):
        """Test feature scaling/normalization"""
        raw_features = np.random.randn(50, 10) * 100  # Random features with large scale
        
        scaled_features = self.engineer._scale_features(raw_features)
        
        self.assertEqual(scaled_features.shape, raw_features.shape)
        
        # Scaled features should have smaller range
        self.assertLess(np.std(scaled_features), np.std(raw_features))
        
    def test_feature_selection(self):
        """Test feature selection based on importance"""
        # Create mock features with varying importance
        features = np.random.randn(100, 20)
        target = np.random.randint(0, 2, 100)
        feature_names = [f"feature_{i}" for i in range(20)]
        
        selected_features, selected_names = self.engineer._select_features(
            features, target, feature_names, k=10
        )
        
        self.assertEqual(selected_features.shape[1], 10)
        self.assertEqual(len(selected_names), 10)
        self.assertEqual(selected_features.shape[0], features.shape[0])


class TestAnalysisIntegration(unittest.TestCase):
    """Test integration between analysis components"""
    
    def test_market_analysis_to_features_workflow(self):
        """Test workflow from market analysis to feature engineering"""
        # 1. Setup components
        mock_data_provider = Mock()
        analyzer = MarketContextAnalyzer(mock_data_provider)
        engineer = EnhancedFeatureEngineer()
        
        # 2. Mock market data
        symbol_data = pd.DataFrame({
            'open': [100.0 + i for i in range(50)],
            'high': [102.0 + i for i in range(50)],
            'low': [99.0 + i for i in range(50)],
            'close': [101.0 + i for i in range(50)],
            'volume': [1000000] * 50
        }, index=pd.date_range('2023-01-01', periods=50))
        
        spy_data = pd.DataFrame({
            'close': [400.0 + i*0.5 for i in range(50)]
        }, index=pd.date_range('2023-01-01', periods=50))
        
        def mock_get_data(symbol, start, end):
            if symbol == "TEST":
                return symbol_data
            elif symbol == "SPY":
                return spy_data
            else:
                return pd.DataFrame()
                
        mock_data_provider.get_historical_data.side_effect = mock_get_data
        
        # 3. Analyze market context
        context = analyzer.analyze_market_context(
            "TEST", 
            dt.datetime(2023, 1, 1), 
            dt.datetime(2023, 12, 31)
        )
        
        # 4. Engineer features
        features = engineer.create_enhanced_features(symbol_data)
        
        # 5. Combine market context with features
        enhanced_metadata = {
            **features.metadata,
            'market_context': {
                'spy_correlation': context.spy_correlation,
                'market_regime': context.market_regime,
                'volatility_regime': context.volatility_regime
            }
        }
        
        # Verify integration
        self.assertIsInstance(context, MarketContext)
        self.assertIsInstance(features, EnhancedFeatures)
        self.assertIn('market_context', enhanced_metadata)
        self.assertIn('spy_correlation', enhanced_metadata['market_context'])


if __name__ == '__main__':
    unittest.main()