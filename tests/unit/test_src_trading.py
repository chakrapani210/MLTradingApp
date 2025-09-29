"""
Unit tests for src/trading modules
Tests trading strategies, order sizing, and position management
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import datetime as dt
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass

# Import trading modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.trading.enhanced_strategies import (
    EnhancedMLTradingStrategy, 
    AutoOrderSizeManager, 
    OrderSizingConfig, 
    OrderSizingStrategy
)
from src.interfaces.trading_strategy import TradingStrategy, Order, OrderType, OrderSide, Position
from src.interfaces.signal_generator import TradingSignal, SignalType


class TestOrderSizingConfig(unittest.TestCase):
    """Test OrderSizingConfig dataclass"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        config = OrderSizingConfig()
        
        self.assertEqual(config.strategy, OrderSizingStrategy.PERCENTAGE)
        self.assertEqual(config.fixed_shares, 100)
        self.assertEqual(config.portfolio_pct, 0.1)
        self.assertEqual(config.min_shares, 10)
        self.assertEqual(config.max_shares, 1000)
        self.assertEqual(config.volatility_window, 20)
        self.assertEqual(config.volatility_target, 0.02)
        self.assertTrue(config.market_conditions_enabled)
        self.assertEqual(config.max_position_pct, 0.25)
        
    def test_custom_configuration(self):
        """Test custom configuration values"""
        config = OrderSizingConfig(
            strategy=OrderSizingStrategy.FIXED,
            fixed_shares=200,
            portfolio_pct=0.2,
            market_conditions_enabled=False
        )
        
        self.assertEqual(config.strategy, OrderSizingStrategy.FIXED)
        self.assertEqual(config.fixed_shares, 200)
        self.assertEqual(config.portfolio_pct, 0.2)
        self.assertFalse(config.market_conditions_enabled)
        
    def test_all_sizing_strategies(self):
        """Test all order sizing strategy options"""
        strategies = [
            OrderSizingStrategy.FIXED,
            OrderSizingStrategy.PERCENTAGE,
            OrderSizingStrategy.VOLATILITY_ADJUSTED,
            OrderSizingStrategy.KELLY_CRITERION,
            OrderSizingStrategy.RISK_PARITY
        ]
        
        for strategy in strategies:
            config = OrderSizingConfig(strategy=strategy)
            self.assertEqual(config.strategy, strategy)


class TestAutoOrderSizeManager(unittest.TestCase):
    """Test AutoOrderSizeManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = OrderSizingConfig()
        self.manager = AutoOrderSizeManager(self.config, starting_portfolio_value=100000)
        
        # Sample stock data
        self.stock_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [102.0, 103.0, 104.0, 105.0, 106.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'volume': [1000000] * 5
        }, index=pd.date_range('2023-01-01', periods=5))
        
        # Sample trading signal
        self.buy_signal = TradingSignal(
            symbol="TEST",
            timestamp=pd.Timestamp.now(),
            signal_type=SignalType.BUY,
            confidence=0.8,
            strength=0.7,
            source="RSI"
        )
        
    def test_initialization(self):
        """Test manager initialization"""
        self.assertEqual(self.manager.current_portfolio_value, 100000)
        self.assertEqual(self.manager.config.strategy, OrderSizingStrategy.PERCENTAGE)
        self.assertEqual(self.manager.current_positions, {})
        
    def test_calculate_order_size_fixed_strategy(self):
        """Test fixed order sizing strategy"""
        config = OrderSizingConfig(strategy=OrderSizingStrategy.FIXED, fixed_shares=250)
        manager = AutoOrderSizeManager(config)
        
        order_size = manager.calculate_order_size("TEST", self.buy_signal, self.stock_data)
        
        self.assertEqual(order_size, 250)
        
    def test_calculate_order_size_percentage_strategy(self):
        """Test percentage-based order sizing strategy"""
        config = OrderSizingConfig(
            strategy=OrderSizingStrategy.PERCENTAGE,
            portfolio_pct=0.1  # 10% of portfolio
        )
        manager = AutoOrderSizeManager(config, starting_portfolio_value=100000)
        
        # Mock current price at $100
        with patch.object(manager, '_get_current_price', return_value=100.0):
            order_size = manager.calculate_order_size("TEST", self.buy_signal, self.stock_data)
            
            # 10% of $100,000 = $10,000 / $100 = 100 shares
            self.assertEqual(order_size, 100)
            
    def test_calculate_order_size_volatility_adjusted(self):
        """Test volatility-adjusted order sizing strategy"""
        config = OrderSizingConfig(
            strategy=OrderSizingStrategy.VOLATILITY_ADJUSTED,
            base_shares=100,
            volatility_target=0.02
        )
        manager = AutoOrderSizeManager(config)
        
        # Create data with known volatility
        volatile_data = pd.DataFrame({
            'close': [100, 105, 95, 110, 90, 115, 85]  # High volatility
        }, index=pd.date_range('2023-01-01', periods=7))
        
        order_size = manager.calculate_order_size("VOLATILE", self.buy_signal, volatile_data)
        
        # Should adjust based on volatility
        self.assertIsInstance(order_size, int)
        self.assertGreater(order_size, 0)
        
    def test_calculate_order_size_kelly_criterion(self):
        """Test Kelly criterion order sizing strategy"""
        config = OrderSizingConfig(
            strategy=OrderSizingStrategy.KELLY_CRITERION,
            win_rate=0.6,
            avg_win=0.05,
            avg_loss=0.03,
            kelly_fraction=0.25
        )
        manager = AutoOrderSizeManager(config, starting_portfolio_value=100000)
        
        with patch.object(manager, '_get_current_price', return_value=100.0):
            order_size = manager.calculate_order_size("KELLY", self.buy_signal, self.stock_data)
            
            # Should calculate Kelly-optimal position size
            self.assertIsInstance(order_size, int)
            self.assertGreater(order_size, 0)
            
    def test_calculate_order_size_risk_parity(self):
        """Test risk parity order sizing strategy"""
        config = OrderSizingConfig(
            strategy=OrderSizingStrategy.RISK_PARITY,
            portfolio_risk_budget=0.1
        )
        manager = AutoOrderSizeManager(config)
        
        order_size = manager.calculate_order_size("RISK_PARITY", self.buy_signal, self.stock_data)
        
        # Should calculate risk-based position size
        self.assertIsInstance(order_size, int)
        self.assertGreater(order_size, 0)
        
    def test_market_conditions_adjustment(self):
        """Test market conditions adjustment"""
        config = OrderSizingConfig(
            strategy=OrderSizingStrategy.PERCENTAGE,
            market_conditions_enabled=True,
            bull_market_multiplier=1.2,
            bear_market_multiplier=0.8
        )
        manager = AutoOrderSizeManager(config, starting_portfolio_value=100000)
        
        # Test with bull market conditions (should increase size)
        with patch.object(manager, '_detect_market_condition', return_value='bull'):
            with patch.object(manager, '_get_current_price', return_value=100.0):
                bull_size = manager.calculate_order_size("BULL", self.buy_signal, self.stock_data)
                
        # Test with bear market conditions (should decrease size)
        with patch.object(manager, '_detect_market_condition', return_value='bear'):
            with patch.object(manager, '_get_current_price', return_value=100.0):
                bear_size = manager.calculate_order_size("BEAR", self.buy_signal, self.stock_data)
                
        # Bull market should result in larger position size than bear market
        self.assertGreater(bull_size, bear_size)
        
    def test_position_limits_enforcement(self):
        """Test that position limits are enforced"""
        config = OrderSizingConfig(
            strategy=OrderSizingStrategy.PERCENTAGE,
            portfolio_pct=0.5,  # 50% - very large
            max_position_pct=0.25  # But limited to 25%
        )
        manager = AutoOrderSizeManager(config, starting_portfolio_value=100000)
        
        with patch.object(manager, '_get_current_price', return_value=100.0):
            order_size = manager.calculate_order_size("LIMITED", self.buy_signal, self.stock_data)
            
            # Should be limited to max_position_pct
            max_shares = int((100000 * 0.25) / 100.0)
            self.assertLessEqual(order_size, max_shares)
            
    def test_minimum_shares_enforcement(self):
        """Test minimum shares enforcement"""
        config = OrderSizingConfig(
            strategy=OrderSizingStrategy.PERCENTAGE,
            portfolio_pct=0.001,  # Very small percentage
            min_shares=50  # But minimum 50 shares
        )
        manager = AutoOrderSizeManager(config, starting_portfolio_value=100000)
        
        with patch.object(manager, '_get_current_price', return_value=100.0):
            order_size = manager.calculate_order_size("MIN_TEST", self.buy_signal, self.stock_data)
            
            # Should enforce minimum shares
            self.assertGreaterEqual(order_size, 50)
            
    def test_update_portfolio_value(self):
        """Test portfolio value updates"""
        initial_value = self.manager.current_portfolio_value
        
        self.manager.update_portfolio_value(120000)
        
        self.assertEqual(self.manager.current_portfolio_value, 120000)
        self.assertNotEqual(self.manager.current_portfolio_value, initial_value)
        
    def test_add_position(self):
        """Test adding positions"""
        position = Position(
            symbol="TEST",
            quantity=100,
            avg_price=150.0,
            market_value=15000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            timestamp=pd.Timestamp.now()
        )
        
        self.manager.add_position("TEST", position)
        
        self.assertIn("TEST", self.manager.current_positions)
        self.assertEqual(self.manager.current_positions["TEST"], position)
        
    def test_get_position_size(self):
        """Test getting position size"""
        # No position initially
        size = self.manager.get_position_size("NEW_SYMBOL")
        self.assertEqual(size, 0)
        
        # Add a position
        position = Position(
            symbol="EXISTING",
            quantity=150,
            avg_price=200.0,
            market_value=30000.0,
            unrealized_pnl=1000.0,
            realized_pnl=0.0,
            timestamp=pd.Timestamp.now()
        )
        self.manager.add_position("EXISTING", position)
        
        size = self.manager.get_position_size("EXISTING")
        self.assertEqual(size, 150)


class TestEnhancedMLTradingStrategy(unittest.TestCase):
    """Test EnhancedMLTradingStrategy functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock dependencies
        self.mock_model_manager = Mock()
        self.mock_risk_manager = Mock()
        self.mock_signal_generators = [Mock(), Mock()]
        
        # Create strategy
        self.strategy = EnhancedMLTradingStrategy(
            model_manager=self.mock_model_manager,
            risk_manager=self.mock_risk_manager,
            signal_generators=self.mock_signal_generators
        )
        
        # Sample market data
        self.market_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [102.0, 103.0, 104.0, 105.0, 106.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'volume': [1000000] * 5
        }, index=pd.date_range('2023-01-01', periods=5))
        
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        self.assertEqual(self.strategy.model_manager, self.mock_model_manager)
        self.assertEqual(self.strategy.risk_manager, self.mock_risk_manager)
        self.assertEqual(len(self.strategy.signal_generators), 2)
        self.assertIsInstance(self.strategy.order_size_manager, AutoOrderSizeManager)
        
    def test_generate_signals(self):
        """Test signal generation from multiple sources"""
        # Setup mock signal generators
        mock_signals_1 = [
            TradingSignal("TEST", pd.Timestamp.now(), SignalType.BUY, 0.8, 0.7, "Gen1")
        ]
        mock_signals_2 = [
            TradingSignal("TEST", pd.Timestamp.now(), SignalType.SELL, 0.6, 0.5, "Gen2")
        ]
        
        self.mock_signal_generators[0].generate_signals.return_value = mock_signals_1
        self.mock_signal_generators[1].generate_signals.return_value = mock_signals_2
        
        # Generate signals
        signals = self.strategy.generate_signals("TEST", self.market_data)
        
        # Should combine signals from all generators
        self.assertEqual(len(signals), 2)
        sources = [signal.source for signal in signals]
        self.assertIn("Gen1", sources)
        self.assertIn("Gen2", sources)
        
    def test_create_orders_from_signals(self):
        """Test order creation from signals"""
        signals = [
            TradingSignal("TEST", pd.Timestamp.now(), SignalType.BUY, 0.8, 0.7, "RSI"),
            TradingSignal("TEST", pd.Timestamp.now(), SignalType.SELL, 0.6, 0.5, "MACD")
        ]
        
        # Mock order size calculation
        with patch.object(self.strategy.order_size_manager, 'calculate_order_size', return_value=100):
            orders = self.strategy.create_orders_from_signals(signals, self.market_data)
            
        self.assertEqual(len(orders), 2)
        
        # Check order details
        buy_order = orders[0]
        sell_order = orders[1]
        
        self.assertEqual(buy_order.side, OrderSide.BUY)
        self.assertEqual(sell_order.side, OrderSide.SELL)
        self.assertEqual(buy_order.quantity, 100)
        self.assertEqual(sell_order.quantity, 100)
        
    def test_apply_risk_management(self):
        """Test risk management application"""
        orders = [
            Order("TEST", OrderSide.BUY, 1000, OrderType.MARKET, 100.0),  # Large order
            Order("TEST", OrderSide.SELL, 50, OrderType.MARKET, 100.0)    # Small order
        ]
        
        # Mock risk manager to reject large order
        self.mock_risk_manager.validate_order.side_effect = [False, True]
        
        filtered_orders = self.strategy.apply_risk_management(orders)
        
        # Should filter out the rejected order
        self.assertEqual(len(filtered_orders), 1)
        self.assertEqual(filtered_orders[0].quantity, 50)
        
    def test_calculate_position_size(self):
        """Test position size calculation"""
        signal = TradingSignal("TEST", pd.Timestamp.now(), SignalType.BUY, 0.8, 0.7, "RSI")
        
        with patch.object(self.strategy.order_size_manager, 'calculate_order_size', return_value=150):
            position_size = self.strategy.calculate_position_size("TEST", signal, self.market_data)
            
        self.assertEqual(position_size, 150)
        
    def test_update_strategy_performance(self):
        """Test strategy performance updates"""
        # Initial performance should be None or default
        initial_performance = getattr(self.strategy, 'performance_metrics', None)
        
        # Mock performance update
        performance_data = {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08
        }
        
        self.strategy.update_performance_metrics(performance_data)
        
        # Should update performance metrics
        self.assertEqual(self.strategy.performance_metrics, performance_data)
        
    def test_strategy_configuration(self):
        """Test strategy configuration management"""
        config = {
            'max_positions': 10,
            'risk_per_trade': 0.02,
            'stop_loss_pct': 0.05
        }
        
        # Create strategy with config
        strategy = EnhancedMLTradingStrategy(
            model_manager=self.mock_model_manager,
            risk_manager=self.mock_risk_manager,
            signal_generators=self.mock_signal_generators,
            config=config
        )
        
        self.assertEqual(strategy.config['max_positions'], 10)
        self.assertEqual(strategy.config['risk_per_trade'], 0.02)
        self.assertEqual(strategy.config['stop_loss_pct'], 0.05)


class TestTradingStrategyIntegration(unittest.TestCase):
    """Test integration between trading strategy components"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.order_config = OrderSizingConfig(strategy=OrderSizingStrategy.PERCENTAGE)
        self.order_manager = AutoOrderSizeManager(self.order_config, 100000)
        
    def test_signal_to_order_workflow(self):
        """Test complete workflow from signal to order"""
        # 1. Create signal
        signal = TradingSignal(
            symbol="INTEG_TEST",
            timestamp=pd.Timestamp.now(),
            signal_type=SignalType.BUY,
            confidence=0.75,
            strength=0.8,
            source="Integration_Test"
        )
        
        # 2. Create market data
        market_data = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0, 104.0]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        # 3. Calculate order size
        with patch.object(self.order_manager, '_get_current_price', return_value=100.0):
            order_size = self.order_manager.calculate_order_size("INTEG_TEST", signal, market_data)
            
        # 4. Create order
        order = Order(
            symbol="INTEG_TEST",
            side=OrderSide.BUY,
            quantity=order_size,
            order_type=OrderType.MARKET,
            timestamp=pd.Timestamp.now()
        )
        
        # Verify end-to-end workflow
        self.assertEqual(order.symbol, signal.symbol)
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertGreater(order.quantity, 0)
        self.assertEqual(order.order_type, OrderType.MARKET)
        
    def test_position_management_lifecycle(self):
        """Test position management lifecycle"""
        # 1. Start with no positions
        self.assertEqual(self.order_manager.get_position_size("LIFECYCLE_TEST"), 0)
        
        # 2. Add a position
        position = Position(
            symbol="LIFECYCLE_TEST",
            quantity=100,
            avg_price=50.0,
            market_value=5000.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            timestamp=pd.Timestamp.now()
        )
        self.order_manager.add_position("LIFECYCLE_TEST", position)
        
        # 3. Verify position exists
        self.assertEqual(self.order_manager.get_position_size("LIFECYCLE_TEST"), 100)
        
        # 4. Update position (simulate price movement)
        updated_position = Position(
            symbol="LIFECYCLE_TEST",
            quantity=100,
            avg_price=50.0,
            market_value=5500.0,  # Price increased
            unrealized_pnl=500.0,  # Gained $500
            realized_pnl=0.0,
            timestamp=pd.Timestamp.now()
        )
        self.order_manager.add_position("LIFECYCLE_TEST", updated_position)
        
        # 5. Verify position update
        current_position = self.order_manager.current_positions["LIFECYCLE_TEST"]
        self.assertEqual(current_position.unrealized_pnl, 500.0)
        self.assertEqual(current_position.market_value, 5500.0)
        
    def test_multi_symbol_portfolio_management(self):
        """Test managing multiple symbols in portfolio"""
        symbols = ["AAPL", "GOOGL", "TSLA"]
        
        # Add positions for each symbol
        for i, symbol in enumerate(symbols):
            position = Position(
                symbol=symbol,
                quantity=100 + i*50,  # Different quantities
                avg_price=100.0 + i*50,  # Different prices
                market_value=(100 + i*50) * (100.0 + i*50),
                unrealized_pnl=i*100,  # Different P&L
                realized_pnl=0.0,
                timestamp=pd.Timestamp.now()
            )
            self.order_manager.add_position(symbol, position)
            
        # Verify all positions
        self.assertEqual(len(self.order_manager.current_positions), 3)
        
        for symbol in symbols:
            self.assertIn(symbol, self.order_manager.current_positions)
            self.assertGreater(self.order_manager.get_position_size(symbol), 0)
            
        # Test portfolio-wide calculations
        total_market_value = sum(
            pos.market_value for pos in self.order_manager.current_positions.values()
        )
        total_unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.order_manager.current_positions.values()
        )
        
        self.assertGreater(total_market_value, 0)
        self.assertGreaterEqual(total_unrealized_pnl, 0)


if __name__ == '__main__':
    unittest.main()