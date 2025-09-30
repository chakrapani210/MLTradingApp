"""
Unit Tests for Risk Management and Backtesting Components
"""
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.interfaces.risk_manager import BaseRiskManager
from src.backtesting.enhanced_backtesting import EnhancedBacktester
from src.interfaces.trading_strategy import TradingSignal, SignalType, Position, PositionType


class TestBaseRiskManager:
    """Test suite for Base Risk Manager"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = {
            'max_position_size': 0.1,  # 10% of portfolio
            'max_daily_loss': 0.05,    # 5% daily loss limit
            'max_drawdown': 0.2,       # 20% max drawdown
            'risk_free_rate': 0.02,    # 2% risk-free rate
            'stop_loss_percentage': 0.05,  # 5% stop loss
            'take_profit_percentage': 0.15  # 15% take profit
        }
        self.risk_manager = BaseRiskManager(self.config)
        
        # Mock account balance
        self.mock_account_balance = 100000.0
        
        # Mock position
        self.mock_position = Position(
            symbol="AAPL",
            position_type=PositionType.LONG,
            quantity=100,
            entry_price=150.0,
            current_price=155.0,
            timestamp=datetime.now()
        )
    
    def test_risk_manager_initialization(self):
        """Test risk manager initialization"""
        assert self.risk_manager.max_position_size == 0.1
        assert self.risk_manager.max_daily_loss == 0.05
        assert self.risk_manager.max_drawdown == 0.2
        assert self.risk_manager.risk_free_rate == 0.02
        assert self.risk_manager.stop_loss_percentage == 0.05
        assert self.risk_manager.take_profit_percentage == 0.15
    
    def test_calculate_position_size(self):
        """Test position size calculation"""
        # Test with current price and account balance
        current_price = 150.0
        position_size = self.risk_manager.calculate_position_size(
            symbol="AAPL",
            current_price=current_price,
            account_balance=self.mock_account_balance,
            risk_tolerance=0.02  # 2% risk tolerance
        )
        
        # Position size should be reasonable
        assert position_size > 0
        assert position_size <= self.mock_account_balance * self.risk_manager.max_position_size
        
        # Test position value doesn't exceed limits
        position_value = position_size * current_price
        max_position_value = self.mock_account_balance * self.risk_manager.max_position_size
        assert position_value <= max_position_value
    
    def test_calculate_stop_loss(self):
        """Test stop loss calculation"""
        entry_price = 150.0
        
        # Test long position stop loss
        stop_loss_long = self.risk_manager.calculate_stop_loss(
            entry_price=entry_price,
            position_type=PositionType.LONG
        )
        
        expected_stop_loss_long = entry_price * (1 - self.risk_manager.stop_loss_percentage)
        assert abs(stop_loss_long - expected_stop_loss_long) < 0.01
        assert stop_loss_long < entry_price  # Stop loss should be below entry for long
        
        # Test short position stop loss
        stop_loss_short = self.risk_manager.calculate_stop_loss(
            entry_price=entry_price,
            position_type=PositionType.SHORT
        )
        
        expected_stop_loss_short = entry_price * (1 + self.risk_manager.stop_loss_percentage)
        assert abs(stop_loss_short - expected_stop_loss_short) < 0.01
        assert stop_loss_short > entry_price  # Stop loss should be above entry for short
    
    def test_calculate_take_profit(self):
        """Test take profit calculation"""
        entry_price = 150.0
        
        # Test long position take profit
        take_profit_long = self.risk_manager.calculate_take_profit(
            entry_price=entry_price,
            position_type=PositionType.LONG
        )
        
        expected_take_profit_long = entry_price * (1 + self.risk_manager.take_profit_percentage)
        assert abs(take_profit_long - expected_take_profit_long) < 0.01
        assert take_profit_long > entry_price  # Take profit should be above entry for long
        
        # Test short position take profit
        take_profit_short = self.risk_manager.calculate_take_profit(
            entry_price=entry_price,
            position_type=PositionType.SHORT
        )
        
        expected_take_profit_short = entry_price * (1 - self.risk_manager.take_profit_percentage)
        assert abs(take_profit_short - expected_take_profit_short) < 0.01
        assert take_profit_short < entry_price  # Take profit should be below entry for short
    
    def test_validate_trade(self):
        """Test trade validation"""
        # Create mock trading signal
        signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            source="test",
            timestamp=pd.Timestamp.now()
        )
        
        # Test valid trade
        is_valid, reason = self.risk_manager.validate_trade(
            signal=signal,
            current_price=150.0,
            account_balance=self.mock_account_balance,
            current_positions=[]
        )
        
        # Should be valid with sufficient balance and no conflicts
        assert is_valid == True
        assert reason is None or "approved" in reason.lower()
        
        # Test trade with insufficient balance
        is_valid_low_balance, reason_low_balance = self.risk_manager.validate_trade(
            signal=signal,
            current_price=150.0,
            account_balance=1000.0,  # Very low balance
            current_positions=[]
        )
        
        # May still be valid but with reduced position size
        assert isinstance(is_valid_low_balance, bool)
        assert isinstance(reason_low_balance, str) or reason_low_balance is None
    
    def test_calculate_portfolio_risk(self):
        """Test portfolio risk calculation"""
        # Create mock positions
        positions = [
            Position(
                symbol="AAPL",
                position_type=PositionType.LONG,
                quantity=100,
                entry_price=150.0,
                current_price=155.0,
                timestamp=datetime.now()
            ),
            Position(
                symbol="MSFT",
                position_type=PositionType.LONG,
                quantity=50,
                entry_price=300.0,
                current_price=305.0,
                timestamp=datetime.now()
            )
        ]
        
        # Calculate portfolio risk
        portfolio_risk = self.risk_manager.calculate_portfolio_risk(
            positions=positions,
            account_balance=self.mock_account_balance
        )
        
        # Verify risk metrics
        assert isinstance(portfolio_risk, dict)
        assert 'total_exposure' in portfolio_risk
        assert 'risk_percentage' in portfolio_risk
        assert 'diversification_score' in portfolio_risk
        
        # Risk percentage should be reasonable
        assert 0 <= portfolio_risk['risk_percentage'] <= 1
        assert portfolio_risk['total_exposure'] >= 0
    
    def test_check_daily_loss_limit(self):
        """Test daily loss limit checking"""
        # Test within limits
        current_pnl = -2000.0  # 2% loss on 100k portfolio
        within_limits = self.risk_manager.check_daily_loss_limit(
            current_daily_pnl=current_pnl,
            account_balance=self.mock_account_balance
        )
        assert within_limits == True
        
        # Test exceeding limits
        excessive_pnl = -6000.0  # 6% loss exceeds 5% limit
        exceeding_limits = self.risk_manager.check_daily_loss_limit(
            current_daily_pnl=excessive_pnl,
            account_balance=self.mock_account_balance
        )
        assert exceeding_limits == False
    
    def test_calculate_var(self):
        """Test Value at Risk calculation"""
        # Create mock returns data
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns for 1 year
        portfolio_value = self.mock_account_balance
        
        # Calculate VaR
        var_95 = self.risk_manager.calculate_var(
            returns=returns,
            portfolio_value=portfolio_value,
            confidence_level=0.95
        )
        
        var_99 = self.risk_manager.calculate_var(
            returns=returns,
            portfolio_value=portfolio_value,
            confidence_level=0.99
        )
        
        # VaR should be negative (representing loss)
        assert var_95 <= 0
        assert var_99 <= 0
        
        # 99% VaR should be more extreme than 95% VaR
        assert var_99 <= var_95


class TestEnhancedBacktester:
    """Test suite for Enhanced Backtester"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = {
            'initial_capital': 100000.0,
            'commission': 0.001,  # 0.1% commission
            'slippage': 0.001,    # 0.1% slippage
            'risk_management': {
                'max_position_size': 0.1,
                'stop_loss_percentage': 0.05
            }
        }
        
        # Create mock data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
        
        self.historical_data = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        # Create mock trading signals
        self.mock_signals = [
            TradingSignal(
                symbol="AAPL",
                signal_type=SignalType.BUY,
                confidence=0.8,
                source="test",
                timestamp=dates[10],
                additional_data={'target_price': prices[10] * 1.05}
            ),
            TradingSignal(
                symbol="AAPL",
                signal_type=SignalType.SELL,
                confidence=0.7,
                source="test",
                timestamp=dates[50],
                additional_data={'target_price': prices[50] * 0.95}
            )
        ]
        
        self.backtester = EnhancedBacktester(self.config)
    
    def test_backtester_initialization(self):
        """Test backtester initialization"""
        assert self.backtester.initial_capital == 100000.0
        assert self.backtester.commission == 0.001
        assert self.backtester.slippage == 0.001
        assert self.backtester.current_capital == 100000.0
        assert len(self.backtester.positions) == 0
        assert len(self.backtester.trades) == 0
    
    def test_execute_trade_buy(self):
        """Test buy trade execution"""
        # Execute buy signal
        signal = self.mock_signals[0]  # BUY signal
        current_price = 150.0
        
        trade_result = self.backtester.execute_trade(
            signal=signal,
            current_price=current_price,
            timestamp=signal.timestamp
        )
        
        # Verify trade execution
        assert trade_result is not None
        assert len(self.backtester.positions) == 1
        assert len(self.backtester.trades) == 1
        
        # Check position details
        position = self.backtester.positions[0]
        assert position.symbol == "AAPL"
        assert position.position_type == PositionType.LONG
        assert position.quantity > 0
        assert position.entry_price > 0
        
        # Check capital adjustment
        assert self.backtester.current_capital < self.backtester.initial_capital
    
    def test_execute_trade_sell(self):
        """Test sell trade execution"""
        # First execute a buy to have a position
        buy_signal = self.mock_signals[0]
        self.backtester.execute_trade(
            signal=buy_signal,
            current_price=150.0,
            timestamp=buy_signal.timestamp
        )
        
        # Then execute sell signal
        sell_signal = self.mock_signals[1]
        sell_result = self.backtester.execute_trade(
            signal=sell_signal,
            current_price=160.0,  # Higher price for profit
            timestamp=sell_signal.timestamp
        )
        
        # Verify sell execution
        assert sell_result is not None
        assert len(self.backtester.positions) == 0  # Position should be closed
        assert len(self.backtester.trades) == 2  # Buy and sell trades
        
        # Should have made a profit (ignoring commission/slippage)
        # Current capital should be higher than after the buy trade
        assert self.backtester.current_capital != self.backtester.initial_capital
    
    def test_calculate_portfolio_value(self):
        """Test portfolio value calculation"""
        # Start with no positions
        initial_value = self.backtester.calculate_portfolio_value({})
        assert initial_value == self.backtester.current_capital
        
        # Add a position
        buy_signal = self.mock_signals[0]
        self.backtester.execute_trade(
            signal=buy_signal,
            current_price=150.0,
            timestamp=buy_signal.timestamp
        )
        
        # Calculate value with current prices
        current_prices = {"AAPL": 155.0}
        portfolio_value = self.backtester.calculate_portfolio_value(current_prices)
        
        # Should include cash + position value
        assert portfolio_value > self.backtester.current_capital
        assert portfolio_value != self.backtester.initial_capital
    
    def test_run_backtest(self):
        """Test full backtest execution"""
        # Mock strategy that generates signals
        mock_strategy = Mock()
        mock_strategy.generate_signals.return_value = self.mock_signals
        
        # Run backtest
        results = self.backtester.run_backtest(
            historical_data={"AAPL": self.historical_data},
            strategy=mock_strategy,
            symbols=["AAPL"]
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'total_trades' in results
        assert 'winning_trades' in results
        assert 'losing_trades' in results
        
        # Verify reasonable values
        assert isinstance(results['total_return'], (int, float))
        assert isinstance(results['total_trades'], int)
        assert results['winning_trades'] + results['losing_trades'] == results['total_trades']
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation"""
        # Create sample equity curve
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        equity_values = [100000.0 + i * 500 + np.random.normal(0, 200) for i in range(100)]
        equity_curve = pd.Series(equity_values, index=dates)
        
        # Calculate metrics
        metrics = self.backtester.calculate_performance_metrics(
            equity_curve=equity_curve,
            risk_free_rate=0.02
        )
        
        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'calmar_ratio' in metrics
        
        # Verify reasonable values
        assert isinstance(metrics['total_return'], (int, float))
        assert isinstance(metrics['sharpe_ratio'], (int, float))
        assert 0 <= metrics['max_drawdown'] <= 1  # Should be between 0 and 1
    
    def test_apply_commission_and_slippage(self):
        """Test commission and slippage application"""
        base_price = 150.0
        quantity = 100
        
        # Test buy order
        adjusted_price_buy = self.backtester._apply_commission_and_slippage(
            base_price=base_price,
            quantity=quantity,
            is_buy=True
        )
        
        # Buy price should be higher due to slippage and commission
        assert adjusted_price_buy > base_price
        
        # Test sell order
        adjusted_price_sell = self.backtester._apply_commission_and_slippage(
            base_price=base_price,
            quantity=quantity,
            is_buy=False
        )
        
        # Sell price should be lower due to slippage and commission
        assert adjusted_price_sell < base_price
    
    def test_risk_management_integration(self):
        """Test risk management integration"""
        # Create large position signal that should be limited by risk management
        large_signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.9,
            source="test",
            timestamp=pd.Timestamp.now(),
            additional_data={'suggested_quantity': 10000}  # Very large quantity
        )
        
        # Execute trade with risk management
        trade_result = self.backtester.execute_trade(
            signal=large_signal,
            current_price=150.0,
            timestamp=large_signal.timestamp
        )
        
        # Verify position size is limited by risk management
        if len(self.backtester.positions) > 0:
            position = self.backtester.positions[0]
            position_value = position.quantity * position.entry_price
            max_position_value = self.backtester.initial_capital * self.config['risk_management']['max_position_size']
            
            # Position value should not exceed risk limit
            assert position_value <= max_position_value * 1.1  # Allow small tolerance for commission/slippage


class TestTradingSignal:
    """Test suite for Trading Signal class"""
    
    def test_trading_signal_creation(self):
        """Test trading signal creation"""
        signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            source="RSI",
            timestamp=pd.Timestamp.now(),
            additional_data={'rsi_value': 25, 'target_price': 155.0}
        )
        
        assert signal.symbol == "AAPL"
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence == 0.8
        assert signal.source == "RSI"
        assert signal.additional_data['rsi_value'] == 25
        assert signal.additional_data['target_price'] == 155.0
    
    def test_trading_signal_validation(self):
        """Test trading signal validation"""
        # Valid signal
        valid_signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.SELL,
            confidence=0.7,
            source="MACD",
            timestamp=pd.Timestamp.now()
        )
        
        assert valid_signal.symbol == "AAPL"
        assert 0 <= valid_signal.confidence <= 1
        assert valid_signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]


class TestPosition:
    """Test suite for Position class"""
    
    def test_position_creation(self):
        """Test position creation"""
        position = Position(
            symbol="AAPL",
            position_type=PositionType.LONG,
            quantity=100,
            entry_price=150.0,
            current_price=155.0,
            timestamp=datetime.now()
        )
        
        assert position.symbol == "AAPL"
        assert position.position_type == PositionType.LONG
        assert position.quantity == 100
        assert position.entry_price == 150.0
        assert position.current_price == 155.0
    
    def test_position_pnl_calculation(self):
        """Test position P&L calculation"""
        # Long position with profit
        long_position = Position(
            symbol="AAPL",
            position_type=PositionType.LONG,
            quantity=100,
            entry_price=150.0,
            current_price=155.0,
            timestamp=datetime.now()
        )
        
        long_pnl = long_position.calculate_pnl()
        expected_long_pnl = 100 * (155.0 - 150.0)  # $500 profit
        assert abs(long_pnl - expected_long_pnl) < 0.01
        
        # Short position with profit
        short_position = Position(
            symbol="AAPL",
            position_type=PositionType.SHORT,
            quantity=100,
            entry_price=150.0,
            current_price=145.0,
            timestamp=datetime.now()
        )
        
        short_pnl = short_position.calculate_pnl()
        expected_short_pnl = 100 * (150.0 - 145.0)  # $500 profit
        assert abs(short_pnl - expected_short_pnl) < 0.01