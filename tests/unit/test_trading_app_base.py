"""
Unit Tests for Trading App Base and Core Trading Components
"""
import sys
import os
import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.trading_app_base import TradingAppBase
from src.interfaces.trading_strategy import TradingSignal, SignalType
from src.interfaces.real_trading import AccountType, AccountBalance


class TestTradingAppBase:
    """Test suite for TradingAppBase"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mock_orchestrator = Mock()
        self.mock_account = Mock()
        
    def test_trading_app_base_initialization(self):
        """Test TradingAppBase initialization"""
        
        class TestApp(TradingAppBase):
            def __init__(self):
                super().__init__(starting_capital=50000, commission_rate=0.002)
                self.account = Mock()
            
            async def connect_account(self):
                return True
            
            async def disconnect_account(self):
                return True
        
        app = TestApp()
        
        # Verify initialization
        assert app.starting_capital == 50000
        assert app.commission_rate == 0.002
        assert len(app.symbols) > 0  # Should have trading symbols
        assert app.trading_strategies == {}

    @pytest.mark.asyncio
    async def test_make_enhanced_decision_success(self):
        """Test successful enhanced decision making"""
        
        class TestApp(TradingAppBase):
            def __init__(self):
                super().__init__(starting_capital=50000, commission_rate=0.002)
                self.account = Mock()
            
            async def connect_account(self):
                return True
            
            async def disconnect_account(self):
                return True
        
        app = TestApp()
        
        # Mock orchestrator response
        mock_signal_result = {
            'success': True,
            'action': 'BUY',
            'confidence': 0.8,
            'strength': 0.7,
            'signal_type': 'BUY',
            'reasoning': 'Strong ML signal'
        }
        
        app.production_orchestrator = Mock()
        app.production_orchestrator.generate_trading_signal = AsyncMock(return_value=mock_signal_result)
        
        # Test enhanced decision
        result = await app.make_enhanced_decision("AAPL", {})
        
        # Verify result
        assert result['symbol'] == "AAPL"
        assert result['action'] == 'BUY'
        assert result['confidence'] == 0.8
        assert result['reasoning'] == 'Strong ML signal'

    @pytest.mark.asyncio
    async def test_make_enhanced_decision_failure(self):
        """Test enhanced decision making with failure"""
        
        class TestApp(TradingAppBase):
            def __init__(self):
                super().__init__(starting_capital=50000, commission_rate=0.002)
                self.account = Mock()
            
            async def connect_account(self):
                return True
            
            async def disconnect_account(self):
                return True
        
        app = TestApp()
        
        # Mock orchestrator failure
        mock_signal_result = {
            'success': False,
            'reasoning': 'Signal generation failed'
        }
        
        app.production_orchestrator = Mock()
        app.production_orchestrator.generate_trading_signal = AsyncMock(return_value=mock_signal_result)
        
        # Test enhanced decision
        result = await app.make_enhanced_decision("AAPL", {})
        
        # Verify failure handling
        assert result['action'] == 'HOLD'
        assert result['confidence'] == 0.0
        assert 'Signal generation failed' in result['reasoning']

    @pytest.mark.asyncio
    async def test_make_production_decision_success(self):
        """Test successful production decision making"""
        
        class TestApp(TradingAppBase):
            def __init__(self):
                super().__init__(starting_capital=50000, commission_rate=0.002)
                self.account = Mock()
            
            async def connect_account(self):
                return True
            
            async def disconnect_account(self):
                return True
        
        app = TestApp()
        
        # Mock orchestrator response with market context
        mock_signal_result = {
            'success': True,
            'action': 'BUY',
            'confidence': 0.85,
            'strength': 0.8,
            'signal_type': 'BUY',
            'reasoning': 'Strong ML signal with bull market context',
            'market_context': {'spy_correlation': 0.7, 'market_regime': 'BULL'},
            'metadata': {'market_adjustments_applied': True}
        }
        
        app.production_orchestrator = Mock()
        app.production_orchestrator.generate_trading_signal = AsyncMock(return_value=mock_signal_result)
        
        # Test production decision
        result = await app.make_production_decision("AAPL", {}, {})
        
        # Verify result
        assert result['action'] == 'BUY'
        assert result['confidence'] == 0.85
        assert result['market_context'] is not None
        assert result['market_adjustments'] == True

    @pytest.mark.asyncio
    async def test_get_real_price_success(self):
        """Test successful real price retrieval"""
        
        class TestApp(TradingAppBase):
            def __init__(self):
                super().__init__(starting_capital=50000, commission_rate=0.002)
                self.account = Mock()
            
            async def connect_account(self):
                return True
            
            async def disconnect_account(self):
                return True
        
        app = TestApp()
        
        # Mock data provider
        app.data_provider = Mock()
        app.data_provider.get_current_price.return_value = 155.50
        
        # Test price retrieval
        price = await app.get_real_price("AAPL")
        
        # Verify price
        assert price == 155.50

    @pytest.mark.asyncio
    async def test_get_real_price_fallback(self):
        """Test real price retrieval with fallback to historical data"""
        
        class TestApp(TradingAppBase):
            def __init__(self):
                super().__init__(starting_capital=50000, commission_rate=0.002)
                self.account = Mock()
            
            async def connect_account(self):
                return True
            
            async def disconnect_account(self):
                return True
        
        app = TestApp()
        
        # Mock data provider - current price fails, historical succeeds
        app.data_provider = Mock()
        app.data_provider.get_current_price.return_value = None
        
        # Mock historical data
        historical_data = pd.DataFrame({
            'Close': [150, 152, 155],
            'close': [150, 152, 155]
        }, index=pd.date_range('2023-01-01', periods=3))
        app.data_provider.get_historical_data.return_value = historical_data
        
        # Test price retrieval
        price = await app.get_real_price("AAPL")
        
        # Verify fallback price (last close)
        assert price == 155.0

    @pytest.mark.asyncio
    async def test_run_trading_mode_simple(self):
        """Test simple trading mode"""
        
        class TestApp(TradingAppBase):
            def __init__(self):
                super().__init__(starting_capital=50000, commission_rate=0.002)
                self.account = Mock()
                self.symbols = ["AAPL", "NVDA"]
            
            async def connect_account(self):
                return True
            
            async def disconnect_account(self):
                return True
            
            async def _run_simple_mode(self):
                self.simple_mode_called = True
        
        app = TestApp()
        app.simple_mode_called = False
        
        # Test simple mode
        await app.run_trading_mode("simple", 2)
        
        # Verify simple mode was called
        assert app.simple_mode_called == True

    @pytest.mark.asyncio
    async def test_run_trading_mode_basic(self):
        """Test basic trading mode"""
        
        class TestApp(TradingAppBase):
            def __init__(self):
                super().__init__(starting_capital=50000, commission_rate=0.002)
                self.account = Mock()
                self.symbols = ["AAPL", "NVDA"]
            
            async def connect_account(self):
                return True
            
            async def disconnect_account(self):
                return True
            
            async def _run_basic_mode(self, symbol_limit):
                self.basic_mode_called = True
                self.basic_mode_symbol_limit = symbol_limit
        
        app = TestApp()
        app.basic_mode_called = False
        
        # Test basic mode
        await app.run_trading_mode("basic", 3)
        
        # Verify basic mode was called with correct parameters
        assert app.basic_mode_called == True
        assert app.basic_mode_symbol_limit == 3

    @pytest.mark.asyncio
    async def test_run_trading_mode_production(self):
        """Test production trading mode"""
        
        class TestApp(TradingAppBase):
            def __init__(self):
                super().__init__(starting_capital=50000, commission_rate=0.002)
                self.account = Mock()
                self.symbols = ["AAPL", "NVDA"]
            
            async def connect_account(self):
                return True
            
            async def disconnect_account(self):
                return True
            
            async def _run_production_mode(self, symbol_limit):
                self.production_mode_called = True
                self.production_mode_symbol_limit = symbol_limit
        
        app = TestApp()
        app.production_mode_called = False
        
        # Test production mode
        await app.run_trading_mode("production", 5)
        
        # Verify production mode was called with correct parameters
        assert app.production_mode_called == True
        assert app.production_mode_symbol_limit == 5

    def test_invalid_trading_mode(self):
        """Test invalid trading mode handling"""
        
        class TestApp(TradingAppBase):
            def __init__(self):
                super().__init__(starting_capital=50000, commission_rate=0.002)
                self.account = Mock()
            
            async def connect_account(self):
                return True
            
            async def disconnect_account(self):
                return True
        
        app = TestApp()
        
        # Test invalid mode should raise ValueError or handle gracefully
        with pytest.raises(ValueError):
            asyncio.run(app.run_trading_mode("invalid_mode", 1))


class TestTradingSignal:
    """Test suite for TradingSignal"""
    
    def test_trading_signal_creation(self):
        """Test TradingSignal creation"""
        
        signal = TradingSignal(
            symbol="AAPL",
            timestamp=pd.Timestamp.now(),
            signal_type=SignalType.BUY,
            confidence=0.8,
            strength=0.7,
            source="TEST",
            metadata={'test': True}
        )
        
        # Verify signal properties
        assert signal.symbol == "AAPL"
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence == 0.8
        assert signal.strength == 0.7
        assert signal.source == "TEST"
        assert signal.metadata == {'test': True}
    
    def test_trading_signal_types(self):
        """Test different signal types"""
        
        # Test BUY signal
        buy_signal = TradingSignal(
            symbol="AAPL",
            timestamp=pd.Timestamp.now(),
            signal_type=SignalType.BUY,
            confidence=0.9,
            strength=0.8,
            source="BUY_TEST"
        )
        assert buy_signal.signal_type == SignalType.BUY
        
        # Test SELL signal
        sell_signal = TradingSignal(
            symbol="AAPL",
            timestamp=pd.Timestamp.now(),
            signal_type=SignalType.SELL,
            confidence=0.7,
            strength=0.6,
            source="SELL_TEST"
        )
        assert sell_signal.signal_type == SignalType.SELL
        
        # Test HOLD signal
        hold_signal = TradingSignal(
            symbol="AAPL",
            timestamp=pd.Timestamp.now(),
            signal_type=SignalType.HOLD,
            confidence=0.5,
            strength=0.3,
            source="HOLD_TEST"
        )
        assert hold_signal.signal_type == SignalType.HOLD


class TestAccountBalance:
    """Test suite for AccountBalance"""
    
    def test_account_balance_creation(self):
        """Test AccountBalance creation"""
        
        balance = AccountBalance(
            cash=10000.0,
            buying_power=20000.0,
            portfolio_value=5000.0,
            day_trade_buying_power=40000.0,
            unsettled_funds=100.0,
            timestamp=pd.Timestamp.now()
        )
        
        # Verify balance properties
        assert balance.cash == 10000.0
        assert balance.buying_power == 20000.0
        assert balance.portfolio_value == 5000.0
        assert balance.day_trade_buying_power == 40000.0
        assert balance.unsettled_funds == 100.0
        assert balance.total_equity == 15000.0  # cash + portfolio_value
    
    def test_account_balance_calculations(self):
        """Test AccountBalance calculations"""
        
        balance = AccountBalance(
            cash=8000.0,
            buying_power=16000.0,
            portfolio_value=12000.0,
            day_trade_buying_power=32000.0,
            unsettled_funds=200.0,
            timestamp=pd.Timestamp.now()
        )
        
        # Test total equity calculation
        assert balance.total_equity == 20000.0  # 8000 + 12000
        
        # Test that buying power is independent of total equity
        assert balance.buying_power == 16000.0


class TestConfigurationManager:
    """Test configuration management functions"""
    
    @patch('config_manager.get_trading_symbols')
    def test_get_trading_symbols_mock(self, mock_get_symbols):
        """Test trading symbols retrieval with mock"""
        
        mock_symbols = ['AAPL', 'NVDA', 'TSLA', 'GOOGL', 'MSFT']
        mock_get_symbols.return_value = mock_symbols
        
        from config_manager import get_trading_symbols
        symbols = get_trading_symbols()
        
        assert symbols == mock_symbols
        assert len(symbols) == 5
    
    @patch('config_manager.get_default_symbol')
    def test_get_default_symbol_mock(self, mock_get_default):
        """Test default symbol retrieval with mock"""
        
        mock_get_default.return_value = 'AAPL'
        
        from config_manager import get_default_symbol
        default_symbol = get_default_symbol()
        
        assert default_symbol == 'AAPL'