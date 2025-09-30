"""
Comprehensive Test Configuration and Fixtures
"""
import sys
import os
from pathlib import Path

# Add project root to path for imports - MUST be done before other imports
project_root = Path(__file__).parent.absolute()
src_path = project_root / "src"

# Ensure our src directory is in Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = str(project_root)

# Now import pytest and other dependencies
try:
    import pytest
except ImportError:
    # If pytest is not available, create minimal mock for testing
    class MockPytest:
        class fixture:
            def __init__(self, scope="function", params=None):
                self.scope = scope
                self.params = params
            def __call__(self, func):
                return func
    pytest = MockPytest()

import asyncio
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import after path is set
from src.interfaces.trading_strategy import TradingSignal, SignalType
from src.interfaces.real_trading import AccountType, AccountBalance, PortfolioSummary
from src.interfaces.data_provider import DataProvider


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_data_provider():
    """Mock data provider for testing"""
    provider = Mock(spec=DataProvider)
    
    # Mock historical data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    mock_data = pd.DataFrame({
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(100, 200, 100),
        'Low': np.random.uniform(100, 200, 100),
        'Close': np.random.uniform(100, 200, 100),
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    provider.get_historical_data.return_value = mock_data
    provider.get_current_price.return_value = 150.0
    provider.get_market_data.return_value = {'AAPL': mock_data}
    
    return provider


@pytest.fixture
def mock_model_manager():
    """Mock model manager for testing"""
    from src.interfaces.model_manager import ModelManagerInterface
    manager = Mock(spec=ModelManagerInterface)
    manager.model_exists.return_value = True
    manager.predict.return_value = Mock(predictions=[0.7], confidence=0.8)
    manager.list_models.return_value = []
    return manager


@pytest.fixture
def sample_trading_signal():
    """Sample trading signal for testing"""
    return TradingSignal(
        symbol="AAPL",
        timestamp=pd.Timestamp.now(),
        signal_type=SignalType.BUY,
        confidence=0.8,
        strength=0.7,
        source="TEST",
        metadata={'test': True}
    )


@pytest.fixture
def mock_account_balance():
    """Mock account balance for testing"""
    return AccountBalance(
        cash=10000.0,
        buying_power=20000.0,
        portfolio_value=5000.0,
        day_trade_buying_power=40000.0,
        unsettled_funds=0.0,
        timestamp=pd.Timestamp.now()
    )


@pytest.fixture
def mock_portfolio_summary():
    """Mock portfolio summary for testing"""
    from src.interfaces.trading_strategy import Position
    return PortfolioSummary(
        total_equity=15000.0,
        cash_balance=10000.0,
        positions_value=5000.0,
        day_pnl=100.0,
        total_pnl=500.0,
        positions=[
            Position(symbol="AAPL", quantity=10, avg_price=150.0, current_price=155.0)
        ],
        timestamp=pd.Timestamp.now()
    )


@pytest.fixture
def mock_orchestrator():
    """Mock production trading orchestrator"""
    from src.enhanced_orchestrator import ProductionTradingOrchestrator
    
    with patch.object(ProductionTradingOrchestrator, '__init__', return_value=None):
        orchestrator = ProductionTradingOrchestrator.__new__(ProductionTradingOrchestrator)
        orchestrator.starting_capital = 100000
        orchestrator.commission_rate = 0.001
        orchestrator.slippage_rate = 0.0005
        orchestrator.symbols = ['AAPL', 'NVDA', 'TSLA']
        orchestrator.strategies = {}
        
        # Mock methods
        orchestrator.generate_trading_signal = Mock(return_value={
            'symbol': 'AAPL',
            'action': 'BUY',
            'confidence': 0.8,
            'strength': 0.7,
            'reasoning': 'Mock signal',
            'success': True
        })
        
        orchestrator.create_enhanced_strategy = Mock()
        orchestrator.train_ml_model = Mock(return_value={'success': True, 'test_accuracy': 0.8})
        orchestrator.analyze_market_context = Mock(return_value={'market_context': {'spy_correlation': 0.7}})
        
        return orchestrator


@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    return pd.DataFrame({
        'Open': np.random.uniform(140, 160, 30),
        'High': np.random.uniform(145, 165, 30),
        'Low': np.random.uniform(135, 155, 30),
        'Close': np.random.uniform(140, 160, 30),
        'Volume': np.random.randint(1000000, 5000000, 30)
    }, index=dates)


@pytest.fixture
def temp_models_dir(tmp_path):
    """Temporary directory for model files"""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return str(models_dir)


@pytest.fixture
def mock_yfinance():
    """Mock yfinance download function"""
    with patch('yfinance.download') as mock_download:
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Adj Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        mock_download.return_value = mock_data
        yield mock_download


# Test utilities
class TestDataGenerator:
    """Helper class to generate test data"""
    
    @staticmethod
    def create_price_data(days=100, start_price=100, symbol="AAPL"):
        """Create realistic price data"""
        dates = pd.date_range('2023-01-01', periods=days, freq='D')
        
        # Generate correlated OHLC data
        returns = np.random.normal(0.001, 0.02, days)
        prices = [start_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        closes = np.array(prices)
        opens = closes * np.random.uniform(0.99, 1.01, days)
        highs = np.maximum(opens, closes) * np.random.uniform(1.0, 1.02, days)
        lows = np.minimum(opens, closes) * np.random.uniform(0.98, 1.0, days)
        volumes = np.random.randint(1000000, 10000000, days)
        
        return pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes
        }, index=dates)
    
    @staticmethod
    def create_signals(count=10, symbol="AAPL"):
        """Create test trading signals"""
        signals = []
        for i in range(count):
            signal = TradingSignal(
                symbol=symbol,
                timestamp=pd.Timestamp.now() - pd.Timedelta(days=i),
                signal_type=np.random.choice([SignalType.BUY, SignalType.SELL, SignalType.HOLD]),
                confidence=np.random.uniform(0.5, 1.0),
                strength=np.random.uniform(0.3, 0.9),
                source=f"TEST_{i}",
                metadata={'test_id': i}
            )
            signals.append(signal)
        return signals


# Error handling for imports
def safe_import(module_path):
    """Safely import modules for testing"""
    try:
        parts = module_path.split('.')
        module = __import__(module_path)
        for part in parts[1:]:
            module = getattr(module, part)
        return module
    except ImportError:
        return None