"""
Unit tests for src/interfaces modules
Tests all interface classes and their contracts
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import datetime as dt
import numpy as np
from typing import Dict, Any, List

# Import interfaces to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.interfaces.data_provider import DataProvider
from src.interfaces.signal_generator import SignalGenerator, TradingSignal, SignalType
from src.interfaces.trading_strategy import TradingStrategy, Order, OrderType, OrderSide, Position, StrategyPerformance
from src.interfaces.model_manager import ModelManagerInterface, ModelMetadata, ModelStatus
from src.interfaces.risk_manager import RiskManager
## Backtester interfaces removed in lean build; import dropped


class TestDataProviderInterface(unittest.TestCase):
    """Test DataProvider interface methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_provider = Mock(spec=DataProvider)
        self.start_date = dt.datetime(2023, 1, 1)
        self.end_date = dt.datetime(2023, 12, 31)
        
    def test_get_historical_data_interface(self):
        """Test get_historical_data interface contract"""
        # Setup mock return value
        expected_data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [102.0, 103.0],
            'low': [99.0, 100.5],
            'close': [101.0, 102.0],
            'volume': [1000000, 1200000]
        })
        self.mock_provider.get_historical_data.return_value = expected_data
        
        # Test call
        result = self.mock_provider.get_historical_data("AAPL", self.start_date, self.end_date)
        
        # Verify
        self.mock_provider.get_historical_data.assert_called_once_with(
            "AAPL", self.start_date, self.end_date
        )
        pd.testing.assert_frame_equal(result, expected_data)
        
    def test_get_market_data_interface(self):
        """Test get_market_data interface contract"""
        symbols = ["AAPL", "GOOGL", "TSLA"]
        expected_data = {symbol: pd.DataFrame() for symbol in symbols}
        self.mock_provider.get_market_data.return_value = expected_data
        
        result = self.mock_provider.get_market_data(symbols, self.start_date, self.end_date)
        
        self.mock_provider.get_market_data.assert_called_once_with(
            symbols, self.start_date, self.end_date
        )
        self.assertEqual(result, expected_data)
        
    def test_get_current_price_interface(self):
        """Test get_current_price interface contract"""
        expected_price = 150.25
        self.mock_provider.get_current_price.return_value = expected_price
        
        result = self.mock_provider.get_current_price("AAPL")
        
        self.mock_provider.get_current_price.assert_called_once_with("AAPL")
        self.assertEqual(result, expected_price)
        
    def test_is_available_interface(self):
        """Test is_available interface contract"""
        self.mock_provider.is_available.return_value = True
        
        result = self.mock_provider.is_available()
        
        self.mock_provider.is_available.assert_called_once()
        self.assertTrue(result)
        
    def test_validate_symbol_interface(self):
        """Test validate_symbol interface contract"""
        self.mock_provider.validate_symbol.return_value = True
        
        result = self.mock_provider.validate_symbol("AAPL")
        
        self.mock_provider.validate_symbol.assert_called_once_with("AAPL")
        self.assertTrue(result)


class TestSignalGeneratorInterface(unittest.TestCase):
    """Test SignalGenerator interface and TradingSignal"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_generator = Mock(spec=SignalGenerator)
        self.mock_generator.name = "MockGenerator"
        self.mock_generator.config = {"param1": "value1"}
        self.mock_generator.enabled = True
        
    def test_trading_signal_creation(self):
        """Test TradingSignal dataclass creation"""
        timestamp = pd.Timestamp.now()
        signal = TradingSignal(
            symbol="AAPL",
            timestamp=timestamp,
            signal_type=SignalType.BUY,
            confidence=0.8,
            strength=0.7,
            source="RSI",
            metadata={"rsi_value": 25.0}
        )
        
        self.assertEqual(signal.symbol, "AAPL")
        self.assertEqual(signal.timestamp, timestamp)
        self.assertEqual(signal.signal_type, SignalType.BUY)
        self.assertEqual(signal.confidence, 0.8)
        self.assertEqual(signal.strength, 0.7)
        self.assertEqual(signal.source, "RSI")
        self.assertEqual(signal.metadata, {"rsi_value": 25.0})
        
    def test_trading_signal_to_dict(self):
        """Test TradingSignal to_dict method"""
        timestamp = pd.Timestamp("2023-01-01 10:00:00")
        signal = TradingSignal(
            symbol="AAPL",
            timestamp=timestamp,
            signal_type=SignalType.SELL,
            confidence=0.9,
            strength=0.8,
            source="MACD"
        )
        
        result = signal.to_dict()
        expected = {
            'symbol': "AAPL",
            'timestamp': timestamp,
            'signal_type': -1,  # SELL = -1
            'confidence': 0.9,
            'strength': 0.8,
            'source': "MACD",
            'metadata': {}
        }
        
        self.assertEqual(result, expected)
        
    def test_signal_type_enum_values(self):
        """Test SignalType enum values"""
        self.assertEqual(SignalType.BUY.value, 1)
        self.assertEqual(SignalType.SELL.value, -1)
        self.assertEqual(SignalType.HOLD.value, 0)
        
    def test_generate_signals_interface(self):
        """Test generate_signals interface contract"""
        data = pd.DataFrame({'close': [100, 101, 102]})
        expected_signals = [
            TradingSignal(
                symbol="TEST",
                timestamp=pd.Timestamp.now(),
                signal_type=SignalType.BUY,
                confidence=0.8,
                strength=0.7,
                source="Mock"
            )
        ]
        self.mock_generator.generate_signals.return_value = expected_signals
        
        result = self.mock_generator.generate_signals(data, "TEST")
        
        self.mock_generator.generate_signals.assert_called_once_with(data, "TEST")
        self.assertEqual(result, expected_signals)
        
    def test_get_required_columns_interface(self):
        """Test get_required_columns interface contract"""
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        self.mock_generator.get_required_columns.return_value = expected_columns
        
        result = self.mock_generator.get_required_columns()
        
        self.mock_generator.get_required_columns.assert_called_once()
        self.assertEqual(result, expected_columns)
        
    def test_validate_data_interface(self):
        """Test validate_data interface contract"""
        data = pd.DataFrame({'close': [100, 101, 102]})
        self.mock_generator.validate_data.return_value = True
        
        result = self.mock_generator.validate_data(data)
        
        self.mock_generator.validate_data.assert_called_once_with(data)
        self.assertTrue(result)


class TestTradingStrategyInterface(unittest.TestCase):
    """Test TradingStrategy interface and related classes"""
    
    def test_order_creation(self):
        """Test Order dataclass creation"""
        timestamp = pd.Timestamp.now()
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            price=150.0,
            timestamp=timestamp,
            metadata={"strategy": "RSI"}
        )
        
        self.assertEqual(order.symbol, "AAPL")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.quantity, 100)
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.price, 150.0)
        self.assertEqual(order.timestamp, timestamp)
        self.assertEqual(order.metadata, {"strategy": "RSI"})
        
    def test_order_to_dict(self):
        """Test Order to_dict method"""
        timestamp = pd.Timestamp("2023-01-01 10:00:00")
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            price=155.0,
            timestamp=timestamp
        )
        
        result = order.to_dict()
        expected = {
            'symbol': "AAPL",
            'side': "sell",
            'quantity': 50,
            'order_type': "limit",
            'price': 155.0,
            'stop_price': None,
            'timestamp': timestamp.isoformat(),
            'metadata': {}
        }
        
        self.assertEqual(result, expected)
        
    def test_position_properties(self):
        """Test Position dataclass and properties"""
        timestamp = pd.Timestamp.now()
        
        # Test long position
        long_position = Position(
            symbol="AAPL",
            quantity=100,
            avg_price=150.0,
            market_value=15000.0,
            unrealized_pnl=500.0,
            realized_pnl=0.0,
            timestamp=timestamp
        )
        
        self.assertTrue(long_position.is_long)
        self.assertFalse(long_position.is_short)
        self.assertFalse(long_position.is_flat)
        
        # Test short position
        short_position = Position(
            symbol="GOOGL",
            quantity=-50,
            avg_price=2800.0,
            market_value=-140000.0,
            unrealized_pnl=-1000.0,
            realized_pnl=0.0,
            timestamp=timestamp
        )
        
        self.assertFalse(short_position.is_long)
        self.assertTrue(short_position.is_short)
        self.assertFalse(short_position.is_flat)
        
        # Test flat position
        flat_position = Position(
            symbol="TSLA",
            quantity=0,
            avg_price=0.0,
            market_value=0.0,
            unrealized_pnl=0.0,
            realized_pnl=100.0,
            timestamp=timestamp
        )
        
        self.assertFalse(flat_position.is_long)
        self.assertFalse(flat_position.is_short)
        self.assertTrue(flat_position.is_flat)
        
    def test_strategy_performance_creation(self):
        """Test StrategyPerformance dataclass creation"""
        performance = StrategyPerformance(
            total_return=0.15,
            annualized_return=0.18,
            volatility=0.12,
            sharpe_ratio=1.5,
            max_drawdown=-0.08,
            win_rate=0.6,
            profit_factor=1.8,
            total_trades=50
        )
        
        self.assertEqual(performance.total_return, 0.15)
        self.assertEqual(performance.annualized_return, 0.18)
        self.assertEqual(performance.volatility, 0.12)
        self.assertEqual(performance.sharpe_ratio, 1.5)
        self.assertEqual(performance.max_drawdown, -0.08)
        self.assertEqual(performance.win_rate, 0.6)
        self.assertEqual(performance.profit_factor, 1.8)
        self.assertEqual(performance.total_trades, 50)


class TestModelManagerInterface(unittest.TestCase):
    """Test ModelManager interface and related classes"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_manager = Mock(spec=ModelManagerInterface)
        
    def test_model_metadata_creation(self):
        """Test ModelMetadata dataclass creation"""
        created_at = dt.datetime.now()
        training_period = (dt.datetime(2023, 1, 1), dt.datetime(2023, 12, 31))
        
        metadata = ModelMetadata(
            symbol="AAPL",
            version="v1.0",
            created_at=created_at,
            model_type="RandomForest",
            performance_metrics={"accuracy": 0.85, "precision": 0.82},
            feature_names=["rsi", "macd", "bb_upper"],
            training_period=training_period,
            config_snapshot={"n_estimators": 100},
            file_path="/models/aapl_v1.pkl",
            status=ModelStatus.TRAINED
        )
        
        self.assertEqual(metadata.symbol, "AAPL")
        self.assertEqual(metadata.version, "v1.0")
        self.assertEqual(metadata.created_at, created_at)
        self.assertEqual(metadata.model_type, "RandomForest")
        self.assertEqual(metadata.performance_metrics, {"accuracy": 0.85, "precision": 0.82})
        self.assertEqual(metadata.feature_names, ["rsi", "macd", "bb_upper"])
        self.assertEqual(metadata.training_period, training_period)
        self.assertEqual(metadata.config_snapshot, {"n_estimators": 100})
        self.assertEqual(metadata.file_path, "/models/aapl_v1.pkl")
        self.assertEqual(metadata.status, ModelStatus.TRAINED)
        
    def test_model_metadata_to_dict(self):
        """Test ModelMetadata to_dict method"""
        created_at = dt.datetime(2023, 6, 1, 10, 30, 45)
        training_period = (dt.datetime(2023, 1, 1), dt.datetime(2023, 5, 31))
        
        metadata = ModelMetadata(
            symbol="GOOGL",
            version="v2.1",
            created_at=created_at,
            model_type="XGBoost",
            performance_metrics={"f1_score": 0.78},
            feature_names=["sma_20", "ema_12"],
            training_period=training_period,
            config_snapshot={"max_depth": 6},
            file_path="/models/googl_v2.pkl"
        )
        
        result = metadata.to_dict()
        expected = {
            'symbol': "GOOGL",
            'version': "v2.1",
            'created_at': created_at.isoformat(),
            'model_type': "XGBoost",
            'performance_metrics': {"f1_score": 0.78},
            'feature_names': ["sma_20", "ema_12"],
            'training_period': [training_period[0].isoformat(), training_period[1].isoformat()],
            'config_snapshot': {"max_depth": 6},
            'file_path': "/models/googl_v2.pkl",
            'status': "trained"
        }
        
        self.assertEqual(result, expected)
        
    def test_model_status_enum_values(self):
        """Test ModelStatus enum values"""
        self.assertEqual(ModelStatus.TRAINED.value, "trained")
        self.assertEqual(ModelStatus.TRAINING.value, "training")
        self.assertEqual(ModelStatus.OUTDATED.value, "outdated")
        self.assertEqual(ModelStatus.ERROR.value, "error")
        self.assertEqual(ModelStatus.NOT_FOUND.value, "not_found")
        
    def test_save_model_interface(self):
        """Test save_model interface contract"""
        mock_model = Mock()
        mock_metadata = Mock(spec=ModelMetadata)
        expected_version = "v1.2.3"
        
        self.mock_manager.save_model.return_value = expected_version
        
        result = self.mock_manager.save_model("TSLA", mock_model, mock_metadata)
        
        self.mock_manager.save_model.assert_called_once_with("TSLA", mock_model, mock_metadata)
        self.assertEqual(result, expected_version)
        
    def test_load_model_interface(self):
        """Test load_model interface contract"""
        mock_model = Mock()
        mock_metadata = Mock(spec=ModelMetadata)
        expected_result = (mock_model, mock_metadata)
        
        self.mock_manager.load_model.return_value = expected_result
        
        result = self.mock_manager.load_model("NVDA", "v1.0")
        
        self.mock_manager.load_model.assert_called_once_with("NVDA", "v1.0")
        self.assertEqual(result, expected_result)
        
    def test_model_exists_interface(self):
        """Test model_exists interface contract"""
        self.mock_manager.model_exists.return_value = True
        
        result = self.mock_manager.model_exists("MSFT")
        
        self.mock_manager.model_exists.assert_called_once_with("MSFT", None)
        self.assertTrue(result)


class TestEnumValueConsistency(unittest.TestCase):
    """Test that all enum values are consistent and correct"""
    
    def test_signal_type_values(self):
        """Test SignalType enum consistency"""
        self.assertEqual(len(SignalType), 3)
        self.assertTrue(hasattr(SignalType, 'BUY'))
        self.assertTrue(hasattr(SignalType, 'SELL'))
        self.assertTrue(hasattr(SignalType, 'HOLD'))
        
    def test_order_type_values(self):
        """Test OrderType enum consistency"""
        self.assertEqual(len(OrderType), 4)
        self.assertTrue(hasattr(OrderType, 'MARKET'))
        self.assertTrue(hasattr(OrderType, 'LIMIT'))
        self.assertTrue(hasattr(OrderType, 'STOP'))
        self.assertTrue(hasattr(OrderType, 'STOP_LIMIT'))
        
    def test_order_side_values(self):
        """Test OrderSide enum consistency"""
        self.assertEqual(len(OrderSide), 2)
        self.assertTrue(hasattr(OrderSide, 'BUY'))
        self.assertTrue(hasattr(OrderSide, 'SELL'))
        
    def test_model_status_values(self):
        """Test ModelStatus enum consistency"""
        self.assertEqual(len(ModelStatus), 5)
        self.assertTrue(hasattr(ModelStatus, 'TRAINED'))
        self.assertTrue(hasattr(ModelStatus, 'TRAINING'))
        self.assertTrue(hasattr(ModelStatus, 'OUTDATED'))
        self.assertTrue(hasattr(ModelStatus, 'ERROR'))
        self.assertTrue(hasattr(ModelStatus, 'NOT_FOUND'))


if __name__ == '__main__':
    unittest.main()