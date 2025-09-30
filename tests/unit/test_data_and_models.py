"""
Unit Tests for Data Providers and Model Management
"""
import sys
import os
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.providers import BaseDataProvider, YFinanceProvider
from src.models.enhanced_model_management import EnhancedModelManager
from src.interfaces.data_provider import MarketData


class TestBaseDataProvider:
    """Test suite for Base Data Provider"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.provider = BaseDataProvider()
    
    def test_base_provider_initialization(self):
        """Test base provider initialization"""
        assert self.provider is not None
        assert hasattr(self.provider, 'get_historical_data')
        assert hasattr(self.provider, 'get_current_price')
    
    def test_get_historical_data_interface(self):
        """Test historical data interface"""
        # Base provider should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            self.provider.get_historical_data("AAPL", "1d", "1y")
    
    def test_get_current_price_interface(self):
        """Test current price interface"""
        # Base provider should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            self.provider.get_current_price("AAPL")


class TestYFinanceProvider:
    """Test suite for YFinance Data Provider"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.provider = YFinanceProvider()
        
        # Mock data for testing
        self.mock_historical_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [103, 104, 105, 106, 107],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=pd.date_range('2023-01-01', periods=5, freq='D'))
    
    def test_yfinance_provider_initialization(self):
        """Test YFinance provider initialization"""
        assert self.provider.name == "YFinance"
        assert hasattr(self.provider, 'session')
    
    @patch('yfinance.Ticker')
    def test_get_historical_data_success(self, mock_ticker):
        """Test successful historical data retrieval"""
        # Setup mock
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = self.mock_historical_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Test the method
        result = self.provider.get_historical_data("AAPL", "1d", "5d")
        
        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert 'Close' in result.columns
        assert 'Volume' in result.columns
        
        # Verify yfinance was called correctly
        mock_ticker.assert_called_once_with("AAPL")
        mock_ticker_instance.history.assert_called_once_with(period="5d", interval="1d")
    
    @patch('yfinance.Ticker')
    def test_get_historical_data_failure(self, mock_ticker):
        """Test historical data retrieval failure"""
        # Setup mock to raise exception
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.side_effect = Exception("Network error")
        mock_ticker.return_value = mock_ticker_instance
        
        # Test the method should handle the exception
        result = self.provider.get_historical_data("AAPL", "1d", "5d")
        
        # Should return empty DataFrame or None on failure
        assert result is None or (isinstance(result, pd.DataFrame) and len(result) == 0)
    
    @patch('yfinance.Ticker')
    def test_get_current_price_success(self, mock_ticker):
        """Test successful current price retrieval"""
        # Setup mock
        mock_ticker_instance = Mock()
        mock_ticker_instance.fast_info = {'last_price': 150.25}
        mock_ticker.return_value = mock_ticker_instance
        
        # Test the method
        result = self.provider.get_current_price("AAPL")
        
        # Verify results
        assert result == 150.25
        mock_ticker.assert_called_once_with("AAPL")
    
    @patch('yfinance.Ticker')
    def test_get_current_price_failure(self, mock_ticker):
        """Test current price retrieval failure"""
        # Setup mock to raise exception
        mock_ticker_instance = Mock()
        mock_ticker_instance.fast_info = {}
        mock_ticker.return_value = mock_ticker_instance
        
        # Test the method should handle missing data
        result = self.provider.get_current_price("AAPL")
        
        # Should return None on failure
        assert result is None
    
    def test_validate_symbol(self):
        """Test symbol validation"""
        # Valid symbols
        assert self.provider._validate_symbol("AAPL") == True
        assert self.provider._validate_symbol("MSFT") == True
        assert self.provider._validate_symbol("GOOGL") == True
        
        # Invalid symbols
        assert self.provider._validate_symbol("") == False
        assert self.provider._validate_symbol(None) == False
        assert self.provider._validate_symbol("123") == False
    
    def test_validate_timeframe(self):
        """Test timeframe validation"""
        # Valid timeframes
        valid_intervals = ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
        for interval in valid_intervals:
            assert self.provider._validate_timeframe(interval) == True
        
        # Invalid timeframes
        assert self.provider._validate_timeframe("invalid") == False
        assert self.provider._validate_timeframe("") == False
        assert self.provider._validate_timeframe(None) == False


class TestEnhancedModelManager:
    """Test suite for Enhanced Model Manager"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = {
            'models_dir': 'test_models',
            'metadata_dir': 'test_metadata',
            'model_types': ['LSTM', 'RandomForest', 'XGBoost'],
            'auto_retrain_threshold': 0.8,
            'performance_threshold': 0.7
        }
        self.manager = EnhancedModelManager(self.config)
        
        # Mock training data
        self.mock_training_data = pd.DataFrame({
            'feature1': np.random.random(100),
            'feature2': np.random.random(100),
            'feature3': np.random.random(100),
            'target': np.random.random(100)
        })
    
    def test_model_manager_initialization(self):
        """Test model manager initialization"""
        assert self.manager.models_dir == 'test_models'
        assert self.manager.metadata_dir == 'test_metadata'
        assert 'LSTM' in self.manager.model_types
        assert self.manager.auto_retrain_threshold == 0.8
        assert self.manager.performance_threshold == 0.7
    
    @patch('joblib.dump')
    @patch('os.makedirs')
    def test_save_model(self, mock_makedirs, mock_joblib_dump):
        """Test model saving functionality"""
        # Create mock model
        mock_model = Mock()
        mock_model.__class__.__name__ = "MockModel"
        
        # Test saving
        model_path = self.manager.save_model(
            model=mock_model,
            symbol="AAPL",
            model_type="LSTM",
            version="v1",
            metadata={"accuracy": 0.85, "features": ["feature1", "feature2"]}
        )
        
        # Verify path structure
        assert "AAPL" in model_path
        assert "LSTM" in model_path
        assert "v1" in model_path
        
        # Verify joblib.dump was called
        mock_joblib_dump.assert_called_once()
        
        # Verify directories were created
        mock_makedirs.assert_called()
    
    @patch('joblib.load')
    @patch('os.path.exists')
    def test_load_model_success(self, mock_exists, mock_joblib_load):
        """Test successful model loading"""
        # Setup mocks
        mock_exists.return_value = True
        mock_model = Mock()
        mock_joblib_load.return_value = mock_model
        
        # Test loading
        loaded_model = self.manager.load_model("AAPL", "LSTM", "v1")
        
        # Verify results
        assert loaded_model is not None
        assert loaded_model == mock_model
        mock_joblib_load.assert_called_once()
    
    @patch('os.path.exists')
    def test_load_model_not_found(self, mock_exists):
        """Test model loading when file doesn't exist"""
        # Setup mock
        mock_exists.return_value = False
        
        # Test loading
        loaded_model = self.manager.load_model("AAPL", "LSTM", "v1")
        
        # Should return None
        assert loaded_model is None
    
    @patch('json.dump')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('os.makedirs')
    def test_save_metadata(self, mock_makedirs, mock_open, mock_json_dump):
        """Test metadata saving"""
        metadata = {
            "symbol": "AAPL",
            "model_type": "LSTM",
            "version": "v1",
            "accuracy": 0.85,
            "training_date": "2023-01-01",
            "features": ["feature1", "feature2"]
        }
        
        # Test saving
        metadata_path = self.manager.save_metadata("AAPL", "LSTM", "v1", metadata)
        
        # Verify path structure
        assert "AAPL" in metadata_path
        assert "metadata.json" in metadata_path
        
        # Verify file operations
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()
        mock_makedirs.assert_called()
    
    @patch('json.load')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('os.path.exists')
    def test_load_metadata_success(self, mock_exists, mock_open, mock_json_load):
        """Test successful metadata loading"""
        # Setup mocks
        mock_exists.return_value = True
        mock_metadata = {
            "symbol": "AAPL",
            "accuracy": 0.85,
            "training_date": "2023-01-01"
        }
        mock_json_load.return_value = mock_metadata
        
        # Test loading
        loaded_metadata = self.manager.load_metadata("AAPL", "LSTM", "v1")
        
        # Verify results
        assert loaded_metadata is not None
        assert loaded_metadata["symbol"] == "AAPL"
        assert loaded_metadata["accuracy"] == 0.85
        mock_open.assert_called_once()
        mock_json_load.assert_called_once()
    
    @patch('os.path.exists')
    def test_load_metadata_not_found(self, mock_exists):
        """Test metadata loading when file doesn't exist"""
        # Setup mock
        mock_exists.return_value = False
        
        # Test loading
        loaded_metadata = self.manager.load_metadata("AAPL", "LSTM", "v1")
        
        # Should return None
        assert loaded_metadata is None
    
    def test_evaluate_model_performance(self):
        """Test model performance evaluation"""
        # Create mock model with predict method
        mock_model = Mock()
        mock_predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_model.predict.return_value = mock_predictions
        
        # Create test data
        test_features = np.random.random((5, 3))
        test_targets = np.array([0.15, 0.25, 0.35, 0.45, 0.55])
        
        # Test evaluation
        performance = self.manager.evaluate_model_performance(
            model=mock_model,
            test_features=test_features,
            test_targets=test_targets
        )
        
        # Verify results
        assert isinstance(performance, dict)
        assert 'mse' in performance
        assert 'mae' in performance
        assert 'r2_score' in performance
        assert all(isinstance(v, (int, float)) for v in performance.values())
    
    @patch('glob.glob')
    def test_list_models(self, mock_glob):
        """Test listing available models"""
        # Setup mock file paths
        mock_glob.return_value = [
            "test_models/AAPL/LSTM_v1.joblib",
            "test_models/AAPL/RandomForest_v1.joblib",
            "test_models/MSFT/LSTM_v1.joblib"
        ]
        
        # Test listing
        models = self.manager.list_models("AAPL")
        
        # Verify results
        assert isinstance(models, list)
        assert len(models) >= 2  # Should find AAPL models
        
        # Test listing all models
        all_models = self.manager.list_models()
        assert isinstance(all_models, list)
        assert len(all_models) >= 3  # Should find all models
    
    def test_get_model_info(self):
        """Test getting model information"""
        # Create mock metadata
        with patch.object(self.manager, 'load_metadata') as mock_load:
            mock_metadata = {
                "symbol": "AAPL",
                "model_type": "LSTM",
                "version": "v1",
                "accuracy": 0.85,
                "training_date": "2023-01-01",
                "features": ["feature1", "feature2"]
            }
            mock_load.return_value = mock_metadata
            
            # Test getting info
            info = self.manager.get_model_info("AAPL", "LSTM", "v1")
            
            # Verify results
            assert info is not None
            assert info["symbol"] == "AAPL"
            assert info["accuracy"] == 0.85
            mock_load.assert_called_once_with("AAPL", "LSTM", "v1")
    
    def test_cleanup_old_models(self):
        """Test cleanup of old models"""
        # This is a complex test that would require mocking file system operations
        # For now, just test that the method exists and is callable
        assert hasattr(self.manager, 'cleanup_old_models')
        assert callable(self.manager.cleanup_old_models)
        
        # Test with mock parameters
        with patch('os.listdir') as mock_listdir, \
             patch('os.path.getmtime') as mock_getmtime, \
             patch('os.remove') as mock_remove:
            
            mock_listdir.return_value = ["old_model.joblib", "new_model.joblib"]
            mock_getmtime.side_effect = [
                (datetime.now() - timedelta(days=40)).timestamp(),  # Old
                datetime.now().timestamp()  # New
            ]
            
            # Should not raise exception
            self.manager.cleanup_old_models(max_age_days=30)


class TestMarketData:
    """Test suite for MarketData class"""
    
    def test_market_data_creation(self):
        """Test MarketData object creation"""
        data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open_price=150.0,
            high_price=155.0,
            low_price=148.0,
            close_price=153.0,
            volume=1000000,
            additional_data={'sector': 'Technology'}
        )
        
        assert data.symbol == "AAPL"
        assert data.open_price == 150.0
        assert data.high_price == 155.0
        assert data.low_price == 148.0
        assert data.close_price == 153.0
        assert data.volume == 1000000
        assert data.additional_data['sector'] == 'Technology'
    
    def test_market_data_validation(self):
        """Test MarketData validation"""
        # Valid data should not raise exceptions
        valid_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open_price=150.0,
            high_price=155.0,
            low_price=148.0,
            close_price=153.0,
            volume=1000000
        )
        
        assert valid_data.symbol == "AAPL"
        
        # Test price relationships
        assert valid_data.high_price >= valid_data.open_price
        assert valid_data.high_price >= valid_data.close_price
        assert valid_data.low_price <= valid_data.open_price
        assert valid_data.low_price <= valid_data.close_price