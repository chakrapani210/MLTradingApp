"""
Unit tests for src/models modules
Tests ML model management, training services, and prediction pipelines
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import pandas as pd
import datetime as dt
import numpy as np
import pickle
import json
import tempfile
import os
from typing import Dict, Any, List, Tuple
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Import model modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.enhanced_model_management import (
    EnhancedModelManager, 
    MLSignalGenerator, 
    ModelTrainingService,
    RNNModelWrapper
)
from src.interfaces.model_manager import ModelManagerInterface, ModelMetadata, ModelStatus
from src.interfaces.signal_generator import TradingSignal, SignalType


class TestEnhancedModelManager(unittest.TestCase):
    """Test EnhancedModelManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = EnhancedModelManager(base_path=self.temp_dir)
        
        # Sample model and metadata
        self.sample_model = DecisionTreeClassifier(random_state=42)
        self.sample_metadata = ModelMetadata(
            symbol="TEST",
            version="v1.0",
            created_at=dt.datetime.now(),
            model_type="DecisionTreeClassifier",
            performance_metrics={"accuracy": 0.85, "precision": 0.82},
            feature_names=["rsi", "macd", "bb_upper"],
            training_period=(dt.datetime(2023, 1, 1), dt.datetime(2023, 12, 31)),
            config_snapshot={"max_depth": 5, "random_state": 42},
            file_path="",
            status=ModelStatus.TRAINED
        )
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_initialization(self):
        """Test manager initialization"""
        self.assertEqual(self.manager.base_path, self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))
        
    def test_save_model_success(self):
        """Test successful model saving"""
        # Train a simple model
        X = np.random.rand(100, 3)
        y = np.random.randint(0, 2, 100)
        self.sample_model.fit(X, y)
        
        version = self.manager.save_model("TEST", self.sample_model, self.sample_metadata)
        
        self.assertIsInstance(version, str)
        self.assertTrue(version.startswith("v"))
        
        # Check that files were created
        model_dir = os.path.join(self.temp_dir, "TEST")
        self.assertTrue(os.path.exists(model_dir))
        
    def test_save_model_creates_directory(self):
        """Test that save_model creates necessary directories"""
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        self.sample_model.fit(X, y)
        
        # Symbol directory shouldn't exist initially
        symbol_dir = os.path.join(self.temp_dir, "NEWTEST")
        self.assertFalse(os.path.exists(symbol_dir))
        
        version = self.manager.save_model("NEWTEST", self.sample_model, self.sample_metadata)
        
        # Should create directory
        self.assertTrue(os.path.exists(symbol_dir))
        
    def test_load_model_success(self):
        """Test successful model loading"""
        # First save a model
        X = np.random.rand(100, 3)
        y = np.random.randint(0, 2, 100)
        self.sample_model.fit(X, y)
        
        version = self.manager.save_model("LOADTEST", self.sample_model, self.sample_metadata)
        
        # Then load it
        loaded_model, loaded_metadata = self.manager.load_model("LOADTEST", version)
        
        self.assertIsInstance(loaded_model, DecisionTreeClassifier)
        self.assertIsInstance(loaded_metadata, ModelMetadata)
        self.assertEqual(loaded_metadata.symbol, "LOADTEST")
        self.assertEqual(loaded_metadata.version, version)
        
    def test_load_model_latest_version(self):
        """Test loading latest version when version not specified"""
        X = np.random.rand(100, 3)
        y = np.random.randint(0, 2, 100)
        
        # Save multiple versions
        model1 = DecisionTreeClassifier(max_depth=3, random_state=42)
        model2 = DecisionTreeClassifier(max_depth=5, random_state=42)
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        metadata1 = self.sample_metadata
        metadata1.config_snapshot = {"max_depth": 3}
        
        metadata2 = self.sample_metadata
        metadata2.config_snapshot = {"max_depth": 5}
        
        version1 = self.manager.save_model("MULTIVERSION", model1, metadata1)
        version2 = self.manager.save_model("MULTIVERSION", model2, metadata2)
        
        # Load latest (should be version2)
        loaded_model, loaded_metadata = self.manager.load_model("MULTIVERSION")
        
        self.assertEqual(loaded_metadata.config_snapshot["max_depth"], 5)
        
    def test_model_exists_true(self):
        """Test model_exists returns True for existing model"""
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        self.sample_model.fit(X, y)
        
        version = self.manager.save_model("EXISTS", self.sample_model, self.sample_metadata)
        
        self.assertTrue(self.manager.model_exists("EXISTS", version))
        
    def test_model_exists_false(self):
        """Test model_exists returns False for non-existing model"""
        self.assertFalse(self.manager.model_exists("NONEXISTENT"))
        self.assertFalse(self.manager.model_exists("EXISTS", "v999.999"))
        
    def test_list_models(self):
        """Test listing available models"""
        # Initially should be empty
        models = self.manager.list_models("LISTTEST")
        self.assertEqual(len(models), 0)
        
        # Save a model
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        self.sample_model.fit(X, y)
        
        version = self.manager.save_model("LISTTEST", self.sample_model, self.sample_metadata)
        
        # Now should have one model
        models = self.manager.list_models("LISTTEST")
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0], version)
        
    def test_delete_model(self):
        """Test model deletion"""
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        self.sample_model.fit(X, y)
        
        version = self.manager.save_model("DELETETEST", self.sample_model, self.sample_metadata)
        
        # Model should exist
        self.assertTrue(self.manager.model_exists("DELETETEST", version))
        
        # Delete model
        self.manager.delete_model("DELETETEST", version)
        
        # Model should no longer exist
        self.assertFalse(self.manager.model_exists("DELETETEST", version))
        
    def test_get_model_info(self):
        """Test getting model information"""
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        self.sample_model.fit(X, y)
        
        version = self.manager.save_model("INFOTEST", self.sample_model, self.sample_metadata)
        
        info = self.manager.get_model_info("INFOTEST", version)
        
        self.assertIsInstance(info, ModelMetadata)
        self.assertEqual(info.symbol, "INFOTEST")
        self.assertEqual(info.version, version)
        self.assertEqual(info.model_type, "DecisionTreeClassifier")
        
    def test_update_model_status(self):
        """Test updating model status"""
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        self.sample_model.fit(X, y)
        
        version = self.manager.save_model("STATUSTEST", self.sample_model, self.sample_metadata)
        
        # Update status
        self.manager.update_model_status("STATUSTEST", version, ModelStatus.OUTDATED)
        
        # Verify status update
        info = self.manager.get_model_info("STATUSTEST", version)
        self.assertEqual(info.status, ModelStatus.OUTDATED)


class TestMLSignalGenerator(unittest.TestCase):
    """Test MLSignalGenerator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock model manager
        self.mock_model_manager = Mock(spec=ModelManagerInterface)
        
        # Mock trained model
        self.mock_model = Mock(spec=BaseEstimator)
        self.mock_model.predict.return_value = np.array([1, 0, 1, 1, 0])
        self.mock_model.predict_proba.return_value = np.array([
            [0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.3, 0.7], [0.6, 0.4]
        ])
        
        # Mock metadata
        self.mock_metadata = Mock()
        self.mock_metadata.feature_names = ["rsi", "macd", "bb_upper"]
        self.mock_metadata.performance_metrics = {"accuracy": 0.85}
        
        # Setup model manager to return mocked model
        self.mock_model_manager.load_model.return_value = (self.mock_model, self.mock_metadata)
        self.mock_model_manager.model_exists.return_value = True
        
        # Create signal generator
        self.generator = MLSignalGenerator(
            model_manager=self.mock_model_manager,
            symbol="TEST",
            confidence_threshold=0.6
        )
        
        # Sample market data with features
        self.market_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [102.0, 103.0, 104.0, 105.0, 106.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [101.0, 102.0, 103.0, 104.0, 105.0],
            'volume': [1000000] * 5,
            'rsi': [30.0, 35.0, 65.0, 70.0, 40.0],
            'macd': [-1.0, -0.5, 0.5, 1.0, 0.0],
            'bb_upper': [105.0, 106.0, 107.0, 108.0, 109.0]
        }, index=pd.date_range('2023-01-01', periods=5))
        
    def test_initialization(self):
        """Test ML signal generator initialization"""
        self.assertEqual(self.generator.symbol, "TEST")
        self.assertEqual(self.generator.confidence_threshold, 0.6)
        self.assertEqual(self.generator.model_manager, self.mock_model_manager)
        
    def test_generate_signals_success(self):
        """Test successful signal generation"""
        signals = self.generator.generate_signals(self.market_data, "TEST")
        
        self.assertIsInstance(signals, list)
        self.assertGreater(len(signals), 0)
        
        for signal in signals:
            self.assertIsInstance(signal, TradingSignal)
            self.assertEqual(signal.symbol, "TEST")
            self.assertEqual(signal.source, "ML_Model")
            self.assertIn(signal.signal_type, [SignalType.BUY, SignalType.SELL])
            
    def test_generate_signals_confidence_filtering(self):
        """Test that signals are filtered by confidence threshold"""
        # Set high confidence threshold
        generator = MLSignalGenerator(
            model_manager=self.mock_model_manager,
            symbol="HIGHCONF",
            confidence_threshold=0.9  # Very high threshold
        )
        
        signals = generator.generate_signals(self.market_data, "HIGHCONF")
        
        # Should have fewer signals due to high confidence threshold
        for signal in signals:
            self.assertGreaterEqual(signal.confidence, 0.9)
            
    def test_generate_signals_no_model(self):
        """Test signal generation when no model exists"""
        # Mock model manager to return False for model_exists
        self.mock_model_manager.model_exists.return_value = False
        
        signals = self.generator.generate_signals(self.market_data, "NOMODEL")
        
        # Should return empty list when no model exists
        self.assertEqual(len(signals), 0)
        
    def test_generate_signals_missing_features(self):
        """Test signal generation with missing features"""
        # Data missing required features
        incomplete_data = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'rsi': [30.0, 35.0, 65.0]
            # Missing macd, bb_upper
        })
        
        signals = self.generator.generate_signals(incomplete_data, "INCOMPLETE")
        
        # Should handle missing features gracefully
        self.assertIsInstance(signals, list)
        
    def test_prepare_features(self):
        """Test feature preparation"""
        features = self.generator._prepare_features(self.market_data)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[1], 3)  # 3 features: rsi, macd, bb_upper
        self.assertEqual(features.shape[0], 5)  # 5 data points
        
    def test_get_required_columns(self):
        """Test required columns specification"""
        required = self.generator.get_required_columns()
        
        self.assertIsInstance(required, list)
        # Should include the feature columns
        expected_features = ["rsi", "macd", "bb_upper"]
        for feature in expected_features:
            self.assertIn(feature, required)
            
    def test_validate_data(self):
        """Test data validation"""
        # Valid data
        self.assertTrue(self.generator.validate_data(self.market_data))
        
        # Invalid data (missing features)
        invalid_data = pd.DataFrame({'close': [100, 101]})
        self.assertFalse(self.generator.validate_data(invalid_data))


class TestModelTrainingService(unittest.TestCase):
    """Test ModelTrainingService functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model_manager = Mock(spec=ModelManagerInterface)
        self.mock_data_provider = Mock()
        
        self.service = ModelTrainingService(
            model_manager=self.mock_model_manager,
            data_provider=self.mock_data_provider
        )
        
        # Sample training data
        self.training_data = pd.DataFrame({
            'open': [100.0 + i for i in range(100)],
            'high': [102.0 + i for i in range(100)],
            'low': [99.0 + i for i in range(100)],
            'close': [101.0 + i + np.sin(i/10)*2 for i in range(100)],
            'volume': [1000000] * 100
        }, index=pd.date_range('2023-01-01', periods=100))
        
    def test_initialization(self):
        """Test training service initialization"""
        self.assertEqual(self.service.model_manager, self.mock_model_manager)
        self.assertEqual(self.service.data_provider, self.mock_data_provider)
        
    def test_prepare_training_data(self):
        """Test training data preparation"""
        # Mock data provider
        self.mock_data_provider.get_historical_data.return_value = self.training_data
        
        start_date = dt.datetime(2023, 1, 1)
        end_date = dt.datetime(2023, 12, 31)
        
        features, labels = self.service.prepare_training_data(
            "TRAIN_TEST", start_date, end_date
        )
        
        self.assertIsInstance(features, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(features), len(labels))
        self.assertGreater(len(features), 0)
        
    def test_train_model_decision_tree(self):
        """Test training a decision tree model"""
        # Prepare mock data
        features = np.random.rand(100, 5)
        labels = np.random.randint(0, 2, 100)
        
        with patch.object(self.service, 'prepare_training_data', return_value=(features, labels)):
            model, metadata = self.service.train_model(
                symbol="DT_TEST",
                model_type="DecisionTree",
                start_date=dt.datetime(2023, 1, 1),
                end_date=dt.datetime(2023, 12, 31)
            )
            
        self.assertIsInstance(model, DecisionTreeClassifier)
        self.assertIsInstance(metadata, ModelMetadata)
        self.assertEqual(metadata.symbol, "DT_TEST")
        self.assertEqual(metadata.model_type, "DecisionTree")
        
    def test_train_model_random_forest(self):
        """Test training a random forest model"""
        features = np.random.rand(100, 5)
        labels = np.random.randint(0, 2, 100)
        
        with patch.object(self.service, 'prepare_training_data', return_value=(features, labels)):
            model, metadata = self.service.train_model(
                symbol="RF_TEST",
                model_type="RandomForest",
                start_date=dt.datetime(2023, 1, 1),
                end_date=dt.datetime(2023, 12, 31)
            )
            
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertEqual(metadata.model_type, "RandomForest")
        
    def test_train_model_unsupported_type(self):
        """Test training with unsupported model type"""
        with self.assertRaises(ValueError):
            self.service.train_model(
                symbol="UNSUPPORTED",
                model_type="UnsupportedModel",
                start_date=dt.datetime(2023, 1, 1),
                end_date=dt.datetime(2023, 12, 31)
            )
            
    def test_evaluate_model(self):
        """Test model evaluation"""
        # Create and train a simple model
        model = DecisionTreeClassifier(random_state=42)
        X = np.random.rand(100, 3)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        
        # Test data
        X_test = np.random.rand(20, 3)
        y_test = np.random.randint(0, 2, 20)
        
        metrics = self.service.evaluate_model(model, X_test, y_test)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        # Check metric ranges
        for metric_name, value in metrics.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
            
    def test_retrain_model(self):
        """Test model retraining"""
        # Mock existing model
        existing_model = DecisionTreeClassifier(random_state=42)
        existing_metadata = ModelMetadata(
            symbol="RETRAIN_TEST",
            version="v1.0",
            created_at=dt.datetime.now(),
            model_type="DecisionTree",
            performance_metrics={"accuracy": 0.75},
            feature_names=["f1", "f2", "f3"],
            training_period=(dt.datetime(2023, 1, 1), dt.datetime(2023, 6, 30)),
            config_snapshot={},
            file_path=""
        )
        
        self.mock_model_manager.load_model.return_value = (existing_model, existing_metadata)
        self.mock_model_manager.model_exists.return_value = True
        
        # Mock training data
        features = np.random.rand(100, 3)
        labels = np.random.randint(0, 2, 100)
        
        with patch.object(self.service, 'prepare_training_data', return_value=(features, labels)):
            new_model, new_metadata = self.service.retrain_model(
                symbol="RETRAIN_TEST",
                start_date=dt.datetime(2023, 7, 1),
                end_date=dt.datetime(2023, 12, 31)
            )
            
        self.assertIsInstance(new_model, DecisionTreeClassifier)
        self.assertIsInstance(new_metadata, ModelMetadata)
        self.assertNotEqual(new_metadata.version, "v1.0")  # Should have new version
        
    def test_auto_retrain_models(self):
        """Test automatic model retraining"""
        # Mock model manager to return models needing retraining
        outdated_models = [
            ("SYMBOL1", "v1.0"),
            ("SYMBOL2", "v1.5")
        ]
        
        with patch.object(self.service, 'get_models_needing_retraining', return_value=outdated_models):
            with patch.object(self.service, 'retrain_model') as mock_retrain:
                mock_retrain.return_value = (Mock(), Mock())
                
                results = self.service.auto_retrain_models()
                
                # Should attempt to retrain all outdated models
                self.assertEqual(mock_retrain.call_count, 2)
                self.assertEqual(len(results), 2)


class TestRNNModelWrapper(unittest.TestCase):
    """Test RNNModelWrapper functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.wrapper = RNNModelWrapper(
            model_type='LSTM',
            sequence_length=10,
            units=32,
            epochs=1  # Minimal for testing
        )
        
    def test_initialization(self):
        """Test RNN wrapper initialization"""
        self.assertEqual(self.wrapper.model_type, 'LSTM')
        self.assertEqual(self.wrapper.sequence_length, 10)
        self.assertEqual(self.wrapper.units, 32)
        self.assertEqual(self.wrapper.epochs, 1)
        
    @patch('src.models.enhanced_model_management.TENSORFLOW_AVAILABLE', True)
    def test_create_rnn_model(self):
        """Test RNN model creation"""
        input_shape = (10, 5)  # sequence_length=10, features=5
        
        try:
            model = self.wrapper._create_rnn_model(input_shape)
            self.assertIsNotNone(model)
        except ImportError:
            # Skip if TensorFlow not available
            self.skipTest("TensorFlow not available")
            
    def test_rnn_without_tensorflow(self):
        """Test RNN functionality without TensorFlow"""
        with patch('src.models.enhanced_model_management.TENSORFLOW_AVAILABLE', False):
            wrapper = RNNModelWrapper()
            
            with self.assertRaises(ImportError):
                wrapper._create_rnn_model((10, 5))


class TestModelIntegration(unittest.TestCase):
    """Test integration between model components"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up integration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_end_to_end_model_workflow(self):
        """Test complete model lifecycle workflow"""
        # 1. Initialize components
        model_manager = EnhancedModelManager(base_path=self.temp_dir)
        mock_data_provider = Mock()
        training_service = ModelTrainingService(model_manager, mock_data_provider)
        
        # 2. Mock training data
        training_data = pd.DataFrame({
            'close': [100 + i + np.sin(i/10)*5 for i in range(100)]
        }, index=pd.date_range('2023-01-01', periods=100))
        mock_data_provider.get_historical_data.return_value = training_data
        
        # 3. Train model
        model, metadata = training_service.train_model(
            symbol="INTEGRATION_TEST",
            model_type="DecisionTree",
            start_date=dt.datetime(2023, 1, 1),
            end_date=dt.datetime(2023, 12, 31)
        )
        
        # 4. Save model
        version = model_manager.save_model("INTEGRATION_TEST", model, metadata)
        
        # 5. Load model
        loaded_model, loaded_metadata = model_manager.load_model("INTEGRATION_TEST", version)
        
        # 6. Create signal generator
        signal_generator = MLSignalGenerator(
            model_manager=model_manager,
            symbol="INTEGRATION_TEST"
        )
        
        # 7. Generate signals (would need proper feature data)
        test_data = pd.DataFrame({
            'close': [105, 106, 107],
            'rsi': [30, 35, 40],
            'macd': [0.5, 0.6, 0.7],
            'bb_upper': [108, 109, 110],
            'bb_lower': [102, 103, 104],
            'sma_20': [104, 105, 106]
        })
        
        # This would generate signals if data had all required features
        signals = signal_generator.generate_signals(test_data, "INTEGRATION_TEST")
        
        # Verify workflow completed
        self.assertIsInstance(loaded_model, DecisionTreeClassifier)
        self.assertEqual(loaded_metadata.symbol, "INTEGRATION_TEST")
        self.assertIsInstance(signals, list)


if __name__ == '__main__':
    unittest.main()