"""
Enhanced ML Model Management Implementation
Implements comprehensive model management with versioning, persistence, and prediction services
"""

import os
import pickle
import json
import datetime as dt
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings

# Advanced ML algorithms (optional imports)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[WARNING] LightGBM not available. Install with: pip install lightgbm")

# Deep Learning frameworks for RNN support (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("[INFO] TensorFlow not available. RNN support disabled. Install with: pip install tensorflow")

from ..interfaces.model_manager import ModelManagerInterface, ModelMetadata, PredictionResult
from ..interfaces.signal_generator import SignalGenerator
from ..interfaces.trading_strategy import TradingSignal

warnings.filterwarnings('ignore')


class RNNModelWrapper(BaseEstimator):
    """
    Scikit-learn compatible wrapper for RNN models (LSTM/GRU)
    Provides standard fit/predict interface for deep learning models
    """
    
    def __init__(self, model_type='LSTM', sequence_length=20, units=50, 
                 dropout_rate=0.2, dense_units=25, epochs=50, batch_size=32, **params):
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.units = units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
        self.params = params
        
    def _create_rnn_model(self, input_shape):
        """Create RNN model architecture"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for RNN models")
            
        model = Sequential()
        
        if self.model_type.upper() == 'LSTM':
            model.add(LSTM(self.units, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(self.dropout_rate))
            model.add(LSTM(self.units // 2))
            model.add(Dropout(self.dropout_rate))
        elif self.model_type.upper() == 'GRU':
            model.add(GRU(self.units, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(self.dropout_rate))
            model.add(GRU(self.units // 2))
            model.add(Dropout(self.dropout_rate))
        else:
            raise ValueError(f"Unsupported RNN type: {self.model_type}")
            
        # Dense layers for classification
        model.add(Dense(self.dense_units, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def _prepare_sequences(self, X, y=None):
        """Convert tabular data to sequences for RNN"""
        if len(X) < self.sequence_length:
            raise ValueError(f"Not enough data points. Need at least {self.sequence_length}")
            
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i-self.sequence_length:i])
            if y is not None:
                targets.append(y[i])
                
        return np.array(sequences), np.array(targets) if y is not None else None
    
    def fit(self, X, y):
        """Train the RNN model"""
        # Initialize scaler
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(X_scaled, y)
        
        # Create model
        input_shape = (self.sequence_length, X.shape[1])
        self.model = self._create_rnn_model(input_shape)
        
        # Train model
        self.model.fit(X_seq, y_seq, epochs=self.epochs, batch_size=self.batch_size, 
                      verbose=0, validation_split=0.2)
        
        return self
    
    def predict(self, X):
        """Make predictions with the RNN model"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._prepare_sequences(X_scaled)
        
        if len(X_seq) == 0:
            # Not enough data for sequences, return last available prediction
            return np.array([0] * len(X))
            
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Convert probabilities to binary predictions
        binary_predictions = (predictions > 0.5).astype(int).flatten()
        
        # Pad with zeros for the initial sequence_length samples
        full_predictions = np.zeros(len(X))
        full_predictions[self.sequence_length:self.sequence_length+len(binary_predictions)] = binary_predictions
        
        return full_predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._prepare_sequences(X_scaled)
        
        if len(X_seq) == 0:
            # Not enough data for sequences, return neutral probabilities
            return np.array([[0.5, 0.5]] * len(X))
            
        predictions = self.model.predict(X_seq, verbose=0).flatten()
        
        # Create probability matrix [prob_class_0, prob_class_1]
        probabilities = np.column_stack([1 - predictions, predictions])
        
        # Pad with neutral probabilities for initial samples
        full_probabilities = np.full((len(X), 2), 0.5)
        full_probabilities[self.sequence_length:self.sequence_length+len(probabilities)] = probabilities
        
        return full_probabilities


class LSTMModelWrapper(RNNModelWrapper):
    """LSTM Model Wrapper"""
    def __init__(self, **params):
        super().__init__(model_type='LSTM', **params)


class GRUModelWrapper(RNNModelWrapper):
    """GRU Model Wrapper"""
    def __init__(self, **params):
        super().__init__(model_type='GRU', **params)


class EnhancedModelManager(ModelManagerInterface):
    """
    Enhanced ML Model Management System
    
    Features:
    - Model persistence and versioning
    - Comprehensive metadata tracking
    - Performance monitoring
    - Model validation and health checks
    - Automatic backup and cleanup
    - Prediction service with caching
    """
    
    def __init__(self, base_path: str = "models"):
        """Initialize Enhanced Model Manager"""
        self.base_path = base_path
        self.models_dir = os.path.join(base_path, "trained_models")
        self.metadata_dir = os.path.join(base_path, "metadata")
        self.backup_dir = os.path.join(base_path, "backups")
        
        # Create directories
        self._create_directories()
        
        # Model cache for performance
        self._model_cache = {}
        self._metadata_cache = {}
        
        print(f"[MODEL_MGR] Enhanced Model Manager initialized at {base_path}")
    
    def _create_directories(self):
        """Create necessary directories"""
        for directory in [self.base_path, self.models_dir, self.metadata_dir, self.backup_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def save_model(self, symbol: str, model: BaseEstimator, metadata: ModelMetadata) -> str:
        """Save model with comprehensive metadata"""
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        version = self._get_next_version(symbol)
        
        # Create symbol directory
        symbol_dir = os.path.join(self.models_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Save model
        model_filename = f"{symbol}_v{version}_{timestamp}.pkl"
        model_path = os.path.join(symbol_dir, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Enhanced metadata
        enhanced_metadata = ModelMetadata(
            symbol=symbol,
            model_type=type(model).__name__,
            version=version,
            created_at=dt.datetime.now(),
            feature_names=metadata.feature_names,
            performance_metrics=metadata.performance_metrics,
            training_params=metadata.training_params,
            model_path=model_path,
            file_size=os.path.getsize(model_path) if os.path.exists(model_path) else 0,
            additional_info={
                'timestamp': timestamp,
                'sklearn_version': getattr(model, '__module__', 'unknown'),
                'feature_count': len(metadata.feature_names) if metadata.feature_names else 0,
                'training_samples': metadata.training_params.get('training_samples', 0),
                'model_parameters': self._extract_model_parameters(model)
            }
        )
        
        # Save metadata
        metadata_path = os.path.join(self.metadata_dir, f"{symbol}_v{version}_{timestamp}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(enhanced_metadata.__dict__, f, indent=2, default=self._json_serializer)
        
        # Update cache
        cache_key = f"{symbol}_v{version}"
        self._model_cache[cache_key] = model
        self._metadata_cache[cache_key] = enhanced_metadata
        
        print(f"[MODEL_SAVE] Model saved for {symbol}")
        print(f"             Version: v{version}")
        print(f"             Path: {model_path}")
        print(f"             Features: {len(metadata.feature_names) if metadata.feature_names else 0}")
        
        return model_path
    
    def load_model(self, symbol: str, version: Optional[str] = None) -> Tuple[BaseEstimator, ModelMetadata]:
        """Load model with metadata"""
        if version is None:
            version = self._get_latest_version(symbol)
        
        cache_key = f"{symbol}_v{version}"
        
        # Check cache first
        if cache_key in self._model_cache and cache_key in self._metadata_cache:
            return self._model_cache[cache_key], self._metadata_cache[cache_key]
        
        # Load from disk
        symbol_dir = os.path.join(self.models_dir, symbol)
        if not os.path.exists(symbol_dir):
            raise FileNotFoundError(f"No models found for symbol {symbol}")
        
        # Find model file
        model_files = [f for f in os.listdir(symbol_dir) if f.startswith(f"{symbol}_v{version}") and f.endswith('.pkl')]
        if not model_files:
            raise FileNotFoundError(f"No model found for {symbol} version {version}")
        
        model_file = model_files[0]  # Take the first match
        model_path = os.path.join(symbol_dir, model_file)
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata_files = [f for f in os.listdir(self.metadata_dir) if f.startswith(f"{symbol}_v{version}") and f.endswith('_metadata.json')]
        if metadata_files:
            metadata_path = os.path.join(self.metadata_dir, metadata_files[0])
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                metadata = ModelMetadata(**metadata_dict)
        else:
            # Create minimal metadata if not found
            metadata = ModelMetadata(
                symbol=symbol,
                model_type=type(model).__name__,
                version=version,
                created_at=dt.datetime.now(),
                feature_names=[],
                performance_metrics={},
                training_params={},
                model_path=model_path
            )
        
        # Update cache
        self._model_cache[cache_key] = model
        self._metadata_cache[cache_key] = metadata
        
        print(f"[MODEL_LOAD] Model loaded for {symbol} v{version}")
        
    def _create_model(self, algorithm: str, **params) -> BaseEstimator:
        """
        Create ML model based on algorithm name with optimized parameters for trading
        
        Supported algorithms:
        - DecisionTree: Fast, interpretable, good baseline
        - RandomForest: Better accuracy, ensemble method
        - XGBoost: Superior performance, industry standard
        - LightGBM: Fast training, memory efficient
        - SVM: Good for high-dimensional data
        - NeuralNetwork: Complex pattern recognition (MLP)
        - LSTM: Long Short-Term Memory for time series patterns
        - GRU: Gated Recurrent Unit, simplified LSTM
        """
        # Default parameters optimized for financial time series
        default_params = {
            'random_state': params.get('random_state', 42),
            'n_jobs': params.get('n_jobs', -1)  # Use all CPU cores
        }
        
        if algorithm.lower() in ['decisiontree', 'dt']:
            model_params = {
                'max_depth': params.get('max_depth', 5),
                'min_samples_split': params.get('min_samples_split', 20),
                'min_samples_leaf': params.get('min_samples_leaf', 10),
                'random_state': default_params['random_state']
            }
            return DecisionTreeClassifier(**model_params)
            
        elif algorithm.lower() in ['randomforest', 'rf']:
            model_params = {
                'n_estimators': params.get('n_estimators', 100),
                'max_depth': params.get('max_depth', 5),
                'min_samples_split': params.get('min_samples_split', 20),
                'min_samples_leaf': params.get('min_samples_leaf', 10),
                'random_state': default_params['random_state'],
                'n_jobs': default_params['n_jobs']
            }
            return RandomForestClassifier(**model_params)
            
        elif algorithm.lower() in ['xgboost', 'xgb']:
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost")
            
            model_params = {
                'n_estimators': params.get('n_estimators', 100),
                'max_depth': params.get('max_depth', 6),
                'learning_rate': params.get('learning_rate', 0.1),
                'subsample': params.get('subsample', 0.8),
                'colsample_bytree': params.get('colsample_bytree', 0.8),
                'reg_alpha': params.get('reg_alpha', 0.1),
                'reg_lambda': params.get('reg_lambda', 1.0),
                'random_state': default_params['random_state'],
                'n_jobs': default_params['n_jobs'],
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }
            return xgb.XGBClassifier(**model_params)
            
        elif algorithm.lower() in ['lightgbm', 'lgb']:
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available. Install with: pip install lightgbm")
            
            model_params = {
                'n_estimators': params.get('n_estimators', 100),
                'max_depth': params.get('max_depth', 6),
                'learning_rate': params.get('learning_rate', 0.1),
                'num_leaves': params.get('num_leaves', 31),
                'subsample': params.get('subsample', 0.8),
                'colsample_bytree': params.get('colsample_bytree', 0.8),
                'reg_alpha': params.get('reg_alpha', 0.1),
                'reg_lambda': params.get('reg_lambda', 1.0),
                'random_state': default_params['random_state'],
                'n_jobs': default_params['n_jobs'],
                'verbose': -1  # Suppress warnings
            }
            return lgb.LGBMClassifier(**model_params)
            
        elif algorithm.lower() in ['svm', 'svc']:
            model_params = {
                'C': params.get('C', 1.0),
                'kernel': params.get('kernel', 'rbf'),
                'gamma': params.get('gamma', 'scale'),
                'probability': True,  # Enable probability estimates
                'random_state': default_params['random_state']
            }
            return SVC(**model_params)
            
        elif algorithm.lower() in ['neuralnetwork', 'mlp', 'nn']:
            model_params = {
                'hidden_layer_sizes': params.get('hidden_layer_sizes', (100, 50)),
                'activation': params.get('activation', 'relu'),
                'solver': params.get('solver', 'adam'),
                'alpha': params.get('alpha', 0.001),
                'learning_rate': params.get('learning_rate', 'constant'),
                'max_iter': params.get('max_iter', 500),
                'random_state': default_params['random_state']
            }
            return MLPClassifier(**model_params)
            
        elif algorithm.lower() in ['lstm']:
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow not available for LSTM. Install with: pip install tensorflow")
            
            # LSTM parameters optimized for trading
            model_params = {
                'sequence_length': params.get('sequence_length', 20),  # 20 days lookback
                'units': params.get('lstm_units', 50),
                'dropout_rate': params.get('dropout_rate', 0.2),
                'dense_units': params.get('dense_units', 25),
                'epochs': params.get('epochs', 50),
                'batch_size': params.get('batch_size', 32)
            }
            return LSTMModelWrapper(**model_params)
            
        elif algorithm.lower() in ['gru']:
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow not available for GRU. Install with: pip install tensorflow")
            
            # GRU parameters optimized for trading
            model_params = {
                'sequence_length': params.get('sequence_length', 20),  # 20 days lookback
                'units': params.get('gru_units', 50),
                'dropout_rate': params.get('dropout_rate', 0.2),
                'dense_units': params.get('dense_units', 25),
                'epochs': params.get('epochs', 50),
                'batch_size': params.get('batch_size', 32)
            }
            return GRUModelWrapper(**model_params)
            
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}. "
                           f"Supported: DecisionTree, RandomForest, XGBoost, LightGBM, SVM, NeuralNetwork, LSTM, GRU")
    
    def predict(self, symbol: str, features: np.ndarray, version: Optional[str] = None) -> PredictionResult:
        """Make predictions using the model"""
        model, metadata = self.load_model(symbol, version)
        
        # Validate features
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        expected_features = len(metadata.feature_names) if metadata.feature_names else features.shape[1]
        if features.shape[1] != expected_features:
            raise ValueError(f"Feature mismatch: expected {expected_features}, got {features.shape[1]}")
        
        # Make predictions
        try:
            predictions = model.predict(features)
            prediction_proba = None
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                try:
                    prediction_proba = model.predict_proba(features)
                except:
                    pass  # Some models may not support predict_proba
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(predictions, prediction_proba)
            
            result = PredictionResult(
                predictions=predictions.tolist(),
                confidence_scores=confidence_scores,
                model_version=metadata.version,
                feature_importance=self._get_feature_importance(model, metadata.feature_names),
                prediction_metadata={
                    'model_type': metadata.model_type,
                    'prediction_timestamp': dt.datetime.now().isoformat(),
                    'feature_count': features.shape[1],
                    'sample_count': features.shape[0],
                    'has_probabilities': prediction_proba is not None
                }
            )
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed for {symbol}: {str(e)}")
    
    def list_models(self, symbol: Optional[str] = None) -> List[ModelMetadata]:
        """List available models"""
        models = []
        
        if symbol:
            # List models for specific symbol
            symbol_dir = os.path.join(self.models_dir, symbol)
            if os.path.exists(symbol_dir):
                for metadata_file in os.listdir(self.metadata_dir):
                    if metadata_file.startswith(f"{symbol}_") and metadata_file.endswith('_metadata.json'):
                        metadata_path = os.path.join(self.metadata_dir, metadata_file)
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata_dict = json.load(f)
                                models.append(ModelMetadata(**metadata_dict))
                        except:
                            continue
        else:
            # List all models
            if os.path.exists(self.metadata_dir):
                for metadata_file in os.listdir(self.metadata_dir):
                    if metadata_file.endswith('_metadata.json'):
                        metadata_path = os.path.join(self.metadata_dir, metadata_file)
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata_dict = json.load(f)
                                models.append(ModelMetadata(**metadata_dict))
                        except:
                            continue
        
        # Sort by created_at descending
        models.sort(key=lambda m: m.created_at, reverse=True)
        return models
    
    def delete_model(self, symbol: str, version: str) -> bool:
        """Delete a specific model version"""
        try:
            # Remove from cache
            cache_key = f"{symbol}_v{version}"
            self._model_cache.pop(cache_key, None)
            self._metadata_cache.pop(cache_key, None)
            
            # Find and delete model file
            symbol_dir = os.path.join(self.models_dir, symbol)
            if os.path.exists(symbol_dir):
                model_files = [f for f in os.listdir(symbol_dir) if f.startswith(f"{symbol}_v{version}") and f.endswith('.pkl')]
                for model_file in model_files:
                    os.remove(os.path.join(symbol_dir, model_file))
            
            # Find and delete metadata file
            metadata_files = [f for f in os.listdir(self.metadata_dir) if f.startswith(f"{symbol}_v{version}") and f.endswith('_metadata.json')]
            for metadata_file in metadata_files:
                os.remove(os.path.join(self.metadata_dir, metadata_file))
            
            print(f"[MODEL_DELETE] Deleted model {symbol} v{version}")
            return True
            
        except Exception as e:
            print(f"[MODEL_DELETE] Failed to delete {symbol} v{version}: {e}")
            return False
    
    def _get_next_version(self, symbol: str) -> int:
        """Get next version number for symbol"""
        existing_models = self.list_models(symbol)
        if not existing_models:
            return 1
        
        max_version = max(int(m.version) for m in existing_models)
        return max_version + 1
    
    def _get_latest_version(self, symbol: str) -> str:
        """Get latest version number for symbol"""
        existing_models = self.list_models(symbol)
        if not existing_models:
            raise FileNotFoundError(f"No models found for symbol {symbol}")
        
        latest_model = max(existing_models, key=lambda m: int(m.version))
        return latest_model.version
    
    def _extract_model_parameters(self, model: BaseEstimator) -> Dict:
        """Extract key model parameters"""
        params = {}
        
        if hasattr(model, 'get_params'):
            all_params = model.get_params()
            # Filter out non-serializable parameters
            for key, value in all_params.items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    params[key] = value
                else:
                    params[key] = str(value)
        
        return params
    
    def _calculate_confidence_scores(self, predictions: np.ndarray, 
                                   probabilities: Optional[np.ndarray]) -> List[float]:
        """Calculate confidence scores for predictions"""
        if probabilities is not None:
            # Use maximum probability as confidence
            confidence_scores = np.max(probabilities, axis=1).tolist()
        else:
            # Use a simple heuristic based on prediction
            confidence_scores = [0.7 if pred != 0 else 0.3 for pred in predictions]
        
        return confidence_scores
    
    def _get_feature_importance(self, model: BaseEstimator, feature_names: List[str]) -> Optional[Dict]:
        """Get feature importance if available"""
        if hasattr(model, 'feature_importances_') and feature_names:
            importances = model.feature_importances_
            if len(importances) == len(feature_names):
                # Return top 10 most important features
                feature_importance = list(zip(feature_names, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                return {
                    'top_features': feature_importance[:10],
                    'all_features': feature_importance
                }
        
        return None
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for datetime and other objects"""
        if isinstance(obj, dt.datetime):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)
    
    def get_latest_version(self, symbol: str) -> Optional[str]:
        """
        Get the latest version for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Latest version string or None if no model exists
        """
        try:
            return self._get_latest_version(symbol)
        except FileNotFoundError:
            return None
    
    def model_exists(self, symbol: str, version: Optional[str] = None) -> bool:
        """
        Check if model exists for symbol and version
        
        Args:
            symbol: Trading symbol
            version: Model version (if None, checks latest)
            
        Returns:
            True if model exists, False otherwise
        """
        try:
            metadata = self.get_model_metadata(symbol, version)
            return metadata is not None
        except Exception:
            return False
    
    def validate_model(self, symbol: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate model integrity and performance
        
        Args:
            symbol: Trading symbol
            version: Model version (if None, validates latest)
            
        Returns:
            Validation results dictionary
        """
        try:
            metadata = self.get_model_metadata(symbol, version)
            if metadata:
                return {
                    'valid': True,
                    'symbol': symbol,
                    'version': metadata.version,
                    'status': metadata.status.value,
                    'performance_metrics': metadata.performance_metrics,
                    'model_type': metadata.model_type
                }
            else:
                return {'valid': False, 'error': 'Model not found'}
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def cleanup_old_versions(self, symbol: str, keep_versions: int = 5) -> int:
        """
        Clean up old model versions
        
        Args:
            symbol: Trading symbol
            keep_versions: Number of versions to keep
            
        Returns:
            Number of models deleted
        """
        try:
            models = self.list_models(symbol)
            if len(models) <= keep_versions:
                return 0
            
            # Sort by version number (keep latest)
            models_sorted = sorted(models, key=lambda m: int(m.version), reverse=True)
            models_to_delete = models_sorted[keep_versions:]
            
            deleted_count = 0
            for model in models_to_delete:
                if self.delete_model(symbol, model.version):
                    deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            print(f"[WARNING] Failed to cleanup old versions for {symbol}: {e}")
            return 0


class MLSignalGenerator(SignalGenerator):
    """ML-based signal generator using trained models"""
    
    def __init__(self, model_manager: EnhancedModelManager, symbol: str, 
                 feature_window: int = 20, confidence_threshold: float = 0.6):
        """
        Initialize ML Signal Generator
        
        Args:
            model_manager: Model manager for loading trained models
            symbol: Trading symbol
            feature_window: Window for feature calculation
            confidence_threshold: Minimum confidence for signal generation
        """
        super().__init__()
        self.model_manager = model_manager
        self.symbol = symbol
        self.feature_window = feature_window
        self.confidence_threshold = confidence_threshold
        self.name = f"ML_{symbol}"
        
        # Try to load existing model
        self.model = None
        self.metadata = None
        try:
            self.model, self.metadata = self.model_manager.load_model(symbol)
            print(f"[ML_SIGNAL] Loaded model for {symbol} v{self.metadata.version}")
        except:
            print(f"[ML_SIGNAL] No trained model found for {symbol}")
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate ML-based trading signals"""
        if self.model is None or len(data) < self.feature_window:
            return []
        
        signals = []
        
        # Generate features for the last few data points
        for i in range(self.feature_window, len(data)):
            try:
                # Calculate features (simplified - would use enhanced_features in practice)
                window_data = data.iloc[i-self.feature_window:i+1]
                features = self._calculate_features(window_data)
                
                if features is not None:
                    # Make prediction
                    prediction_result = self.model_manager.predict(
                        symbol=self.symbol,
                        features=features.reshape(1, -1)
                    )
                    
                    prediction = prediction_result.predictions[0]
                    confidence = prediction_result.confidence_scores[0]
                    
                    # Generate signal if confidence is high enough
                    if abs(prediction) > 0 and confidence >= self.confidence_threshold:
                        signal = TradingSignal(
                            timestamp=data.index[i],
                            symbol=self.symbol,
                            direction=int(prediction),
                            confidence=confidence,
                            signal_type='ML_PREDICTION',
                            metadata={
                                'model_version': prediction_result.model_version,
                                'model_type': self.metadata.model_type if self.metadata else 'Unknown',
                                'feature_importance': prediction_result.feature_importance,
                                'prediction_confidence': confidence,
                                'feature_window': self.feature_window
                            }
                        )
                        signals.append(signal)
            
            except Exception as e:
                print(f"[ML_SIGNAL] Prediction failed at index {i}: {e}")
                continue
        
        return signals
    
    def _calculate_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Calculate features for ML prediction (simplified implementation)"""
        try:
            if len(data) < 5:
                return None
            
            # Simple feature calculation - would use enhanced_features in practice
            features = []
            
            # Price-based features
            current_price = data.iloc[-1]
            price_change_1d = (data.iloc[-1] - data.iloc[-2]) / data.iloc[-2]
            price_change_5d = (data.iloc[-1] - data.iloc[-6]) / data.iloc[-6] if len(data) >= 6 else 0
            
            # Moving averages
            sma_5 = data.rolling(5).mean().iloc[-1] if len(data) >= 5 else current_price
            sma_10 = data.rolling(10).mean().iloc[-1] if len(data) >= 10 else current_price
            
            # Volatility
            volatility = data.rolling(10).std().iloc[-1] if len(data) >= 10 else 0.02
            
            # RSI (simplified)
            returns = data.pct_change().dropna()
            if len(returns) >= 14:
                up_moves = returns.where(returns > 0, 0)
                down_moves = returns.where(returns < 0, 0).abs()
                avg_up = up_moves.rolling(14).mean().iloc[-1]
                avg_down = down_moves.rolling(14).mean().iloc[-1]
                rs = avg_up / avg_down if avg_down != 0 else 100
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50
            
            # Assemble features
            features = [
                price_change_1d,
                price_change_5d,
                (sma_5 - current_price) / current_price,
                (sma_10 - current_price) / current_price,
                volatility,
                rsi / 100.0,  # Normalize RSI
                (current_price - data.min()) / (data.max() - data.min()) if data.max() != data.min() else 0.5
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"[FEATURE_CALC] Feature calculation failed: {e}")
            return None


class ModelTrainingService:
    """Service for training ML models with enhanced features"""
    
    def __init__(self, model_manager: EnhancedModelManager, data_provider):
        """Initialize Model Training Service"""
        self.model_manager = model_manager
        self.data_provider = data_provider
    
    def train_model(self, symbol: str, algorithm: str = 'LightGBM',
                   training_period_days: int = 365,
                   test_split: float = 0.2) -> Dict[str, Any]:
        """
        Train ML model for symbol with enhanced features
        
        Args:
            symbol: Trading symbol
            algorithm: ML algorithm to use
            training_period_days: Days of historical data for training
            test_split: Fraction of data for testing
            
        Returns:
            Training results and metrics
        """
        print(f"[MODEL_TRAIN] Training {algorithm} model for {symbol}")
        
        # Get training data
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=training_period_days)
        
        training_data = self.data_provider.get_historical_data(
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date
        )
        
        if symbol not in training_data or len(training_data[symbol]) < 100:
            raise ValueError(f"Insufficient training data for {symbol}")
        
        # Calculate features and labels
        features, labels, feature_names = self._prepare_training_data(training_data[symbol])
        
        # Split data
        split_idx = int(len(features) * (1 - test_split))
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]
        
        # Create and train model using factory method
        model = self.model_manager._create_model(algorithm)
        
        # Train model
        print(f"[MODEL_TRAIN] Training {type(model).__name__} with {len(X_train)} samples")
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        # Create metadata
        metadata = ModelMetadata(
            symbol=symbol,
            model_type=type(model).__name__,
            version="1",  # Will be updated by model manager
            created_at=dt.datetime.now(),
            feature_names=feature_names,
            performance_metrics={
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(feature_names)
            },
            training_params={
                'algorithm': algorithm,
                'training_period_days': training_period_days,
                'test_split': test_split,
                'training_start': start_date.isoformat(),
                'training_end': end_date.isoformat(),
                'training_samples': len(features)
            }
        )
        
        # Save model
        model_path = self.model_manager.save_model(symbol, model, metadata)
        
        results = {
            'success': True,
            'model_path': model_path,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_count': len(feature_names),
            'training_samples': len(features),
            'model_type': algorithm
        }
        
        print(f"[MODEL_TRAIN] Training completed for {symbol}")
        print(f"              Train Accuracy: {train_accuracy:.3f}")
        print(f"              Test Accuracy: {test_accuracy:.3f}")
        print(f"              Features: {len(feature_names)}")
        
        return results
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training features and labels (simplified implementation)"""
        # This is a simplified implementation - would use enhanced_features in practice
        features_list = []
        labels_list = []
        feature_names = [
            'price_change_1d', 'price_change_5d', 'sma_5_diff', 'sma_10_diff',
            'volatility', 'rsi', 'price_position'
        ]
        
        for i in range(20, len(data) - 3):  # Need lookback and lookahead
            # Calculate features (same as in MLSignalGenerator)
            window_data = data.iloc[i-20:i+1]
            
            # Features
            current_price = data.iloc[i]
            price_change_1d = (data.iloc[i] - data.iloc[i-1]) / data.iloc[i-1]
            price_change_5d = (data.iloc[i] - data.iloc[i-5]) / data.iloc[i-5]
            
            sma_5 = data.iloc[i-4:i+1].mean()
            sma_10 = data.iloc[i-9:i+1].mean()
            
            volatility = data.iloc[i-9:i+1].std()
            
            # RSI calculation (simplified)
            returns = data.iloc[i-14:i+1].pct_change().dropna()
            up_moves = returns.where(returns > 0, 0)
            down_moves = returns.where(returns < 0, 0).abs()
            avg_up = up_moves.mean()
            avg_down = down_moves.mean()
            rs = avg_up / avg_down if avg_down != 0 else 100
            rsi = 100 - (100 / (1 + rs))
            
            price_position = (current_price - window_data.min()) / (window_data.max() - window_data.min()) if window_data.max() != window_data.min() else 0.5
            
            features = [
                price_change_1d,
                price_change_5d,
                (sma_5 - current_price) / current_price,
                (sma_10 - current_price) / current_price,
                volatility,
                rsi / 100.0,
                price_position
            ]
            
            # Label (future return)
            future_price = data.iloc[i+3]
            future_return = (future_price - current_price) / current_price
            
            if future_return > 0.02:
                label = 1  # BUY
            elif future_return < -0.02:
                label = -1  # SELL
            else:
                label = 0  # HOLD
            
            features_list.append(features)
            labels_list.append(label)
        
        return np.array(features_list), np.array(labels_list), feature_names