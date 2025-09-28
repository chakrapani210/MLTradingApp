"""
ML Model Management System for Enhanced Trading Strategy
Handles model persistence, versioning, loading, and metadata tracking
"""

import os
import pickle
import json
import datetime as dt
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages ML model lifecycle for trading strategies
    
    Features:
    - Model persistence and loading
    - Metadata tracking (performance, features, dates)
    - Version management
    - Model validation and health checks
    - Automatic backup and cleanup
    """
    
    def __init__(self, base_path: str = "models"):
        """
        Initialize Model Manager
        
        Args:
            base_path (str): Base directory for storing models
        """
        self.base_path = base_path
        self.models_dir = os.path.join(base_path, "trained_models")
        self.metadata_dir = os.path.join(base_path, "metadata")
        self.backup_dir = os.path.join(base_path, "backups")
        
        # Create directories
        self._create_directories()
        
        # Model cache for quick access
        self._model_cache = {}
        self._metadata_cache = {}
    
    def _create_directories(self) -> None:
        """Create necessary directories for model storage"""
        directories = [self.base_path, self.models_dir, self.metadata_dir, self.backup_dir]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        logger.info(f"Model management directories created at: {self.base_path}")
    
    def save_model(self, symbol: str, model: BaseEstimator, 
                   training_data: Dict[str, Any], performance_metrics: Dict[str, float],
                   feature_names: list, config_snapshot: Dict[str, Any]) -> str:
        """
        Save trained model with comprehensive metadata
        
        Args:
            symbol (str): Trading symbol (e.g., 'TSLA')
            model (BaseEstimator): Trained ML model
            training_data (Dict): Training data information
            performance_metrics (Dict): Model performance metrics
            feature_names (list): List of feature names used
            config_snapshot (Dict): Configuration used for training
            
        Returns:
            str: Path to saved model
        """
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        version = self._get_next_version(symbol)
        
        # Create symbol directory
        symbol_dir = os.path.join(self.models_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Model filename with version
        model_filename = f"{symbol}_v{version}_{timestamp}.pkl"
        model_path = os.path.join(symbol_dir, model_filename)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Create comprehensive metadata
        metadata = {
            'symbol': symbol,
            'version': version,
            'timestamp': timestamp,
            'model_path': model_path,
            'model_type': type(model).__name__,
            'training_info': {
                'train_start': training_data.get('train_start').isoformat() if training_data.get('train_start') else None,
                'train_end': training_data.get('train_end').isoformat() if training_data.get('train_end') else None,
                'training_samples': training_data.get('training_samples', 0),
                'feature_count': len(feature_names)
            },
            'performance': performance_metrics,
            'features': {
                'names': feature_names,
                'count': len(feature_names),
                'enhanced_features': len(feature_names) > 10  # Assume enhanced if >10 features
            },
            'config_snapshot': config_snapshot,
            'created_at': dt.datetime.now().isoformat(),
            'model_size_bytes': os.path.getsize(model_path)
        }
        
        # Save metadata
        metadata_filename = f"{symbol}_v{version}_{timestamp}_metadata.json"
        metadata_path = os.path.join(self.metadata_dir, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Update latest model symlink/reference
        self._update_latest_model_reference(symbol, model_path, metadata_path)
        
        # Cache the model and metadata
        self._model_cache[symbol] = model
        self._metadata_cache[symbol] = metadata
        
        logger.info(f"Model saved for {symbol}: {model_filename}")
        logger.info(f"Training accuracy: {performance_metrics.get('training_accuracy', 'N/A'):.3f}")
        logger.info(f"Features used: {len(feature_names)}")
        
        return model_path
    
    def load_model(self, symbol: str, version: Optional[str] = None) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Load trained model for a symbol
        
        Args:
            symbol (str): Trading symbol
            version (str, optional): Specific version to load. If None, loads latest
            
        Returns:
            Tuple[BaseEstimator, Dict]: (model, metadata)
        """
        # Check cache first
        if version is None and symbol in self._model_cache:
            logger.info(f"Loading {symbol} model from cache")
            return self._model_cache[symbol], self._metadata_cache[symbol]
        
        if version is None:
            # Load latest model
            model_path, metadata_path = self._get_latest_model_paths(symbol)
        else:
            # Load specific version
            model_path, metadata_path = self._get_version_model_paths(symbol, version)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found for symbol {symbol} (version: {version})")
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Cache the loaded model
        if version is None:  # Only cache latest version
            self._model_cache[symbol] = model
            self._metadata_cache[symbol] = metadata
        
        logger.info(f"Model loaded for {symbol}: {os.path.basename(model_path)}")
        logger.info(f"Model type: {metadata['model_type']}")
        logger.info(f"Features: {metadata['features']['count']}")
        logger.info(f"Training accuracy: {metadata['performance'].get('training_accuracy', 'N/A')}")
        
        return model, metadata
    
    def model_exists(self, symbol: str) -> bool:
        """Check if a model exists for the given symbol"""
        try:
            self._get_latest_model_paths(symbol)
            return True
        except FileNotFoundError:
            return False
    
    def get_model_info(self, symbol: str) -> Dict[str, Any]:
        """Get model information without loading the actual model"""
        if symbol in self._metadata_cache:
            return self._metadata_cache[symbol]
        
        try:
            _, metadata_path = self._get_latest_model_paths(symbol)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except FileNotFoundError:
            return {}
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models with their information"""
        models = {}
        
        if os.path.exists(self.models_dir):
            for symbol_dir in os.listdir(self.models_dir):
                symbol_path = os.path.join(self.models_dir, symbol_dir)
                if os.path.isdir(symbol_path):
                    try:
                        info = self.get_model_info(symbol_dir)
                        if info:
                            models[symbol_dir] = {
                                'version': info.get('version', 'unknown'),
                                'created_at': info.get('created_at', 'unknown'),
                                'model_type': info.get('model_type', 'unknown'),
                                'features': info.get('features', {}).get('count', 0),
                                'training_accuracy': info.get('performance', {}).get('training_accuracy', 'N/A')
                            }
                    except Exception as e:
                        logger.warning(f"Could not load info for {symbol_dir}: {e}")
        
        return models
    
    def validate_model(self, symbol: str, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """
        Validate model performance on test data
        
        Args:
            symbol (str): Trading symbol
            test_data (np.ndarray): Test feature data
            test_labels (np.ndarray): Test labels
            
        Returns:
            Dict[str, float]: Validation metrics
        """
        model, metadata = self.load_model(symbol)
        
        # Make predictions
        predictions = model.predict(test_data)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        
        # Generate detailed report
        class_report = classification_report(test_labels, predictions, output_dict=True)
        
        validation_metrics = {
            'validation_accuracy': accuracy,
            'precision_macro': class_report['macro avg']['precision'],
            'recall_macro': class_report['macro avg']['recall'],
            'f1_macro': class_report['macro avg']['f1-score'],
            'samples_tested': len(test_labels)
        }
        
        logger.info(f"Model validation for {symbol}:")
        logger.info(f"  Validation accuracy: {accuracy:.3f}")
        logger.info(f"  Training accuracy: {metadata['performance'].get('training_accuracy', 'N/A')}")
        
        return validation_metrics
    
    def cleanup_old_models(self, symbol: str, keep_versions: int = 5) -> None:
        """
        Clean up old model versions, keeping only the most recent ones
        
        Args:
            symbol (str): Trading symbol
            keep_versions (int): Number of versions to keep
        """
        symbol_dir = os.path.join(self.models_dir, symbol)
        if not os.path.exists(symbol_dir):
            return
        
        # Get all model files for this symbol
        model_files = []
        for filename in os.listdir(symbol_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(symbol_dir, filename)
                model_files.append((filepath, os.path.getctime(filepath)))
        
        # Sort by creation time (newest first)
        model_files.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old versions
        removed_count = 0
        for filepath, _ in model_files[keep_versions:]:
            # Move to backup before deleting
            backup_path = os.path.join(self.backup_dir, os.path.basename(filepath))
            os.rename(filepath, backup_path)
            
            # Also move corresponding metadata
            base_name = os.path.basename(filepath).replace('.pkl', '')
            metadata_file = f"{base_name}_metadata.json"
            metadata_path = os.path.join(self.metadata_dir, metadata_file)
            if os.path.exists(metadata_path):
                backup_metadata_path = os.path.join(self.backup_dir, metadata_file)
                os.rename(metadata_path, backup_metadata_path)
            
            removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old model versions for {symbol}")
    
    def get_model_performance_history(self, symbol: str) -> Dict[str, Any]:
        """Get performance history for a symbol across all versions"""
        history = []
        
        # Look through metadata files for this symbol
        for filename in os.listdir(self.metadata_dir):
            if filename.startswith(f"{symbol}_v") and filename.endswith('_metadata.json'):
                metadata_path = os.path.join(self.metadata_dir, filename)
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    history.append({
                        'version': metadata.get('version'),
                        'timestamp': metadata.get('timestamp'),
                        'training_accuracy': metadata.get('performance', {}).get('training_accuracy'),
                        'feature_count': metadata.get('features', {}).get('count'),
                        'model_type': metadata.get('model_type')
                    })
                except Exception as e:
                    logger.warning(f"Could not parse metadata file {filename}: {e}")
        
        # Sort by version
        history.sort(key=lambda x: x['version'] if x['version'] else 0)
        
        return {
            'symbol': symbol,
            'total_versions': len(history),
            'history': history
        }
    
    def _get_next_version(self, symbol: str) -> int:
        """Get the next version number for a symbol"""
        symbol_dir = os.path.join(self.models_dir, symbol)
        if not os.path.exists(symbol_dir):
            return 1
        
        max_version = 0
        for filename in os.listdir(symbol_dir):
            if filename.startswith(f"{symbol}_v") and filename.endswith('.pkl'):
                try:
                    # Extract version number from filename
                    version_part = filename.split('_v')[1].split('_')[0]
                    version = int(version_part)
                    max_version = max(max_version, version)
                except (IndexError, ValueError):
                    continue
        
        return max_version + 1
    
    def _get_latest_model_paths(self, symbol: str) -> Tuple[str, str]:
        """Get paths to the latest model and metadata files"""
        symbol_dir = os.path.join(self.models_dir, symbol)
        if not os.path.exists(symbol_dir):
            raise FileNotFoundError(f"No models found for symbol: {symbol}")
        
        # Find the latest model file
        latest_model = None
        latest_time = 0
        
        for filename in os.listdir(symbol_dir):
            if filename.startswith(f"{symbol}_v") and filename.endswith('.pkl'):
                filepath = os.path.join(symbol_dir, filename)
                file_time = os.path.getctime(filepath)
                if file_time > latest_time:
                    latest_time = file_time
                    latest_model = filepath
        
        if latest_model is None:
            raise FileNotFoundError(f"No model files found for symbol: {symbol}")
        
        # Get corresponding metadata file
        base_name = os.path.basename(latest_model).replace('.pkl', '')
        metadata_filename = f"{base_name}_metadata.json"
        metadata_path = os.path.join(self.metadata_dir, metadata_filename)
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_filename}")
        
        return latest_model, metadata_path
    
    def _get_version_model_paths(self, symbol: str, version: str) -> Tuple[str, str]:
        """Get paths to specific version model and metadata files"""
        symbol_dir = os.path.join(self.models_dir, symbol)
        if not os.path.exists(symbol_dir):
            raise FileNotFoundError(f"No models found for symbol: {symbol}")
        
        # Find the specific version
        for filename in os.listdir(symbol_dir):
            if filename.startswith(f"{symbol}_v{version}_") and filename.endswith('.pkl'):
                model_path = os.path.join(symbol_dir, filename)
                
                # Get corresponding metadata
                base_name = os.path.basename(model_path).replace('.pkl', '')
                metadata_filename = f"{base_name}_metadata.json"
                metadata_path = os.path.join(self.metadata_dir, metadata_filename)
                
                return model_path, metadata_path
        
        raise FileNotFoundError(f"Version {version} not found for symbol: {symbol}")
    
    def _update_latest_model_reference(self, symbol: str, model_path: str, metadata_path: str) -> None:
        """Update reference to the latest model for quick access"""
        # This could be enhanced to create symlinks on Unix systems
        # For now, we rely on the _get_latest_model_paths method
        pass


class ModelPredictionService:
    """
    Service for making predictions using managed models
    Handles model loading, caching, and real-time predictions
    """
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize prediction service
        
        Args:
            model_manager (ModelManager): Model management instance
        """
        self.model_manager = model_manager
        self.loaded_models = {}  # Cache for loaded models
        self.model_metadata = {}  # Cache for model metadata
    
    def predict(self, symbol: str, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Make predictions for a symbol using its trained model
        
        Args:
            symbol (str): Trading symbol
            features (np.ndarray): Feature data for prediction
            
        Returns:
            Tuple[np.ndarray, Dict]: (predictions, prediction_info)
        """
        # Load model if not cached
        if symbol not in self.loaded_models:
            try:
                model, metadata = self.model_manager.load_model(symbol)
                self.loaded_models[symbol] = model
                self.model_metadata[symbol] = metadata
                logger.info(f"Loaded model for {symbol} into prediction service")
            except FileNotFoundError:
                raise ValueError(f"No trained model available for symbol: {symbol}")
        
        model = self.loaded_models[symbol]
        metadata = self.model_metadata[symbol]
        
        # Validate feature dimensions
        expected_features = metadata['features']['count']
        if features.shape[1] != expected_features:
            raise ValueError(f"Feature dimension mismatch. Expected {expected_features}, got {features.shape[1]}")
        
        # Make predictions
        predictions = model.predict(features)
        
        # Get prediction probabilities if available
        prediction_info = {
            'symbol': symbol,
            'model_version': metadata['version'],
            'model_type': metadata['model_type'],
            'features_used': expected_features,
            'predictions_count': len(predictions),
            'prediction_timestamp': dt.datetime.now().isoformat()
        }
        
        # Add prediction probabilities for probabilistic models
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(features)
                prediction_info['prediction_probabilities'] = probabilities.tolist()
                prediction_info['confidence_scores'] = np.max(probabilities, axis=1).tolist()
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")
        
        logger.debug(f"Generated {len(predictions)} predictions for {symbol}")
        
        return predictions, prediction_info
    
    def get_prediction_summary(self, symbol: str, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Generate summary statistics for predictions
        
        Args:
            symbol (str): Trading symbol
            predictions (np.ndarray): Array of predictions
            
        Returns:
            Dict[str, Any]: Prediction summary
        """
        unique, counts = np.unique(predictions, return_counts=True)
        prediction_counts = dict(zip(unique.astype(int), counts.astype(int)))
        
        summary = {
            'symbol': symbol,
            'total_predictions': int(len(predictions)),
            'buy_signals': int(prediction_counts.get(1, 0)),
            'sell_signals': int(prediction_counts.get(-1, 0)),
            'hold_signals': int(prediction_counts.get(0, 0)),
            'buy_percentage': float(prediction_counts.get(1, 0) / len(predictions) * 100),
            'signal_distribution': prediction_counts,
            'model_info': self.model_metadata.get(symbol, {})
        }
        
        return summary
    
    def refresh_model(self, symbol: str) -> bool:
        """
        Refresh cached model for a symbol (reload from disk)
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            bool: True if refresh successful
        """
        try:
            # Remove from cache
            if symbol in self.loaded_models:
                del self.loaded_models[symbol]
            if symbol in self.model_metadata:
                del self.model_metadata[symbol]
            
            # Reload model
            model, metadata = self.model_manager.load_model(symbol)
            self.loaded_models[symbol] = model
            self.model_metadata[symbol] = metadata
            
            logger.info(f"Refreshed model for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to refresh model for {symbol}: {e}")
            return False


# Global instances for easy import
model_manager = ModelManager()
prediction_service = ModelPredictionService(model_manager)