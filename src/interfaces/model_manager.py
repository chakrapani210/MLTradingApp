"""
Model Manager Interface
Abstract base class for ML model management in the trading system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from dataclasses import dataclass
from enum import Enum
import datetime as dt


class ModelStatus(Enum):
    """Enumeration of model status types"""
    TRAINED = "trained"
    TRAINING = "training"
    OUTDATED = "outdated"
    ERROR = "error"
    NOT_FOUND = "not_found"


@dataclass
class ModelMetadata:
    """
    Standardized model metadata structure
    """
    symbol: str
    version: str
    created_at: dt.datetime
    model_type: str
    performance_metrics: Dict[str, float]
    feature_names: List[str]
    training_period: Tuple[dt.datetime, dt.datetime]
    config_snapshot: Dict[str, Any]
    file_path: str
    status: ModelStatus = ModelStatus.TRAINED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format"""
        return {
            'symbol': self.symbol,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'model_type': self.model_type,
            'performance_metrics': self.performance_metrics,
            'feature_names': self.feature_names,
            'training_period': [self.training_period[0].isoformat(), 
                              self.training_period[1].isoformat()],
            'config_snapshot': self.config_snapshot,
            'file_path': self.file_path,
            'status': self.status.value
        }


class ModelManagerInterface(ABC):
    """
    Abstract base class for ML model management
    
    Implements Repository Pattern for model persistence
    Follows Single Responsibility Principle for model lifecycle management
    """
    
    @abstractmethod
    def save_model(self, symbol: str, model: BaseEstimator, 
                   metadata: ModelMetadata) -> str:
        """
        Save a trained model with metadata
        
        Args:
            symbol: Trading symbol
            model: Trained ML model
            metadata: Model metadata
            
        Returns:
            Model version string
        """
        pass
    
    @abstractmethod
    def load_model(self, symbol: str, version: Optional[str] = None) -> Tuple[BaseEstimator, ModelMetadata]:
        """
        Load a model and its metadata
        
        Args:
            symbol: Trading symbol
            version: Specific version (if None, loads latest)
            
        Returns:
            Tuple of (model, metadata)
        """
        pass
    
    @abstractmethod
    def model_exists(self, symbol: str, version: Optional[str] = None) -> bool:
        """
        Check if a model exists
        
        Args:
            symbol: Trading symbol
            version: Specific version (if None, checks latest)
            
        Returns:
            True if model exists, False otherwise
        """
        pass
    
    @abstractmethod
    def list_models(self, symbol: Optional[str] = None) -> List[ModelMetadata]:
        """
        List available models
        
        Args:
            symbol: Filter by symbol (if None, returns all)
            
        Returns:
            List of ModelMetadata objects
        """
        pass
    
    @abstractmethod
    def delete_model(self, symbol: str, version: str) -> bool:
        """
        Delete a specific model version
        
        Args:
            symbol: Trading symbol
            version: Model version to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_latest_version(self, symbol: str) -> Optional[str]:
        """
        Get the latest version for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Latest version string or None if no model exists
        """
        pass
    
    @abstractmethod
    def validate_model(self, symbol: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate model integrity and performance
        
        Args:
            symbol: Trading symbol
            version: Model version (if None, validates latest)
            
        Returns:
            Validation results dictionary
        """
        pass
    
    @abstractmethod
    def cleanup_old_versions(self, symbol: str, keep_versions: int = 5) -> int:
        """
        Clean up old model versions
        
        Args:
            symbol: Trading symbol
            keep_versions: Number of versions to keep
            
        Returns:
            Number of models deleted
        """
        pass