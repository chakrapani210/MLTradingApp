"""
Signal Generator Interface
Abstract base class for all signal generators in the trading system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Enumeration of signal types"""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradingSignal:
    """
    Standardized trading signal structure
    """
    symbol: str
    timestamp: pd.Timestamp
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    strength: float   # Signal strength/magnitude
    source: str       # Source of the signal (e.g., 'RSI', 'ML_Model', 'GoldenCross')
    metadata: Dict[str, Any] = None  # Additional signal-specific data
    
    @property
    def direction(self) -> int:
        """Convert signal_type to direction for backward compatibility"""
        if self.signal_type == SignalType.BUY:
            return 1
        elif self.signal_type == SignalType.SELL:
            return -1
        else:  # HOLD
            return 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary format"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'strength': self.strength,
            'source': self.source,
            'metadata': self.metadata or {}
        }


class SignalGenerator(ABC):
    """
    Abstract base class for signal generators
    
    Implements the Command Pattern for signal generation
    Follows Single Responsibility Principle - each generator handles one signal type
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize signal generator
        
        Args:
            name: Human-readable name for this generator
            config: Configuration parameters
        """
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """
        Generate trading signals from market data
        
        Args:
            data: Market data DataFrame with OHLCV columns
            symbol: Trading symbol
            
        Returns:
            List of TradingSignal objects
        """
        pass
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        Get list of required DataFrame columns for signal generation
        
        Returns:
            List of required column names
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that input data has required structure
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if this signal generator is enabled"""
        return self.enabled
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable this signal generator"""
        self.enabled = enabled
    
    def get_name(self) -> str:
        """Get the name of this signal generator"""
        return self.name
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of this signal generator"""
        return self.config.copy()
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration parameters"""
        self.config.update(config)