"""
Data Provider Interface
Abstract base class for all data providers in the trading system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import datetime as dt


class DataProvider(ABC):
    """
    Abstract base class for data providers
    
    Follows the Strategy Pattern to allow different data sources
    while maintaining a consistent interface
    """
    
    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: dt.datetime, 
                          end_date: dt.datetime, **kwargs) -> pd.DataFrame:
        """
        Get historical price data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            start_date: Start date for data
            end_date: End date for data
            **kwargs: Additional parameters specific to data provider
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def get_market_data(self, symbols: List[str], start_date: dt.datetime,
                       end_date: dt.datetime) -> Dict[str, pd.DataFrame]:
        """
        Get market data for multiple symbols
        
        Args:
            symbols: List of trading symbols
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Get current/latest price for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price as float
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the data provider is available/connected
        
        Returns:
            True if available, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists and is tradeable
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass