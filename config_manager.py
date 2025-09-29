"""
Enhanced Configuration Manager for ML Trading System
Provides centralized access to all configuration values including technical analysis periods
"""

import yaml
import os
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Centralized configuration management for the ML Trading System"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager
        
        Args:
            config_path: Path to the YAML configuration file
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                logger.info(f"Configuration loaded successfully from {self.config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            return {}
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., 'technical_analysis.rsi.period')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.debug(f"Configuration key '{key_path}' not found, using default: {default}")
            return default
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration"""
        return self.get('trading', {})
    
    def get_technical_analysis_config(self) -> Dict[str, Any]:
        """Get technical analysis configuration"""
        return self.get('technical_analysis', {})
        
    def get_signal_generator_config(self, generator_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific signal generator
        
        Args:
            generator_name: Name of the signal generator (rsi, macd, bollinger_bands, etc.)
            
        Returns:
            Configuration dictionary for the signal generator
        """
        return self.get(f'technical_analysis.signal_generators.{generator_name}', {})
    
    def get_technical_indicators_config(self) -> Dict[str, Any]:
        """Get technical indicators configuration"""
        return self.get('technical_analysis.technical_indicators', {})
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get machine learning configuration"""
        return self.get('machine_learning', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.get('data', {})
    
    def get_backtesting_config(self) -> Dict[str, Any]:
        """Get backtesting configuration"""
        return self.get('backtesting', {})
    
    def get_risk_management_config(self) -> Dict[str, Any]:
        """Get risk management configuration"""
        return self.get('risk_management', {})
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration"""
        return self.get('environment', {})
    
    # Convenience methods for commonly used technical analysis periods
    def get_rsi_period(self) -> int:
        """Get RSI period"""
        return self.get('technical_analysis.signal_generators.rsi.period', 14)
    
    def get_macd_periods(self) -> Dict[str, int]:
        """Get MACD periods"""
        return {
            'fast': self.get('technical_analysis.signal_generators.macd.fast_period', 12),
            'slow': self.get('technical_analysis.signal_generators.macd.slow_period', 26),
            'signal': self.get('technical_analysis.signal_generators.macd.signal_period', 9)
        }
    
    def get_bollinger_config(self) -> Dict[str, Any]:
        """Get Bollinger Bands configuration"""
        return {
            'period': self.get('technical_analysis.signal_generators.bollinger_bands.period', 20),
            'std_dev': self.get('technical_analysis.signal_generators.bollinger_bands.std_dev', 2)
        }
    
    def get_sma_periods(self) -> List[int]:
        """Get SMA periods for technical indicators"""
        return self.get('technical_analysis.technical_indicators.sma_periods', [5, 10, 20, 50])
    
    def get_ema_periods(self) -> List[int]:
        """Get EMA periods for technical indicators"""
        return self.get('technical_analysis.technical_indicators.ema_periods', [12, 26])
    
    def get_volume_config(self) -> Dict[str, Any]:
        """Get volume analysis configuration"""
        return {
            'volume_sma_period': self.get('technical_analysis.signal_generators.volume.volume_sma_period', 20),
            'obv_period': self.get('technical_analysis.signal_generators.volume.obv_period', 10),
            'volume_spike_threshold': self.get('technical_analysis.signal_generators.volume.volume_spike_threshold', 2.0)
        }
    
    def get_trading_symbols(self) -> List[str]:
        """Get list of trading symbols from configuration"""
        assets = self.get('trading.assets.stocks', [])
        symbols = [asset.get('symbol') for asset in assets if asset.get('symbol')]
        
        # Return default symbols if none configured
        if not symbols:
            logger.warning("No symbols found in config, using defaults")
            return ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL']
        
        return symbols
    
    def get_default_symbol(self) -> str:
        """Get default trading symbol"""
        return self.get('trading.default_symbol', 'AAPL')
    
    def reload_config(self):
        """Reload configuration from file"""
        self.config = self._load_config()
        logger.info("Configuration reloaded")
    
    def validate_config(self) -> bool:
        """
        Validate configuration completeness
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_sections = [
            'technical_analysis',
            'trading', 
            'machine_learning',
            'data'
        ]
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Required configuration section '{section}' is missing")
                return False
        
        logger.info("Configuration validation passed")
        return True


# Global configuration manager instance
_config_manager = None

def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get the global configuration manager instance (backward compatibility)
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None or config_path is not None:
        _config_manager = ConfigManager(config_path)
    return _config_manager

def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get the global configuration manager instance
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager instance
    """
    return get_config(config_path)

def reload_config():
    """Reload the global configuration"""
    global _config_manager
    if _config_manager:
        _config_manager.reload_config()

# Convenience functions for commonly used configuration values
def get_rsi_period() -> int:
    """Get RSI period from configuration"""
    return get_config_manager().get_rsi_period()

def get_macd_periods() -> Dict[str, int]:
    """Get MACD periods from configuration"""
    return get_config_manager().get_macd_periods()

def get_bollinger_config() -> Dict[str, Any]:
    """Get Bollinger Bands configuration"""
    return get_config_manager().get_bollinger_config()

def get_sma_periods() -> List[int]:
    """Get SMA periods from configuration"""
    return get_config_manager().get_sma_periods()

def get_ema_periods() -> List[int]:
    """Get EMA periods from configuration"""
    return get_config_manager().get_ema_periods()

def get_volume_config() -> Dict[str, Any]:
    """Get volume analysis configuration"""
    return get_config_manager().get_volume_config()

def get_trading_symbols() -> List[str]:
    """Get list of trading symbols from configuration"""
    return get_config_manager().get_trading_symbols()

def get_default_symbol() -> str:
    """Get default trading symbol"""
    return get_config_manager().get_default_symbol()