"""
Factory Classes for Trading System Components
Implements Factory Pattern for creating components with proper configuration
"""

from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
import logging

# Import interfaces
from src.interfaces.data_provider import DataProvider
from src.interfaces.signal_generator import SignalGenerator
from src.interfaces.trading_strategy import TradingStrategy
from src.interfaces.risk_manager import RiskManager

# Import concrete implementations
from src.data.providers import YFinanceProvider, RobinhoodProvider
from src.signals.technical import RSISignalGenerator, MACDSignalGenerator, BollingerBandsSignalGenerator

logger = logging.getLogger(__name__)


class ComponentFactory(ABC):
    """
    Abstract base factory for creating trading system components
    """
    
    @abstractmethod
    def create(self, component_type: str, config: Dict[str, Any]) -> Any:
        """
        Create a component of the specified type
        
        Args:
            component_type: Type of component to create
            config: Component configuration
            
        Returns:
            Created component instance
        """
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """
        Get list of supported component types
        
        Returns:
            List of supported type names
        """
        pass


class DataProviderFactory(ComponentFactory):
    """
    Factory for creating data provider instances
    """
    
    SUPPORTED_PROVIDERS = {
        'yfinance': YFinanceProvider,
        'yahoo': YFinanceProvider,
        'robinhood': RobinhoodProvider,
        'rh': RobinhoodProvider
    }
    
    def create(self, provider_type: str, config: Dict[str, Any]) -> DataProvider:
        """
        Create a data provider instance
        
        Args:
            provider_type: Type of data provider ('yfinance', 'robinhood')
            config: Provider configuration
            
        Returns:
            DataProvider instance
        """
        provider_type_lower = provider_type.lower()
        
        if provider_type_lower not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported data provider type: {provider_type}. "
                           f"Supported types: {list(self.SUPPORTED_PROVIDERS.keys())}")
        
        provider_class = self.SUPPORTED_PROVIDERS[provider_type_lower]
        provider = provider_class(config)
        
        logger.info(f"Created {provider_class.__name__} data provider")
        return provider
    
    def get_supported_types(self) -> List[str]:
        """Get supported data provider types"""
        return list(self.SUPPORTED_PROVIDERS.keys())
    
    def get_default_config(self, provider_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a provider type
        
        Args:
            provider_type: Data provider type
            
        Returns:
            Default configuration dictionary
        """
        defaults = {
            'yfinance': {
                'auto_adjust': True,
                'prepost': False,
                'threads': True,
                'proxy': None
            },
            'robinhood': {
                'username': None,
                'password': None,
                'mfa_code': None,
                'pickle_name': 'robinhood.pickle'
            }
        }
        
        return defaults.get(provider_type.lower(), {})


class SignalGeneratorFactory(ComponentFactory):
    """
    Factory for creating signal generator instances
    """
    
    SUPPORTED_GENERATORS = {
        'rsi': RSISignalGenerator,
        'macd': MACDSignalGenerator,
        'bollinger_bands': BollingerBandsSignalGenerator,
        'bb': BollingerBandsSignalGenerator,
        # Add more signal generators here as they're implemented
    }
    
    def create(self, generator_type: str, config: Dict[str, Any]) -> SignalGenerator:
        """
        Create a signal generator instance
        
        Args:
            generator_type: Type of signal generator
            config: Generator configuration
            
        Returns:
            SignalGenerator instance
        """
        generator_type_lower = generator_type.lower()
        
        if generator_type_lower not in self.SUPPORTED_GENERATORS:
            raise ValueError(f"Unsupported signal generator type: {generator_type}. "
                           f"Supported types: {list(self.SUPPORTED_GENERATORS.keys())}")
        
        generator_class = self.SUPPORTED_GENERATORS[generator_type_lower]
        generator = generator_class(config)
        
        logger.info(f"Created {generator_class.__name__} signal generator")
        return generator
    
    def create_multiple(self, generators_config: Dict[str, Dict[str, Any]]) -> List[SignalGenerator]:
        """
        Create multiple signal generators from configuration
        
        Args:
            generators_config: Dictionary mapping generator types to their configs
            
        Returns:
            List of SignalGenerator instances
        """
        generators = []
        
        for gen_type, gen_config in generators_config.items():
            try:
                if gen_config.get('enabled', True):
                    generator = self.create(gen_type, gen_config)
                    generators.append(generator)
                else:
                    logger.info(f"Skipping disabled signal generator: {gen_type}")
            except Exception as e:
                logger.error(f"Failed to create signal generator {gen_type}: {str(e)}")
                continue
        
        logger.info(f"Created {len(generators)} signal generators")
        return generators
    
    def get_supported_types(self) -> List[str]:
        """Get supported signal generator types"""
        return list(self.SUPPORTED_GENERATORS.keys())
    
    def get_default_config(self, generator_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a signal generator type
        
        Args:
            generator_type: Signal generator type
            
        Returns:
            Default configuration dictionary
        """
        defaults = {
            'rsi': {
                'period': 14,
                'oversold_threshold': 30,
                'overbought_threshold': 70,
                'enabled': True
            },
            'macd': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9,
                'enabled': True
            },
            'bollinger_bands': {
                'period': 20,
                'std_dev': 2.0,
                'touch_threshold': 0.01,
                'enabled': True
            }
        }
        
        return defaults.get(generator_type.lower(), {'enabled': True})


class TradingStrategyFactory(ComponentFactory):
    """
    Factory for creating trading strategy instances
    Placeholder for future implementation
    """
    
    SUPPORTED_STRATEGIES = {
        # Will be populated as strategies are implemented
        # 'momentum': MomentumStrategy,
        # 'mean_reversion': MeanReversionStrategy,
        # 'ml_enhanced': MLEnhancedStrategy,
    }
    
    def create(self, strategy_type: str, config: Dict[str, Any]) -> TradingStrategy:
        """
        Create a trading strategy instance
        
        Args:
            strategy_type: Type of trading strategy
            config: Strategy configuration
            
        Returns:
            TradingStrategy instance
        """
        raise NotImplementedError("Trading strategy factory not yet implemented")
    
    def get_supported_types(self) -> List[str]:
        """Get supported trading strategy types"""
        return list(self.SUPPORTED_STRATEGIES.keys())




class RiskManagerFactory(ComponentFactory):
    """
    Factory for creating risk manager instances
    Placeholder for future implementation
    """
    
    SUPPORTED_RISK_MANAGERS = {
        # Will be populated as risk managers are implemented
        # 'basic': BasicRiskManager,
        # 'var_based': VaRRiskManager,
        # 'portfolio': PortfolioRiskManager,
    }
    
    def create(self, risk_manager_type: str, config: Dict[str, Any]) -> RiskManager:
        """
        Create a risk manager instance
        
        Args:
            risk_manager_type: Type of risk manager
            config: Risk manager configuration
            
        Returns:
            RiskManager instance
        """
        raise NotImplementedError("Risk manager factory not yet implemented")
    
    def get_supported_types(self) -> List[str]:
        """Get supported risk manager types"""
        return list(self.SUPPORTED_RISK_MANAGERS.keys())


class TradingSystemFactory:
    """
    Master factory for creating complete trading system components
    Coordinates all individual factories
    """
    
    def __init__(self):
        """
        Initialize the master factory with all component factories
        """
        self.data_provider_factory = DataProviderFactory()
        self.signal_generator_factory = SignalGeneratorFactory()
        self.trading_strategy_factory = TradingStrategyFactory()
        self.risk_manager_factory = RiskManagerFactory()
        
        logger.info("TradingSystemFactory initialized")
    
    def create_data_provider(self, provider_type: str, config: Dict[str, Any] = None) -> DataProvider:
        """
        Create a data provider with default config merge
        
        Args:
            provider_type: Type of data provider
            config: Optional configuration (merged with defaults)
            
        Returns:
            DataProvider instance
        """
        # Merge with default configuration
        default_config = self.data_provider_factory.get_default_config(provider_type)
        merged_config = {**default_config, **(config or {})}
        
        return self.data_provider_factory.create(provider_type, merged_config)
    
    def create_signal_generators(self, generators_config: Dict[str, Dict[str, Any]]) -> List[SignalGenerator]:
        """
        Create signal generators with default config merge
        
        Args:
            generators_config: Configuration for multiple generators
            
        Returns:
            List of SignalGenerator instances
        """
        # Merge each generator config with defaults
        merged_configs = {}
        
        for gen_type, gen_config in generators_config.items():
            default_config = self.signal_generator_factory.get_default_config(gen_type)
            merged_configs[gen_type] = {**default_config, **gen_config}
        
        return self.signal_generator_factory.create_multiple(merged_configs)
    
    def get_available_components(self) -> Dict[str, List[str]]:
        """
        Get all available component types
        
        Returns:
            Dictionary mapping component categories to available types
        """
        return {
            'data_providers': self.data_provider_factory.get_supported_types(),
            'signal_generators': self.signal_generator_factory.get_supported_types(),
            'trading_strategies': self.trading_strategy_factory.get_supported_types(),
            'backtesters': [],  # removed in lean build
            'risk_managers': self.risk_manager_factory.get_supported_types()
        }
    
    def validate_configuration(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate a complete system configuration
        
        Args:
            config: System configuration to validate
            
        Returns:
            Dictionary with validation errors by category
        """
        errors = {
            'data_provider': [],
            'signal_generators': [],
            'trading_strategy': [],
            'backtester': [],
            'risk_manager': []
        }
        
        # Validate data provider
        if 'data_provider' in config:
            provider_config = config['data_provider']
            provider_type = provider_config.get('type')
            
            if not provider_type:
                errors['data_provider'].append("Missing 'type' in data_provider config")
            elif provider_type not in self.data_provider_factory.get_supported_types():
                errors['data_provider'].append(f"Unsupported data provider type: {provider_type}")
        
        # Validate signal generators
        if 'signal_generators' in config:
            for gen_type, gen_config in config['signal_generators'].items():
                if gen_type not in self.signal_generator_factory.get_supported_types():
                    errors['signal_generators'].append(f"Unsupported signal generator type: {gen_type}")
        
        # Remove empty error lists
        errors = {k: v for k, v in errors.items() if v}
        
        return errors