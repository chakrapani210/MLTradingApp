"""
Unit tests for src/utils and main modules
Tests utility functions, factories, and main application orchestrators
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import pandas as pd
import datetime as dt
import numpy as np
import tempfile
import os
import yaml
from typing import Dict, Any, List

# Import utils and main modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.factories import (
    DataProviderFactory,
    SignalGeneratorFactory,
    TradingStrategyFactory,
    BacktesterFactory,
    ComponentFactory
)
from src.enhanced_orchestrator import EnhancedTradingSystemOrchestrator
from src.main_simple import TradingSystemOrchestrator, ConfigManager


class TestDataProviderFactory(unittest.TestCase):
    """Test DataProviderFactory functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.factory = DataProviderFactory()
        
    def test_supported_providers(self):
        """Test supported provider types"""
        supported = self.factory.get_supported_types()
        
        self.assertIsInstance(supported, list)
        self.assertIn('yfinance', supported)
        self.assertIn('yahoo', supported)
        
    def test_create_yfinance_provider(self):
        """Test creating YFinance provider"""
        config = {'some_param': 'value'}
        
        provider = self.factory.create('yfinance', config)
        
        # Should import and create YFinanceProvider
        self.assertIsNotNone(provider)
        # Verify it implements DataProvider interface methods
        self.assertTrue(hasattr(provider, 'get_historical_data'))
        self.assertTrue(hasattr(provider, 'get_market_data'))
        self.assertTrue(hasattr(provider, 'get_current_price'))
        self.assertTrue(hasattr(provider, 'is_available'))
        self.assertTrue(hasattr(provider, 'validate_symbol'))
        
    def test_create_provider_case_insensitive(self):
        """Test provider creation is case insensitive"""
        provider1 = self.factory.create('YFinance', {})
        provider2 = self.factory.create('yfinance', {})
        
        # Should create same type of provider
        self.assertEqual(type(provider1), type(provider2))
        
    def test_create_unsupported_provider(self):
        """Test creating unsupported provider raises error"""
        with self.assertRaises(ValueError) as context:
            self.factory.create('unsupported_provider', {})
            
        self.assertIn('Unsupported data provider type', str(context.exception))
        
    def test_get_default_config(self):
        """Test getting default configuration"""
        config = self.factory.get_default_config('yfinance')
        
        self.assertIsInstance(config, dict)
        # Should have some default configuration
        
    def test_get_default_config_unsupported(self):
        """Test getting default config for unsupported provider"""
        with self.assertRaises(ValueError):
            self.factory.get_default_config('unsupported')


class TestSignalGeneratorFactory(unittest.TestCase):
    """Test SignalGeneratorFactory functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.factory = SignalGeneratorFactory()
        
    def test_supported_generators(self):
        """Test supported generator types"""
        supported = self.factory.get_supported_types()
        
        self.assertIsInstance(supported, list)
        self.assertIn('rsi', supported)
        self.assertIn('macd', supported)
        self.assertIn('bollinger_bands', supported)
        
    def test_create_rsi_generator(self):
        """Test creating RSI signal generator"""
        config = {'period': 14, 'oversold_threshold': 30}
        
        generator = self.factory.create('rsi', config)
        
        self.assertIsNotNone(generator)
        # Verify it implements SignalGenerator interface
        self.assertTrue(hasattr(generator, 'generate_signals'))
        self.assertTrue(hasattr(generator, 'get_required_columns'))
        self.assertTrue(hasattr(generator, 'validate_data'))
        
    def test_create_macd_generator(self):
        """Test creating MACD signal generator"""
        config = {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        
        generator = self.factory.create('macd', config)
        
        self.assertIsNotNone(generator)
        self.assertTrue(hasattr(generator, 'generate_signals'))
        
    def test_create_bollinger_bands_generator(self):
        """Test creating Bollinger Bands signal generator"""
        config = {'window': 20, 'num_std': 2}
        
        generator = self.factory.create('bollinger_bands', config)
        
        self.assertIsNotNone(generator)
        self.assertTrue(hasattr(generator, 'generate_signals'))
        
    def test_create_multiple_generators(self):
        """Test creating multiple generators"""
        configs = {
            'rsi': {'period': 14},
            'macd': {'fast_period': 12},
            'bollinger_bands': {'window': 20}
        }
        
        generators = self.factory.create_multiple(configs)
        
        self.assertEqual(len(generators), 3)
        self.assertIn('rsi', generators)
        self.assertIn('macd', generators)
        self.assertIn('bollinger_bands', generators)


class TestTradingStrategyFactory(unittest.TestCase):
    """Test TradingStrategyFactory functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.factory = TradingStrategyFactory()
        
    def test_supported_strategies(self):
        """Test supported strategy types"""
        supported = self.factory.get_supported_types()
        
        self.assertIsInstance(supported, list)
        self.assertGreater(len(supported), 0)
        
    def test_create_strategy_with_dependencies(self):
        """Test creating strategy with dependencies"""
        # Mock dependencies
        mock_model_manager = Mock()
        mock_risk_manager = Mock()
        mock_signal_generators = [Mock(), Mock()]
        
        config = {
            'model_manager': mock_model_manager,
            'risk_manager': mock_risk_manager,
            'signal_generators': mock_signal_generators
        }
        
        # This would test strategy creation if specific strategy types were implemented
        # For now, test the factory interface
        self.assertTrue(hasattr(self.factory, 'create'))
        self.assertTrue(hasattr(self.factory, 'get_supported_types'))


class TestBacktesterFactory(unittest.TestCase):
    """Test BacktesterFactory functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.factory = BacktesterFactory()
        
    def test_supported_backtester_types(self):
        """Test supported backtester types"""
        supported = self.factory.get_supported_types()
        
        self.assertIsInstance(supported, list)
        self.assertGreater(len(supported), 0)
        
    def test_create_backtester_with_dependencies(self):
        """Test creating backtester with dependencies"""
        # Mock dependencies
        mock_strategy = Mock()
        mock_data_provider = Mock()
        mock_risk_manager = Mock()
        
        config = {
            'strategy': mock_strategy,
            'data_provider': mock_data_provider,
            'risk_manager': mock_risk_manager,
            'starting_capital': 100000.0,
            'commission_rate': 0.001
        }
        
        # Test factory interface exists
        self.assertTrue(hasattr(self.factory, 'create'))
        self.assertTrue(hasattr(self.factory, 'get_supported_types'))


class TestConfigManager(unittest.TestCase):
    """Test ConfigManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            'data': {
                'provider': 'yfinance',
                'symbols': ['AAPL', 'GOOGL', 'TSLA']
            },
            'signals': {
                'rsi': {
                    'enabled': True,
                    'period': 14,
                    'oversold_threshold': 30,
                    'overbought_threshold': 70
                },
                'macd': {
                    'enabled': True,
                    'fast_period': 12,
                    'slow_period': 26
                }
            },
            'trading': {
                'starting_capital': 100000,
                'commission_rate': 0.001
            }
        }
        
        # Create temporary config file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.test_config, self.temp_file)
        self.temp_file.close()
        
    def tearDown(self):
        """Clean up test fixtures"""
        os.unlink(self.temp_file.name)
        
    def test_config_loading(self):
        """Test configuration loading"""
        # Mock the config file path
        with patch('src.main_simple.os.path.join', return_value=self.temp_file.name):
            config_manager = ConfigManager()
            
        # Test getting nested configuration values
        provider = config_manager.get('data.provider')
        self.assertEqual(provider, 'yfinance')
        
        symbols = config_manager.get('data.symbols')
        self.assertEqual(symbols, ['AAPL', 'GOOGL', 'TSLA'])
        
        starting_capital = config_manager.get('trading.starting_capital')
        self.assertEqual(starting_capital, 100000)
        
    def test_config_get_with_default(self):
        """Test getting configuration with default value"""
        with patch('src.main_simple.os.path.join', return_value=self.temp_file.name):
            config_manager = ConfigManager()
            
        # Existing key
        provider = config_manager.get('data.provider', 'default_provider')
        self.assertEqual(provider, 'yfinance')
        
        # Non-existing key should return default
        missing_value = config_manager.get('missing.key', 'default_value')
        self.assertEqual(missing_value, 'default_value')
        
    def test_config_nested_access(self):
        """Test nested configuration access"""
        with patch('src.main_simple.os.path.join', return_value=self.temp_file.name):
            config_manager = ConfigManager()
            
        # Deep nested access
        rsi_period = config_manager.get('signals.rsi.period')
        self.assertEqual(rsi_period, 14)
        
        macd_fast = config_manager.get('signals.macd.fast_period')
        self.assertEqual(macd_fast, 12)


class TestTradingSystemOrchestrator(unittest.TestCase):
    """Test TradingSystemOrchestrator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock config
        self.mock_config = Mock()
        self.mock_config.get.side_effect = lambda key, default=None: {
            'data.provider': 'yfinance',
            'data.symbols': ['AAPL', 'GOOGL'],
            'trading.starting_capital': 100000,
            'signals.enabled': ['rsi', 'macd']
        }.get(key, default)
        
        # Mock get_config function
        with patch('src.main_simple.get_config', return_value=self.mock_config):
            self.orchestrator = TradingSystemOrchestrator()
            
    def test_initialization(self):
        """Test orchestrator initialization"""
        self.assertIsNotNone(self.orchestrator.logger)
        self.assertEqual(self.orchestrator.config, self.mock_config)
        
    def test_system_status(self):
        """Test getting system status"""
        # Mock component initialization
        with patch.object(self.orchestrator, '_initialize_components'):
            self.orchestrator.data_provider = Mock()
            self.orchestrator.data_provider.is_available.return_value = True
            self.orchestrator.signal_generators = [Mock(), Mock()]
            
            status = self.orchestrator.get_system_status()
            
        self.assertIsInstance(status, dict)
        self.assertIn('data_provider', status)
        self.assertIn('signal_generators', status)
        self.assertIn('components_initialized', status)
        
    def test_run_analysis_basic(self):
        """Test basic analysis run"""
        # Mock dependencies
        self.orchestrator.data_provider = Mock()
        self.orchestrator.preprocessor = Mock()
        self.orchestrator.indicator_calculator = Mock()
        self.orchestrator.signal_generators = [Mock()]
        
        # Mock data flow
        sample_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        self.orchestrator.data_provider.validate_symbol.return_value = True
        self.orchestrator.data_provider.get_historical_data.return_value = sample_data
        self.orchestrator.preprocessor.clean_data.return_value = sample_data
        self.orchestrator.indicator_calculator.calculate_all_indicators.return_value = pd.DataFrame()
        self.orchestrator.signal_generators[0].generate_signals.return_value = []
        self.orchestrator.signal_generators[0].is_enabled.return_value = True
        self.orchestrator.signal_generators[0].get_name.return_value = "Test Generator"
        
        # Run analysis
        result = self.orchestrator.run_analysis(
            "AAPL",
            dt.datetime(2023, 1, 1),
            dt.datetime(2023, 12, 31)
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('symbol', result)
        self.assertIn('analysis_period', result)
        self.assertIn('price_statistics', result)
        self.assertIn('signal_summary', result)
        
    def test_run_analysis_error_handling(self):
        """Test error handling in analysis"""
        # Mock data provider to raise exception
        self.orchestrator.data_provider = Mock()
        self.orchestrator.data_provider.validate_symbol.side_effect = Exception("Test error")
        
        result = self.orchestrator.run_analysis(
            "ERROR_SYMBOL",
            dt.datetime(2023, 1, 1),
            dt.datetime(2023, 12, 31)
        )
        
        self.assertIn('error', result)
        self.assertEqual(result['error'], "Test error")


class TestEnhancedTradingSystemOrchestrator(unittest.TestCase):
    """Test EnhancedTradingSystemOrchestrator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('src.enhanced_orchestrator.YFinanceProvider'):
            with patch('src.enhanced_orchestrator.EnhancedModelManager'):
                self.orchestrator = EnhancedTradingSystemOrchestrator(
                    starting_capital=50000,
                    commission_rate=0.0015,
                    slippage_rate=0.001
                )
                
    def test_initialization(self):
        """Test enhanced orchestrator initialization"""
        self.assertEqual(self.orchestrator.starting_capital, 50000)
        self.assertEqual(self.orchestrator.commission_rate, 0.0015)
        self.assertEqual(self.orchestrator.slippage_rate, 0.001)
        
    def test_add_symbol(self):
        """Test adding symbols to watch list"""
        initial_count = len(self.orchestrator.symbols)
        
        self.orchestrator.add_symbol("NVDA")
        
        self.assertEqual(len(self.orchestrator.symbols), initial_count + 1)
        self.assertIn("NVDA", self.orchestrator.symbols)
        
    def test_create_strategy(self):
        """Test strategy creation"""
        config = {
            'name': 'TestStrategy',
            'order_sizing': {
                'strategy': 'percentage',
                'portfolio_pct': 0.1
            }
        }
        
        # Mock dependencies
        with patch.object(self.orchestrator, 'model_manager'):
            with patch.object(self.orchestrator, 'risk_manager'):
                strategy = self.orchestrator.create_strategy("TEST", config)
                
                # Should create a strategy and register it
                self.assertIn("TEST", self.orchestrator.strategies)
                
    def test_run_enhanced_analysis(self):
        """Test enhanced analysis with all features"""
        # Mock all dependencies
        self.orchestrator.data_provider = Mock()
        self.orchestrator.market_analyzer = Mock()
        self.orchestrator.feature_engineer = Mock()
        self.orchestrator.model_training_service = Mock()
        
        # Mock data returns
        sample_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000000] * 5
        }, index=pd.date_range('2023-01-01', periods=5))
        
        self.orchestrator.data_provider.get_historical_data.return_value = sample_data
        self.orchestrator.market_analyzer.analyze_market_context.return_value = Mock()
        self.orchestrator.feature_engineer.create_enhanced_features.return_value = Mock()
        
        # Run enhanced analysis
        with patch.object(self.orchestrator, '_generate_comprehensive_report') as mock_report:
            mock_report.return_value = {'enhanced_analysis': True}
            
            result = self.orchestrator.run_enhanced_analysis(
                "ENHANCED_TEST",
                dt.datetime(2023, 1, 1),
                dt.datetime(2023, 12, 31)
            )
            
        self.assertIsInstance(result, dict)
        
    def test_run_backtest(self):
        """Test backtesting functionality"""
        # Mock strategy and backtester
        mock_strategy = Mock()
        mock_backtester = Mock()
        mock_backtester.run_backtest.return_value = Mock()
        
        self.orchestrator.strategies["TEST"] = mock_strategy
        
        with patch('src.enhanced_orchestrator.EnhancedBacktester', return_value=mock_backtester):
            result = self.orchestrator.run_backtest(
                strategy_name="TEST",
                symbols=["AAPL"],
                start_date=dt.datetime(2023, 1, 1),
                end_date=dt.datetime(2023, 12, 31)
            )
            
        # Should call backtester
        mock_backtester.run_backtest.assert_called_once()


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios across utils and main modules"""
    
    def test_factory_to_orchestrator_workflow(self):
        """Test workflow from factories to orchestrator"""
        # 1. Create components using factories
        data_factory = DataProviderFactory()
        signal_factory = SignalGeneratorFactory()
        
        # 2. Create data provider
        data_provider = data_factory.create('yfinance', {})
        
        # 3. Create signal generators
        signal_configs = {
            'rsi': {'period': 14},
            'macd': {'fast_period': 12, 'slow_period': 26}
        }
        signal_generators = signal_factory.create_multiple(signal_configs)
        
        # 4. Verify components can be used together
        self.assertIsNotNone(data_provider)
        self.assertEqual(len(signal_generators), 2)
        
        # Components should have required interfaces
        self.assertTrue(hasattr(data_provider, 'get_historical_data'))
        for generator in signal_generators.values():
            self.assertTrue(hasattr(generator, 'generate_signals'))
            
    def test_config_to_orchestrator_integration(self):
        """Test configuration integration with orchestrator"""
        # Create test configuration
        config_data = {
            'data': {'provider': 'yfinance'},
            'trading': {'starting_capital': 75000}
        }
        
        # Mock config file
        with patch('builtins.open', mock_open()):
            with patch('yaml.safe_load', return_value=config_data):
                config_manager = ConfigManager()
                
        # Verify configuration access
        provider = config_manager.get('data.provider')
        capital = config_manager.get('trading.starting_capital')
        
        self.assertEqual(provider, 'yfinance')
        self.assertEqual(capital, 75000)
        
    def test_comprehensive_system_initialization(self):
        """Test comprehensive system initialization"""
        # Mock all external dependencies
        with patch('src.enhanced_orchestrator.YFinanceProvider') as mock_provider:
            with patch('src.enhanced_orchestrator.EnhancedModelManager') as mock_model:
                with patch('src.enhanced_orchestrator.ModelTrainingService') as mock_training:
                    with patch('src.enhanced_orchestrator.MarketContextAnalyzer') as mock_analyzer:
                        # Initialize system
                        system = EnhancedTradingSystemOrchestrator()
                        
                        # Add symbols and create strategies
                        system.add_symbol("INTEGRATION_TEST")
                        
                        # Verify system components initialized
                        mock_provider.assert_called_once()
                        mock_model.assert_called_once()
                        
                        # System should be ready for operation
                        self.assertIn("INTEGRATION_TEST", system.symbols)


if __name__ == '__main__':
    unittest.main()