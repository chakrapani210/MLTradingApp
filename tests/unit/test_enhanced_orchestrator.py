"""
Comprehensive Unit Tests for Enhanced Trading Orchestrator
Target: 90% Code Coverage
"""
import sys
import os
import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import modules to test
import src.enhanced_orchestrator as orch_module
from src.interfaces.trading_strategy import TradingSignal, SignalType


class TestProductionTradingOrchestrator:
    """Test suite for ProductionTradingOrchestrator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mock_data_provider = Mock()
        self.mock_model_manager = Mock()
        self.mock_market_analyzer = Mock()
        self.mock_feature_engineer = Mock()
        
    @patch('src.enhanced_orchestrator.YFinanceProvider')
    @patch('src.enhanced_orchestrator.DataPreprocessor')
    @patch('src.enhanced_orchestrator.TechnicalIndicatorCalculator')
    @patch('src.enhanced_orchestrator.MarketContextAnalyzer')
    @patch('src.enhanced_orchestrator.EnhancedFeatureEngineer')
    @patch('src.enhanced_orchestrator.EnhancedModelManager')
    @patch('src.enhanced_orchestrator.ModelTrainingService')
    def test_orchestrator_initialization(self, mock_model_training, 
                                       mock_model_manager, mock_feature_engineer,
                                       mock_market_analyzer, mock_indicator_calc,
                                       mock_preprocessor, mock_data_provider):
        """Test orchestrator initialization"""
        
        # Create orchestrator
        orchestrator = orch_module.ProductionTradingOrchestrator(
            starting_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        # Verify initialization
        assert orchestrator.starting_capital == 100000
        assert orchestrator.commission_rate == 0.001
        assert orchestrator.slippage_rate == 0.0005
        assert orchestrator.symbols == []
        assert orchestrator.strategies == {}
        
        # Verify components are initialized
        mock_data_provider.assert_called_once()
        mock_preprocessor.assert_called_once()
        mock_indicator_calc.assert_called_once()
        mock_market_analyzer.assert_called_once()
        mock_feature_engineer.assert_called_once()
        mock_model_manager.assert_called_once()
    # Backtester removed in lean build

    @patch('src.enhanced_orchestrator.YFinanceProvider')
    @patch('src.enhanced_orchestrator.DataPreprocessor')
    @patch('src.enhanced_orchestrator.TechnicalIndicatorCalculator')
    @patch('src.enhanced_orchestrator.MarketContextAnalyzer')
    @patch('src.enhanced_orchestrator.EnhancedFeatureEngineer')
    @patch('src.enhanced_orchestrator.EnhancedModelManager')
    @patch('src.enhanced_orchestrator.ModelTrainingService')
    def test_create_enhanced_strategy(self, mock_model_training, 
                                    mock_model_manager, mock_feature_engineer,
                                    mock_market_analyzer, mock_indicator_calc,
                                    mock_preprocessor, mock_data_provider):
        """Test enhanced strategy creation"""
        
        orchestrator = orch_module.ProductionTradingOrchestrator()
        
        # Mock EnhancedMLTradingStrategy
        with patch('src.enhanced_orchestrator.EnhancedMLTradingStrategy') as mock_strategy_class:
            mock_strategy = Mock()
            mock_strategy_class.return_value = mock_strategy
            
            # Create strategy
            strategy = orchestrator.create_enhanced_strategy(
                symbol="AAPL",
                order_sizing_strategy="percentage",
                golden_cross_enabled=True,
                short_term_patterns_enabled=True
            )
            
            # Verify strategy creation
            assert strategy == mock_strategy
            assert "AAPL" in orchestrator.strategies
            assert "AAPL" in orchestrator.symbols
            mock_strategy_class.assert_called_once()

    @patch('src.enhanced_orchestrator.YFinanceProvider')
    @patch('src.enhanced_orchestrator.DataPreprocessor')
    @patch('src.enhanced_orchestrator.TechnicalIndicatorCalculator')
    @patch('src.enhanced_orchestrator.MarketContextAnalyzer')
    @patch('src.enhanced_orchestrator.EnhancedFeatureEngineer')
    @patch('src.enhanced_orchestrator.EnhancedModelManager')
    @patch('src.enhanced_orchestrator.ModelTrainingService')
    def test_train_ml_model_existing(self, mock_model_training, 
                                   mock_model_manager, mock_feature_engineer,
                                   mock_market_analyzer, mock_indicator_calc,
                                   mock_preprocessor, mock_data_provider):
        """Test ML model training with existing model"""
        
        orchestrator = orch_module.ProductionTradingOrchestrator()
        
        # Mock existing model
        mock_model = Mock()
        mock_model.version = "v1"
        mock_model.performance_metrics = {'train_accuracy': 0.8, 'test_accuracy': 0.75}
        mock_model.feature_names = ['feature1', 'feature2']
        
        orchestrator.model_manager.list_models.return_value = [mock_model]
        
        # Train model (should return existing)
        result = orchestrator.train_ml_model("AAPL", force_retrain=False)
        
        # Verify result
        assert result['success'] == True
        assert result['model_exists'] == True
        assert result['model_version'] == "v1"
        assert result['train_accuracy'] == 0.8
        assert result['test_accuracy'] == 0.75

    @patch('src.enhanced_orchestrator.YFinanceProvider')
    @patch('src.enhanced_orchestrator.DataPreprocessor')
    @patch('src.enhanced_orchestrator.TechnicalIndicatorCalculator')
    @patch('src.enhanced_orchestrator.MarketContextAnalyzer')
    @patch('src.enhanced_orchestrator.EnhancedFeatureEngineer')
    @patch('src.enhanced_orchestrator.EnhancedModelManager')
    @patch('src.enhanced_orchestrator.ModelTrainingService')
    def test_train_ml_model_new(self, mock_model_training, 
                              mock_model_manager, mock_feature_engineer,
                              mock_market_analyzer, mock_indicator_calc,
                              mock_preprocessor, mock_data_provider):
        """Test ML model training for new model"""
        
        orchestrator = orch_module.ProductionTradingOrchestrator()
        
        # Mock no existing models
        orchestrator.model_manager.list_models.return_value = []
        
        # Mock training service result
        training_result = {
            'success': True,
            'train_accuracy': 0.85,
            'test_accuracy': 0.80,
            'model_version': 'v1'
        }
        orchestrator.model_training_service.train_model.return_value = training_result
        
        # Train model
        result = orchestrator.train_ml_model("AAPL", algorithm="RandomForest")
        
        # Verify training was called
        orchestrator.model_training_service.train_model.assert_called_once()
        assert result == training_result

    @patch('src.enhanced_orchestrator.YFinanceProvider')
    @patch('src.enhanced_orchestrator.DataPreprocessor')
    @patch('src.enhanced_orchestrator.TechnicalIndicatorCalculator')
    @patch('src.enhanced_orchestrator.MarketContextAnalyzer')
    @patch('src.enhanced_orchestrator.EnhancedFeatureEngineer')
    @patch('src.enhanced_orchestrator.EnhancedModelManager')
    @patch('src.enhanced_orchestrator.ModelTrainingService')
    def test_analyze_market_context(self, mock_model_training, 
                                  mock_model_manager, mock_feature_engineer,
                                  mock_market_analyzer, mock_indicator_calc,
                                  mock_preprocessor, mock_data_provider):
        """Test market context analysis"""
        
        orchestrator = orch_module.ProductionTradingOrchestrator()
        
        # Mock market context
        mock_context = Mock()
        mock_context.spy_correlation = 0.7
        mock_context.qqq_correlation = 0.6
        mock_context.spy_beta = 1.2
        mock_context.qqq_beta = 1.1
        mock_context.market_regime = "BULL"
        mock_context.volatility_regime = "NORMAL"
        mock_context.sector_strength = 0.8
        mock_context.market_indicators = {}
        
        orchestrator.market_analyzer.analyze_market_context.return_value = mock_context
        
        # Mock enhanced features
        mock_features = Mock()
        mock_features.feature_names = ['feature1', 'feature2', 'feature3']
        mock_features.features = np.array([[1, 2, 3], [4, 5, 6]])
        mock_features.target_labels = np.array([1, -1])
        mock_features.metadata = {}
        
        orchestrator.feature_engineer.create_enhanced_features.return_value = mock_features
        
        # Analyze market context
        result = orchestrator.analyze_market_context("AAPL", analysis_period_days=180)
        
        # Verify result structure
        assert result['symbol'] == "AAPL"
        assert 'analysis_period' in result
        assert 'market_context' in result
        assert 'enhanced_features' in result
        
        # Verify market context data
        market_ctx = result['market_context']
        assert market_ctx['spy_correlation'] == 0.7
        assert market_ctx['market_regime'] == "BULL"
        assert market_ctx['spy_beta'] == 1.2

    @pytest.mark.asyncio
    @patch('src.enhanced_orchestrator.YFinanceProvider')
    @patch('src.enhanced_orchestrator.DataPreprocessor')
    @patch('src.enhanced_orchestrator.TechnicalIndicatorCalculator')
    @patch('src.enhanced_orchestrator.MarketContextAnalyzer')
    @patch('src.enhanced_orchestrator.EnhancedFeatureEngineer')
    @patch('src.enhanced_orchestrator.EnhancedModelManager')
    @patch('src.enhanced_orchestrator.ModelTrainingService')
    async def test_generate_trading_signal_success(self, mock_model_training, 
                                                 mock_model_manager, mock_feature_engineer,
                                                 mock_market_analyzer, mock_indicator_calc,
                                                 mock_preprocessor, mock_data_provider):
        """Test successful trading signal generation"""
        
        orchestrator = orch_module.ProductionTradingOrchestrator()
        
        # Mock strategy
        mock_strategy = Mock()
        mock_signal = TradingSignal(
            symbol="AAPL",
            timestamp=pd.Timestamp.now(),
            signal_type=SignalType.BUY,
            confidence=0.8,
            strength=0.7,
            source="TEST",
            metadata={}
        )
        mock_strategy.generate_signal.return_value = mock_signal
        orchestrator.strategies["AAPL"] = mock_strategy
        
        # Mock market data
        mock_market_data = pd.DataFrame({
            'Close': [150, 155, 160, 152, 158],
            'Volume': [1000000, 1200000, 900000, 1100000, 1050000]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        orchestrator.data_provider.get_market_data.return_value = {"AAPL": mock_market_data}
        
        # Mock market context analysis
        mock_market_context = {
            'spy_correlation': 0.7,
            'market_regime': 'BULL',
            'spy_beta': 1.2,
            'feature_analysis': {'strength': 0.8}
        }
        
        with patch.object(orchestrator, 'analyze_symbol_market_context', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = mock_market_context
            
            # Generate signal
            result = await orchestrator.generate_trading_signal("AAPL")
            
            # Verify result
            assert result['success'] == True
            assert result['action'] == 'BUY'
            assert result['symbol'] == 'AAPL'
            assert result['confidence'] > 0.8  # Should be boosted by market context

    @pytest.mark.asyncio
    @patch('src.enhanced_orchestrator.YFinanceProvider')
    @patch('src.enhanced_orchestrator.DataPreprocessor')
    @patch('src.enhanced_orchestrator.TechnicalIndicatorCalculator')
    @patch('src.enhanced_orchestrator.MarketContextAnalyzer')
    @patch('src.enhanced_orchestrator.EnhancedFeatureEngineer')
    @patch('src.enhanced_orchestrator.EnhancedModelManager')
    @patch('src.enhanced_orchestrator.ModelTrainingService')
    async def test_generate_trading_signal_insufficient_data(self, mock_model_training, 
                                                           mock_model_manager, mock_feature_engineer,
                                                           mock_market_analyzer, mock_indicator_calc,
                                                           mock_preprocessor, mock_data_provider):
        """Test trading signal generation with insufficient data"""
        
        orchestrator = orch_module.ProductionTradingOrchestrator()
        
        # Mock insufficient market data
        orchestrator.data_provider.get_market_data.return_value = {}
        
        # Generate signal
        result = await orchestrator.generate_trading_signal("AAPL")
        
        # Verify error handling
        assert result['success'] == False
        assert result['action'] == 'HOLD'
        assert 'Insufficient market data' in result['reasoning']

    @pytest.mark.asyncio
    @patch('src.enhanced_orchestrator.YFinanceProvider')
    @patch('src.enhanced_orchestrator.DataPreprocessor')
    @patch('src.enhanced_orchestrator.TechnicalIndicatorCalculator')
    @patch('src.enhanced_orchestrator.MarketContextAnalyzer')
    @patch('src.enhanced_orchestrator.EnhancedFeatureEngineer')
    @patch('src.enhanced_orchestrator.EnhancedModelManager')
    @patch('src.enhanced_orchestrator.ModelTrainingService')
    async def test_analyze_symbol_market_context(self, mock_model_training, 
                                               mock_model_manager, mock_feature_engineer,
                                               mock_market_analyzer, mock_indicator_calc,
                                               mock_preprocessor, mock_data_provider):
        """Test symbol market context analysis"""
        
        orchestrator = orch_module.ProductionTradingOrchestrator()
        
        # Mock market context
        mock_context = Mock()
        mock_context.spy_correlation = 0.6
        mock_context.qqq_correlation = 0.5
        mock_context.spy_beta = 1.1
        mock_context.qqq_beta = 1.0
        mock_context.market_regime = "NEUTRAL"
        mock_context.volatility_regime = "LOW"
        mock_context.sector_strength = 0.7
        mock_context.market_indicators = {'vix': 15.0}
        
        orchestrator.market_analyzer.analyze_market_context.return_value = mock_context
        
        # Mock enhanced features
        mock_features = Mock()
        mock_features.features = np.array([[1, 2], [3, 4]])
        mock_features.feature_names = ['feature1', 'feature2']
        
        orchestrator.feature_engineer.create_enhanced_features.return_value = mock_features
        
        # Analyze context
        result = await orchestrator.analyze_symbol_market_context("AAPL")
        
        # Verify result
        assert result['spy_correlation'] == 0.6
        assert result['market_regime'] == "NEUTRAL"
        assert result['spy_beta'] == 1.1
        assert 'feature_analysis' in result
        assert result['feature_analysis']['feature_count'] == 2

    @patch('src.enhanced_orchestrator.YFinanceProvider')
    @patch('src.enhanced_orchestrator.DataPreprocessor')
    @patch('src.enhanced_orchestrator.TechnicalIndicatorCalculator')
    @patch('src.enhanced_orchestrator.MarketContextAnalyzer')
    @patch('src.enhanced_orchestrator.EnhancedFeatureEngineer')
    @patch('src.enhanced_orchestrator.EnhancedModelManager')
    @patch('src.enhanced_orchestrator.ModelTrainingService')
    def test_get_system_status(self, mock_model_training, 
                             mock_model_manager, mock_feature_engineer,
                             mock_market_analyzer, mock_indicator_calc,
                             mock_preprocessor, mock_data_provider):
        """Test system status retrieval"""
        
        orchestrator = orch_module.ProductionTradingOrchestrator()
        orchestrator.symbols = ["AAPL", "NVDA"]
        orchestrator.strategies = {"AAPL": Mock(), "NVDA": Mock()}
        
        # Mock model manager
        orchestrator.model_manager.list_models.return_value = [Mock(), Mock(), Mock()]
        
        # Get status
        status = orchestrator.get_system_status()
        
        # Verify status structure
        assert status['system_initialized'] == True
        assert status['architecture'] == 'Production-Ready Modular Design'
        assert 'layers' in status
        assert 'configuration' in status
        assert 'runtime_state' in status
        
        # Verify runtime state
        runtime = status['runtime_state']
        assert runtime['active_symbols'] == ["AAPL", "NVDA"]
        assert runtime['active_strategies'] == ["AAPL", "NVDA"]
        assert runtime['available_models'] == 3

    @patch('src.enhanced_orchestrator.YFinanceProvider')
    @patch('src.enhanced_orchestrator.DataPreprocessor')
    @patch('src.enhanced_orchestrator.TechnicalIndicatorCalculator')
    @patch('src.enhanced_orchestrator.MarketContextAnalyzer')
    @patch('src.enhanced_orchestrator.EnhancedFeatureEngineer')
    @patch('src.enhanced_orchestrator.EnhancedModelManager')
    @patch('src.enhanced_orchestrator.ModelTrainingService')
    def test_categorize_features(self, mock_model_training, 
                                mock_model_manager, mock_feature_engineer,
                                mock_market_analyzer, mock_indicator_calc,
                                mock_preprocessor, mock_data_provider):
        """Test feature categorization"""
        
        orchestrator = orch_module.ProductionTradingOrchestrator()
        
        feature_names = [
            'spy_correlation', 'rsi_14', 'macd_signal', 'bb_upper',
            'sma_20', 'ema_12', 'vix_level', 'beta_spy',
            'volume_ratio', 'price_momentum'
        ]
        
        categories = orchestrator._categorize_features(feature_names)
        
        # Verify categorization
        assert categories['market_context'] > 0  # spy_correlation, vix_level, beta_spy
        assert categories['momentum'] > 0  # rsi_14, macd_signal, price_momentum
        assert categories['volatility'] > 0  # bb_upper
        assert categories['trend'] > 0  # sma_20, ema_12
        assert categories['technical'] > 0  # volume_ratio

    @patch('src.enhanced_orchestrator.YFinanceProvider')
    @patch('src.enhanced_orchestrator.DataPreprocessor')
    @patch('src.enhanced_orchestrator.TechnicalIndicatorCalculator')
    @patch('src.enhanced_orchestrator.MarketContextAnalyzer')
    @patch('src.enhanced_orchestrator.EnhancedFeatureEngineer')
    @patch('src.enhanced_orchestrator.EnhancedModelManager')
    @patch('src.enhanced_orchestrator.ModelTrainingService')
    def test_validate_system_health(self, mock_model_training, 
                                   mock_model_manager, mock_feature_engineer,
                                   mock_market_analyzer, mock_indicator_calc,
                                   mock_preprocessor, mock_data_provider):
        """Test system health validation"""
        
        orchestrator = orch_module.ProductionTradingOrchestrator()
        
        # Mock healthy data provider
        orchestrator.data_provider.get_current_price.return_value = 150.0
        
        # Mock model manager
        orchestrator.model_manager.list_models.return_value = [Mock()]
        
        # Mock chart generator (optional)
        orchestrator.chart_generator = Mock()
        
        # Validate health
        health = orchestrator.validate_system_health()
        
        # Verify health report
        assert health['overall_status'] == 'healthy'
        assert health['component_health']['data_provider'] == 'healthy'
        assert health['component_health']['model_manager'] == 'healthy'
        assert health['component_health']['visualization'] == 'healthy'

    def test_demonstrate_enhanced_features(self):
        """Test demonstration function"""
        
        # Mock the entire demonstration
        with patch('src.enhanced_orchestrator.ProductionTradingOrchestrator') as mock_orch_class:
            mock_orchestrator = Mock()
            mock_orchestrator.run_complete_enhanced_simulation.return_value = {'success': True}
            mock_orchestrator.get_system_status.return_value = {'status': 'running'}
            mock_orch_class.return_value = mock_orchestrator
            
            # Run demonstration
            orch_module.demonstrate_enhanced_features()
            
            # Verify orchestrator was created and used
            mock_orch_class.assert_called_once()
            assert mock_orchestrator.run_complete_enhanced_simulation.call_count >= 1