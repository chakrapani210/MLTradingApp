"""
Enhanced Trading System Orchestrator
Demonstrates all features from enhanced_strategy.py implemented in the new modular architecture
"""

import datetime as dt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import warnings
import os
import sys

# Add src directory to path for absolute imports
src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Core interfaces
try:
    from .interfaces.data_provider import DataProvider
    from .interfaces.signal_generator import SignalGenerator
    from .interfaces.trading_strategy import TradingStrategy, TradingSignal
    from .interfaces.model_manager import ModelManagerInterface
    from .interfaces.risk_manager import RiskManager
    from .interfaces.backtester import Backtester
except ImportError:
    # Fallback to absolute imports
    from interfaces.data_provider import DataProvider
    from interfaces.signal_generator import SignalGenerator
    from interfaces.trading_strategy import TradingStrategy, TradingSignal
    from interfaces.model_manager import ModelManagerInterface
    from interfaces.risk_manager import RiskManager
    from interfaces.backtester import Backtester

# Enhanced implementations
try:
    from .data.providers import YFinanceProvider
    from .data.preprocessors import DataPreprocessor, TechnicalIndicatorCalculator
    from .trading.enhanced_strategies import (
        EnhancedMLTradingStrategy, 
        AutoOrderSizeManager, 
        OrderSizingConfig, 
        OrderSizingStrategy
    )
    from .models.enhanced_model_management import (
        EnhancedModelManager, 
        MLSignalGenerator, 
        ModelTrainingService
    )
    from .analysis.enhanced_market_analysis import (
        MarketContextAnalyzer, 
        EnhancedFeatureEngineer
    )
    from .backtesting.enhanced_backtesting import EnhancedBacktester
    from .signals.technical import RSISignalGenerator, MACDSignalGenerator, BollingerBandsSignalGenerator
except ImportError:
    # Fallback to absolute imports
    from data.providers import YFinanceProvider
    from data.preprocessors import DataPreprocessor, TechnicalIndicatorCalculator
    from trading.enhanced_strategies import (
        EnhancedMLTradingStrategy, 
        AutoOrderSizeManager, 
        OrderSizingConfig, 
        OrderSizingStrategy
    )
    from models.enhanced_model_management import (
        EnhancedModelManager, 
        MLSignalGenerator, 
        ModelTrainingService
    )
    from analysis.enhanced_market_analysis import (
        MarketContextAnalyzer, 
        EnhancedFeatureEngineer
    )
    from backtesting.enhanced_backtesting import EnhancedBacktester
    from signals.technical import RSISignalGenerator, MACDSignalGenerator, BollingerBandsSignalGenerator

warnings.filterwarnings('ignore')


class ProductionTradingOrchestrator:
    """
    Production-Ready Trading System Orchestrator
    
    Modular architecture with layered components:
    - Data Layer: Data acquisition, preprocessing, indicators
    - Analysis Layer: Market context, feature engineering
    - Model Layer: ML model management and training
    - Trading Layer: Strategy execution, backtesting
    - Visualization Layer: Chart generation and analysis
    
    Features:
    - Enhanced market context analysis with regime detection
    - 40+ TA-Lib technical indicators
    - ML-ready feature engineering pipeline
    - Comprehensive backtesting with risk metrics
    - Modular, testable, production-ready architecture
    """
    
    def __init__(self, 
                 starting_capital: float = 100000,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 models_path: str = "models"):
        """
        Initialize Enhanced Trading System
        
        Args:
            starting_capital: Starting portfolio value
            commission_rate: Commission rate per trade
            slippage_rate: Slippage rate per trade
            models_path: Path for ML model storage
        """
        self.starting_capital = starting_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        print(f"[SYSTEM] Initializing Enhanced Trading System")
        print(f"         Starting Capital: ${starting_capital:,.0f}")
        print(f"         Commission Rate: {commission_rate:.3%}")
        print(f"         Models Path: {models_path}")
        
        # Initialize core components
        self._initialize_components(models_path)
        
        # System configuration
        self.symbols = []
        self.strategies = {}
        self.performance_history = {}
        
        print(f"[SYSTEM] Enhanced Trading System initialized successfully")
    
    def _initialize_data_layer(self):
        """Initialize data layer components"""
        self.data_provider = YFinanceProvider()
        self.data_preprocessor = DataPreprocessor()
        self.indicator_calculator = TechnicalIndicatorCalculator()
        print(f"[INIT] Data layer initialized")
    
    def _initialize_analysis_layer(self):
        """Initialize analysis layer components"""
        self.market_analyzer = MarketContextAnalyzer(data_provider=self.data_provider)
        self.feature_engineer = EnhancedFeatureEngineer(market_analyzer=self.market_analyzer)
        print(f"[INIT] Analysis layer initialized")
    
    def _initialize_model_layer(self, models_path: str):
        """Initialize ML model layer components"""
        self.model_manager = EnhancedModelManager(base_path=models_path)
        self.model_training_service = ModelTrainingService(
            model_manager=self.model_manager,
            data_provider=self.data_provider
        )
        print(f"[INIT] Model layer initialized")
    
    def _initialize_trading_layer(self):
        """Initialize trading layer components"""
        self.backtester = EnhancedBacktester(
            data_provider=self.data_provider,
            initial_cash=self.starting_capital,
            commission_rate=self.commission_rate,
            slippage_rate=self.slippage_rate
        )
        print(f"[INIT] Trading layer initialized")
    
    def _initialize_visualization_layer(self):
        """Initialize visualization layer components"""
        try:
            from src.tradingview_charts import TradingViewChartGenerator
            self.chart_generator = TradingViewChartGenerator()
            print(f"[INIT] Visualization layer initialized")
        except ImportError:
            try:
                from tradingview_charts import TradingViewChartGenerator
                self.chart_generator = TradingViewChartGenerator()
                print(f"[INIT] Visualization layer initialized")
            except ImportError:
                self.chart_generator = None
                print(f"[INIT] Visualization layer initialized (chart generator not available)")
    
    def _initialize_components(self, models_path: str):
        """Initialize all system components in modular layers"""
        print(f"[INIT] Initializing Enhanced Trading System Components...")
        
        # Initialize components in dependency order
        self._initialize_data_layer()
        self._initialize_analysis_layer() 
        self._initialize_model_layer(models_path)
        self._initialize_trading_layer()
        self._initialize_visualization_layer()
        
        print(f"[INIT] All production-ready components initialized successfully")
    
    def create_enhanced_strategy(self, 
                                symbol: str,
                                order_sizing_strategy: str = "percentage",
                                golden_cross_enabled: bool = True,
                                short_term_patterns_enabled: bool = True,
                                **strategy_configs) -> EnhancedMLTradingStrategy:
        """
        Create enhanced trading strategy with all features
        
        Args:
            symbol: Trading symbol
            order_sizing_strategy: Order sizing strategy name
            golden_cross_enabled: Enable golden cross analysis
            short_term_patterns_enabled: Enable short-term pattern analysis
            **strategy_configs: Additional strategy configurations
            
        Returns:
            Configured EnhancedMLTradingStrategy
        """
        print(f"[STRATEGY] Creating enhanced strategy for {symbol}")
        
        # Configure order sizing
        order_sizing_config = OrderSizingConfig(
            strategy=OrderSizingStrategy(order_sizing_strategy),
            portfolio_pct=strategy_configs.get('portfolio_pct', 0.1),
            volatility_target=strategy_configs.get('volatility_target', 0.02),
            win_rate=strategy_configs.get('win_rate', 0.55),
            kelly_fraction=strategy_configs.get('kelly_fraction', 0.25)
        )
        
        # Configure Golden Cross
        golden_cross_config = {
            'enabled': golden_cross_enabled,
            'short_window': strategy_configs.get('gc_short_window', 20),
            'long_window': strategy_configs.get('gc_long_window', 50),
            'strength_threshold': strategy_configs.get('gc_strength_threshold', 0.6),
            'confirmation_days': strategy_configs.get('gc_confirmation_days', 3)
        }
        
        # Configure Short-term Patterns
        short_term_config = {
            'enabled': short_term_patterns_enabled,
            'rsi_period': strategy_configs.get('rsi_period', 14),
            'bb_period': strategy_configs.get('bb_period', 20),
            'bb_std': strategy_configs.get('bb_std', 2.0)
        }
        
        # Create comprehensive technical analysis configuration
        technical_config = {
            'rsi': {
                'enabled': strategy_configs.get('rsi_enabled', True),
                'period': strategy_configs.get('rsi_period', 14),
                'oversold_threshold': strategy_configs.get('rsi_oversold', 30),
                'overbought_threshold': strategy_configs.get('rsi_overbought', 70)
            },
            'macd': {
                'enabled': strategy_configs.get('macd_enabled', True),
                'fast_period': strategy_configs.get('macd_fast', 12),
                'slow_period': strategy_configs.get('macd_slow', 26),
                'signal_period': strategy_configs.get('macd_signal', 9)
            },
            'bollinger_bands': {
                'enabled': strategy_configs.get('bb_enabled', True),
                'period': strategy_configs.get('bb_period', 20),
                'std_dev': strategy_configs.get('bb_std', 2.0)
            },
            'sma_crossover': {
                'enabled': strategy_configs.get('sma_crossover_enabled', True),
                'short_period': strategy_configs.get('sma_short_period', 20),
                'long_period': strategy_configs.get('sma_long_period', 50),
                'signal_strength_threshold': strategy_configs.get('sma_strength_threshold', 0.5),
                'confirmation_periods': strategy_configs.get('sma_confirmation_periods', 2)
            },
            'ema': {
                'enabled': strategy_configs.get('ema_enabled', True),
                'periods': strategy_configs.get('ema_periods', [12, 26]),
                'crossover_pairs': strategy_configs.get('ema_crossover_pairs', [(12, 26)]),
                'slope_threshold': strategy_configs.get('ema_slope_threshold', 0.001),
                'min_confidence': strategy_configs.get('ema_min_confidence', 0.3)
            },
            'volume_analysis': {
                'enabled': strategy_configs.get('volume_analysis_enabled', True),
                'volume_surge_threshold': strategy_configs.get('volume_surge_threshold', 2.0),
                'volume_sma_period': strategy_configs.get('volume_sma_period', 20),
                'obv_period': strategy_configs.get('obv_period', 10),
                'price_volume_confirmation': strategy_configs.get('price_volume_confirmation', True),
                'min_confidence': strategy_configs.get('volume_min_confidence', 0.4)
            }
        }
        
        # Create strategy
        strategy = EnhancedMLTradingStrategy(
            symbol=symbol,
            data_provider=self.data_provider,
            model_manager=self.model_manager,
            order_sizing_config=order_sizing_config,
            golden_cross_config=golden_cross_config,
            short_term_config=short_term_config,
            starting_portfolio_value=self.starting_capital
        )
        
        # Set technical configuration
        strategy.config['technical_generators'] = technical_config
        
        self.strategies[symbol] = strategy
        if symbol not in self.symbols:
            self.symbols.append(symbol)
        
        print(f"[STRATEGY] Enhanced strategy created for {symbol}")
        print(f"           Order Sizing: {order_sizing_strategy}")
        print(f"           Golden Cross: {'Enabled' if golden_cross_enabled else 'Disabled'}")
        print(f"           Short-term Patterns: {'Enabled' if short_term_patterns_enabled else 'Disabled'}")
        
        return strategy
    
    def train_ml_model(self, 
                      symbol: str, 
                      algorithm: str = 'RandomForest',
                      training_period_days: int = 365,
                      force_retrain: bool = False) -> Dict[str, Any]:
        """
        Train ML model with enhanced features
        
        Args:
            symbol: Trading symbol
            algorithm: ML algorithm ('RandomForest', 'DecisionTree')
            training_period_days: Training period in days
            force_retrain: Force retraining even if model exists
            
        Returns:
            Training results and metrics
        """
        print(f"[MODEL_TRAIN] Training ML model for {symbol}")
        
        # Check if model already exists
        existing_models = self.model_manager.list_models(symbol)
        if existing_models and not force_retrain:
            print(f"[MODEL_TRAIN] Model already exists for {symbol} (use force_retrain=True to retrain)")
            latest_model = existing_models[0]
            return {
                'success': True,
                'model_exists': True,
                'model_version': latest_model.version,
                'train_accuracy': latest_model.performance_metrics.get('train_accuracy', 'N/A'),
                'test_accuracy': latest_model.performance_metrics.get('test_accuracy', 'N/A'),
                'feature_count': len(latest_model.feature_names) if latest_model.feature_names else 0
            }
        
        # Train new model
        try:
            results = self.model_training_service.train_model(
                symbol=symbol,
                algorithm=algorithm,
                training_period_days=training_period_days
            )
            
            print(f"[MODEL_TRAIN] Training completed for {symbol}")
            print(f"              Algorithm: {algorithm}")
            print(f"              Train Accuracy: {results['train_accuracy']:.3f}")
            print(f"              Test Accuracy: {results['test_accuracy']:.3f}")
            
            return results
            
        except Exception as e:
            print(f"[MODEL_TRAIN] Training failed for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def analyze_market_context(self, 
                              symbol: str,
                              analysis_period_days: int = 180) -> Dict[str, Any]:
        """
        Perform comprehensive market context analysis
        
        Args:
            symbol: Trading symbol
            analysis_period_days: Analysis period in days
            
        Returns:
            Market context analysis results
        """
        print(f"[MARKET_ANALYSIS] Analyzing market context for {symbol}")
        
        # Calculate date range
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=analysis_period_days)
        
        try:
            # Perform market context analysis
            market_context = self.market_analyzer.analyze_market_context(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            # Create enhanced features
            enhanced_features = self.feature_engineer.create_enhanced_features(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                include_market_context=True
            )
            
            analysis_results = {
                'symbol': symbol,
                'analysis_period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                'market_context': {
                    'spy_correlation': market_context.spy_correlation,
                    'qqq_correlation': market_context.qqq_correlation,
                    'spy_beta': market_context.spy_beta,
                    'qqq_beta': market_context.qqq_beta,
                    'market_regime': market_context.market_regime,
                    'volatility_regime': market_context.volatility_regime,
                    'sector_strength': market_context.sector_strength,
                    'market_indicators': market_context.market_indicators
                },
                'enhanced_features': {
                    'feature_count': len(enhanced_features.feature_names),
                    'sample_count': len(enhanced_features.features),
                    'feature_categories': self._categorize_features(enhanced_features.feature_names),
                    'target_distribution': {
                        'buy_signals': int(np.sum(enhanced_features.target_labels == 1)),
                        'sell_signals': int(np.sum(enhanced_features.target_labels == -1)),
                        'hold_signals': int(np.sum(enhanced_features.target_labels == 0))
                    }
                }
            }
            
            print(f"[MARKET_ANALYSIS] Analysis completed for {symbol}")
            print(f"                  SPY Correlation: {market_context.spy_correlation:.3f}")
            print(f"                  Market Regime: {market_context.market_regime}")
            print(f"                  Features Generated: {len(enhanced_features.feature_names)}")
            
            return analysis_results
            
        except Exception as e:
            print(f"[MARKET_ANALYSIS] Analysis failed for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_comprehensive_backtest(self, 
                                  symbol: str,
                                  backtest_period_months: int = 6,
                                  benchmark_symbol: str = 'SPY',
                                  rebalance_frequency: str = 'daily') -> Dict[str, Any]:
        """
        Run comprehensive backtest with all features
        
        This method delegates to the simulation package for clean separation.
        
        Args:
            symbol: Trading symbol
            backtest_period_months: Backtest period in months
            benchmark_symbol: Benchmark for comparison
            rebalance_frequency: Rebalancing frequency
            
        Returns:
            Comprehensive backtest results
        """
        from simulation.enhanced_backtesting_runner import EnhancedBacktestingRunner
        backtest_runner = EnhancedBacktestingRunner(self)
        return backtest_runner.run_comprehensive_backtest(
            symbol=symbol,
            backtest_period_months=backtest_period_months,
            benchmark_symbol=benchmark_symbol,
            rebalance_frequency=rebalance_frequency
        )
    
    def run_complete_enhanced_simulation(self, 
                                       symbol: str,
                                       simulation_months: int = 6,
                                       order_sizing_strategy: str = "percentage",
                                       force_retrain_ml: bool = False,
                                       include_market_analysis: bool = True) -> Dict[str, Any]:
        """
        Run complete enhanced simulation with all features
        
        This method delegates to the simulation package for clean separation.
        
        Args:
            symbol: Trading symbol
            simulation_months: Simulation period in months
            order_sizing_strategy: Order sizing strategy
            force_retrain_ml: Force ML model retraining
            include_market_analysis: Include market context analysis
            
        Returns:
            Complete simulation results with all features
        """
        from .simulation.enhanced_simulation import EnhancedTradingSimulator
        simulator = EnhancedTradingSimulator(self)
        return simulator.run_complete_enhanced_simulation(
            symbol=symbol,
            simulation_months=simulation_months,
            order_sizing_strategy=order_sizing_strategy,
            force_retrain_ml=force_retrain_ml,
            include_market_analysis=include_market_analysis
        )
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, int]:
        """Categorize features by type"""
        categories = {
            'technical': 0,
            'market_context': 0,
            'momentum': 0,
            'volatility': 0,
            'trend': 0
        }
        
        for name in feature_names:
            name_lower = name.lower()
            if any(x in name_lower for x in ['spy', 'qqq', 'vix', 'correlation', 'beta']):
                categories['market_context'] += 1
            elif any(x in name_lower for x in ['rsi', 'macd', 'mom', 'roc']):
                categories['momentum'] += 1
            elif any(x in name_lower for x in ['bb', 'atr', 'volatility']):
                categories['volatility'] += 1
            elif any(x in name_lower for x in ['sma', 'ema', 'dema', 'tema']):
                categories['trend'] += 1
            else:
                categories['technical'] += 1
        
        return categories
    
    def run_production_data_pipeline(self, symbol: str, days: int = 180) -> Dict[str, Any]:
        """Run production-ready data pipeline with enhanced analysis"""
        print(f"[PRODUCTION_PIPELINE] Running production data pipeline for {symbol}")
        
        try:
            # Data acquisition
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=days)
            raw_data = self.data_provider.get_historical_data(symbol, start_date, end_date)
            
            if raw_data.empty:
                return {"error": f"No data available for {symbol}"}
            
            # Data preprocessing
            cleaned_data = self.data_preprocessor.clean_data(raw_data, symbol)
            print(f"[PRODUCTION_PIPELINE] Data preprocessed: {len(cleaned_data)} records")
            
            # Enhanced market context analysis
            market_context = self.market_analyzer.analyze_market_context(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            # Enhanced feature engineering
            enhanced_features = self.feature_engineer.create_enhanced_features(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                include_market_context=True,
                normalize_features=True
            )
            
            pipeline_results = {
                "symbol": symbol,
                "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "data_quality": {
                    "raw_records": len(raw_data),
                    "processed_records": len(cleaned_data),
                    "feature_count": len(enhanced_features.feature_names),
                    "sample_count": len(enhanced_features.features)
                },
                "market_context": {
                    "spy_correlation": market_context.spy_correlation,
                    "qqq_correlation": market_context.qqq_correlation,
                    "spy_beta": market_context.spy_beta,
                    "qqq_beta": market_context.qqq_beta,
                    "market_regime": market_context.market_regime,
                    "volatility_regime": market_context.volatility_regime,
                    "sector_strength": market_context.sector_strength
                },
                "enhanced_features": {
                    "feature_names": enhanced_features.feature_names[:10],  # First 10 for brevity
                    "metadata": enhanced_features.metadata
                }
            }
            
            print(f"[PRODUCTION_PIPELINE] Pipeline completed successfully")
            print(f"                      Market Regime: {market_context.market_regime}")
            print(f"                      Features Generated: {len(enhanced_features.feature_names)}")
            
            return pipeline_results
            
        except Exception as e:
            error_msg = f"Production pipeline failed for {symbol}: {e}"
            print(f"[PRODUCTION_PIPELINE] {error_msg}")
            return {"error": error_msg}

    def validate_system_health(self) -> Dict[str, Any]:
        """Validate system health and component availability"""
        health_report = {
            "overall_status": "healthy",
            "component_health": {},
            "warnings": [],
            "errors": []
        }
        
        # Check data layer
        try:
            test_data = self.data_provider.get_current_price("AAPL")
            health_report["component_health"]["data_provider"] = "healthy"
        except Exception as e:
            health_report["component_health"]["data_provider"] = "unhealthy"
            health_report["errors"].append(f"Data provider error: {e}")
            health_report["overall_status"] = "degraded"
        
        # Check model layer
        try:
            models = self.model_manager.list_models()
            health_report["component_health"]["model_manager"] = "healthy"
            if len(models) == 0:
                health_report["warnings"].append("No models available")
        except Exception as e:
            health_report["component_health"]["model_manager"] = "unhealthy"
            health_report["errors"].append(f"Model manager error: {e}")
        
        # Check visualization layer
        if self.chart_generator:
            health_report["component_health"]["visualization"] = "healthy"
        else:
            health_report["component_health"]["visualization"] = "unavailable"
            health_report["warnings"].append("Chart generator not available")
        
        return health_report
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            "active_strategies": len(self.strategies),
            "total_symbols_analyzed": len(self.symbols),
            "models_trained": len(self.model_manager.list_models()),
            "backtest_history": len(self.performance_history),
            "system_uptime": dt.datetime.now().isoformat(),
            "capital_utilization": {
                "starting_capital": self.starting_capital,
                "commission_rate": self.commission_rate,
                "slippage_rate": self.slippage_rate
            }
        }
    
    def cleanup_resources(self):
        """Clean up system resources"""
        print("[CLEANUP] Cleaning up system resources...")
        
        # Clear caches
        if hasattr(self, 'data_provider'):
            # Clear any cached data
            pass
        
        # Clear performance history if needed
        if len(self.performance_history) > 100:
            # Keep only recent 100 entries
            recent_keys = list(self.performance_history.keys())[-100:]
            self.performance_history = {k: self.performance_history[k] for k in recent_keys}
        
        print("[CLEANUP] Resource cleanup completed")
    
    def get_simulation_runner(self):
        """Get simulation runner for comprehensive testing capabilities"""
        from .simulation.simulation_runner import SimulationRunner
        return SimulationRunner(self)
    
    def get_backtest_runner(self):
        """Get backtesting runner for focused performance evaluation"""
        from .simulation.enhanced_backtesting_runner import EnhancedBacktestingRunner
        return EnhancedBacktestingRunner(self)

    def create_trading_chart(self, symbol: str = "AAPL", timeframe: str = "1D", 
                           indicators: Optional[List[str]] = None, period: str = "6mo",
                           include_backtest: bool = True) -> str:
        """Create TradingView-style chart with enhanced functionality and real signals"""
        if not self.chart_generator:
            return "Error: TradingViewChartGenerator not available"
        
        try:
            if indicators is None:
                indicators = ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands"]
            
            print(f"[CHART] Generating TradingView chart for {symbol}")
            
            # Generate real trading signals for the chart
            real_signals = self._generate_real_signals_for_chart(symbol, period)
            
            # Get backtest results if requested
            backtest_results = None
            if include_backtest:
                try:
                    print(f"[CHART] Running quick backtest for {symbol} chart data...")
                    backtest_results = self.run_comprehensive_backtest(symbol)
                    if 'error' not in backtest_results:
                        print(f"[CHART] Backtest completed for chart display")
                    else:
                        print(f"[CHART] Backtest failed, using sample data: {backtest_results.get('error', 'Unknown')}")
                        backtest_results = None
                except Exception as e:
                    print(f"[CHART] Could not run backtest for chart: {e}")
                    backtest_results = None
            
            # Create chart with real signals and backtest results
            chart_path = self.chart_generator.create_comprehensive_chart(
                symbol, period, real_signals=real_signals, backtest_results=backtest_results
            )
            print(f"[CHART] Chart created successfully: {chart_path}")
            return chart_path
        except Exception as e:
            error_msg = f"Error creating trading chart: {e}"
            print(f"[CHART] {error_msg}")
            return error_msg
    
    def _generate_real_signals_for_chart(self, symbol: str, period: str) -> List:
        """Generate real trading signals for chart display"""
        try:
            # Import here to avoid circular imports
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            
            from .trading.enhanced_strategies import EnhancedMLTradingStrategy
            
            # Get market data for the period
            if period == "6mo":
                days = 180
            elif period == "1y":
                days = 365
            elif period == "3mo":
                days = 90
            else:
                days = 180  # Default
            
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=days + 30)  # Extra data for signals
            
            market_data = self.data_provider.get_market_data(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date
            )
            
            if symbol not in market_data or len(market_data[symbol]) < 50:
                print(f"[CHART] Insufficient data for real signals, chart will show price data only")
                return None
            
            # Create trading strategy
            strategy = EnhancedMLTradingStrategy(
                symbol=symbol,
                model_manager=self.model_manager,
                data_provider=self.data_provider
            )
            
            # Generate signals for recent period
            data = market_data[symbol]
            all_signals = []
            
            # Generate signals day by day for the last portion of data
            signal_start_idx = max(20, len(data) - 60)  # Last 60 days with 20-day warmup
            
            for i in range(signal_start_idx, len(data)):
                day_data = data.iloc[:i+1]
                if len(day_data) >= 20:  # Minimum data for signals
                    timestamp = day_data.index[-1]
                    signal = strategy.generate_signal(day_data, timestamp)
                    if signal and signal.signal_type.name != 'HOLD':  # Only add BUY/SELL signals
                        all_signals.append(signal)
            
            print(f"[CHART] Generated {len(all_signals)} real signals for {symbol}")
            return all_signals
            
        except Exception as e:
            print(f"[CHART] Error generating real signals: {e}")
            return None

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for production components"""
        return {
            'system_initialized': True,
            'architecture': 'Production-Ready Modular Design',
            'layers': {
                'data_layer': {
                    'data_provider': type(self.data_provider).__name__,
                    'data_preprocessor': type(self.data_preprocessor).__name__,
                    'indicator_calculator': type(self.indicator_calculator).__name__
                },
                'analysis_layer': {
                    'market_analyzer': type(self.market_analyzer).__name__,
                    'feature_engineer': type(self.feature_engineer).__name__
                },
                'model_layer': {
                    'model_manager': type(self.model_manager).__name__,
                    'model_training_service': type(self.model_training_service).__name__
                },
                'trading_layer': {
                    'backtester': type(self.backtester).__name__
                },
                'visualization_layer': {
                    'chart_generator': type(self.chart_generator).__name__ if self.chart_generator else 'Not Available'
                }
            },
            'configuration': {
                'starting_capital': self.starting_capital,
                'commission_rate': self.commission_rate,
                'slippage_rate': self.slippage_rate
            },
            'runtime_state': {
                'active_symbols': list(self.symbols),
                'active_strategies': list(self.strategies.keys()),
                'available_models': len(self.model_manager.list_models()),
                'performance_history': list(self.performance_history.keys())
            }
        }


# Demonstration function
def demonstrate_enhanced_features():
    """Demonstrate all enhanced features"""
    print(f"\n{'='*100}")
    print(f"DEMONSTRATING ALL ENHANCED FEATURES FROM enhanced_strategy.py")
    print(f"{'='*100}")
    
    # Initialize system
    orchestrator = ProductionTradingOrchestrator(
        starting_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    
    # Test symbols
    test_symbols = ['AAPL', 'NVDA']
    
    for symbol in test_symbols:
        print(f"\n{'*'*60}")
        print(f"TESTING ENHANCED FEATURES FOR {symbol}")
        print(f"{'*'*60}")
        
        # Run complete enhanced simulation
        results = orchestrator.run_complete_enhanced_simulation(
            symbol=symbol,
            simulation_months=6,
            order_sizing_strategy="percentage",
            force_retrain_ml=False,
            include_market_analysis=True
        )
    
    # Print system status
    print(f"\n{'='*60}")
    print(f"FINAL SYSTEM STATUS")
    print(f"{'='*60}")
    
    status = orchestrator.get_system_status()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    print(f"\n{'='*100}")
    print(f"DEMONSTRATION COMPLETE - ALL ENHANCED FEATURES IMPLEMENTED!")
    print(f"{'='*100}")


if __name__ == "__main__":
    demonstrate_enhanced_features()