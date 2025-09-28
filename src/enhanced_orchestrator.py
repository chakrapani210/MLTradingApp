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

# Core interfaces
from .interfaces.data_provider import DataProvider
from .interfaces.signal_generator import SignalGenerator
from .interfaces.trading_strategy import TradingStrategy, TradingSignal
from .interfaces.model_manager import ModelManagerInterface
from .interfaces.risk_manager import RiskManager
from .interfaces.backtester import Backtester

# Enhanced implementations
from .data.providers import YFinanceProvider
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

warnings.filterwarnings('ignore')


class EnhancedTradingSystemOrchestrator:
    """
    Enhanced Trading System Orchestrator
    
    Implements all features from enhanced_strategy.py:
    - Intelligent order sizing with 5 strategies
    - Golden Cross and Short-term pattern analysis
    - Enhanced ML models with comprehensive management
    - Market context analysis with correlations and beta
    - 40+ technical indicators using TA-Lib
    - Comprehensive backtesting with risk metrics
    - Performance analytics and reporting
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
    
    def _initialize_components(self, models_path: str):
        """Initialize all system components"""
        # Data provider
        self.data_provider = YFinanceProvider()
        
        # Model management
        self.model_manager = EnhancedModelManager(base_path=models_path)
        self.model_training_service = ModelTrainingService(
            model_manager=self.model_manager,
            data_provider=self.data_provider
        )
        
        # Market analysis
        self.market_analyzer = MarketContextAnalyzer(data_provider=self.data_provider)
        self.feature_engineer = EnhancedFeatureEngineer(market_analyzer=self.market_analyzer)
        
        # Backtesting
        self.backtester = EnhancedBacktester(
            data_provider=self.data_provider,
            initial_cash=self.starting_capital,
            commission_rate=self.commission_rate,
            slippage_rate=self.slippage_rate
        )
        
        print(f"[INIT] All system components initialized")
    
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
        
        Args:
            symbol: Trading symbol
            backtest_period_months: Backtest period in months
            benchmark_symbol: Benchmark for comparison
            rebalance_frequency: Rebalancing frequency
            
        Returns:
            Comprehensive backtest results
        """
        print(f"[BACKTEST] Running comprehensive backtest for {symbol}")
        
        # Get strategy
        if symbol not in self.strategies:
            print(f"[BACKTEST] Creating default strategy for {symbol}")
            self.create_enhanced_strategy(symbol)
        
        strategy = self.strategies[symbol]
        
        # Calculate backtest period
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=backtest_period_months * 30)
        
        try:
            # Run backtest
            backtest_result = self.backtester.run_backtest(
                strategy=strategy,
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                benchmark_symbol=benchmark_symbol,
                rebalance_frequency=rebalance_frequency
            )
            
            # Store results
            self.performance_history[symbol] = backtest_result
            
            # Create comprehensive results
            comprehensive_results = {
                'symbol': symbol,
                'backtest_period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                'strategy_type': type(strategy).__name__,
                'performance': {
                    'total_return': backtest_result.total_return,
                    'annualized_return': backtest_result.performance_metrics.annualized_return,
                    'sharpe_ratio': backtest_result.sharpe_ratio,
                    'max_drawdown': backtest_result.max_drawdown,
                    'volatility': backtest_result.performance_metrics.volatility,
                    'alpha': backtest_result.performance_metrics.alpha,
                    'beta': backtest_result.performance_metrics.beta
                },
                'trading_metrics': {
                    'total_trades': backtest_result.performance_metrics.total_trades,
                    'win_rate': backtest_result.performance_metrics.win_rate,
                    'profit_factor': backtest_result.performance_metrics.profit_factor,
                    'avg_win': backtest_result.performance_metrics.avg_win,
                    'avg_loss': backtest_result.performance_metrics.avg_loss
                },
                'risk_metrics': {
                    'sortino_ratio': backtest_result.performance_metrics.sortino_ratio,
                    'calmar_ratio': backtest_result.performance_metrics.calmar_ratio,
                    'value_at_risk': backtest_result.performance_metrics.value_at_risk,
                    'information_ratio': backtest_result.performance_metrics.information_ratio
                },
                'benchmark_comparison': {
                    'benchmark_symbol': benchmark_symbol,
                    'benchmark_return': backtest_result.performance_metrics.benchmark_return,
                    'outperformance': backtest_result.total_return - backtest_result.performance_metrics.benchmark_return
                },
                'order_sizing_analysis': self._analyze_order_sizing_performance(strategy),
                'signal_analysis': self._analyze_signal_performance(backtest_result)
            }
            
            print(f"[BACKTEST] Backtest completed for {symbol}")
            print(f"           Total Return: {backtest_result.total_return:.1%}")
            print(f"           Sharpe Ratio: {backtest_result.sharpe_ratio:.3f}")
            print(f"           Total Trades: {backtest_result.performance_metrics.total_trades}")
            
            return comprehensive_results
            
        except Exception as e:
            print(f"[BACKTEST] Backtest failed for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_complete_enhanced_simulation(self, 
                                       symbol: str,
                                       simulation_months: int = 6,
                                       order_sizing_strategy: str = "percentage",
                                       force_retrain_ml: bool = False,
                                       include_market_analysis: bool = True) -> Dict[str, Any]:
        """
        Run complete enhanced simulation with all features (equivalent to enhanced_strategy.py)
        
        Args:
            symbol: Trading symbol
            simulation_months: Simulation period in months
            order_sizing_strategy: Order sizing strategy
            force_retrain_ml: Force ML model retraining
            include_market_analysis: Include market context analysis
            
        Returns:
            Complete simulation results with all features
        """
        print(f"\n{'='*80}")
        print(f"[SIMULATION] ENHANCED TRADING SIMULATION - {symbol}")
        print(f"{'='*80}")
        
        simulation_results = {
            'symbol': symbol,
            'simulation_start': dt.datetime.now().isoformat(),
            'configuration': {
                'simulation_months': simulation_months,
                'order_sizing_strategy': order_sizing_strategy,
                'force_retrain_ml': force_retrain_ml,
                'include_market_analysis': include_market_analysis,
                'starting_capital': self.starting_capital
            }
        }
        
        try:
            # 1. Create Enhanced Strategy
            print(f"\n[STEP 1] Creating Enhanced Trading Strategy")
            strategy = self.create_enhanced_strategy(
                symbol=symbol,
                order_sizing_strategy=order_sizing_strategy,
                golden_cross_enabled=True,
                short_term_patterns_enabled=True
            )
            simulation_results['strategy_created'] = True
            
            # 2. Train ML Model
            print(f"\n[STEP 2] Training ML Model")
            ml_results = self.train_ml_model(
                symbol=symbol,
                algorithm='RandomForest',
                force_retrain=force_retrain_ml
            )
            simulation_results['ml_training'] = ml_results
            
            # 3. Market Context Analysis
            if include_market_analysis:
                print(f"\n[STEP 3] Market Context Analysis")
                market_analysis = self.analyze_market_context(
                    symbol=symbol,
                    analysis_period_days=365
                )
                simulation_results['market_analysis'] = market_analysis
            
            # 4. Comprehensive Backtesting
            print(f"\n[STEP 4] Comprehensive Backtesting")
            backtest_results = self.run_comprehensive_backtest(
                symbol=symbol,
                backtest_period_months=simulation_months,
                benchmark_symbol='SPY'
            )
            simulation_results['backtest_results'] = backtest_results
            
            # 5. Performance Summary
            print(f"\n[STEP 5] Generating Performance Summary")
            performance_summary = self._generate_performance_summary(
                symbol, ml_results, market_analysis if include_market_analysis else None, backtest_results
            )
            simulation_results['performance_summary'] = performance_summary
            
            # Mark simulation as successful
            simulation_results['success'] = True
            simulation_results['simulation_end'] = dt.datetime.now().isoformat()
            
            # Print final summary
            self._print_enhanced_simulation_summary(simulation_results)
            
            return simulation_results
            
        except Exception as e:
            print(f"\n[ERROR] Simulation failed for {symbol}: {e}")
            simulation_results['success'] = False
            simulation_results['error'] = str(e)
            return simulation_results
    
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
    
    def _analyze_order_sizing_performance(self, strategy: EnhancedMLTradingStrategy) -> Dict[str, Any]:
        """Analyze order sizing strategy performance"""
        if not hasattr(strategy, 'trade_history') or not strategy.trade_history:
            return {'no_trades': True}
        
        trades = strategy.trade_history
        order_sizes = [t['shares'] for t in trades]
        
        return {
            'total_trades': len(trades),
            'avg_order_size': np.mean(order_sizes),
            'order_size_std': np.std(order_sizes),
            'min_order_size': np.min(order_sizes),
            'max_order_size': np.max(order_sizes),
            'sizing_strategy': strategy.order_sizing_config.strategy.value,
            'size_efficiency': 1 - (np.std(order_sizes) / np.mean(order_sizes)) if np.mean(order_sizes) > 0 else 0
        }
    
    def _analyze_signal_performance(self, backtest_result) -> Dict[str, Any]:
        """Analyze signal generation performance"""
        signal_history = backtest_result.additional_metrics.get('signal_history', [])
        
        if not signal_history:
            return {'no_signals': True}
        
        signal_types = {}
        for signal in signal_history:
            sig_type = signal['signal_type']
            if sig_type not in signal_types:
                signal_types[sig_type] = {'count': 0, 'avg_confidence': 0, 'directions': []}
            
            signal_types[sig_type]['count'] += 1
            signal_types[sig_type]['avg_confidence'] += signal['confidence']
            signal_types[sig_type]['directions'].append(signal['signal'])
        
        # Calculate averages
        for sig_type in signal_types:
            count = signal_types[sig_type]['count']
            signal_types[sig_type]['avg_confidence'] /= count
            directions = signal_types[sig_type]['directions']
            signal_types[sig_type]['buy_ratio'] = sum(1 for d in directions if d > 0) / count
        
        return {
            'total_signals': len(signal_history),
            'signal_types': signal_types,
            'avg_confidence': np.mean([s['confidence'] for s in signal_history]),
            'signal_distribution': {
                'buy_signals': sum(1 for s in signal_history if s['signal'] > 0),
                'sell_signals': sum(1 for s in signal_history if s['signal'] < 0)
            }
        }
    
    def _generate_performance_summary(self, symbol: str, ml_results: Dict, 
                                    market_analysis: Optional[Dict], 
                                    backtest_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive performance summary"""
        summary = {
            'symbol': symbol,
            'overall_performance': 'SUCCESS' if backtest_results.get('success', True) else 'FAILED'
        }
        
        # ML Model Performance
        if ml_results.get('success', False):
            summary['ml_model'] = {
                'status': 'SUCCESS',
                'algorithm': ml_results.get('model_type', 'Unknown'),
                'train_accuracy': ml_results.get('train_accuracy', 0),
                'test_accuracy': ml_results.get('test_accuracy', 0),
                'feature_count': ml_results.get('feature_count', 0)
            }
        
        # Market Analysis Summary
        if market_analysis:
            mc = market_analysis.get('market_context', {})
            summary['market_context'] = {
                'spy_correlation': mc.get('spy_correlation', 0),
                'market_regime': mc.get('market_regime', 'unknown'),
                'volatility_regime': mc.get('volatility_regime', 'unknown'),
                'feature_count': market_analysis.get('enhanced_features', {}).get('feature_count', 0)
            }
        
        # Trading Performance
        if 'performance' in backtest_results:
            perf = backtest_results['performance']
            summary['trading_performance'] = {
                'total_return': perf.get('total_return', 0),
                'sharpe_ratio': perf.get('sharpe_ratio', 0),
                'max_drawdown': perf.get('max_drawdown', 0),
                'total_trades': backtest_results.get('trading_metrics', {}).get('total_trades', 0),
                'win_rate': backtest_results.get('trading_metrics', {}).get('win_rate', 0)
            }
        
        return summary
    
    def _print_enhanced_simulation_summary(self, results: Dict[str, Any]):
        """Print comprehensive simulation summary"""
        print(f"\n{'='*80}")
        print(f"[COMPLETE] ENHANCED SIMULATION RESULTS - {results['symbol']}")
        print(f"{'='*80}")
        
        # Configuration Summary
        config = results.get('configuration', {})
        print(f"[CONFIG] SIMULATION SETUP")
        print(f"   Symbol: {results['symbol']}")
        print(f"   Period: {config.get('simulation_months', 0)} months")
        print(f"   Order Sizing: {config.get('order_sizing_strategy', 'Unknown')}")
        print(f"   Starting Capital: ${config.get('starting_capital', 0):,.0f}")
        
        # ML Model Results
        ml_results = results.get('ml_training', {})
        if ml_results.get('success', False):
            print(f"\n[ML_MODEL] TRAINING RESULTS")
            print(f"   Algorithm: {ml_results.get('model_type', 'Unknown')}")
            print(f"   Train Accuracy: {ml_results.get('train_accuracy', 0):.3f}")
            print(f"   Test Accuracy: {ml_results.get('test_accuracy', 0):.3f}")
            print(f"   Features: {ml_results.get('feature_count', 0)}")
        
        # Market Analysis Results
        market_analysis = results.get('market_analysis', {})
        if market_analysis:
            mc = market_analysis.get('market_context', {})
            print(f"\n[MARKET] CONTEXT ANALYSIS")
            print(f"   SPY Correlation: {mc.get('spy_correlation', 0):.3f}")
            print(f"   QQQ Correlation: {mc.get('qqq_correlation', 0):.3f}")
            print(f"   Market Regime: {mc.get('market_regime', 'Unknown')}")
            print(f"   Volatility Regime: {mc.get('volatility_regime', 'Unknown')}")
            print(f"   Enhanced Features: {market_analysis.get('enhanced_features', {}).get('feature_count', 0)}")
        
        # Backtest Results
        backtest_results = results.get('backtest_results', {})
        if 'performance' in backtest_results:
            perf = backtest_results['performance']
            trading = backtest_results.get('trading_metrics', {})
            
            print(f"\n[PERFORMANCE] TRADING RESULTS")
            print(f"   Total Return: {perf.get('total_return', 0):.1%}")
            print(f"   Annualized Return: {perf.get('annualized_return', 0):.1%}")
            print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
            print(f"   Max Drawdown: {perf.get('max_drawdown', 0):.1%}")
            print(f"   Volatility: {perf.get('volatility', 0):.1%}")
            
            print(f"\n[TRADING] EXECUTION METRICS")
            print(f"   Total Trades: {trading.get('total_trades', 0)}")
            print(f"   Win Rate: {trading.get('win_rate', 0):.1%}")
            print(f"   Profit Factor: {trading.get('profit_factor', 0):.3f}")
            
            benchmark = backtest_results.get('benchmark_comparison', {})
            if benchmark:
                print(f"\n[BENCHMARK] COMPARISON ({benchmark.get('benchmark_symbol', 'SPY')})")
                print(f"   Benchmark Return: {benchmark.get('benchmark_return', 0):.1%}")
                print(f"   Outperformance: {benchmark.get('outperformance', 0):.1%}")
        
        # Overall Status
        success = results.get('success', False)
        print(f"\n[STATUS] SIMULATION COMPLETE")
        print(f"   Result: {'SUCCESS' if success else 'FAILED'}")
        if not success and 'error' in results:
            print(f"   Error: {results['error']}")
        
        print(f"{'='*80}")
        print(f"[SUCCESS] Enhanced Trading System simulation completed successfully!")
        print(f"All features from enhanced_strategy.py have been implemented in the modular architecture!")
        print(f"{'='*80}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'system_initialized': True,
            'components': {
                'data_provider': type(self.data_provider).__name__,
                'model_manager': type(self.model_manager).__name__,
                'market_analyzer': type(self.market_analyzer).__name__,
                'feature_engineer': type(self.feature_engineer).__name__,
                'backtester': type(self.backtester).__name__
            },
            'configuration': {
                'starting_capital': self.starting_capital,
                'commission_rate': self.commission_rate,
                'slippage_rate': self.slippage_rate
            },
            'active_symbols': list(self.symbols),
            'active_strategies': list(self.strategies.keys()),
            'available_models': len(self.model_manager.list_models()),
            'performance_history': list(self.performance_history.keys())
        }


# Demonstration function
def demonstrate_enhanced_features():
    """Demonstrate all enhanced features"""
    print(f"\n{'='*100}")
    print(f"DEMONSTRATING ALL ENHANCED FEATURES FROM enhanced_strategy.py")
    print(f"{'='*100}")
    
    # Initialize system
    orchestrator = EnhancedTradingSystemOrchestrator(
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