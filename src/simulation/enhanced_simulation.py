"""
Enhanced Trading System Simulation

This module provides comprehensive simulation capabilities that encompass
the complete trading system workflow: strategy creation, ML training,
market analysis, backtesting, and performance evaluation.

Simulations differ from backtesting by including multiple system components
and parameter variations to test overall system behavior.
"""

import datetime as dt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..enhanced_orchestrator import ProductionTradingOrchestrator

class EnhancedTradingSimulator:
    """
    Enhanced Trading System Simulator
    
    Provides comprehensive simulation capabilities separate from production orchestrator.
    Handles end-to-end system testing including strategy optimization, ML training,
    market analysis integration, and performance evaluation.
    """
    
    def __init__(self, orchestrator: 'ProductionTradingOrchestrator'):
        """
        Initialize simulator with orchestrator reference
        
        Args:
            orchestrator: Production trading orchestrator instance
        """
        self.orchestrator = orchestrator
        self.simulation_history = {}
        
    def run_complete_enhanced_simulation(self, 
                                       symbol: str,
                                       simulation_months: int = 6,
                                       order_sizing_strategy: str = "percentage",
                                       force_retrain_ml: bool = False,
                                       include_market_analysis: bool = True) -> Dict[str, Any]:
        """
        Run complete enhanced simulation with all features
        
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
                'starting_capital': self.orchestrator.starting_capital
            }
        }
        
        try:
            # 1. Create Enhanced Strategy
            print(f"\n[STEP 1] Creating Enhanced Trading Strategy")
            strategy = self.orchestrator.create_enhanced_strategy(
                symbol=symbol,
                order_sizing_strategy=order_sizing_strategy,
                golden_cross_enabled=True,
                short_term_patterns_enabled=True
            )
            simulation_results['strategy_created'] = True
            
            # 2. Train ML Model
            print(f"\n[STEP 2] Training ML Model")
            ml_results = self.orchestrator.train_ml_model(
                symbol=symbol,
                algorithm='RandomForest',
                force_retrain=force_retrain_ml
            )
            simulation_results['ml_training'] = ml_results
            
            # 3. Market Context Analysis
            if include_market_analysis:
                print(f"\n[STEP 3] Market Context Analysis")
                market_analysis = self.orchestrator.analyze_market_context(
                    symbol=symbol,
                    analysis_period_days=365
                )
                simulation_results['market_analysis'] = market_analysis
            
            # 4. Comprehensive Backtesting
            print(f"\n[STEP 4] Comprehensive Backtesting")
            from .enhanced_backtesting_runner import EnhancedBacktestingRunner
            backtest_runner = EnhancedBacktestingRunner(self.orchestrator)
            backtest_results = backtest_runner.run_comprehensive_backtest(
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
            
            # Store simulation history
            self.simulation_history[f"{symbol}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"] = simulation_results
            
            return simulation_results
            
        except Exception as e:
            print(f"\n[ERROR] Simulation failed for {symbol}: {e}")
            simulation_results['success'] = False
            simulation_results['error'] = str(e)
            return simulation_results
    
    def run_multi_symbol_simulation(self,
                                   symbols: List[str],
                                   simulation_months: int = 6,
                                   **simulation_kwargs) -> Dict[str, Any]:
        """
        Run simulation across multiple symbols
        
        Args:
            symbols: List of trading symbols
            simulation_months: Simulation period in months
            **simulation_kwargs: Additional simulation parameters
            
        Returns:
            Multi-symbol simulation results
        """
        print(f"\n{'='*100}")
        print(f"[MULTI_SIMULATION] Running simulation for {len(symbols)} symbols")
        print(f"{'='*100}")
        
        multi_results = {
            'symbols': symbols,
            'simulation_start': dt.datetime.now().isoformat(),
            'individual_results': {},
            'comparative_analysis': {}
        }
        
        # Run individual simulations
        for symbol in symbols:
            print(f"\n{'*'*60}")
            print(f"[SYMBOL] {symbol} Simulation")
            print(f"{'*'*60}")
            
            symbol_result = self.run_complete_enhanced_simulation(
                symbol=symbol,
                simulation_months=simulation_months,
                **simulation_kwargs
            )
            multi_results['individual_results'][symbol] = symbol_result
        
        # Comparative analysis
        multi_results['comparative_analysis'] = self._generate_comparative_analysis(
            multi_results['individual_results']
        )
        
        multi_results['simulation_end'] = dt.datetime.now().isoformat()
        return multi_results
    
    def run_parameter_sensitivity_analysis(self,
                                          symbol: str,
                                          parameter_ranges: Dict[str, List[Any]],
                                          simulation_months: int = 6) -> Dict[str, Any]:
        """
        Run parameter sensitivity analysis
        
        Args:
            symbol: Trading symbol
            parameter_ranges: Dictionary of parameter names to value ranges
            simulation_months: Simulation period in months
            
        Returns:
            Parameter sensitivity analysis results
        """
        print(f"\n{'='*100}")
        print(f"[SENSITIVITY] Parameter Sensitivity Analysis - {symbol}")
        print(f"{'='*100}")
        
        sensitivity_results = {
            'symbol': symbol,
            'parameter_ranges': parameter_ranges,
            'simulation_start': dt.datetime.now().isoformat(),
            'parameter_results': {},
            'sensitivity_metrics': {}
        }
        
        # Run simulations for each parameter combination
        for param_name, param_values in parameter_ranges.items():
            print(f"\n[PARAM] Testing {param_name} sensitivity")
            param_results = []
            
            for value in param_values:
                print(f"  Testing {param_name} = {value}")
                
                # Create simulation kwargs with this parameter value
                sim_kwargs = {param_name: value}
                
                result = self.run_complete_enhanced_simulation(
                    symbol=symbol,
                    simulation_months=simulation_months,
                    **sim_kwargs
                )
                
                param_results.append({
                    'parameter_value': value,
                    'simulation_result': result
                })
            
            sensitivity_results['parameter_results'][param_name] = param_results
        
        # Calculate sensitivity metrics
        sensitivity_results['sensitivity_metrics'] = self._calculate_sensitivity_metrics(
            sensitivity_results['parameter_results']
        )
        
        sensitivity_results['simulation_end'] = dt.datetime.now().isoformat()
        return sensitivity_results
    
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
        print(f"All features implemented in modular simulation architecture!")
        print(f"{'='*80}")
    
    def _generate_comparative_analysis(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis across multiple symbols"""
        comparative = {
            'performance_ranking': [],
            'risk_comparison': {},
            'strategy_effectiveness': {},
            'ml_model_comparison': {}
        }
        
        # Extract performance metrics for comparison
        symbol_performance = {}
        for symbol, result in individual_results.items():
            if result.get('success') and 'backtest_results' in result:
                perf = result['backtest_results'].get('performance', {})
                symbol_performance[symbol] = {
                    'total_return': perf.get('total_return', 0),
                    'sharpe_ratio': perf.get('sharpe_ratio', 0),
                    'max_drawdown': perf.get('max_drawdown', 0)
                }
        
        # Rank by total return
        comparative['performance_ranking'] = sorted(
            symbol_performance.items(),
            key=lambda x: x[1]['total_return'],
            reverse=True
        )
        
        return comparative
    
    def _calculate_sensitivity_metrics(self, parameter_results: Dict[str, List]) -> Dict[str, Any]:
        """Calculate parameter sensitivity metrics"""
        sensitivity_metrics = {}
        
        for param_name, param_data in parameter_results.items():
            # Extract performance metrics
            returns = []
            sharpe_ratios = []
            
            for result in param_data:
                sim_result = result['simulation_result']
                if sim_result.get('success') and 'backtest_results' in sim_result:
                    perf = sim_result['backtest_results'].get('performance', {})
                    returns.append(perf.get('total_return', 0))
                    sharpe_ratios.append(perf.get('sharpe_ratio', 0))
            
            if returns and sharpe_ratios:
                sensitivity_metrics[param_name] = {
                    'return_sensitivity': {
                        'mean': np.mean(returns),
                        'std': np.std(returns),
                        'range': max(returns) - min(returns)
                    },
                    'sharpe_sensitivity': {
                        'mean': np.mean(sharpe_ratios),
                        'std': np.std(sharpe_ratios),
                        'range': max(sharpe_ratios) - min(sharpe_ratios)
                    }
                }
        
        return sensitivity_metrics
    
    def get_simulation_history(self) -> Dict[str, Any]:
        """Get simulation history"""
        return self.simulation_history
    
    def clear_simulation_history(self):
        """Clear simulation history"""
        self.simulation_history = {}