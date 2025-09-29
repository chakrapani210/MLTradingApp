"""
Simulation Runner

This module provides a high-level interface for running various types
of simulations and backtests with the trading system.
"""

import datetime as dt
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from .enhanced_simulation import EnhancedTradingSimulator
from .enhanced_backtesting_runner import EnhancedBacktestingRunner

if TYPE_CHECKING:
    from ..enhanced_orchestrator import ProductionTradingOrchestrator

class SimulationRunner:
    """
    High-level simulation runner interface
    
    Provides convenient methods for running different types of simulations
    and backtests without directly interacting with individual components.
    """
    
    def __init__(self, orchestrator: 'ProductionTradingOrchestrator'):
        """
        Initialize simulation runner
        
        Args:
            orchestrator: Production trading orchestrator instance
        """
        self.orchestrator = orchestrator
        self.simulator = EnhancedTradingSimulator(orchestrator)
        self.backtest_runner = EnhancedBacktestingRunner(orchestrator)
        self.run_history = {}
    
    def run_quick_simulation(self, symbol: str, months: int = 3) -> Dict[str, Any]:
        """
        Run a quick simulation for rapid testing
        
        Args:
            symbol: Trading symbol
            months: Simulation period in months
            
        Returns:
            Quick simulation results
        """
        print(f"[QUICK_SIM] Running quick simulation for {symbol}")
        
        return self.simulator.run_complete_enhanced_simulation(
            symbol=symbol,
            simulation_months=months,
            force_retrain_ml=False,
            include_market_analysis=True
        )
    
    def run_comprehensive_simulation(self, symbol: str, months: int = 6) -> Dict[str, Any]:
        """
        Run a comprehensive simulation with all features
        
        Args:
            symbol: Trading symbol
            months: Simulation period in months
            
        Returns:
            Comprehensive simulation results
        """
        print(f"[COMP_SIM] Running comprehensive simulation for {symbol}")
        
        return self.simulator.run_complete_enhanced_simulation(
            symbol=symbol,
            simulation_months=months,
            force_retrain_ml=True,
            include_market_analysis=True
        )
    
    def run_portfolio_simulation(self, symbols: List[str], months: int = 6) -> Dict[str, Any]:
        """
        Run simulation across a portfolio of symbols
        
        Args:
            symbols: List of trading symbols
            months: Simulation period in months
            
        Returns:
            Portfolio simulation results
        """
        print(f"[PORTFOLIO_SIM] Running portfolio simulation for {len(symbols)} symbols")
        
        return self.simulator.run_multi_symbol_simulation(
            symbols=symbols,
            simulation_months=months,
            force_retrain_ml=False,
            include_market_analysis=True
        )
    
    def run_strategy_optimization(self, symbol: str, months: int = 6) -> Dict[str, Any]:
        """
        Run strategy optimization by testing different parameter combinations
        
        Args:
            symbol: Trading symbol
            months: Simulation period in months
            
        Returns:
            Strategy optimization results
        """
        print(f"[STRATEGY_OPT] Running strategy optimization for {symbol}")
        
        # Define parameter ranges for optimization
        parameter_ranges = {
            'order_sizing_strategy': ['percentage', 'kelly', 'volatility'],
            'portfolio_pct': [0.05, 0.1, 0.15, 0.2],
            'volatility_target': [0.015, 0.02, 0.025],
        }
        
        return self.simulator.run_parameter_sensitivity_analysis(
            symbol=symbol,
            parameter_ranges=parameter_ranges,
            simulation_months=months
        )
    
    def run_backtest_only(self, symbol: str, months: int = 6) -> Dict[str, Any]:
        """
        Run backtest only (no ML training or market analysis)
        
        Args:
            symbol: Trading symbol
            months: Backtest period in months
            
        Returns:
            Backtest results
        """
        print(f"[BACKTEST_ONLY] Running backtest for {symbol}")
        
        return self.backtest_runner.run_comprehensive_backtest(
            symbol=symbol,
            backtest_period_months=months
        )
    
    def run_period_comparison(self, symbol: str, periods: List[int] = None) -> Dict[str, Any]:
        """
        Run backtests across different time periods for comparison
        
        Args:
            symbol: Trading symbol
            periods: List of periods in months (default: [3, 6, 12])
            
        Returns:
            Period comparison results
        """
        if periods is None:
            periods = [3, 6, 12]
        
        print(f"[PERIOD_COMP] Running period comparison for {symbol}")
        
        return self.backtest_runner.run_multi_period_backtest(
            symbol=symbol,
            period_months=periods
        )
    
    def run_strategy_comparison(self, symbol: str, months: int = 6) -> Dict[str, Any]:
        """
        Compare different strategy configurations
        
        Args:
            symbol: Trading symbol
            months: Backtest period in months
            
        Returns:
            Strategy comparison results
        """
        print(f"[STRATEGY_COMP] Running strategy comparison for {symbol}")
        
        # Define different strategy configurations to compare
        strategy_configs = [
            {
                'order_sizing_strategy': 'percentage',
                'portfolio_pct': 0.1,
                'golden_cross_enabled': True,
                'short_term_patterns_enabled': True
            },
            {
                'order_sizing_strategy': 'kelly',
                'win_rate': 0.55,
                'kelly_fraction': 0.25,
                'golden_cross_enabled': True,
                'short_term_patterns_enabled': False
            },
            {
                'order_sizing_strategy': 'volatility',
                'volatility_target': 0.02,
                'golden_cross_enabled': False,
                'short_term_patterns_enabled': True
            }
        ]
        
        return self.backtest_runner.run_strategy_comparison_backtest(
            symbol=symbol,
            strategy_configs=strategy_configs,
            backtest_period_months=months
        )
    
    def run_full_analysis(self, symbol: str, months: int = 6) -> Dict[str, Any]:
        """
        Run complete analysis including simulation, optimization, and comparison
        
        Args:
            symbol: Trading symbol
            months: Analysis period in months
            
        Returns:
            Complete analysis results
        """
        print(f"[FULL_ANALYSIS] Running complete analysis for {symbol}")
        
        analysis_results = {
            'symbol': symbol,
            'analysis_start': dt.datetime.now().isoformat(),
            'analysis_months': months
        }
        
        # 1. Comprehensive Simulation
        print("\n[STEP 1] Comprehensive Simulation")
        analysis_results['simulation'] = self.run_comprehensive_simulation(symbol, months)
        
        # 2. Strategy Optimization
        print("\n[STEP 2] Strategy Optimization")
        analysis_results['optimization'] = self.run_strategy_optimization(symbol, months)
        
        # 3. Period Comparison
        print("\n[STEP 3] Period Comparison")
        analysis_results['period_comparison'] = self.run_period_comparison(symbol)
        
        # 4. Strategy Comparison
        print("\n[STEP 4] Strategy Comparison")
        analysis_results['strategy_comparison'] = self.run_strategy_comparison(symbol, months)
        
        analysis_results['analysis_end'] = dt.datetime.now().isoformat()
        
        # Store in run history
        self.run_history[f"full_analysis_{symbol}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"] = analysis_results
        
        self._print_full_analysis_summary(analysis_results)
        
        return analysis_results
    
    def _print_full_analysis_summary(self, results: Dict[str, Any]):
        """Print summary of full analysis results"""
        print(f"\n{'='*100}")
        print(f"[COMPLETE] FULL ANALYSIS SUMMARY - {results['symbol']}")
        print(f"{'='*100}")
        
        symbol = results['symbol']
        
        # Simulation Summary
        sim_result = results.get('simulation', {})
        if sim_result.get('success'):
            print(f"[SIMULATION] ✅ Complete simulation successful")
            if 'performance_summary' in sim_result:
                perf = sim_result['performance_summary'].get('trading_performance', {})
                print(f"   Return: {perf.get('total_return', 0):.1%}")
                print(f"   Sharpe: {perf.get('sharpe_ratio', 0):.3f}")
        else:
            print(f"[SIMULATION] ❌ Simulation failed")
        
        # Optimization Summary
        opt_result = results.get('optimization', {})
        if opt_result.get('sensitivity_metrics'):
            print(f"\n[OPTIMIZATION] ✅ Parameter optimization completed")
            best_params = self._find_best_parameters(opt_result)
            if best_params:
                print(f"   Best parameters: {best_params}")
        
        # Period Comparison Summary
        period_result = results.get('period_comparison', {})
        if period_result.get('comparative_analysis'):
            print(f"\n[PERIOD_COMPARISON] ✅ Multi-period analysis completed")
            comp_analysis = period_result['comparative_analysis']
            if 'best_return_period' in comp_analysis.get('period_comparison', {}):
                best_period = comp_analysis['period_comparison']['best_return_period']
                print(f"   Best performing period: {best_period[0]} ({best_period[1]:.1%} return)")
        
        # Strategy Comparison Summary
        strategy_result = results.get('strategy_comparison', {})
        if strategy_result.get('performance_ranking'):
            print(f"\n[STRATEGY_COMPARISON] ✅ Strategy comparison completed")
            best_strategy = strategy_result['performance_ranking'][0]
            print(f"   Best strategy: {best_strategy['strategy_name']}")
            print(f"   Score: {best_strategy['composite_score']:.3f}")
        
        print(f"\n{'='*100}")
        print(f"[SUCCESS] Complete analysis finished for {symbol}!")
        print(f"{'='*100}")
    
    def _find_best_parameters(self, optimization_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find best parameter combination from optimization results"""
        param_results = optimization_result.get('parameter_results', {})
        
        best_performance = -float('inf')
        best_params = None
        
        for param_name, param_data in param_results.items():
            for result in param_data:
                sim_result = result.get('simulation_result', {})
                if sim_result.get('success'):
                    perf_summary = sim_result.get('performance_summary', {})
                    trading_perf = perf_summary.get('trading_performance', {})
                    
                    # Use Sharpe ratio as performance metric
                    sharpe = trading_perf.get('sharpe_ratio', 0)
                    if sharpe > best_performance:
                        best_performance = sharpe
                        best_params = {param_name: result['parameter_value']}
        
        return best_params
    
    def get_run_history(self) -> Dict[str, Any]:
        """Get simulation run history"""
        return self.run_history
    
    def get_simulator(self) -> EnhancedTradingSimulator:
        """Get the simulator instance"""
        return self.simulator
    
    def get_backtest_runner(self) -> EnhancedBacktestingRunner:
        """Get the backtest runner instance"""
        return self.backtest_runner