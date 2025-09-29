"""
Enhanced Backtesting Runner

This module provides comprehensive backtesting capabilities separate from
the production orchestrator, focusing on strategy performance evaluation
against historical market data.
"""

import datetime as dt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..enhanced_orchestrator import ProductionTradingOrchestrator
    from ..trading.enhanced_strategies import EnhancedMLTradingStrategy

class EnhancedBacktestingRunner:
    """
    Enhanced Backtesting Runner
    
    Provides comprehensive backtesting capabilities separate from production orchestrator.
    Focuses on historical performance evaluation with detailed analytics.
    """
    
    def __init__(self, orchestrator: 'ProductionTradingOrchestrator'):
        """
        Initialize backtesting runner with orchestrator reference
        
        Args:
            orchestrator: Production trading orchestrator instance
        """
        self.orchestrator = orchestrator
        self.backtest_history = {}
    
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
        if symbol not in self.orchestrator.strategies:
            print(f"[BACKTEST] Creating default strategy for {symbol}")
            self.orchestrator.create_enhanced_strategy(symbol)
        
        strategy = self.orchestrator.strategies[symbol]
        
        # Calculate backtest period
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=backtest_period_months * 30)
        
        try:
            # Run backtest
            backtest_result = self.orchestrator.backtester.run_backtest(
                strategy=strategy,
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                benchmark_symbol=benchmark_symbol,
                rebalance_frequency=rebalance_frequency
            )
            
            # Store results
            self.orchestrator.performance_history[symbol] = backtest_result
            
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
            
            # Store in backtest history
            self.backtest_history[f"{symbol}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"] = comprehensive_results
            
            return comprehensive_results
            
        except Exception as e:
            print(f"[BACKTEST] Backtest failed for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_multi_period_backtest(self,
                                 symbol: str,
                                 period_months: List[int],
                                 benchmark_symbol: str = 'SPY') -> Dict[str, Any]:
        """
        Run backtests across multiple time periods
        
        Args:
            symbol: Trading symbol
            period_months: List of backtest periods in months
            benchmark_symbol: Benchmark for comparison
            
        Returns:
            Multi-period backtest results
        """
        print(f"[MULTI_BACKTEST] Running multi-period backtests for {symbol}")
        
        multi_results = {
            'symbol': symbol,
            'periods': period_months,
            'backtest_start': dt.datetime.now().isoformat(),
            'period_results': {},
            'comparative_analysis': {}
        }
        
        for period in period_months:
            print(f"\n[PERIOD] {period}-month backtest")
            result = self.run_comprehensive_backtest(
                symbol=symbol,
                backtest_period_months=period,
                benchmark_symbol=benchmark_symbol
            )
            multi_results['period_results'][f"{period}m"] = result
        
        # Generate comparative analysis
        multi_results['comparative_analysis'] = self._analyze_period_performance(
            multi_results['period_results']
        )
        
        multi_results['backtest_end'] = dt.datetime.now().isoformat()
        return multi_results
    
    def run_strategy_comparison_backtest(self,
                                        symbol: str,
                                        strategy_configs: List[Dict[str, Any]],
                                        backtest_period_months: int = 6) -> Dict[str, Any]:
        """
        Run backtests comparing different strategy configurations
        
        Args:
            symbol: Trading symbol
            strategy_configs: List of strategy configuration dictionaries
            backtest_period_months: Backtest period in months
            
        Returns:
            Strategy comparison backtest results
        """
        print(f"[STRATEGY_COMPARISON] Comparing {len(strategy_configs)} strategies for {symbol}")
        
        comparison_results = {
            'symbol': symbol,
            'strategy_configs': strategy_configs,
            'backtest_start': dt.datetime.now().isoformat(),
            'strategy_results': {},
            'performance_ranking': []
        }
        
        for i, config in enumerate(strategy_configs):
            strategy_name = f"strategy_{i+1}_{config.get('order_sizing_strategy', 'default')}"
            print(f"\n[STRATEGY] {strategy_name}")
            
            # Create strategy with this configuration
            strategy = self.orchestrator.create_enhanced_strategy(
                symbol=symbol,
                **config
            )
            
            # Run backtest
            result = self.run_comprehensive_backtest(
                symbol=symbol,
                backtest_period_months=backtest_period_months
            )
            
            comparison_results['strategy_results'][strategy_name] = {
                'config': config,
                'results': result
            }
        
        # Rank strategies by performance
        comparison_results['performance_ranking'] = self._rank_strategies(
            comparison_results['strategy_results']
        )
        
        comparison_results['backtest_end'] = dt.datetime.now().isoformat()
        return comparison_results
    
    def _analyze_order_sizing_performance(self, strategy: 'EnhancedMLTradingStrategy') -> Dict[str, Any]:
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
    
    def _analyze_period_performance(self, period_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance across different time periods"""
        period_analysis = {
            'consistency_metrics': {},
            'period_comparison': {},
            'risk_adjusted_performance': {}
        }
        
        returns = {}
        sharpe_ratios = {}
        max_drawdowns = {}
        
        for period, result in period_results.items():
            if result.get('success', True) and 'performance' in result:
                perf = result['performance']
                returns[period] = perf.get('total_return', 0)
                sharpe_ratios[period] = perf.get('sharpe_ratio', 0)
                max_drawdowns[period] = perf.get('max_drawdown', 0)
        
        if returns:
            period_analysis['consistency_metrics'] = {
                'return_consistency': 1 - (np.std(list(returns.values())) / np.mean(list(returns.values()))) if np.mean(list(returns.values())) > 0 else 0,
                'sharpe_consistency': 1 - (np.std(list(sharpe_ratios.values())) / np.mean(list(sharpe_ratios.values()))) if np.mean(list(sharpe_ratios.values())) > 0 else 0
            }
            
            period_analysis['period_comparison'] = {
                'best_return_period': max(returns.items(), key=lambda x: x[1]),
                'best_sharpe_period': max(sharpe_ratios.items(), key=lambda x: x[1]),
                'lowest_drawdown_period': min(max_drawdowns.items(), key=lambda x: x[1])
            }
        
        return period_analysis
    
    def _rank_strategies(self, strategy_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank strategies by performance metrics"""
        strategy_scores = []
        
        for strategy_name, strategy_data in strategy_results.items():
            result = strategy_data['results']
            if result.get('success', True) and 'performance' in result:
                perf = result['performance']
                
                # Composite score: weighted combination of return, Sharpe, and drawdown
                return_score = perf.get('total_return', 0)
                sharpe_score = perf.get('sharpe_ratio', 0)
                drawdown_penalty = abs(perf.get('max_drawdown', 0))
                
                composite_score = (return_score * 0.4) + (sharpe_score * 0.4) - (drawdown_penalty * 0.2)
                
                strategy_scores.append({
                    'strategy_name': strategy_name,
                    'composite_score': composite_score,
                    'total_return': return_score,
                    'sharpe_ratio': sharpe_score,
                    'max_drawdown': drawdown_penalty,
                    'config': strategy_data['config']
                })
        
        # Sort by composite score (highest first)
        return sorted(strategy_scores, key=lambda x: x['composite_score'], reverse=True)
    
    def get_backtest_history(self) -> Dict[str, Any]:
        """Get backtest history"""
        return self.backtest_history
    
    def clear_backtest_history(self):
        """Clear backtest history"""
        self.backtest_history = {}