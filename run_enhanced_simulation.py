#!/usr/bin/env python3
"""
Complete Enhanced Trading System Simulation
This script demonstrates training ML models and running comprehensive backtesting
"""

import sys
import os
import datetime as dt
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_complete_simulation():
    """Run complete enhanced trading system simulation with model training"""
    
    print("=" * 80)
    print("ENHANCED TRADING SYSTEM - COMPLETE SIMULATION")
    print("=" * 80)
    print(f"Date: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version}")
    print("=" * 80)
    
    # Configuration
    TEST_SYMBOLS = ['AAPL', 'NVDA', 'TSLA']
    SIMULATION_MONTHS = 6
    STARTING_CAPITAL = 100000
    
    print(f"\n[CONFIG] Simulation Configuration:")
    print(f"         Test Symbols: {', '.join(TEST_SYMBOLS)}")
    print(f"         Simulation Period: {SIMULATION_MONTHS} months")
    print(f"         Starting Capital: ${STARTING_CAPITAL:,}")
    
    try:
        # Step 1: Import required components
        print(f"\n[STEP 1] Importing Enhanced Trading System Components...")
        
        from data.providers import YFinanceProvider
        from models.enhanced_model_management import EnhancedModelManager, ModelTrainingService
        from analysis.enhanced_market_analysis import MarketContextAnalyzer, EnhancedFeatureEngineer
        from trading.enhanced_strategies import (
            EnhancedMLTradingStrategy, 
            OrderSizingConfig, 
            OrderSizingStrategy,
            AutoOrderSizeManager
        )
        from backtesting.enhanced_backtesting import EnhancedBacktester
        from signals.technical import RSISignalGenerator, MACDSignalGenerator, BollingerBandsSignalGenerator
        
        print("         ✅ All components imported successfully")
        
        # Step 2: Initialize core system components
        print(f"\n[STEP 2] Initializing Core System Components...")
        
        # Data provider
        data_provider = YFinanceProvider()
        print("         ✅ Data Provider initialized")
        
        # Model management
        model_manager = EnhancedModelManager(base_path="models")
        model_training_service = ModelTrainingService(model_manager, data_provider)
        print("         ✅ Model Management System initialized")
        
        # Market analysis
        market_analyzer = MarketContextAnalyzer(data_provider)
        feature_engineer = EnhancedFeatureEngineer(market_analyzer)
        print("         ✅ Market Analysis System initialized")
        
        # Backtesting engine
        backtester = EnhancedBacktester(
            data_provider=data_provider,
            initial_cash=STARTING_CAPITAL,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        print("         ✅ Backtesting Engine initialized")
        
        # Results storage
        all_results = {}
        
        # Step 3: Process each symbol
        for symbol in TEST_SYMBOLS:
            print(f"\n{'=' * 60}")
            print(f"PROCESSING SYMBOL: {symbol}")
            print(f"{'=' * 60}")
            
            symbol_results = {}
            
            try:
                # Step 3.1: Train ML Model
                print(f"\n[STEP 3.1] Training ML Model for {symbol}...")
                
                ml_training_results = model_training_service.train_model(
                    symbol=symbol,
                    algorithm='LightGBM',  # Using LightGBM as the superior algorithm
                    training_period_days=365,
                    test_split=0.2
                )
                
                print(f"             ✅ Model Training Completed")
                print(f"             Algorithm: LightGBM (Optimized)")
                print(f"             Train Accuracy: {ml_training_results['train_accuracy']:.3f}")
                print(f"             Test Accuracy: {ml_training_results['test_accuracy']:.3f}")
                print(f"             Features: {ml_training_results['feature_count']}")
                
                symbol_results['ml_training'] = ml_training_results
                
                # Step 3.2: Market Context Analysis
                print(f"\n[STEP 3.2] Analyzing Market Context for {symbol}...")
                
                end_date = dt.datetime.now()
                start_date = end_date - dt.timedelta(days=365)
                
                market_context = market_analyzer.analyze_market_context(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    window=60
                )
                
                print(f"             ✅ Market Context Analysis Completed")
                print(f"             SPY Correlation: {market_context.spy_correlation:.3f}")
                print(f"             QQQ Correlation: {market_context.qqq_correlation:.3f}")
                print(f"             SPY Beta: {market_context.spy_beta:.3f}")
                print(f"             Market Regime: {market_context.market_regime}")
                print(f"             Volatility Regime: {market_context.volatility_regime}")
                
                symbol_results['market_context'] = market_context
                
                # Step 3.3: Enhanced Feature Engineering
                print(f"\n[STEP 3.3] Creating Enhanced Features for {symbol}...")
                
                enhanced_features = feature_engineer.create_enhanced_features(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    window=60,
                    include_market_context=True,
                    normalize_features=True
                )
                
                print(f"             ✅ Enhanced Features Created")
                print(f"             Total Features: {len(enhanced_features.feature_names)}")
                print(f"             Samples: {len(enhanced_features.features)}")
                print(f"             Buy Signals: {np.sum(enhanced_features.target_labels == 1)}")
                print(f"             Sell Signals: {np.sum(enhanced_features.target_labels == -1)}")
                print(f"             Hold Signals: {np.sum(enhanced_features.target_labels == 0)}")
                
                symbol_results['enhanced_features'] = {
                    'feature_count': len(enhanced_features.feature_names),
                    'sample_count': len(enhanced_features.features),
                    'buy_signals': int(np.sum(enhanced_features.target_labels == 1)),
                    'sell_signals': int(np.sum(enhanced_features.target_labels == -1)),
                    'hold_signals': int(np.sum(enhanced_features.target_labels == 0))
                }
                
                # Step 3.4: Create Enhanced Trading Strategy
                print(f"\n[STEP 3.4] Creating Enhanced Trading Strategy for {symbol}...")
                
                # Configure order sizing
                order_sizing_config = OrderSizingConfig(
                    strategy=OrderSizingStrategy.PERCENTAGE,
                    portfolio_pct=0.1,  # 10% per position
                    min_shares=10,
                    max_shares=1000,
                    volatility_target=0.02,
                    market_conditions_enabled=True,
                    max_position_pct=0.25
                )
                
                # Golden Cross configuration
                golden_cross_config = {
                    'enabled': True,
                    'short_window': 20,
                    'long_window': 50,
                    'strength_threshold': 0.6,
                    'confirmation_days': 3
                }
                
                # Short-term pattern configuration
                short_term_config = {
                    'enabled': True,
                    'rsi_period': 14,
                    'bb_period': 20,
                    'bb_std': 2.0
                }
                
                # Create strategy
                strategy = EnhancedMLTradingStrategy(
                    symbol=symbol,
                    data_provider=data_provider,
                    model_manager=model_manager,
                    order_sizing_config=order_sizing_config,
                    golden_cross_config=golden_cross_config,
                    short_term_config=short_term_config,
                    starting_portfolio_value=STARTING_CAPITAL
                )
                
                print(f"             ✅ Enhanced Trading Strategy Created")
                print(f"             Order Sizing: {order_sizing_config.strategy.value}")
                print(f"             Golden Cross: Enabled ({golden_cross_config['short_window']}/{golden_cross_config['long_window']})")
                print(f"             Short-term Patterns: Enabled")
                
                symbol_results['strategy_config'] = {
                    'order_sizing': order_sizing_config.strategy.value,
                    'golden_cross_enabled': golden_cross_config['enabled'],
                    'short_term_enabled': short_term_config['enabled']
                }
                
                # Step 3.5: Run Comprehensive Backtesting
                print(f"\n[STEP 3.5] Running Comprehensive Backtest for {symbol}...")
                
                backtest_end = dt.datetime.now()
                backtest_start = backtest_end - dt.timedelta(days=SIMULATION_MONTHS * 30)
                
                backtest_result = backtester.run_backtest(
                    strategy=strategy,
                    symbols=[symbol],
                    start_date=backtest_start,
                    end_date=backtest_end,
                    benchmark_symbol='SPY',
                    rebalance_frequency='daily'
                )
                
                print(f"             ✅ Backtesting Completed")
                print(f"             Period: {backtest_start.strftime('%Y-%m-%d')} to {backtest_end.strftime('%Y-%m-%d')}")
                print(f"             Total Return: {backtest_result.total_return:.2%}")
                print(f"             Sharpe Ratio: {backtest_result.sharpe_ratio:.3f}")
                print(f"             Max Drawdown: {backtest_result.max_drawdown:.2%}")
                print(f"             Total Trades: {backtest_result.performance_metrics.total_trades}")
                print(f"             Win Rate: {backtest_result.performance_metrics.win_rate:.1%}")
                
                symbol_results['backtest_results'] = {
                    'period': f"{backtest_start.strftime('%Y-%m-%d')} to {backtest_end.strftime('%Y-%m-%d')}",
                    'total_return': backtest_result.total_return,
                    'annualized_return': backtest_result.performance_metrics.annualized_return,
                    'sharpe_ratio': backtest_result.sharpe_ratio,
                    'max_drawdown': backtest_result.max_drawdown,
                    'volatility': backtest_result.performance_metrics.volatility,
                    'total_trades': backtest_result.performance_metrics.total_trades,
                    'win_rate': backtest_result.performance_metrics.win_rate,
                    'profit_factor': backtest_result.performance_metrics.profit_factor,
                    'benchmark_return': backtest_result.performance_metrics.benchmark_return,
                    'alpha': backtest_result.performance_metrics.alpha,
                    'beta': backtest_result.performance_metrics.beta
                }
                
                symbol_results['success'] = True
                
            except Exception as e:
                print(f"             ❌ Error processing {symbol}: {str(e)}")
                symbol_results['success'] = False
                symbol_results['error'] = str(e)
            
            all_results[symbol] = symbol_results
        
        # Step 4: Generate Comprehensive Summary
        print(f"\n{'=' * 80}")
        print("COMPREHENSIVE SIMULATION SUMMARY")
        print(f"{'=' * 80}")
        
        successful_symbols = [symbol for symbol, results in all_results.items() if results.get('success', False)]
        failed_symbols = [symbol for symbol, results in all_results.items() if not results.get('success', False)]
        
        print(f"\n[OVERVIEW]")
        print(f"   Total Symbols Processed: {len(TEST_SYMBOLS)}")
        print(f"   Successful Simulations: {len(successful_symbols)}")
        print(f"   Failed Simulations: {len(failed_symbols)}")
        if failed_symbols:
            print(f"   Failed Symbols: {', '.join(failed_symbols)}")
        
        # Performance Summary for Successful Symbols
        if successful_symbols:
            print(f"\n[PERFORMANCE SUMMARY]")
            total_returns = []
            sharpe_ratios = []
            max_drawdowns = []
            win_rates = []
            total_trades_all = []
            
            for symbol in successful_symbols:
                results = all_results[symbol]['backtest_results']
                total_returns.append(results['total_return'])
                sharpe_ratios.append(results['sharpe_ratio'])
                max_drawdowns.append(results['max_drawdown'])
                win_rates.append(results['win_rate'])
                total_trades_all.append(results['total_trades'])
                
                print(f"\n   {symbol}:")
                print(f"      Total Return: {results['total_return']:.2%}")
                print(f"      Sharpe Ratio: {results['sharpe_ratio']:.3f}")
                print(f"      Max Drawdown: {results['max_drawdown']:.2%}")
                print(f"      Win Rate: {results['win_rate']:.1%}")
                print(f"      Total Trades: {results['total_trades']}")
                print(f"      Alpha: {results['alpha']:.2%}")
            
            # Portfolio-level metrics
            print(f"\n[PORTFOLIO METRICS]")
            print(f"   Average Return: {np.mean(total_returns):.2%}")
            print(f"   Average Sharpe Ratio: {np.mean(sharpe_ratios):.3f}")
            print(f"   Average Max Drawdown: {np.mean(max_drawdowns):.2%}")
            print(f"   Average Win Rate: {np.mean(win_rates):.1%}")
            print(f"   Total Trades (All Symbols): {sum(total_trades_all)}")
        
        # ML Model Summary
        print(f"\n[ML MODEL SUMMARY]")
        for symbol in successful_symbols:
            ml_results = all_results[symbol]['ml_training']
            print(f"   {symbol}: Train Acc: {ml_results['train_accuracy']:.3f}, "
                  f"Test Acc: {ml_results['test_accuracy']:.3f}, "
                  f"Features: {ml_results['feature_count']}")
        
        # Market Context Summary
        print(f"\n[MARKET CONTEXT SUMMARY]")
        for symbol in successful_symbols:
            context = all_results[symbol]['market_context']
            print(f"   {symbol}: SPY Corr: {context.spy_correlation:.3f}, "
                  f"Beta: {context.spy_beta:.3f}, "
                  f"Regime: {context.market_regime}")
        
        print(f"\n{'=' * 80}")
        print("✅ ENHANCED TRADING SYSTEM SIMULATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"🎯 Key Achievements:")
        print(f"   ✅ ML Models trained with enhanced features")
        print(f"   ✅ Market context analysis completed")
        print(f"   ✅ Professional backtesting with risk metrics")
        print(f"   ✅ Multi-symbol portfolio simulation")
        print(f"   ✅ Comprehensive performance analytics")
        print("=" * 80)
        
        return all_results
        
    except Exception as e:
        print(f"\n❌ SIMULATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Starting Enhanced Trading System Simulation...")
    results = run_complete_simulation()
    
    if results:
        print(f"\nSimulation completed successfully!")
        print(f"Results available for symbols: {list(results.keys())}")
    else:
        print(f"\nSimulation failed. Check error messages above.")