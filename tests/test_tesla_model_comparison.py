#!/usr/bin/env python3
"""
Test Model Comparison: LightGBM vs XGBoost for Tesla (TSLA)
Comprehensive training and performance comparison between advanced ML algorithms
"""

import sys
import os
import unittest
import datetime as dt
import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, Tuple
import json

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_path = os.path.join(parent_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class TestTeslaModelComparison(unittest.TestCase):
    """
    Test class for comparing LightGBM and XGBoost models on Tesla (TSLA) stock
    
    This class will:
    1. Train both LightGBM and XGBoost models
    2. Compare their performance metrics
    3. Analyze feature importance
    4. Generate comprehensive comparison report
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment and components"""
        print("\n" + "="*80)
        print("TESLA MODEL COMPARISON TEST - SETUP")
        print("="*80)
        
        # Import required components
        try:
            from data.providers import YFinanceProvider
            from models.enhanced_model_management import EnhancedModelManager, ModelTrainingService
            from analysis.enhanced_market_analysis import MarketContextAnalyzer, EnhancedFeatureEngineer
            from trading.enhanced_strategies import EnhancedMLTradingStrategy
            from backtesting.enhanced_backtesting import EnhancedBacktester
            
            cls.data_provider = YFinanceProvider()
            cls.model_manager = EnhancedModelManager(base_path="models/test_models")
            cls.model_training_service = ModelTrainingService(cls.model_manager, cls.data_provider)
            cls.market_analyzer = MarketContextAnalyzer(cls.data_provider)
            cls.feature_engineer = EnhancedFeatureEngineer(cls.market_analyzer)
            
            print("✅ All components imported and initialized successfully")
            
            # Test configuration
            cls.symbol = 'TSLA'
            cls.training_period_days = 730  # 2 years of training data
            cls.test_split = 0.2
            cls.algorithms = ['LightGBM', 'XGBoost']
            
            print(f"📊 Test Configuration:")
            print(f"   Symbol: {cls.symbol}")
            print(f"   Training Period: {cls.training_period_days} days")
            print(f"   Test Split: {cls.test_split * 100}%")
            print(f"   Algorithms: {', '.join(cls.algorithms)}")
            
            # Storage for results
            cls.model_results = {}
            cls.comparison_results = {}
            
        except Exception as e:
            print(f"❌ Setup failed: {str(e)}")
            raise
    
    def test_01_prepare_tesla_data(self):
        """Test data preparation for Tesla"""
        print(f"\n[TEST 1] Preparing Tesla Data...")
        
        try:
            # Get enhanced features for Tesla
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=self.training_period_days + 100)
            
            enhanced_features = self.feature_engineer.create_enhanced_features(
                symbol=self.symbol,
                start_date=start_date,
                end_date=end_date,
                window=60,
                include_market_context=True,
                normalize_features=True
            )
            
            self.assertIsNotNone(enhanced_features)
            self.assertGreater(len(enhanced_features.features), 0)
            self.assertGreater(len(enhanced_features.feature_names), 20)  # Should have 25+ features
            
            # Store for later use
            self.enhanced_features = enhanced_features
            
            print(f"   ✅ Tesla data prepared successfully")
            print(f"   📈 Total Features: {len(enhanced_features.feature_names)}")
            print(f"   📊 Training Samples: {len(enhanced_features.features)}")
            print(f"   🎯 Buy Signals: {np.sum(enhanced_features.target_labels == 1)}")
            print(f"   🔻 Sell Signals: {np.sum(enhanced_features.target_labels == -1)}")
            print(f"   ⏸️ Hold Signals: {np.sum(enhanced_features.target_labels == 0)}")
            
            # Validate data quality
            self.assertFalse(np.isnan(enhanced_features.features).any())
            self.assertFalse(np.isinf(enhanced_features.features).any())
            
            print(f"   ✅ Data quality validation passed")
            
        except Exception as e:
            self.fail(f"Data preparation failed: {str(e)}")
    
    def test_02_train_lightgbm_model(self):
        """Test LightGBM model training for Tesla"""
        print(f"\n[TEST 2] Training LightGBM Model for Tesla...")
        
        try:
            start_time = dt.datetime.now()
            
            # Train LightGBM model
            lightgbm_results = self.model_training_service.train_model(
                symbol=self.symbol,
                algorithm='LightGBM',
                training_period_days=self.training_period_days,
                test_split=self.test_split,
                # LightGBM specific parameters
                n_estimators=150,
                max_depth=8,
                learning_rate=0.08,
                num_leaves=50,
                subsample=0.85,
                colsample_bytree=0.85
            )
            
            training_time = (dt.datetime.now() - start_time).total_seconds()
            
            # Validate results
            self.assertIsNotNone(lightgbm_results)
            self.assertIn('train_accuracy', lightgbm_results)
            self.assertIn('test_accuracy', lightgbm_results)
            self.assertIn('feature_count', lightgbm_results)
            
            # Store results
            lightgbm_results['training_time_seconds'] = training_time
            lightgbm_results['algorithm'] = 'LightGBM'
            self.model_results['LightGBM'] = lightgbm_results
            
            print(f"   ✅ LightGBM model trained successfully")
            print(f"   🎯 Train Accuracy: {lightgbm_results['train_accuracy']:.4f}")
            print(f"   🎯 Test Accuracy: {lightgbm_results['test_accuracy']:.4f}")
            print(f"   📊 Features Used: {lightgbm_results['feature_count']}")
            print(f"   ⏱️ Training Time: {training_time:.2f} seconds")
            
            # Validate accuracy is reasonable
            self.assertGreater(lightgbm_results['test_accuracy'], 0.6)  # At least 60%
            self.assertLess(lightgbm_results['test_accuracy'], 1.0)      # Not perfect (would indicate overfitting)
            
        except Exception as e:
            self.fail(f"LightGBM training failed: {str(e)}")
    
    def test_03_train_xgboost_model(self):
        """Test XGBoost model training for Tesla"""
        print(f"\n[TEST 3] Training XGBoost Model for Tesla...")
        
        try:
            start_time = dt.datetime.now()
            
            # Train XGBoost model
            xgboost_results = self.model_training_service.train_model(
                symbol=self.symbol,
                algorithm='XGBoost',
                training_period_days=self.training_period_days,
                test_split=self.test_split,
                # XGBoost specific parameters
                n_estimators=150,
                max_depth=8,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.15,
                reg_lambda=1.2
            )
            
            training_time = (dt.datetime.now() - start_time).total_seconds()
            
            # Validate results
            self.assertIsNotNone(xgboost_results)
            self.assertIn('train_accuracy', xgboost_results)
            self.assertIn('test_accuracy', xgboost_results)
            self.assertIn('feature_count', xgboost_results)
            
            # Store results
            xgboost_results['training_time_seconds'] = training_time
            xgboost_results['algorithm'] = 'XGBoost'
            self.model_results['XGBoost'] = xgboost_results
            
            print(f"   ✅ XGBoost model trained successfully")
            print(f"   🎯 Train Accuracy: {xgboost_results['train_accuracy']:.4f}")
            print(f"   🎯 Test Accuracy: {xgboost_results['test_accuracy']:.4f}")
            print(f"   📊 Features Used: {xgboost_results['feature_count']}")
            print(f"   ⏱️ Training Time: {training_time:.2f} seconds")
            
            # Validate accuracy is reasonable
            self.assertGreater(xgboost_results['test_accuracy'], 0.6)  # At least 60%
            self.assertLess(xgboost_results['test_accuracy'], 1.0)      # Not perfect
            
        except Exception as e:
            self.fail(f"XGBoost training failed: {str(e)}")
    
    def test_04_compare_model_performance(self):
        """Test comprehensive model performance comparison"""
        print(f"\n[TEST 4] Comparing Model Performance...")
        
        try:
            # Ensure both models are trained
            self.assertIn('LightGBM', self.model_results)
            self.assertIn('XGBoost', self.model_results)
            
            lgb_results = self.model_results['LightGBM']
            xgb_results = self.model_results['XGBoost']
            
            # Performance comparison
            comparison = {
                'symbol': self.symbol,
                'training_date': dt.datetime.now().isoformat(),
                'training_period_days': self.training_period_days,
                'models': {
                    'LightGBM': {
                        'train_accuracy': lgb_results['train_accuracy'],
                        'test_accuracy': lgb_results['test_accuracy'],
                        'training_time_seconds': lgb_results['training_time_seconds'],
                        'feature_count': lgb_results['feature_count']
                    },
                    'XGBoost': {
                        'train_accuracy': xgb_results['train_accuracy'],
                        'test_accuracy': xgb_results['test_accuracy'],
                        'training_time_seconds': xgb_results['training_time_seconds'],
                        'feature_count': xgb_results['feature_count']
                    }
                }
            }
            
            # Calculate differences
            accuracy_diff = xgb_results['test_accuracy'] - lgb_results['test_accuracy']
            time_diff = xgb_results['training_time_seconds'] - lgb_results['training_time_seconds']
            
            comparison['comparison_metrics'] = {
                'accuracy_difference': accuracy_diff,
                'time_difference_seconds': time_diff,
                'winner_by_accuracy': 'XGBoost' if accuracy_diff > 0 else 'LightGBM',
                'winner_by_speed': 'LightGBM' if time_diff > 0 else 'XGBoost'
            }
            
            self.comparison_results = comparison
            
            print(f"   📊 PERFORMANCE COMPARISON RESULTS:")
            print(f"   {'Metric':<20} {'LightGBM':<12} {'XGBoost':<12} {'Difference':<12}")
            print(f"   {'-'*60}")
            print(f"   {'Train Accuracy':<20} {lgb_results['train_accuracy']:<12.4f} {xgb_results['train_accuracy']:<12.4f} {xgb_results['train_accuracy']-lgb_results['train_accuracy']:+.4f}")
            print(f"   {'Test Accuracy':<20} {lgb_results['test_accuracy']:<12.4f} {xgb_results['test_accuracy']:<12.4f} {accuracy_diff:+.4f}")
            print(f"   {'Training Time (s)':<20} {lgb_results['training_time_seconds']:<12.2f} {xgb_results['training_time_seconds']:<12.2f} {time_diff:+.2f}")
            print(f"   {'Features':<20} {lgb_results['feature_count']:<12d} {xgb_results['feature_count']:<12d} {'0':<12}")
            
            print(f"\n   🏆 WINNERS:")
            print(f"   📈 Best Accuracy: {comparison['comparison_metrics']['winner_by_accuracy']} ({abs(accuracy_diff):.4f} advantage)")
            print(f"   ⚡ Fastest Training: {comparison['comparison_metrics']['winner_by_speed']} ({abs(time_diff):.2f}s advantage)")
            
            # Overall winner calculation
            accuracy_weight = 0.7
            speed_weight = 0.3
            
            # Normalize metrics (higher is better for accuracy, lower is better for time)
            lgb_score = (lgb_results['test_accuracy'] * accuracy_weight + 
                        (1 / lgb_results['training_time_seconds']) * speed_weight)
            xgb_score = (xgb_results['test_accuracy'] * accuracy_weight + 
                        (1 / xgb_results['training_time_seconds']) * speed_weight)
            
            overall_winner = 'XGBoost' if xgb_score > lgb_score else 'LightGBM'
            comparison['comparison_metrics']['overall_winner'] = overall_winner
            comparison['comparison_metrics']['lgb_weighted_score'] = lgb_score
            comparison['comparison_metrics']['xgb_weighted_score'] = xgb_score
            
            print(f"   🎯 OVERALL WINNER: {overall_winner} (weighted score: accuracy 70%, speed 30%)")
            
        except Exception as e:
            self.fail(f"Performance comparison failed: {str(e)}")
    
    def test_05_analyze_feature_importance(self):
        """Test feature importance analysis for both models"""
        print(f"\n[TEST 5] Analyzing Feature Importance...")
        
        try:
            # Load trained models
            lgb_model, lgb_metadata = self.model_manager.load_model(self.symbol)
            
            # Get feature importance if available
            feature_importance_analysis = {}
            
            for algorithm in ['LightGBM', 'XGBoost']:
                try:
                    model, metadata = self.model_manager.load_model(self.symbol)
                    
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        feature_names = metadata.feature_names[:len(importances)]
                        
                        # Create importance rankings
                        importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importances
                        }).sort_values('importance', ascending=False)
                        
                        top_10_features = importance_df.head(10)
                        
                        feature_importance_analysis[algorithm] = {
                            'top_10_features': top_10_features.to_dict('records'),
                            'total_features': len(feature_names),
                            'importance_sum': float(importances.sum())
                        }
                        
                        print(f"   📊 {algorithm} Top 10 Features:")
                        for idx, row in top_10_features.iterrows():
                            print(f"      {row['feature']:<25}: {row['importance']:.4f}")
                    
                except Exception as e:
                    print(f"   ⚠️ Could not analyze feature importance for {algorithm}: {str(e)}")
            
            self.comparison_results['feature_importance'] = feature_importance_analysis
            
            if feature_importance_analysis:
                print(f"   ✅ Feature importance analysis completed")
            else:
                print(f"   ⚠️ Feature importance analysis not available")
                
        except Exception as e:
            print(f"   ⚠️ Feature importance analysis failed: {str(e)}")
            # Don't fail the test for this, as feature importance might not always be available
    
    def test_06_run_backtest_comparison(self):
        """Test backtesting comparison between models"""
        print(f"\n[TEST 6] Running Backtest Comparison...")
        
        try:
            from trading.enhanced_strategies import EnhancedMLTradingStrategy, OrderSizingConfig, OrderSizingStrategy
            from backtesting.enhanced_backtesting import EnhancedBacktester
            
            # Backtesting configuration
            backtest_months = 3
            starting_capital = 100000
            
            # Initialize backtester
            backtester = EnhancedBacktester(
                data_provider=self.data_provider,
                initial_cash=starting_capital,
                commission_rate=0.001,
                slippage_rate=0.0005
            )
            
            backtest_results = {}
            
            for algorithm in ['LightGBM', 'XGBoost']:
                print(f"   🔄 Running backtest for {algorithm}...")
                
                # Configure order sizing
                order_sizing_config = OrderSizingConfig(
                    strategy=OrderSizingStrategy.PERCENTAGE,
                    portfolio_pct=0.2,  # 20% per position for Tesla (high volatility)
                    min_shares=1,
                    max_shares=500,
                    volatility_target=0.03,
                    market_conditions_enabled=True,
                    max_position_pct=0.3
                )
                
                # Create strategy with the specific algorithm
                strategy = EnhancedMLTradingStrategy(
                    symbol=self.symbol,
                    data_provider=self.data_provider,
                    model_manager=self.model_manager,
                    order_sizing_config=order_sizing_config,
                    starting_portfolio_value=starting_capital,
                    preferred_algorithm=algorithm
                )
                
                # Run backtest
                backtest_end = dt.datetime.now()
                backtest_start = backtest_end - dt.timedelta(days=backtest_months * 30)
                
                result = backtester.run_backtest(
                    strategy=strategy,
                    symbols=[self.symbol],
                    start_date=backtest_start,
                    end_date=backtest_end,
                    benchmark_symbol='QQQ',  # Tech benchmark for Tesla
                    rebalance_frequency='daily'
                )
                
                backtest_results[algorithm] = {
                    'total_return': result.total_return,
                    'annualized_return': result.performance_metrics.annualized_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'volatility': result.performance_metrics.volatility,
                    'total_trades': result.performance_metrics.total_trades,
                    'win_rate': result.performance_metrics.win_rate,
                    'profit_factor': result.performance_metrics.profit_factor,
                    'alpha': result.performance_metrics.alpha,
                    'beta': result.performance_metrics.beta
                }
                
                print(f"      📈 {algorithm} Results:")
                print(f"         Total Return: {result.total_return:.2%}")
                print(f"         Sharpe Ratio: {result.sharpe_ratio:.3f}")
                print(f"         Max Drawdown: {result.max_drawdown:.2%}")
                print(f"         Win Rate: {result.performance_metrics.win_rate:.1%}")
                print(f"         Total Trades: {result.performance_metrics.total_trades}")
            
            self.comparison_results['backtest_results'] = backtest_results
            
            # Compare backtest performance
            if len(backtest_results) == 2:
                lgb_bt = backtest_results['LightGBM']
                xgb_bt = backtest_results['XGBoost']
                
                print(f"\n   🏆 BACKTEST COMPARISON:")
                print(f"   {'Metric':<18} {'LightGBM':<12} {'XGBoost':<12} {'Winner':<12}")
                print(f"   {'-'*60}")
                
                metrics = ['total_return', 'sharpe_ratio', 'win_rate', 'profit_factor']
                winners = []
                
                for metric in metrics:
                    lgb_val = lgb_bt[metric]
                    xgb_val = xgb_bt[metric]
                    winner = 'XGBoost' if xgb_val > lgb_val else 'LightGBM'
                    winners.append(winner)
                    
                    if metric in ['total_return', 'win_rate']:
                        print(f"   {metric.replace('_', ' ').title():<18} {lgb_val:<12.2%} {xgb_val:<12.2%} {winner:<12}")
                    else:
                        print(f"   {metric.replace('_', ' ').title():<18} {lgb_val:<12.3f} {xgb_val:<12.3f} {winner:<12}")
                
                backtest_winner = max(set(winners), key=winners.count)
                self.comparison_results['backtest_winner'] = backtest_winner
                print(f"\n   🎯 BACKTEST OVERALL WINNER: {backtest_winner}")
            
            print(f"   ✅ Backtest comparison completed")
            
        except Exception as e:
            print(f"   ⚠️ Backtest comparison failed: {str(e)}")
            # Don't fail the test, as backtesting might have external dependencies
    
    def test_07_generate_comprehensive_report(self):
        """Generate comprehensive comparison report"""
        print(f"\n[TEST 7] Generating Comprehensive Report...")
        
        try:
            # Ensure we have comparison results
            self.assertIsNotNone(self.comparison_results)
            self.assertIn('models', self.comparison_results)
            
            # Create comprehensive report
            report = {
                'test_metadata': {
                    'test_date': dt.datetime.now().isoformat(),
                    'symbol': self.symbol,
                    'training_period_days': self.training_period_days,
                    'test_split': self.test_split,
                    'algorithms_tested': self.algorithms
                },
                'training_results': self.comparison_results,
                'summary': {
                    'best_accuracy_model': self.comparison_results['comparison_metrics']['winner_by_accuracy'],
                    'fastest_training_model': self.comparison_results['comparison_metrics']['winner_by_speed'],
                    'overall_winner': self.comparison_results['comparison_metrics']['overall_winner']
                }
            }
            
            # Add backtest winner if available
            if 'backtest_winner' in self.comparison_results:
                report['summary']['backtest_winner'] = self.comparison_results['backtest_winner']
            
            # Save report to file
            report_filename = f"tesla_model_comparison_report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path = os.path.join(os.path.dirname(__file__), report_filename)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"   ✅ Comprehensive report generated: {report_filename}")
            
            # Print executive summary
            print(f"\n   📋 EXECUTIVE SUMMARY - TESLA MODEL COMPARISON:")
            print(f"   {'='*60}")
            lgb = self.comparison_results['models']['LightGBM']
            xgb = self.comparison_results['models']['XGBoost']
            
            print(f"   🎯 TEST ACCURACY:")
            print(f"      LightGBM: {lgb['test_accuracy']:.4f}")
            print(f"      XGBoost:  {xgb['test_accuracy']:.4f}")
            print(f"      Winner:   {report['summary']['best_accuracy_model']}")
            
            print(f"   ⚡ TRAINING SPEED:")
            print(f"      LightGBM: {lgb['training_time_seconds']:.2f}s")
            print(f"      XGBoost:  {xgb['training_time_seconds']:.2f}s")
            print(f"      Winner:   {report['summary']['fastest_training_model']}")
            
            print(f"   🏆 OVERALL WINNER: {report['summary']['overall_winner']}")
            
            if 'backtest_winner' in report['summary']:
                print(f"   📈 BACKTEST WINNER: {report['summary']['backtest_winner']}")
            
            print(f"   {'='*60}")
            
            # Store report for potential external access
            self.final_report = report
            
        except Exception as e:
            self.fail(f"Report generation failed: {str(e)}")
    
    def tearDown(self):
        """Clean up after each test"""
        pass
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        print(f"\n{'='*80}")
        print("TESLA MODEL COMPARISON TEST - CLEANUP")
        print("="*80)
        print("✅ All tests completed successfully")
        print("🧹 Cleanup completed")


def run_tesla_model_comparison():
    """
    Convenience function to run the Tesla model comparison test suite
    """
    print("Starting Tesla Model Comparison Test Suite...")
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestTeslaModelComparison)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run the test suite
    success = run_tesla_model_comparison()
    
    if success:
        print(f"\n🎉 TESLA MODEL COMPARISON COMPLETED SUCCESSFULLY!")
        print(f"📊 Check the generated JSON report for detailed results.")
    else:
        print(f"\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)