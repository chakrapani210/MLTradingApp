#!/usr/bin/env python3
"""
Direct Tesla Model Comparison: LightGBM vs XGBoost
Direct implementation without complex imports - focuses on training and comparison
"""

import sys
import os
import datetime as dt
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

warnings.filterwarnings('ignore')

def run_tesla_comparison():
    """Run direct Tesla model comparison"""
    
    print("=" * 80)
    print("🚀 TESLA MODEL COMPARISON: LightGBM vs XGBoost")
    print("=" * 80)
    print(f"📅 Date: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    symbol = 'TSLA'
    training_period_days = 730
    test_split = 0.2
    
    print(f"\n[CONFIG] Test Configuration:")
    print(f"         Symbol: {symbol}")
    print(f"         Training Period: {training_period_days} days")
    print(f"         Test Split: {test_split * 100}%")
    
    results = {}
    
    try:
        # Import components with error handling
        print(f"\n[STEP 1] Importing Enhanced Trading Components...")
        
        try:
            from data.providers import YFinanceProvider
            from models.enhanced_model_management import EnhancedModelManager, ModelTrainingService
            from analysis.enhanced_market_analysis import MarketContextAnalyzer, EnhancedFeatureEngineer
            
            print(f"         ✅ All components imported successfully")
        except ImportError as e:
            print(f"         ❌ Import error: {e}")
            print(f"         💡 Trying alternative approach...")
            
            # Alternative direct imports
            sys.path.append(os.path.join(src_path, 'data'))
            sys.path.append(os.path.join(src_path, 'models'))
            sys.path.append(os.path.join(src_path, 'analysis'))
            
            # Try simpler approach with direct Python execution
            return run_simplified_comparison()
        
        # Initialize components
        print(f"\n[STEP 2] Initializing Components...")
        data_provider = YFinanceProvider()
        model_manager = EnhancedModelManager(base_path="models/tesla_comparison")
        model_training_service = ModelTrainingService(model_manager, data_provider)
        market_analyzer = MarketContextAnalyzer(data_provider)
        feature_engineer = EnhancedFeatureEngineer(market_analyzer)
        
        print(f"         ✅ Components initialized")
        
        # Test both algorithms
        algorithms = ['LightGBM', 'XGBoost']
        
        for algorithm in algorithms:
            print(f"\n[STEP 3.{algorithms.index(algorithm) + 1}] Training {algorithm} Model...")
            
            try:
                start_time = dt.datetime.now()
                
                # Train model with optimized parameters
                training_results = model_training_service.train_model(
                    symbol=symbol,
                    algorithm=algorithm,
                    training_period_days=training_period_days,
                    test_split=test_split,
                    # Optimized parameters for both algorithms
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.08,
                    subsample=0.85,
                    colsample_bytree=0.85
                )
                
                training_time = (dt.datetime.now() - start_time).total_seconds()
                
                # Store results
                results[algorithm] = {
                    'train_accuracy': training_results['train_accuracy'],
                    'test_accuracy': training_results['test_accuracy'],
                    'feature_count': training_results['feature_count'],
                    'training_time_seconds': training_time,
                    'algorithm': algorithm
                }
                
                print(f"         ✅ {algorithm} Training Completed")
                print(f"         🎯 Train Accuracy: {training_results['train_accuracy']:.4f}")
                print(f"         🎯 Test Accuracy: {training_results['test_accuracy']:.4f}")
                print(f"         📊 Features: {training_results['feature_count']}")
                print(f"         ⏱️ Training Time: {training_time:.2f}s")
                
            except Exception as e:
                print(f"         ❌ {algorithm} training failed: {str(e)}")
                results[algorithm] = {'error': str(e)}
        
        # Generate comparison if both models trained successfully
        if len([r for r in results.values() if 'error' not in r]) == 2:
            print(f"\n[STEP 4] Performance Comparison...")
            generate_comparison_report(results, symbol)
        else:
            print(f"\n❌ Cannot compare models - some training failed")
            
    except Exception as e:
        print(f"\n❌ Tesla comparison failed: {str(e)}")
        return False
    
    return True

def run_simplified_comparison():
    """Simplified comparison using direct algorithm implementation"""
    
    print(f"\n🔄 Running Simplified Tesla Comparison...")
    
    try:
        # Check if packages are available
        try:
            import lightgbm as lgb
            import xgboost as xgb
            print(f"         ✅ LightGBM v{lgb.__version__} available")
            print(f"         ✅ XGBoost v{xgb.__version__} available")
        except ImportError as e:
            print(f"         ❌ Required packages not available: {e}")
            return False
        
        # Get Tesla data directly
        print(f"\n[DATA] Fetching Tesla Data...")
        try:
            import yfinance as yf
            
            # Fetch Tesla data
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=1000)  # More data for better features
            
            tesla = yf.Ticker('TSLA')
            data = tesla.history(start=start_date, end=end_date)
            
            if len(data) < 100:
                print(f"         ❌ Insufficient data: {len(data)} days")
                return False
            
            print(f"         ✅ Tesla data fetched: {len(data)} days")
            
        except Exception as e:
            print(f"         ❌ Data fetch failed: {e}")
            return False
        
        # Create simple features
        print(f"\n[FEATURES] Creating Features...")
        features_df = create_simple_features(data)
        
        if len(features_df) < 50:
            print(f"         ❌ Insufficient feature data: {len(features_df)} samples")
            return False
            
        print(f"         ✅ Features created: {features_df.shape[1]} features, {len(features_df)} samples")
        
        # Prepare data for ML
        X = features_df.drop(['target'], axis=1).values
        y = features_df['target'].values
        
        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"         📊 Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Test both algorithms
        results = {}
        
        # LightGBM
        print(f"\n[LIGHTGBM] Training LightGBM...")
        start_time = dt.datetime.now()
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.08,
            num_leaves=50,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbose=-1
        )
        
        lgb_model.fit(X_train, y_train)
        lgb_train_acc = lgb_model.score(X_train, y_train)
        lgb_test_acc = lgb_model.score(X_test, y_test)
        lgb_time = (dt.datetime.now() - start_time).total_seconds()
        
        results['LightGBM'] = {
            'train_accuracy': lgb_train_acc,
            'test_accuracy': lgb_test_acc,
            'training_time_seconds': lgb_time
        }
        
        print(f"            ✅ LightGBM completed")
        print(f"            🎯 Train Accuracy: {lgb_train_acc:.4f}")
        print(f"            🎯 Test Accuracy: {lgb_test_acc:.4f}")
        print(f"            ⏱️ Training Time: {lgb_time:.2f}s")
        
        # XGBoost
        print(f"\n[XGBOOST] Training XGBoost...")
        start_time = dt.datetime.now()
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        xgb_model.fit(X_train, y_train)
        xgb_train_acc = xgb_model.score(X_train, y_train)
        xgb_test_acc = xgb_model.score(X_test, y_test)
        xgb_time = (dt.datetime.now() - start_time).total_seconds()
        
        results['XGBoost'] = {
            'train_accuracy': xgb_train_acc,
            'test_accuracy': xgb_test_acc,
            'training_time_seconds': xgb_time
        }
        
        print(f"            ✅ XGBoost completed")
        print(f"            🎯 Train Accuracy: {xgb_train_acc:.4f}")
        print(f"            🎯 Test Accuracy: {xgb_test_acc:.4f}")
        print(f"            ⏱️ Training Time: {xgb_time:.2f}s")
        
        # Generate comparison
        print(f"\n[COMPARISON] Model Performance Analysis...")
        generate_comparison_report(results, 'TSLA')
        
        return True
        
    except Exception as e:
        print(f"         ❌ Simplified comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_features(data):
    """Create simple but effective trading features"""
    
    df = data.copy()
    
    # Price-based features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_10'] = df['Close'].rolling(10).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    
    # Moving average signals
    df['ma_signal_short'] = (df['sma_5'] > df['sma_10']).astype(int)
    df['ma_signal_long'] = (df['sma_20'] > df['sma_50']).astype(int)
    
    # Volatility
    df['volatility_10'] = df['returns'].rolling(10).std()
    df['volatility_20'] = df['returns'].rolling(20).std()
    
    # Volume features
    df['volume_sma_10'] = df['Volume'].rolling(10).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_10']
    
    # Price position features
    df['high_low_ratio'] = df['High'] / df['Low']
    df['close_high_ratio'] = df['Close'] / df['High']
    df['close_low_ratio'] = df['Close'] / df['Low']
    
    # Momentum features
    df['momentum_5'] = df['Close'] / df['Close'].shift(5)
    df['momentum_10'] = df['Close'] / df['Close'].shift(10)
    
    # RSI (simplified)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Target: next day positive return
    df['target'] = (df['returns'].shift(-1) > 0).astype(int)
    
    # Select features
    feature_columns = [
        'returns', 'log_returns', 'ma_signal_short', 'ma_signal_long',
        'volatility_10', 'volatility_20', 'volume_ratio', 
        'high_low_ratio', 'close_high_ratio', 'close_low_ratio',
        'momentum_5', 'momentum_10', 'rsi', 'target'
    ]
    
    features_df = df[feature_columns].dropna()
    
    return features_df

def generate_comparison_report(results, symbol):
    """Generate comprehensive comparison report"""
    
    if len(results) < 2:
        print("❌ Need at least 2 models to compare")
        return
    
    lgb_results = results.get('LightGBM', {})
    xgb_results = results.get('XGBoost', {})
    
    if 'error' in lgb_results or 'error' in xgb_results:
        print("❌ Cannot compare - one or more models failed")
        return
    
    print(f"   📊 TESLA MODEL COMPARISON RESULTS")
    print(f"   {'='*60}")
    print(f"   {'Metric':<20} {'LightGBM':<12} {'XGBoost':<12} {'Winner':<12}")
    print(f"   {'-'*60}")
    
    # Compare metrics
    metrics = ['train_accuracy', 'test_accuracy', 'training_time_seconds']
    winners = []
    
    for metric in metrics:
        if metric in lgb_results and metric in xgb_results:
            lgb_val = lgb_results[metric]
            xgb_val = xgb_results[metric]
            
            if metric == 'training_time_seconds':
                # For time, lower is better
                winner = 'LightGBM' if lgb_val < xgb_val else 'XGBoost'
                print(f"   {metric.replace('_', ' ').title():<20} {lgb_val:<12.3f} {xgb_val:<12.3f} {winner:<12}")
            else:
                # For accuracy, higher is better
                winner = 'LightGBM' if lgb_val > xgb_val else 'XGBoost'
                print(f"   {metric.replace('_', ' ').title():<20} {lgb_val:<12.4f} {xgb_val:<12.4f} {winner:<12}")
                
            winners.append(winner)
    
    # Overall winner
    if winners:
        overall_winner = max(set(winners), key=winners.count)
        print(f"   {'-'*60}")
        print(f"   {'OVERALL WINNER':<20} {'':<12} {'':<12} {overall_winner:<12}")
    
    # Performance improvement
    if 'test_accuracy' in lgb_results and 'test_accuracy' in xgb_results:
        accuracy_diff = abs(xgb_results['test_accuracy'] - lgb_results['test_accuracy'])
        improvement_pct = accuracy_diff * 100
        print(f"\n   📈 ACCURACY DIFFERENCE: {improvement_pct:.2f} percentage points")
    
    # Speed comparison
    if 'training_time_seconds' in lgb_results and 'training_time_seconds' in xgb_results:
        time_diff = abs(xgb_results['training_time_seconds'] - lgb_results['training_time_seconds'])
        faster_model = 'LightGBM' if lgb_results['training_time_seconds'] < xgb_results['training_time_seconds'] else 'XGBoost'
        print(f"   ⚡ SPEED ADVANTAGE: {faster_model} is {time_diff:.2f}s faster")
    
    print(f"\n   🎯 RECOMMENDATION FOR TESLA TRADING:")
    if winners.count('LightGBM') > winners.count('XGBoost'):
        print(f"      🏆 Use LightGBM for Tesla trading")
        print(f"      ✅ Better overall performance balance")
    else:
        print(f"      🏆 Use XGBoost for Tesla trading")
        print(f"      ✅ Superior accuracy performance")
    
    print(f"   {'='*60}")

if __name__ == "__main__":
    print("Starting Tesla Model Comparison...")
    success = run_tesla_comparison()
    
    if success:
        print(f"\n🎉 Tesla model comparison completed successfully!")
    else:
        print(f"\n❌ Tesla model comparison failed.")
    
    print(f"\nComparison completed at: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")