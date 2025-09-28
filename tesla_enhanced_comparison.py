#!/usr/bin/env python3
"""
Tesla Model Comparison: Enhanced Version with Regularization
Addresses overfitting issues and provides more robust comparison
"""

import sys
import os
import datetime as dt
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

warnings.filterwarnings('ignore')

def run_enhanced_tesla_comparison():
    """Run enhanced Tesla comparison with better regularization and cross-validation"""
    
    print("=" * 80)
    print("🚀 TESLA MODEL COMPARISON: Enhanced with Regularization")
    print("=" * 80)
    print(f"📅 Date: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Import required packages
        import lightgbm as lgb
        import xgboost as xgb
        import yfinance as yf
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        print(f"✅ Packages: LightGBM v{lgb.__version__}, XGBoost v{xgb.__version__}")
        
        # Enhanced configuration
        symbol = 'TSLA'
        print(f"\n[CONFIG] Enhanced Test Configuration:")
        print(f"         Symbol: {symbol}")
        print(f"         Cross-validation: 5-fold Time Series Split")
        print(f"         Regularization: Enhanced to prevent overfitting")
        
        # Fetch Tesla data
        print(f"\n[DATA] Fetching Tesla Data...")
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=1200)  # More historical data
        
        tesla = yf.Ticker(symbol)
        data = tesla.history(start=start_date, end=end_date)
        
        print(f"         ✅ Tesla data fetched: {len(data)} days")
        print(f"         📊 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Create enhanced features
        print(f"\n[FEATURES] Creating Enhanced Features...")
        features_df = create_enhanced_features(data)
        
        print(f"         ✅ Enhanced features created: {features_df.shape[1]-1} features, {len(features_df)} samples")
        
        # Prepare data
        X = features_df.drop(['target'], axis=1).values
        y = features_df['target'].values
        
        print(f"         📊 Feature matrix: {X.shape}")
        print(f"         🎯 Target distribution: {np.sum(y==1)} positive, {np.sum(y==0)} negative")
        
        # Enhanced model configurations with better regularization
        models_config = {
            'LightGBM_Regular': {
                'model': lgb.LGBMClassifier(
                    n_estimators=100,      # Reduced from 150
                    max_depth=5,           # Reduced from 8
                    learning_rate=0.05,    # Reduced from 0.08
                    num_leaves=20,         # Reduced from 50
                    subsample=0.8,         # More regularization
                    colsample_bytree=0.8,  # More regularization
                    reg_alpha=0.3,         # L1 regularization
                    reg_lambda=1.5,        # L2 regularization
                    min_child_samples=50,  # Prevent overfitting
                    random_state=42,
                    verbose=-1
                ),
                'description': 'LightGBM with Enhanced Regularization'
            },
            'XGBoost_Regular': {
                'model': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.3,
                    reg_lambda=1.5,
                    min_child_weight=10,   # Prevent overfitting
                    gamma=1.0,             # Minimum loss reduction
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'description': 'XGBoost with Enhanced Regularization'
            },
            'LightGBM_Aggressive': {
                'model': lgb.LGBMClassifier(
                    n_estimators=50,       # Very conservative
                    max_depth=3,           # Very shallow
                    learning_rate=0.03,    # Very low learning rate
                    num_leaves=10,         # Very few leaves
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    min_child_samples=100,
                    random_state=42,
                    verbose=-1
                ),
                'description': 'LightGBM with Aggressive Regularization'
            },
            'XGBoost_Aggressive': {
                'model': xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    min_child_weight=20,
                    gamma=2.0,
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'description': 'XGBoost with Aggressive Regularization'
            }
        }
        
        results = {}
        
        # Train and evaluate all models
        print(f"\n[TRAINING] Training Models with Cross-Validation...")
        
        # Time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for model_name, config in models_config.items():
            print(f"\n   🔄 Training {model_name}...")
            
            start_time = dt.datetime.now()
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                config['model'], X, y, 
                cv=tscv, 
                scoring='accuracy',
                n_jobs=-1
            )
            
            # Train on full dataset
            config['model'].fit(X, y)
            
            # Calculate training accuracy
            train_predictions = config['model'].predict(X)
            train_accuracy = accuracy_score(y, train_predictions)
            
            training_time = (dt.datetime.now() - start_time).total_seconds()
            
            results[model_name] = {
                'model': config['model'],
                'description': config['description'],
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_accuracy': train_accuracy,
                'training_time_seconds': training_time,
                'overfitting_score': train_accuracy - cv_scores.mean()  # Measure of overfitting
            }
            
            print(f"      ✅ {model_name} completed")
            print(f"      🎯 Cross-Val Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"      📊 Train Accuracy: {train_accuracy:.4f}")
            print(f"      ⚠️ Overfitting Score: {train_accuracy - cv_scores.mean():.4f}")
            print(f"      ⏱️ Training Time: {training_time:.2f}s")
        
        # Comprehensive comparison
        print(f"\n[COMPARISON] Enhanced Model Performance Analysis...")
        generate_enhanced_comparison_report(results, symbol)
        
        # Feature importance analysis
        print(f"\n[FEATURES] Feature Importance Analysis...")
        analyze_feature_importance(results, features_df.drop(['target'], axis=1).columns)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced Tesla comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_enhanced_features(data):
    """Create enhanced feature set for better trading signals"""
    
    df = data.copy()
    
    # Basic price features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Multiple timeframe moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['Close'].rolling(period).mean()
        df[f'close_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
    
    # Moving average crossover signals
    df['golden_cross'] = ((df['sma_20'] > df['sma_50']) & 
                         (df['sma_20'].shift(1) <= df['sma_50'].shift(1))).astype(int)
    df['death_cross'] = ((df['sma_20'] < df['sma_50']) & 
                        (df['sma_20'].shift(1) >= df['sma_50'].shift(1))).astype(int)
    
    # Volatility features
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        df[f'volatility_{period}_norm'] = df[f'volatility_{period}'] / df[f'volatility_{period}'].rolling(50).mean()
    
    # Volume analysis
    df['volume_sma_10'] = df['Volume'].rolling(10).mean()
    df['volume_sma_20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
    df['volume_breakout'] = (df['volume_ratio'] > 2.0).astype(int)
    
    # Price position and momentum
    df['high_low_ratio'] = df['High'] / df['Low']
    df['close_high_ratio'] = df['Close'] / df['High']
    df['close_low_ratio'] = df['Close'] / df['Low']
    
    # Multiple timeframe momentum
    for period in [3, 5, 10, 20]:
        df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period)
        df[f'momentum_{period}_above_1'] = (df[f'momentum_{period}'] > 1.05).astype(int)
    
    # Technical indicators
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    
    # Bollinger Bands
    bb_period = 20
    df['bb_middle'] = df['Close'].rolling(bb_period).mean()
    bb_std = df['Close'].rolling(bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
    df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_squeeze'] = (bb_std / df['bb_middle'] < 0.1).astype(int)
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['macd_bullish'] = ((df['macd'] > df['macd_signal']) & 
                         (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    
    # Enhanced target with forward-looking multiple periods
    df['next_1_return'] = df['Close'].pct_change().shift(-1)
    df['next_2_return'] = df['Close'].pct_change(2).shift(-2)
    df['next_3_return'] = df['Close'].pct_change(3).shift(-3)
    
    # Multi-horizon target (profitable over next 1-3 days)
    df['target'] = ((df['next_1_return'] > 0.01) | 
                   (df['next_2_return'] > 0.02) | 
                   (df['next_3_return'] > 0.03)).astype(int)
    
    # Select relevant features (remove intermediate calculations)
    feature_columns = [
        'returns', 'log_returns',
        'close_sma_5_ratio', 'close_sma_10_ratio', 'close_sma_20_ratio', 'close_sma_50_ratio',
        'golden_cross', 'death_cross',
        'volatility_5', 'volatility_10', 'volatility_20',
        'volatility_5_norm', 'volatility_10_norm', 'volatility_20_norm',
        'volume_ratio', 'volume_breakout',
        'high_low_ratio', 'close_high_ratio', 'close_low_ratio',
        'momentum_3', 'momentum_5', 'momentum_10', 'momentum_20',
        'momentum_3_above_1', 'momentum_5_above_1', 'momentum_10_above_1', 'momentum_20_above_1',
        'rsi', 'rsi_oversold', 'rsi_overbought',
        'bb_position', 'bb_squeeze',
        'macd', 'macd_histogram', 'macd_bullish',
        'target'
    ]
    
    enhanced_features = df[feature_columns].dropna()
    
    return enhanced_features

def generate_enhanced_comparison_report(results, symbol):
    """Generate comprehensive comparison report with overfitting analysis"""
    
    print(f"   📊 TESLA ENHANCED MODEL COMPARISON")
    print(f"   {'='*80}")
    print(f"   {'Model':<20} {'CV Score':<12} {'CV Std':<10} {'Train Acc':<10} {'Overfit':<8} {'Time(s)':<8}")
    print(f"   {'-'*80}")
    
    # Sort by cross-validation score (most important metric)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
    
    for model_name, result in sorted_results:
        print(f"   {model_name:<20} {result['cv_mean']:<12.4f} {result['cv_std']:<10.4f} "
              f"{result['train_accuracy']:<10.4f} {result['overfitting_score']:<8.4f} {result['training_time_seconds']:<8.2f}")
    
    print(f"   {'-'*80}")
    
    # Analysis
    best_cv = sorted_results[0]
    best_generalization = min(results.items(), key=lambda x: x[1]['overfitting_score'])
    fastest = min(results.items(), key=lambda x: x[1]['training_time_seconds'])
    
    print(f"\n   🏆 WINNERS:")
    print(f"   🎯 Best CV Score:      {best_cv[0]} ({best_cv[1]['cv_mean']:.4f})")
    print(f"   🧠 Best Generalization: {best_generalization[0]} (overfit: {best_generalization[1]['overfitting_score']:.4f})")
    print(f"   ⚡ Fastest Training:    {fastest[0]} ({fastest[1]['training_time_seconds']:.2f}s)")
    
    # Recommendation logic
    print(f"\n   💡 RECOMMENDATIONS:")
    
    # Find model with best balance of CV score and low overfitting
    balanced_score = {}
    for name, result in results.items():
        # Penalize overfitting heavily
        balanced_score[name] = result['cv_mean'] - (2 * result['overfitting_score'])
    
    best_balanced = max(balanced_score.items(), key=lambda x: x[1])
    
    print(f"   🎯 BEST OVERALL: {best_balanced[0]}")
    print(f"      - CV Score: {results[best_balanced[0]]['cv_mean']:.4f}")
    print(f"      - Overfitting: {results[best_balanced[0]]['overfitting_score']:.4f}")
    print(f"      - Training Time: {results[best_balanced[0]]['training_time_seconds']:.2f}s")
    
    # Compare regular vs aggressive regularization
    print(f"\n   📈 REGULARIZATION ANALYSIS:")
    lgb_regular = results.get('LightGBM_Regular', {})
    lgb_aggressive = results.get('LightGBM_Aggressive', {})
    xgb_regular = results.get('XGBoost_Regular', {})
    xgb_aggressive = results.get('XGBoost_Aggressive', {})
    
    if lgb_regular and lgb_aggressive:
        lgb_improvement = lgb_aggressive['cv_mean'] - lgb_regular['cv_mean']
        lgb_overfit_reduction = lgb_regular['overfitting_score'] - lgb_aggressive['overfitting_score']
        print(f"      LightGBM Aggressive vs Regular:")
        print(f"         CV Score Change: {lgb_improvement:+.4f}")
        print(f"         Overfitting Reduction: {lgb_overfit_reduction:+.4f}")
    
    if xgb_regular and xgb_aggressive:
        xgb_improvement = xgb_aggressive['cv_mean'] - xgb_regular['cv_mean']
        xgb_overfit_reduction = xgb_regular['overfitting_score'] - xgb_aggressive['overfitting_score']
        print(f"      XGBoost Aggressive vs Regular:")
        print(f"         CV Score Change: {xgb_improvement:+.4f}")
        print(f"         Overfitting Reduction: {xgb_overfit_reduction:+.4f}")
    
    print(f"   {'='*80}")

def analyze_feature_importance(results, feature_names):
    """Analyze feature importance across models"""
    
    print(f"   📊 FEATURE IMPORTANCE ANALYSIS")
    print(f"   {'-'*60}")
    
    # Get feature importances from tree-based models
    for model_name, result in results.items():
        model = result['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\n   {model_name} - Top 10 Features:")
            for idx, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"      {idx+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")

if __name__ == "__main__":
    print("Starting Enhanced Tesla Model Comparison...")
    success = run_enhanced_tesla_comparison()
    
    if success:
        print(f"\n🎉 Enhanced Tesla model comparison completed successfully!")
        print(f"📊 Key improvements:")
        print(f"   ✅ Better regularization to prevent overfitting")
        print(f"   ✅ Cross-validation for robust performance measurement")
        print(f"   ✅ Enhanced feature engineering")
        print(f"   ✅ Multiple model configurations tested")
    else:
        print(f"\n❌ Enhanced Tesla model comparison failed.")
    
    print(f"\nAnalysis completed at: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")