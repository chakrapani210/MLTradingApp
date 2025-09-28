#!/usr/bin/env python3
"""
Apple Comprehensive Backtesting System
Implements sophisticated backtesting with LightGBM:
- Train on first 6 months of 1-year period
- Test on next 6 months
- Retrain every 2 weeks during second 6 months
"""

import sys
import os
import datetime as dt
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional
import logging

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

warnings.filterwarnings('ignore')

class AppleBacktestingSystem:
    """
    Comprehensive Apple backtesting system with adaptive retraining
    """
    
    def __init__(self):
        self.symbol = 'AAPL'
        self.results = {}
        self.models = {}
        self.predictions = {}
        
        # Backtesting configuration
        self.config = {
            'initial_training_months': 6,
            'testing_months': 6,
            'retrain_frequency_days': 14,  # 2 weeks
            'min_training_days': 120,      # Minimum data for training
            'feature_window': 50,          # Days needed for feature calculation
            'prediction_horizon': 3        # Multi-day prediction horizon
        }
        
        # LightGBM configuration (optimized from Tesla results)
        self.model_configs = {
            'aggressive': {
                'n_estimators': 50,
                'max_depth': 3,
                'learning_rate': 0.03,
                'num_leaves': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 2.0,
                'min_child_samples': 100,
                'random_state': 42,
                'verbose': -1
            },
            'moderate': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.05,
                'num_leaves': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.3,
                'reg_lambda': 1.5,
                'min_child_samples': 50,
                'random_state': 42,
                'verbose': -1
            }
        }
        
        # Trading configuration
        self.trading_config = {
            'transaction_cost': 0.001,     # 0.1% transaction cost
            'position_size': 0.10,         # 10% of portfolio per trade
            'stop_loss': 0.03,             # 3% stop loss
            'take_profit': 0.06,           # 6% take profit
            'min_confidence': 0.55,        # Minimum prediction confidence
            'holding_period_days': 5       # Maximum holding period
        }
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for backtesting operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('apple_backtesting.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AppleBacktest')
    
    def create_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set optimized for Apple trading
        Based on Tesla winning features but adapted for Apple's characteristics
        """
        
        df = data.copy()
        
        # Basic price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Multiple timeframe moving averages (Apple responds well to trend following)
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'close_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
            df[f'sma_{period}_slope'] = df[f'sma_{period}'].diff(5) / df[f'sma_{period}']
        
        # Moving average crossover signals (important for Apple)
        df['golden_cross'] = ((df['sma_20'] > df['sma_50']) & 
                             (df['sma_20'].shift(1) <= df['sma_50'].shift(1))).astype(int)
        df['death_cross'] = ((df['sma_20'] < df['sma_50']) & 
                            (df['sma_20'].shift(1) >= df['sma_50'].shift(1))).astype(int)
        
        # Apple-specific trend strength
        df['trend_strength'] = (df['sma_5'] - df['sma_20']) / df['sma_20']
        df['trend_acceleration'] = df['trend_strength'].diff(3)
        
        # Volatility features (Apple has different volatility patterns than Tesla)
        for period in [5, 10, 20, 30]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'volatility_{period}_norm'] = df[f'volatility_{period}'] / df[f'volatility_{period}'].rolling(100).mean()
        
        # Volatility breakouts and contractions
        df['volatility_expansion'] = (df['volatility_5'] > df['volatility_20'] * 1.5).astype(int)
        df['volatility_contraction'] = (df['volatility_5'] < df['volatility_20'] * 0.5).astype(int)
        
        # Volume analysis (Apple has distinct volume patterns)
        for period in [10, 20, 50]:
            df[f'volume_sma_{period}'] = df['Volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['Volume'] / df[f'volume_sma_{period}']
        
        df['volume_breakout'] = (df['volume_ratio_20'] > 2.0).astype(int)
        df['volume_surge'] = (df['volume_ratio_10'] > df['volume_ratio_20'] * 1.5).astype(int)
        
        # Price position features
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_high_ratio'] = df['Close'] / df['High']
        df['close_low_ratio'] = df['Close'] / df['Low']
        
        # Intraday strength
        df['intraday_strength'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-10)
        df['gap_strength'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Multi-timeframe momentum (Apple shows strong momentum patterns)
        for period in [3, 5, 10, 20, 50]:
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            df[f'momentum_{period}_strength'] = (df[f'momentum_{period}'] > 0.02).astype(int)
        
        # Rate of change indicators
        df['roc_5'] = df['Close'].pct_change(5)
        df['roc_10'] = df['Close'].pct_change(10)
        df['roc_acceleration'] = df['roc_5'] - df['roc_10']
        
        # Technical indicators optimized for Apple
        # RSI with multiple timeframes
        for period in [14, 21]:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
            df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
        
        # Bollinger Bands (effective for Apple's mean reversion)
        for period in [20, 50]:
            bb_middle = df['Close'].rolling(period).mean()
            bb_std = df['Close'].rolling(period).std()
            df[f'bb_upper_{period}'] = bb_middle + (2 * bb_std)
            df[f'bb_lower_{period}'] = bb_middle - (2 * bb_std)
            df[f'bb_position_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
            df[f'bb_squeeze_{period}'] = (bb_std / bb_middle < 0.1).astype(int)
            df[f'bb_breakout_{period}'] = ((df['Close'] > df[f'bb_upper_{period}']) | 
                                          (df['Close'] < df[f'bb_lower_{period}'])).astype(int)
        
        # MACD with signal analysis
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_bullish'] = ((df['macd'] > df['macd_signal']) & 
                             (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_bearish'] = ((df['macd'] < df['macd_signal']) & 
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        
        # Apple-specific seasonality (tech earnings, iPhone releases, etc.)
        df['month'] = df.index.month
        df['quarter_end'] = df['month'].isin([3, 6, 9, 12]).astype(int)
        df['september_effect'] = (df['month'] == 9).astype(int)  # iPhone launch season
        df['january_effect'] = (df['month'] == 1).astype(int)    # New year optimism
        
        # Market microstructure
        df['price_level'] = pd.qcut(df['Close'], q=5, labels=False, duplicates='drop')
        df['volume_level'] = pd.qcut(df['Volume'], q=5, labels=False, duplicates='drop')
        
        # Select feature columns (optimized set based on Apple characteristics)
        feature_columns = [
            # Returns and momentum
            'returns', 'log_returns',
            'momentum_3', 'momentum_5', 'momentum_10', 'momentum_20',
            'momentum_3_strength', 'momentum_5_strength', 'momentum_10_strength',
            'roc_5', 'roc_10', 'roc_acceleration',
            
            # Moving averages and trends
            'close_sma_5_ratio', 'close_sma_10_ratio', 'close_sma_20_ratio', 'close_sma_50_ratio',
            'sma_5_slope', 'sma_10_slope', 'sma_20_slope',
            'golden_cross', 'death_cross',
            'trend_strength', 'trend_acceleration',
            
            # Volatility
            'volatility_5', 'volatility_10', 'volatility_20',
            'volatility_5_norm', 'volatility_10_norm', 'volatility_20_norm',
            'volatility_expansion', 'volatility_contraction',
            
            # Volume
            'volume_ratio_10', 'volume_ratio_20',
            'volume_breakout', 'volume_surge',
            
            # Price action
            'high_low_ratio', 'close_high_ratio', 'close_low_ratio',
            'intraday_strength', 'gap_strength',
            
            # Technical indicators
            'rsi_14', 'rsi_21',
            'rsi_14_oversold', 'rsi_14_overbought',
            'bb_position_20', 'bb_position_50',
            'bb_squeeze_20', 'bb_breakout_20',
            'macd', 'macd_histogram', 'macd_bullish', 'macd_bearish',
            
            # Seasonality and structure
            'quarter_end', 'september_effect', 'january_effect',
            'price_level', 'volume_level'
        ]
        
        enhanced_features = df[feature_columns].dropna()
        
        return enhanced_features
    
    def create_target(self, data: pd.DataFrame) -> pd.Series:
        """Create multi-horizon target optimized for Apple's price movements"""
        df = data.copy()
        
        # Multi-horizon returns with Apple-specific thresholds
        df['next_1_return'] = df['Close'].pct_change().shift(-1)
        df['next_3_return'] = df['Close'].pct_change(3).shift(-3)
        df['next_5_return'] = df['Close'].pct_change(5).shift(-5)
        
        # Apple-specific profit thresholds (conservative for large cap)
        target = ((df['next_1_return'] > 0.008) |   # 0.8% in 1 day
                 (df['next_3_return'] > 0.015) |   # 1.5% in 3 days
                 (df['next_5_return'] > 0.025)).astype(int)  # 2.5% in 5 days
        
        return target
    
    def run_comprehensive_backtest(self, start_date: str = None, end_date: str = None) -> Dict:
        """
        Run the comprehensive Apple backtesting system
        
        Args:
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            
        Returns:
            Dictionary with comprehensive backtesting results
        """
        
        print("=" * 80)
        print("🍎 APPLE COMPREHENSIVE BACKTESTING SYSTEM")
        print("=" * 80)
        print(f"📅 Started: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            import lightgbm as lgb
            import yfinance as yf
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.model_selection import TimeSeriesSplit
            
            # Set default date range (1 year)
            if end_date is None:
                end_date = dt.datetime.now()
            else:
                end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
            
            if start_date is None:
                start_date = end_date - dt.timedelta(days=400)  # Extra buffer for features
            else:
                start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
            
            self.logger.info(f"Backtesting period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Phase 1: Data Collection and Preparation
            print(f"\n[PHASE 1] Data Collection and Preparation")
            print(f"          Fetching Apple data...")
            
            apple = yf.Ticker(self.symbol)
            data = apple.history(start=start_date, end=end_date)
            
            print(f"          ✅ Fetched {len(data)} days of Apple data")
            print(f"          📊 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            
            # Create features and target
            print(f"          Creating enhanced features...")
            features_df = self.create_enhanced_features(data)
            target_series = self.create_target(data)
            
            # Align features and target
            common_index = features_df.index.intersection(target_series.index)
            X = features_df.loc[common_index]
            y = target_series.loc[common_index]
            
            # Remove any remaining NaN values
            valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[valid_mask]
            y = y[valid_mask]
            
            print(f"          ✅ Features created: {X.shape[1]} features, {len(X)} samples")
            print(f"          🎯 Target distribution: {y.sum()} positive ({y.mean():.2%}), {(~y.astype(bool)).sum()} negative")
            
            # Calculate split points
            total_days = len(X)
            training_days = int(total_days * 0.5)  # First 6 months (50%)
            
            train_end_idx = training_days
            
            print(f"          📋 Backtest structure:")
            print(f"             Total samples: {total_days}")
            print(f"             Training period: {training_days} days (first 50%)")
            print(f"             Testing period: {total_days - training_days} days (remaining)")
            
            # Phase 2: Initial Training (First 6 Months)
            print(f"\n[PHASE 2] Initial Training Phase")
            print(f"          Training on first {training_days} days...")
            
            X_train_initial = X.iloc[:train_end_idx]
            y_train_initial = y.iloc[:train_end_idx]
            
            # Train both model configurations
            models = {}
            training_results = {}
            
            for config_name, config in self.model_configs.items():
                print(f"          🔄 Training {config_name} model...")
                
                model = lgb.LGBMClassifier(**config)
                
                # Cross-validation on training data
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(X_train_initial):
                    cv_train_X, cv_val_X = X_train_initial.iloc[train_idx], X_train_initial.iloc[val_idx]
                    cv_train_y, cv_val_y = y_train_initial.iloc[train_idx], y_train_initial.iloc[val_idx]
                    
                    cv_model = lgb.LGBMClassifier(**config)
                    cv_model.fit(cv_train_X, cv_train_y)
                    cv_pred = cv_model.predict(cv_val_X)
                    cv_scores.append(accuracy_score(cv_val_y, cv_pred))
                
                # Train final model
                start_time = dt.datetime.now()
                model.fit(X_train_initial, y_train_initial)
                training_time = (dt.datetime.now() - start_time).total_seconds()
                
                # Training accuracy
                train_pred = model.predict(X_train_initial)
                train_accuracy = accuracy_score(y_train_initial, train_pred)
                
                models[config_name] = model
                training_results[config_name] = {
                    'cv_accuracy': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'train_accuracy': train_accuracy,
                    'overfitting': train_accuracy - np.mean(cv_scores),
                    'training_time': training_time
                }
                
                print(f"             ✅ {config_name} completed:")
                print(f"                CV Accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
                print(f"                Train Accuracy: {train_accuracy:.4f}")
                print(f"                Overfitting: {train_accuracy - np.mean(cv_scores):.4f}")
                print(f"                Training Time: {training_time:.2f}s")
            
            # Select best model based on CV score and overfitting
            best_model_name = min(training_results.keys(), 
                                key=lambda x: training_results[x]['overfitting'] - training_results[x]['cv_accuracy'])
            
            best_model = models[best_model_name]
            print(f"          🏆 Best model selected: {best_model_name}")
            
            # Phase 3: Static Testing (Next 6 Months)
            print(f"\n[PHASE 3] Static Testing Phase")
            print(f"          Testing on remaining {total_days - training_days} days...")
            
            X_test_static = X.iloc[train_end_idx:]
            y_test_static = y.iloc[train_end_idx:]
            
            static_predictions = best_model.predict(X_test_static)
            static_probabilities = best_model.predict_proba(X_test_static)
            
            static_accuracy = accuracy_score(y_test_static, static_predictions)
            
            print(f"          📊 Static testing results:")
            print(f"             Accuracy: {static_accuracy:.4f}")
            print(f"             Samples: {len(y_test_static)}")
            
            # Phase 4: Adaptive Testing (2-Week Retraining)
            print(f"\n[PHASE 4] Adaptive Testing Phase")
            print(f"          Implementing 2-week retraining strategy...")
            
            # Split test period into 2-week chunks
            retrain_frequency = self.config['retrain_frequency_days']
            adaptive_results = []
            
            current_train_end = train_end_idx
            test_start_idx = train_end_idx
            
            retrain_count = 0
            
            while test_start_idx < len(X):
                retrain_count += 1
                test_end_idx = min(test_start_idx + retrain_frequency, len(X))
                
                print(f"          🔄 Retrain cycle {retrain_count}:")
                print(f"             Training data: samples 0 to {current_train_end}")
                print(f"             Testing data: samples {test_start_idx} to {test_end_idx}")
                
                # Prepare training data (expanding window)
                X_train_adaptive = X.iloc[:current_train_end]
                y_train_adaptive = y.iloc[:current_train_end]
                
                # Prepare testing data
                X_test_chunk = X.iloc[test_start_idx:test_end_idx]
                y_test_chunk = y.iloc[test_start_idx:test_end_idx]
                
                if len(X_test_chunk) == 0:
                    break
                
                # Retrain model
                adaptive_model = lgb.LGBMClassifier(**self.model_configs[best_model_name])
                adaptive_model.fit(X_train_adaptive, y_train_adaptive)
                
                # Make predictions
                chunk_predictions = adaptive_model.predict(X_test_chunk)
                chunk_probabilities = adaptive_model.predict_proba(X_test_chunk)
                chunk_accuracy = accuracy_score(y_test_chunk, chunk_predictions)
                
                # Store results
                adaptive_results.append({
                    'cycle': retrain_count,
                    'test_start': test_start_idx,
                    'test_end': test_end_idx,
                    'samples': len(X_test_chunk),
                    'accuracy': chunk_accuracy,
                    'predictions': chunk_predictions,
                    'probabilities': chunk_probabilities[:, 1],  # Probability of positive class
                    'actual': y_test_chunk.values
                })
                
                print(f"             ✅ Cycle {retrain_count} completed: {chunk_accuracy:.4f} accuracy on {len(X_test_chunk)} samples")
                
                # Update for next cycle
                current_train_end = test_end_idx
                test_start_idx = test_end_idx
            
            # Calculate overall adaptive results
            all_adaptive_predictions = np.concatenate([r['predictions'] for r in adaptive_results])
            all_adaptive_actual = np.concatenate([r['actual'] for r in adaptive_results])
            adaptive_accuracy = accuracy_score(all_adaptive_actual, all_adaptive_predictions)
            
            print(f"          📊 Adaptive testing results:")
            print(f"             Overall Accuracy: {adaptive_accuracy:.4f}")
            print(f"             Retrain Cycles: {retrain_count}")
            print(f"             Total Samples: {len(all_adaptive_predictions)}")
            
            # Phase 5: Performance Analysis and Trading Simulation
            print(f"\n[PHASE 5] Performance Analysis and Trading Simulation")
            
            trading_results = self.simulate_trading_performance(
                static_predictions, static_probabilities[:, 1], y_test_static,
                all_adaptive_predictions, np.concatenate([r['probabilities'] for r in adaptive_results]), all_adaptive_actual,
                X_test_static.index.tolist() + [X.index[i] for r in adaptive_results for i in range(r['test_start'], r['test_end'])],
                data
            )
            
            # Compile comprehensive results
            results = {
                'metadata': {
                    'symbol': self.symbol,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'total_days': total_days,
                    'training_days': training_days,
                    'features': X.columns.tolist(),
                    'best_model': best_model_name,
                    'retrain_cycles': retrain_count
                },
                'training_phase': training_results,
                'static_testing': {
                    'accuracy': static_accuracy,
                    'samples': len(y_test_static),
                    'predictions': static_predictions.tolist(),
                    'probabilities': static_probabilities[:, 1].tolist(),
                    'actual': y_test_static.tolist()
                },
                'adaptive_testing': {
                    'accuracy': adaptive_accuracy,
                    'samples': len(all_adaptive_predictions),
                    'cycles': adaptive_results,
                    'improvement_over_static': adaptive_accuracy - static_accuracy
                },
                'trading_simulation': trading_results,
                'feature_importance': dict(zip(X.columns, best_model.feature_importances_))
            }
            
            # Generate comprehensive report
            self.generate_comprehensive_report(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Backtesting failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def simulate_trading_performance(self, static_pred, static_prob, static_actual,
                                   adaptive_pred, adaptive_prob, adaptive_actual,
                                   dates, price_data) -> Dict:
        """Simulate trading performance for both strategies"""
        
        print(f"          🎯 Simulating trading performance...")
        
        # Trading simulation parameters
        initial_capital = 100000  # $100k
        transaction_cost = self.trading_config['transaction_cost']
        min_confidence = self.trading_config['min_confidence']
        
        def simulate_strategy(predictions, probabilities, actual, strategy_name):
            capital = initial_capital
            positions = []
            trades = []
            daily_returns = []
            
            for i, (pred, prob, actual_val) in enumerate(zip(predictions, probabilities, actual)):
                # Only trade if confidence is high enough
                if prob >= min_confidence and pred == 1:  # Buy signal
                    # Calculate position size
                    position_size = capital * self.trading_config['position_size']
                    
                    # Account for transaction cost
                    actual_position = position_size * (1 - transaction_cost)
                    
                    # Simulate holding for prediction horizon days
                    if i < len(actual) - self.config['prediction_horizon']:
                        # Use actual future performance
                        future_return = actual_val  # Binary target, simulate return
                        
                        if future_return:  # Positive outcome
                            profit = actual_position * 0.02  # Simulate 2% average profit
                        else:  # Negative outcome
                            profit = actual_position * -0.015  # Simulate 1.5% average loss
                        
                        capital += profit
                        daily_returns.append(profit / initial_capital)
                        
                        trades.append({
                            'date_index': i,
                            'confidence': prob,
                            'predicted': pred,
                            'actual': actual_val,
                            'profit': profit,
                            'capital': capital
                        })
                else:
                    daily_returns.append(0)  # No trade
            
            total_return = (capital - initial_capital) / initial_capital
            num_trades = len(trades)
            win_rate = sum(1 for t in trades if t['profit'] > 0) / num_trades if num_trades > 0 else 0
            
            return {
                'strategy': strategy_name,
                'final_capital': capital,
                'total_return': total_return,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'daily_returns': daily_returns,
                'trades': trades
            }
        
        # Simulate both strategies
        static_results = simulate_strategy(static_pred, static_prob, static_actual, 'Static')
        adaptive_results = simulate_strategy(adaptive_pred, adaptive_prob, adaptive_actual, 'Adaptive')
        
        print(f"             ✅ Static strategy: {static_results['total_return']:+.2%} return, {static_results['num_trades']} trades")
        print(f"             ✅ Adaptive strategy: {adaptive_results['total_return']:+.2%} return, {adaptive_results['num_trades']} trades")
        
        return {
            'static': static_results,
            'adaptive': adaptive_results,
            'improvement': adaptive_results['total_return'] - static_results['total_return']
        }
    
    def generate_comprehensive_report(self, results: Dict):
        """Generate comprehensive backtesting report"""
        
        print(f"\n{'='*80}")
        print(f"📊 APPLE BACKTESTING COMPREHENSIVE RESULTS")
        print(f"{'='*80}")
        
        metadata = results['metadata']
        training = results['training_phase']
        static = results['static_testing']
        adaptive = results['adaptive_testing']
        trading = results['trading_simulation']
        
        print(f"\n📋 BACKTEST CONFIGURATION")
        print(f"   Symbol: {metadata['symbol']}")
        print(f"   Period: {metadata['start_date']} to {metadata['end_date']}")
        print(f"   Total Days: {metadata['total_days']}")
        print(f"   Training Days: {metadata['training_days']}")
        print(f"   Features: {len(metadata['features'])}")
        print(f"   Best Model: {metadata['best_model']}")
        print(f"   Retrain Cycles: {metadata['retrain_cycles']}")
        
        print(f"\n🏆 MODEL PERFORMANCE COMPARISON")
        print(f"   {'Model':<15} {'CV Acc':<8} {'Train Acc':<10} {'Overfit':<8} {'Time(s)':<8}")
        print(f"   {'-'*55}")
        
        for model_name, metrics in training.items():
            print(f"   {model_name:<15} {metrics['cv_accuracy']:<8.4f} {metrics['train_accuracy']:<10.4f} "
                  f"{metrics['overfitting']:<8.4f} {metrics['training_time']:<8.2f}")
        
        print(f"\n📊 TESTING PERFORMANCE")
        print(f"   Static Testing (No Retraining):")
        print(f"      Accuracy: {static['accuracy']:.4f}")
        print(f"      Samples: {static['samples']}")
        print(f"   ")
        print(f"   Adaptive Testing (2-week Retraining):")
        print(f"      Accuracy: {adaptive['accuracy']:.4f}")
        print(f"      Samples: {adaptive['samples']}")
        print(f"      Improvement: {adaptive['improvement_over_static']:+.4f}")
        
        print(f"\n💰 TRADING SIMULATION RESULTS")
        static_trading = trading['static']
        adaptive_trading = trading['adaptive']
        
        print(f"   Static Strategy:")
        print(f"      Total Return: {static_trading['total_return']:+.2%}")
        print(f"      Final Capital: ${static_trading['final_capital']:,.2f}")
        print(f"      Number of Trades: {static_trading['num_trades']}")
        print(f"      Win Rate: {static_trading['win_rate']:.2%}")
        print(f"   ")
        print(f"   Adaptive Strategy:")
        print(f"      Total Return: {adaptive_trading['total_return']:+.2%}")
        print(f"      Final Capital: ${adaptive_trading['final_capital']:,.2f}")
        print(f"      Number of Trades: {adaptive_trading['num_trades']}")
        print(f"      Win Rate: {adaptive_trading['win_rate']:.2%}")
        print(f"   ")
        print(f"   Performance Improvement:")
        print(f"      Return Difference: {trading['improvement']:+.2%}")
        print(f"      Capital Difference: ${adaptive_trading['final_capital'] - static_trading['final_capital']:+,.2f}")
        
        print(f"\n🔍 TOP FEATURE IMPORTANCE")
        feature_importance = results['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for i, (feature, importance) in enumerate(sorted_features):
            print(f"   {i+1:2d}. {feature:<25}: {importance:.4f}")
        
        print(f"\n💡 KEY INSIGHTS")
        
        # Performance insights
        if adaptive['improvement_over_static'] > 0.01:
            print(f"   ✅ Adaptive retraining provides significant improvement (+{adaptive['improvement_over_static']:.3f})")
        elif adaptive['improvement_over_static'] > 0:
            print(f"   ✅ Adaptive retraining provides modest improvement (+{adaptive['improvement_over_static']:.3f})")
        else:
            print(f"   ⚠️ Adaptive retraining shows minimal benefit ({adaptive['improvement_over_static']:+.3f})")
        
        # Trading insights
        if trading['improvement'] > 0.05:
            print(f"   💰 Adaptive trading strategy significantly outperforms (+{trading['improvement']:.1%})")
        elif trading['improvement'] > 0:
            print(f"   💰 Adaptive trading strategy modestly outperforms (+{trading['improvement']:.1%})")
        else:
            print(f"   ⚠️ Static strategy performs similarly or better ({trading['improvement']:+.1%})")
        
        # Model insights
        best_model_metrics = training[metadata['best_model']]
        if best_model_metrics['overfitting'] < 0.1:
            print(f"   🎯 Excellent model generalization (overfitting: {best_model_metrics['overfitting']:.3f})")
        elif best_model_metrics['overfitting'] < 0.2:
            print(f"   🎯 Good model generalization (overfitting: {best_model_metrics['overfitting']:.3f})")
        else:
            print(f"   ⚠️ Model shows signs of overfitting ({best_model_metrics['overfitting']:.3f})")
        
        print(f"\n📈 RECOMMENDATIONS")
        
        if adaptive['improvement_over_static'] > 0:
            print(f"   🎯 Deploy adaptive retraining strategy for Apple")
            print(f"   📅 Retrain model every 2 weeks with expanding window")
        else:
            print(f"   🎯 Static model sufficient for Apple - save computational costs")
        
        print(f"   🏆 Use {metadata['best_model']} model configuration")
        print(f"   📊 Focus on top features: {', '.join([f[0] for f in sorted_features[:5]])}")
        print(f"   💰 Minimum confidence threshold: {self.trading_config['min_confidence']}")
        
        print(f"\n{'='*80}")

def run_apple_backtest_demo():
    """Run comprehensive Apple backtesting demonstration"""
    
    print("Starting Apple Comprehensive Backtesting System...")
    
    # Initialize backtesting system
    apple_system = AppleBacktestingSystem()
    
    # Set date range (last year)
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=365)
    
    # Run comprehensive backtest
    results = apple_system.run_comprehensive_backtest(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    if 'error' in results:
        print(f"\n❌ Backtesting failed: {results['error']}")
        return False
    
    print(f"\n🎉 Apple comprehensive backtesting completed successfully!")
    return True

if __name__ == "__main__":
    success = run_apple_backtest_demo()
    
    if success:
        print(f"\n✅ Apple backtesting system demonstration completed!")
    else:
        print(f"\n❌ Apple backtesting system failed.")
    
    print(f"\nCompleted at: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")