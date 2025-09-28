#!/usr/bin/env python3
"""
Tesla Production Trading Model
Integration of the winning LightGBM Aggressive model into the trading system
"""

import sys
import os
import datetime as dt
import numpy as np
import pandas as pd
import pickle
import warnings
from typing import Dict, Tuple, Optional, List
import logging

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

warnings.filterwarnings('ignore')

class TeslaProductionModel:
    """
    Production-ready Tesla trading model using LightGBM with aggressive regularization
    Based on enhanced comparison results showing 53.76% CV accuracy with excellent generalization
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.symbol = 'TSLA'
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        self.model_path = model_path or 'models/tesla_production_model.pkl'
        self.performance_threshold = 0.48  # Minimum acceptable accuracy
        
        # Model configuration (winner from comparison)
        self.model_config = {
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
        }
        
        # Trading configuration
        self.trading_config = {
            'max_position_size': 0.10,      # Max 10% portfolio allocation
            'min_confidence': 0.55,         # Minimum prediction probability
            'stop_loss_pct': 0.025,         # 2.5% stop loss
            'take_profit_pct': 0.05,        # 5% take profit
            'retraining_days': 7,           # Retrain weekly
            'min_data_days': 200            # Minimum historical data
        }
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for model operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('tesla_model.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('TeslaModel')
    
    def create_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create the same enhanced feature set used in winning model
        35 features with multi-timeframe analysis
        """
        
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
        
        # Select feature columns (same as winning model)
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
            'macd', 'macd_histogram', 'macd_bullish'
        ]
        
        features_df = df[feature_columns].dropna()
        return features_df
    
    def create_target(self, data: pd.DataFrame) -> pd.Series:
        """Create multi-horizon target matching winning model"""
        df = data.copy()
        
        # Multi-horizon returns
        df['next_1_return'] = df['Close'].pct_change().shift(-1)
        df['next_2_return'] = df['Close'].pct_change(2).shift(-2)
        df['next_3_return'] = df['Close'].pct_change(3).shift(-3)
        
        # Multi-horizon target (profitable over next 1-3 days)
        target = ((df['next_1_return'] > 0.01) | 
                 (df['next_2_return'] > 0.02) | 
                 (df['next_3_return'] > 0.03)).astype(int)
        
        return target
    
    def train_model(self, data: pd.DataFrame, retrain: bool = False) -> Dict:
        """
        Train the Tesla model with the winning configuration
        
        Args:
            data: Historical price data
            retrain: Force retrain even if model exists
            
        Returns:
            Training results dictionary
        """
        
        try:
            import lightgbm as lgb
            from sklearn.model_selection import TimeSeriesSplit, cross_val_score
            from sklearn.metrics import accuracy_score, classification_report
            
            self.logger.info(f"Training Tesla model (retrain={retrain})")
            
            # Check if model already exists and is recent
            if not retrain and self.load_model():
                self.logger.info("Using existing trained model")
                return {'status': 'loaded', 'accuracy': 'unknown'}
            
            # Create features and target
            self.logger.info("Creating enhanced features...")
            features = self.create_enhanced_features(data)
            target_full = self.create_target(data)
            
            # Align features and target
            common_index = features.index.intersection(target_full.index)
            X = features.loc[common_index]
            y = target_full.loc[common_index]
            
            # Remove any remaining NaN values
            valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[valid_mask]
            y = y[valid_mask]
            
            self.logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
            self.logger.info(f"Target distribution: {y.sum()} positive, {(~y.astype(bool)).sum()} negative")
            
            # Initialize model with winning configuration
            self.model = lgb.LGBMClassifier(**self.model_config)
            
            # Cross-validation for performance estimation
            self.logger.info("Running cross-validation...")
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(self.model, X.values, y.values, cv=tscv, scoring='accuracy')
            
            # Train on full dataset
            self.logger.info("Training final model...")
            start_time = dt.datetime.now()
            self.model.fit(X.values, y.values)
            training_time = (dt.datetime.now() - start_time).total_seconds()
            
            # Calculate training accuracy
            train_predictions = self.model.predict(X.values)
            train_accuracy = accuracy_score(y.values, train_predictions)
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            results = {
                'status': 'trained',
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_accuracy': train_accuracy,
                'overfitting_score': train_accuracy - cv_scores.mean(),
                'training_time': training_time,
                'samples': len(X),
                'features': len(self.feature_columns)
            }
            
            self.logger.info(f"Training completed:")
            self.logger.info(f"  CV Accuracy: {results['cv_accuracy']:.4f} (±{results['cv_std']:.4f})")
            self.logger.info(f"  Train Accuracy: {results['train_accuracy']:.4f}")
            self.logger.info(f"  Overfitting: {results['overfitting_score']:.4f}")
            self.logger.info(f"  Training Time: {results['training_time']:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def predict_signal(self, data: pd.DataFrame, return_confidence: bool = True) -> Dict:
        """
        Generate trading signal for Tesla
        
        Args:
            data: Recent price data (should include at least 50+ days for features)
            return_confidence: Whether to return prediction confidence
            
        Returns:
            Dictionary with signal, confidence, and metadata
        """
        
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            # Create features
            features = self.create_enhanced_features(data)
            
            if features.empty:
                return {'error': 'No features created from data'}
            
            # Get latest features
            latest_features = features.iloc[-1:][self.feature_columns].values
            
            # Check for NaN values
            if np.isnan(latest_features).any():
                return {'error': 'NaN values in features'}
            
            # Make prediction
            prediction = self.model.predict(latest_features)[0]
            prediction_proba = self.model.predict_proba(latest_features)[0]
            
            confidence = max(prediction_proba)
            signal_strength = 'STRONG' if confidence >= 0.70 else 'MODERATE' if confidence >= 0.60 else 'WEAK'
            
            # Position sizing based on confidence
            base_position = self.trading_config['max_position_size']
            confidence_multiplier = min(confidence / 0.55, 1.0)  # Scale down if low confidence
            suggested_position = base_position * confidence_multiplier
            
            result = {
                'signal': 'BUY' if prediction == 1 else 'SELL',
                'confidence': confidence,
                'signal_strength': signal_strength,
                'suggested_position': suggested_position,
                'timestamp': dt.datetime.now().isoformat(),
                'model_ready': confidence >= self.trading_config['min_confidence']
            }
            
            if return_confidence:
                result['probabilities'] = {
                    'buy': prediction_proba[1],
                    'sell': prediction_proba[0]
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {'error': str(e)}
    
    def save_model(self):
        """Save trained model to disk"""
        if self.model and self.feature_columns:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'model_config': self.model_config,
                'trading_config': self.trading_config,
                'timestamp': dt.datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            if not os.path.exists(self.model_path):
                return False
            
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = True
            
            # Check model age
            model_timestamp = dt.datetime.fromisoformat(model_data['timestamp'])
            age_days = (dt.datetime.now() - model_timestamp).days
            
            if age_days > self.trading_config['retraining_days']:
                self.logger.warning(f"Model is {age_days} days old, consider retraining")
            
            self.logger.info(f"Model loaded from {self.model_path} (age: {age_days} days)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_status(self) -> Dict:
        """Get current model status and configuration"""
        return {
            'is_trained': self.is_trained,
            'model_path': self.model_path,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'model_config': self.model_config,
            'trading_config': self.trading_config,
            'performance_threshold': self.performance_threshold
        }

def demo_tesla_production_model():
    """Demonstration of the Tesla production model"""
    
    print("=" * 80)
    print("🚀 TESLA PRODUCTION MODEL DEMO")
    print("=" * 80)
    
    try:
        import yfinance as yf
        
        # Initialize model
        tesla_model = TeslaProductionModel()
        
        # Fetch Tesla data
        print("\n[1] Fetching Tesla data for training...")
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=800)
        
        tesla = yf.Ticker('TSLA')
        data = tesla.history(start=start_date, end=end_date)
        print(f"    ✅ Fetched {len(data)} days of Tesla data")
        
        # Train model
        print("\n[2] Training production model...")
        training_results = tesla_model.train_model(data)
        
        if training_results['status'] == 'trained':
            print(f"    ✅ Model training successful!")
            print(f"    📊 CV Accuracy: {training_results['cv_accuracy']:.4f}")
            print(f"    🎯 Overfitting: {training_results['overfitting_score']:.4f}")
            print(f"    ⏱️ Training Time: {training_results['training_time']:.2f}s")
        else:
            print(f"    ❌ Training failed: {training_results.get('error', 'Unknown error')}")
            return
        
        # Generate current signal
        print("\n[3] Generating current trading signal...")
        signal_result = tesla_model.predict_signal(data)
        
        if 'error' not in signal_result:
            print(f"    🎯 Signal: {signal_result['signal']}")
            print(f"    🎲 Confidence: {signal_result['confidence']:.4f}")
            print(f"    💪 Strength: {signal_result['signal_strength']}")
            print(f"    💰 Suggested Position: {signal_result['suggested_position']:.2%}")
            print(f"    ✅ Ready to Trade: {signal_result['model_ready']}")
        else:
            print(f"    ❌ Signal generation failed: {signal_result['error']}")
        
        # Show model status
        print("\n[4] Model Status:")
        status = tesla_model.get_model_status()
        print(f"    📊 Features: {status['feature_count']}")
        print(f"    🎯 Performance Threshold: {status['performance_threshold']:.4f}")
        print(f"    ⚙️ Max Position Size: {status['trading_config']['max_position_size']:.2%}")
        print(f"    🛡️ Stop Loss: {status['trading_config']['stop_loss_pct']:.2%}")
        
        print(f"\n🎉 Tesla production model demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_tesla_production_model()