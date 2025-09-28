"""
Enhanced Trading Strategy with Market Context
Clean, production-ready implementation of the ML trading system with market indicators
"""

import datetime as dt
import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Optional, List, Dict
from marketsimcode import compute_stats, trades2orders, compute_portvals
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from config_manager import get_config
from market_indicators import get_enhanced_features, get_market_data, analyze_market_correlation
from model_management import ModelManager, ModelPredictionService

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class EnhancedTradingStrategy:
    """
    Enhanced ML Trading Strategy with Market Context and Model Management
    
    Features:
    - Market index integration (SPY, QQQ)
    - Enhanced feature engineering (17 vs 5 features)
    - Configurable ML algorithms
    - Comprehensive performance analysis
    - Risk management through market context
    - Model persistence and versioning
    - Real-time prediction service
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the enhanced trading strategy with model management"""
        self.config = get_config()
        self.symbol = None
        self.model = None
        self.feature_names = None
        self.training_data = None
        self.performance_metrics = {}
        
        # Initialize model management
        self.model_manager = ModelManager()
        self.prediction_service = ModelPredictionService(self.model_manager)
        print("[INIT] Enhanced Trading Strategy with Model Management initialized")
        print(f"       Models directory: {self.model_manager.models_dir}")
        
    def check_existing_model(self, symbol: str) -> bool:
        """Check if a trained model exists for the symbol"""
        return self.model_manager.model_exists(symbol)
        
    def get_model_info(self, symbol: str) -> Dict:
        """Get information about existing model for the symbol"""
        return self.model_manager.get_model_info(symbol)
        
    def analyze_market_context(self, symbol: str, start_date: dt.datetime, 
                             end_date: dt.datetime) -> Dict:
        """Analyze market context and correlations for the symbol"""
        print(f"[INFO] Analyzing Market Context for {symbol}")
        print(f"       Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        correlations, beta_spy, beta_qqq, market_indicators = analyze_market_correlation(
            symbol, start_date, end_date, window=self.config.get_indicator_window()
        )
        
        return {
            'correlations': correlations,
            'beta_spy': beta_spy,
            'beta_qqq': beta_qqq,
            'market_indicators': market_indicators,
            'analysis_period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        }
    
    def prepare_training_data(self, symbol: str, train_start: dt.datetime, 
                            train_end: dt.datetime) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare enhanced training data with market context"""
        print(f"[TRAIN] Preparing Enhanced Training Data")
        print(f"        Training Period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        
        # Get enhanced features
        enhanced_features = get_enhanced_features(
            symbol, train_start, train_end, 
            window=self.config.get_indicator_window(), 
            train=True
        )
        
        # Get stock data for labels
        stock_data = get_market_data([symbol], train_start, train_end)[symbol]
        labels = self._generate_labels(stock_data)
        
        print(f"        [OK] Training samples: {enhanced_features.shape[0]}")
        print(f"        [OK] Features: {enhanced_features.shape[1]} (enhanced)")
        print(f"        [OK] Labels: {len(labels)}")
        
        # Store feature names for later analysis
        from market_indicators import compute_enhanced_indicators
        sample_data = get_market_data([symbol, 'SPY', 'QQQ'], train_start, train_end)
        sample_features = compute_enhanced_indicators(
            sample_data[symbol], sample_data['SPY'], sample_data['QQQ'], 
            window=self.config.get_indicator_window()
        )
        if 'Upper_BB' in sample_features.columns:
            sample_features.drop('Upper_BB', axis=1, inplace=True)
        if 'Down_BB' in sample_features.columns:
            sample_features.drop('Down_BB', axis=1, inplace=True)
        self.feature_names = sample_features.columns.tolist()
        
        self.training_data = {
            'features': enhanced_features,
            'labels': labels,
            'feature_names': self.feature_names,
            'train_start': train_start,
            'train_end': train_end
        }
        
        return enhanced_features, labels
    
    def train_model(self, X: np.ndarray, y: np.ndarray, algorithm: str = None, 
                   save_model: bool = True) -> None:
        """Train the ML model with enhanced features and save to disk"""
        print(f"[MODEL] Training Enhanced ML Model for {self.symbol}")
        
        ml_config = self.config.get_ml_config()
        algorithm = algorithm or ml_config['algorithm']
        
        if algorithm == 'RandomForest':
            self.model = RandomForestClassifier(
                n_estimators=ml_config['n_estimators'],
                max_depth=ml_config['max_depth'],
                random_state=ml_config['random_state'],
                n_jobs=-1  # Use all CPU cores
            )
        else:
            self.model = DecisionTreeClassifier(
                max_depth=ml_config['max_depth'],
                random_state=ml_config['random_state']
            )
        
        # Train the model
        self.model.fit(X, y)
        
        # Analyze training performance
        train_predictions = self.model.predict(X)
        train_accuracy = accuracy_score(y, train_predictions)
        
        # Store performance metrics
        self.performance_metrics = {
            'training_accuracy': train_accuracy,
            'algorithm': algorithm,
            'training_samples': len(X),
            'feature_count': len(self.feature_names)
        }
        
        print(f"        [OK] Algorithm: {algorithm}")
        print(f"        [OK] Training accuracy: {train_accuracy:.3f}")
        print(f"        [OK] Training samples: {len(X)}")
        print(f"        [OK] Features: {len(self.feature_names)}")
        
        # Save the model
        if save_model and self.symbol:
            self._save_trained_model()
    
    def _save_trained_model(self) -> None:
        """Save the trained model using model management system"""
        if not self.model or not self.symbol:
            print("[WARNING] No trained model or symbol to save")
            return
            
        print(f"[SAVE] Saving model for {self.symbol}")
        
        # Prepare training data info
        training_info = {
            'train_start': self.training_data.get('train_start'),
            'train_end': self.training_data.get('train_end'),
            'training_samples': len(self.training_data.get('features', [])),
        }
        
        # Save model with comprehensive metadata
        model_path = self.model_manager.save_model(
            symbol=self.symbol,
            model=self.model,
            training_data=training_info,
            performance_metrics=self.performance_metrics,
            feature_names=self.feature_names,
            config_snapshot=self.config.get_all_config()
        )
        
        print(f"       [OK] Model saved: {model_path}")
    
    def load_existing_model(self, symbol: str) -> bool:
        """Load existing trained model for the symbol"""
        try:
            print(f"[LOAD] Loading existing model for {symbol}")
            model, metadata = self.model_manager.load_model(symbol)
            
            self.model = model
            self.symbol = symbol
            self.feature_names = metadata['features']['names']
            self.performance_metrics = metadata['performance']
            
            print(f"       [OK] Model loaded successfully")
            print(f"       [OK] Version: {metadata['version']}")
            print(f"       [OK] Training accuracy: {metadata['performance'].get('training_accuracy', 'N/A'):.3f}")
            print(f"       [OK] Features: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            print(f"       [ERROR] Failed to load model: {e}")
            return False
    
    def generate_predictions(self, symbol: str, test_start: dt.datetime, 
                           test_end: dt.datetime, use_prediction_service: bool = True) -> Tuple[np.ndarray, Dict]:
        """Generate trading predictions for test period using model management"""
        print(f"[PREDICT] Generating Trading Predictions for {symbol}")
        print(f"          Test Period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
        
        # Get test features
        test_features = get_enhanced_features(
            symbol, test_start, test_end,
            window=self.config.get_indicator_window(),
            train=False
        )
        
        if use_prediction_service and symbol:
            # Use prediction service for better model management
            try:
                predictions, pred_info = self.prediction_service.predict(symbol, test_features[:-3])
                signal_analysis = self.prediction_service.get_prediction_summary(symbol, predictions)
                
                print(f"          [OK] Using prediction service")
                print(f"          [OK] Model version: {pred_info.get('model_version', 'Unknown')}")
                
            except Exception as e:
                print(f"          [WARNING] Prediction service failed: {e}")
                print(f"          [INFO] Falling back to direct model prediction")
                predictions = self.model.predict(test_features[:-3])
                signal_analysis = self._analyze_signal_distribution(predictions)
        else:
            # Direct model prediction
            predictions = self.model.predict(test_features[:-3])
            signal_analysis = self._analyze_signal_distribution(predictions)
        
        print(f"          [OK] Predictions: {signal_analysis['total_predictions']}")
        print(f"          [OK] BUY: {signal_analysis['buy_signals']} ({signal_analysis.get('buy_percentage', 0):.1f}%)")
        print(f"          [OK] SELL: {signal_analysis['sell_signals']}")
        print(f"          [OK] HOLD: {signal_analysis['hold_signals']}")
        
        return predictions, signal_analysis
    
    def _analyze_signal_distribution(self, predictions: np.ndarray) -> Dict:
        """Analyze signal distribution in predictions"""
        return {
            'total_predictions': len(predictions),
            'buy_signals': int(np.sum(predictions == 1)),
            'sell_signals': int(np.sum(predictions == -1)),
            'hold_signals': int(np.sum(predictions == 0)),
            'buy_percentage': float(np.sum(predictions == 1) / len(predictions) * 100)
        }
    
    def backtest_strategy(self, symbol: str, test_start: dt.datetime, 
                         test_end: dt.datetime, predictions: np.ndarray) -> Dict:
        """Run comprehensive backtest of the trading strategy"""
        print(f"[BACKTEST] Running Strategy Backtest")
        
        # Get test stock data
        stock_data = get_market_data([symbol], test_start, test_end)[symbol]
        
        # Create trades
        trades_df = self._create_trades_dataframe(stock_data, symbol, predictions)
        orders = trades2orders(trades_df, symbol)
        
        trading_summary = {
            'total_orders': len(orders),
            'buy_orders': len(orders[orders['ORDER'] == 'BUY']) if len(orders) > 0 else 0,
            'sell_orders': len(orders[orders['ORDER'] == 'SELL']) if len(orders) > 0 else 0
        }
        
        print(f"           [OK] Orders executed: {trading_summary['total_orders']}")
        print(f"           [OK] Buys: {trading_summary['buy_orders']}, Sells: {trading_summary['sell_orders']}")
        
        # Calculate performance
        if len(orders) > 0:
            portfolio_config = self.config.get_portfolio_config()
            portfolio_values = compute_portvals(
                stock_data, orders,
                start_val=portfolio_config['starting_value'],
                commission=portfolio_config['commission'],
                impact=portfolio_config['impact']
            )
            
            # Calculate metrics
            cr, adr, sddr, sr = compute_stats(portfolio_values)
            
            # Benchmark comparison
            initial_price = stock_data.iloc[0, 0]
            final_price = stock_data.iloc[-1, 0]
            buy_hold_return = (final_price - initial_price) / initial_price
            
            performance = {
                'strategy_return': float(cr),
                'daily_return': float(adr),
                'volatility': float(sddr),
                'sharpe_ratio': float(sr),
                'starting_value': portfolio_config['starting_value'],
                'final_value': float(portfolio_values.iloc[-1]),
                'profit_loss': float(portfolio_values.iloc[-1] - portfolio_config['starting_value']),
                'buy_hold_return': float(buy_hold_return),
                'outperformance': float(cr - buy_hold_return),
                'portfolio_values': portfolio_values
            }
            
            print(f"           [OK] Strategy Return: {performance['strategy_return']:.1%}")
            print(f"           [OK] Buy-Hold Return: {performance['buy_hold_return']:.1%}")
            print(f"           [OK] Outperformance: {performance['outperformance']:.1%}")
            print(f"           [OK] Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
            
        else:
            # No trades executed
            performance = {
                'strategy_return': 0.0,
                'daily_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'buy_hold_return': float(buy_hold_return),
                'outperformance': float(-buy_hold_return)
            }
            print(f"           [WARN] No trades executed - remained in cash")
        
        performance.update(trading_summary)
        self.performance_metrics = performance
        return performance
    
    def analyze_feature_importance(self) -> Dict:
        """Analyze which features are most important for predictions"""
        if self.model is None or self.feature_names is None:
            return {}
        
        print(f"[FEATURES] Analyzing Feature Importance")
        
        importances = self.model.feature_importances_
        feature_importance = list(zip(self.feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Categorize features
        market_features = []
        technical_features = []
        
        for name, importance in feature_importance:
            if any(x in name.lower() for x in ['spy', 'qqq', 'beta', 'market', 'tech']):
                market_features.append((name, importance))
            else:
                technical_features.append((name, importance))
        
        analysis = {
            'feature_importance': feature_importance,
            'top_10': feature_importance[:10],
            'market_features': market_features,
            'technical_features': technical_features,
            'market_dominance': len([f for f in feature_importance[:10] if any(x in f[0].lower() for x in ['spy', 'qqq', 'beta', 'market', 'tech'])])
        }
        
        print(f"           [OK] Top 10 features analyzed")
        print(f"           [OK] Market features in top 10: {analysis['market_dominance']}/10")
        
        return analysis
    
    def run_complete_simulation(self, symbol: str, months_back: int = 6, 
                               force_retrain: bool = False) -> Dict:
        """Run a complete simulation with intelligent model management"""
        self.symbol = symbol
        
        # Calculate dates
        current_date = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        test_start = current_date - dt.timedelta(days=months_back * 30)
        train_start = test_start - dt.timedelta(days=365)  # 12 months training
        train_end = test_start
        
        print(f"[SIMULATION] Running Complete Enhanced Trading Simulation with Model Management")
        print(f"             Symbol: {symbol}")
        print(f"             Simulation Period: {months_back} months")
        print(f"             Test Period: {test_start.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}")
        print("=" * 70)
        
        # Check for existing model
        model_exists = self.check_existing_model(symbol)
        use_existing_model = model_exists and not force_retrain
        
        if use_existing_model:
            print(f"[MODEL] Existing model found for {symbol}")
            model_info = self.get_model_info(symbol)
            print(f"        Version: {model_info.get('version', 'Unknown')}")
            print(f"        Training Accuracy: {model_info.get('performance', {}).get('training_accuracy', 'N/A')}")
            print(f"        Features: {model_info.get('features', {}).get('count', 'Unknown')}")
            print(f"        Created: {model_info.get('created_at', 'Unknown')}")
            
            # Load existing model
            model_loaded = self.load_existing_model(symbol)
            if not model_loaded:
                print("[WARNING] Failed to load existing model, will train new one")
                use_existing_model = False
        
        if not use_existing_model:
            print(f"[TRAINING] {'Retraining' if force_retrain else 'Training new'} model for {symbol}")
            
            # 1. Market Context Analysis
            market_context = self.analyze_market_context(symbol, train_start, current_date)
            
            print()
            # 2. Prepare Training Data
            X_train, y_train = self.prepare_training_data(symbol, train_start, train_end)
            
            print()
            # 3. Train Model (automatically saves)
            self.train_model(X_train, y_train)
        else:
            print(f"[REUSE] Using existing model for {symbol}")
            # Still analyze market context for reporting
            market_context = self.analyze_market_context(symbol, train_start, current_date)
        
        print()
        # 4. Generate Predictions (uses prediction service if available)
        predictions, signal_analysis = self.generate_predictions(symbol, test_start, current_date)
        
        print()
        # 5. Backtest Strategy
        performance = self.backtest_strategy(symbol, test_start, current_date, predictions)
        
        print()
        # 6. Feature Importance Analysis
        feature_analysis = self.analyze_feature_importance()
        
        # Compile complete results
        results = {
            'symbol': symbol,
            'model_management': {
                'existing_model_used': use_existing_model,
                'force_retrain': force_retrain,
                'model_version': self.get_model_info(symbol).get('version', 'New') if use_existing_model else 'New'
            },
            'dates': {
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': current_date
            },
            'market_context': market_context,
            'signal_analysis': signal_analysis,
            'performance': performance,
            'feature_analysis': feature_analysis,
            'model_type': type(self.model).__name__
        }
        
        self._print_summary_report(results)
        return results
    
    def _generate_labels(self, prices: pd.DataFrame, market_impact: float = None) -> np.ndarray:
        """Generate trading labels based on future returns"""
        if market_impact is None:
            market_impact = self.config.get_market_impact()
        
        labels = np.zeros(prices.shape[0] - 3)
        for i in range(prices.shape[0] - 3):
            ret = (prices.values[i+3] - prices.values[i])
            if ret / prices.values[i] < (-1 * market_impact - 0.02):
                labels[i] = -1  # SELL
            elif ret / prices.values[i] > (market_impact + 0.02):
                labels[i] = 1   # BUY
        return labels
    
    def _create_trades_dataframe(self, prices: pd.DataFrame, symbol: str, 
                               predictions: np.ndarray) -> pd.DataFrame:
        """Create trades DataFrame from predictions"""
        trades = pd.DataFrame(index=prices.index)
        trades[symbol] = 0
        shares = 0
        shares_per_trade = self.config.get_shares_per_trade()
        
        for i in range(len(predictions)):
            if shares == 0 and predictions[i] == 1:  # BUY signal
                trades.iloc[i, 0] = shares_per_trade
                shares = shares_per_trade
            elif shares > 0 and predictions[i] == -1:  # SELL signal
                trades.iloc[i, 0] = -shares_per_trade
                shares = 0
        
        return trades
    
    def _print_summary_report(self, results: Dict) -> None:
        """Print a comprehensive summary report"""
        print()
        print("=" * 60)
        print("[COMPLETE] ENHANCED TRADING SIMULATION COMPLETE")
        print("=" * 60)
        
        # Performance Summary
        perf = results['performance']
        print(f"[PERFORMANCE] SUMMARY")
        print(f"   Strategy Return: {perf.get('strategy_return', 0):.1%}")
        print(f"   Buy-Hold Return: {perf.get('buy_hold_return', 0):.1%}")
        print(f"   Outperformance: {perf.get('outperformance', 0):.1%}")
        print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
        
        if perf.get('final_value'):
            print(f"   Portfolio: ${perf['starting_value']:,} -> ${perf['final_value']:,.0f}")
            print(f"   Profit/Loss: ${perf['profit_loss']:,.0f}")
        
        # Market Context
        mc = results['market_context']
        print(f"\n[MARKET] CONTEXT")
        print(f"   Beta vs SPY: {mc['beta_spy']:.2f}")
        print(f"   Beta vs QQQ: {mc['beta_qqq']:.2f}")
        
        # Feature Analysis
        fa = results['feature_analysis']
        if fa:
            print(f"\n[FEATURES] ANALYSIS")
            print(f"   Market features in top 10: {fa['market_dominance']}/10")
            print(f"   Total features used: {len(fa['feature_importance'])}")
        
        # Signal Analysis
        sa = results['signal_analysis']
        print(f"\n[SIGNALS] ANALYSIS")
        print(f"   Buy signals: {sa['buy_signals']} ({sa['buy_percentage']:.1f}%)")
        print(f"   Sell signals: {sa['sell_signals']}")
        print(f"   Total predictions: {sa['total_predictions']}")
        
        print("=" * 60)
        print("[SUCCESS] Enhanced market indicators successfully integrated!")


if __name__ == "__main__":
    # Example usage
    strategy = EnhancedTradingStrategy()
    results = strategy.run_complete_simulation("NVDA", months_back=6)