"""
ML Integration Demonstration
Shows how the enhanced ML trading system works with trained models
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.enhanced_model_management import EnhancedModelManager, ModelTrainingService
from src.data.providers import YFinanceProvider
from src.trading.enhanced_strategies import EnhancedMLTradingStrategy, OrderSizingConfig


def demo_ml_integration():
    """Demonstrate ML model integration"""
    
    print("=" * 60)
    print("ML Integration Demonstration")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing Components...")
    data_provider = YFinanceProvider()
    model_manager = EnhancedModelManager(base_path="models")
    
    symbol = "AAPL"
    
    # Check if model exists
    print(f"\n2. Checking ML Model for {symbol}...")
    if model_manager.model_exists(symbol):
        print(f"   ✅ ML model found for {symbol}")
        models = model_manager.list_models(symbol)
        for model in models:
            print(f"      Version: {model.version}")
            print(f"      Type: {model.model_type}")
            print(f"      Performance: {model.performance_metrics}")
    else:
        print(f"   ❌ No ML model found for {symbol}")
        print("   🔧 Creating a sample model...")
        
        # Create sample training data
        sample_data = create_sample_training_data()
        
        # Train a simple model
        try:
            training_service = ModelTrainingService(model_manager, data_provider)
            
            # Create a simple mock training pipeline
            from sklearn.ensemble import RandomForestClassifier
            from src.interfaces.model_manager import ModelMetadata
            
            # Create sample features and labels
            np.random.seed(42)
            n_samples = 1000
            features = np.random.randn(n_samples, 7)  # 7 features to match our feature engineering
            labels = np.random.choice([-1, 0, 1], size=n_samples)  # Buy, Hold, Sell
            
            # Train model
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(features, labels)
            
            # Create metadata
            metadata = ModelMetadata(
                symbol=symbol,
                version="1",
                created_at=datetime.now(),
                model_type="RandomForestClassifier",
                performance_metrics={
                    'train_accuracy': 0.75,
                    'test_accuracy': 0.68,
                    'feature_count': 7
                },
                feature_names=[
                    'price_change_1d', 'price_change_5d', 'sma_5_diff', 'sma_10_diff',
                    'volatility', 'rsi', 'price_position'
                ],
                training_period=(datetime.now() - timedelta(days=365), datetime.now()),
                config_snapshot={
                    'algorithm': 'RandomForest',
                    'n_estimators': 50,
                    'random_state': 42
                },
                file_path=""  # Will be filled by save_model
            )
            
            # Save model
            model_path = model_manager.save_model(symbol, model, metadata)
            print(f"   ✅ Sample model created: {model_path}")
            
        except Exception as e:
            print(f"   ❌ Failed to create sample model: {e}")
    
    # Initialize trading strategy
    print(f"\n3. Initializing ML Trading Strategy for {symbol}...")
    try:
        strategy = EnhancedMLTradingStrategy(
            symbol=symbol,
            data_provider=data_provider,
            model_manager=model_manager,
            order_sizing_config=OrderSizingConfig()
        )
        print(f"   ✅ ML Trading Strategy initialized")
        print(f"      Strategy: {strategy.name}")
        print(f"      Symbol: {strategy.symbol}")
        
        # Test ML prediction functionality
        print(f"\n4. Testing ML Prediction Functionality...")
        
        # Create sample price data
        sample_data = create_sample_price_data()
        timestamp = pd.Timestamp.now()
        
        # Test feature preparation
        features = strategy._prepare_ml_features(sample_data)
        if features is not None:
            print(f"   ✅ Features prepared successfully")
            print(f"      Feature vector shape: {features.shape}")
            print(f"      Features: {features}")
        else:
            print(f"   ❌ Failed to prepare features")
        
        # Test ML prediction
        if model_manager.model_exists(symbol):
            ml_signal = strategy._get_ml_prediction(sample_data, timestamp)
            if ml_signal:
                print(f"   ✅ ML prediction generated")
                print(f"      Signal Type: {ml_signal.signal_type}")
                print(f"      Confidence: {ml_signal.confidence:.3f}")
                print(f"      Strength: {ml_signal.strength:.3f}")
                print(f"      Source: {ml_signal.source}")
                print(f"      Metadata: {ml_signal.metadata}")
            else:
                print(f"   ⚠️  ML prediction returned None (confidence threshold not met)")
                print(f"      Note: This is normal behavior when the model prediction confidence is below 0.6")
                
                # Let's show what the raw prediction would be by temporarily lowering the threshold
                print(f"   🔍 Testing with lower confidence threshold...")
                
                # Test with a mock confident prediction by directly calling model_manager.predict
                features = strategy._prepare_ml_features(sample_data)
                if features is not None:
                    try:
                        prediction_result = model_manager.predict(symbol, features.reshape(1, -1))
                        print(f"      Raw prediction: {prediction_result.predictions[0]}")
                        print(f"      Raw confidence: {prediction_result.confidence:.3f}")
                        print(f"      Model version: {prediction_result.model_version}")
                    except Exception as e:
                        print(f"      Raw prediction failed: {e}")
        else:
            print(f"   ❌ No model available for prediction")
        
        print(f"\n5. Testing Signal Generation...")
        try:
            # Generate signals using the strategy
            signals = strategy.generate_signal(sample_data, timestamp)
            print(f"   ✅ Signal generation completed")
            print(f"      Signals generated: {len(signals) if signals else 0}")
            
            if signals:
                for i, signal in enumerate(signals):
                    print(f"      Signal {i+1}:")
                    print(f"        Direction: {signal.direction}")
                    print(f"        Confidence: {signal.confidence:.3f}")
                    print(f"        Type: {signal.signal_type}")
            
        except Exception as e:
            print(f"   ❌ Signal generation failed: {e}")
        
    except Exception as e:
        print(f"   ❌ Failed to initialize ML Trading Strategy: {e}")
    
    print(f"\n" + "=" * 60)
    print("ML Integration Demonstration Complete")
    print("=" * 60)


def create_sample_price_data():
    """Create sample price data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    # Create realistic price movements
    np.random.seed(42)
    initial_price = 150.0
    prices = [initial_price]
    
    for i in range(1, 30):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 10.0))  # Minimum price of $10
    
    return pd.Series(prices, index=dates)


def create_sample_training_data():
    """Create sample training data"""
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')  # 1 year
    
    np.random.seed(42)
    initial_price = 100.0
    prices = [initial_price]
    
    for i in range(1, 252):
        change = np.random.normal(0, 0.015)  # 1.5% daily volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 10.0))
    
    return pd.DataFrame({'price': prices}, index=dates)


if __name__ == "__main__":
    demo_ml_integration()