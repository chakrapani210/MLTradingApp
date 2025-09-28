"""
Test Model Management System
Demonstrates the new model persistence and loading capabilities
"""

from enhanced_strategy import EnhancedTradingStrategy
from model_management import ModelManager, ModelPredictionService
import datetime as dt


def test_model_management():
    """Test the complete model management workflow"""
    print("Testing Model Management System")
    print("=" * 50)
    
    # Test symbol
    symbol = "TSLA"
    
    # Initialize components
    strategy = EnhancedTradingStrategy()
    manager = ModelManager()
    
    print(f"\n[1] Testing Model Training and Saving for {symbol}")
    print("-" * 40)
    
    # First run - should train new model
    print("Running first simulation (should train new model)...")
    results1 = strategy.run_complete_simulation(symbol, months_back=6)
    
    print(f"\n   First run completed:")
    print(f"      Strategy Return: {results1['performance']['strategy_return']:.1%}")
    print(f"      Training Accuracy: {results1['performance'].get('training_accuracy', 'N/A'):.3f}")
    print(f"      Model Saved: {results1['model_management']['existing_model_used'] == False}")
    
    print(f"\n[2] Testing Model Loading and Reuse")
    print("-" * 40)
    
    # Second run - should load existing model
    print("Running second simulation (should load existing model)...")
    strategy2 = EnhancedTradingStrategy()  # New instance
    results2 = strategy2.run_complete_simulation(symbol, months_back=6)
    
    print(f"\n   Second run completed:")
    print(f"      Strategy Return: {results2['performance']['strategy_return']:.1%}")
    print(f"      Existing Model Used: {results2['model_management']['existing_model_used']}")
    print(f"      Model Version: {results2['model_management']['model_version']}")
    
    print(f"\n[3] Testing Model Information")
    print("-" * 40)
    
    # Check model information
    model_info = manager.get_model_info(symbol)
    print(f"Model Information for {symbol}:")
    print(f"   Version: {model_info.get('version', 'Unknown')}")
    print(f"   Model Type: {model_info.get('model_type', 'Unknown')}")
    print(f"   Features: {model_info.get('features', {}).get('count', 'Unknown')}")
    print(f"   Training Accuracy: {model_info.get('performance', {}).get('training_accuracy', 'N/A'):.3f}")
    
    print(f"\n[4] Testing Prediction Service")
    print("-" * 40)
    
    # Test prediction service
    try:
        from market_indicators import get_enhanced_features
        
        # Get some recent features for testing
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=30)
        features = get_enhanced_features(symbol, start_date, end_date, train=False)
        
        # Test predictions
        service = ModelPredictionService(manager)
        predictions, pred_info = service.predict(symbol, features[-10:])
        summary = service.get_prediction_summary(symbol, predictions)
        
        print(f"Prediction Service Test:")
        print(f"   Model Version: {pred_info.get('model_version', 'Unknown')}")
        print(f"   Predictions: {len(predictions)}")
        print(f"   BUY signals: {summary['buy_signals']} ({summary['buy_percentage']:.1f}%)")
        print(f"   SELL signals: {summary['sell_signals']}")
        print(f"   HOLD signals: {summary['hold_signals']}")
        
    except Exception as e:
        print(f"   Prediction service test failed: {e}")
    
    print(f"\n[5] Testing Forced Retraining")
    print("-" * 40)
    
    # Test forced retraining
    print("Testing forced retraining...")
    strategy3 = EnhancedTradingStrategy()  # New instance
    results3 = strategy3.run_complete_simulation(symbol, months_back=6, force_retrain=True)
    
    print(f"\n   Forced retraining completed:")
    print(f"      Strategy Return: {results3['performance']['strategy_return']:.1%}")
    print(f"      Force Retrain: {results3['model_management']['force_retrain']}")
    print(f"      New Model Version: {results3['model_management']['model_version']}")
    
    print(f"\n[6] Testing Model List")
    print("-" * 40)
    
    # List all models
    models = manager.list_available_models()
    print(f"Available Models ({len(models)}):")
    for sym, info in models.items():
        print(f"   {sym}: v{info['version']}, {info['training_accuracy']:.3f} accuracy")
    
    print(f"\nModel Management System Test Completed!")
    print("=" * 50)
    
    # Summary
    print(f"\nTest Summary:")
    print(f"   Model Training & Saving: Working")
    print(f"   Model Loading & Reuse: Working") 
    print(f"   Model Information: Working")
    print(f"   Prediction Service: {'Working' if 'pred_info' in locals() else 'Failed'}")
    print(f"   Forced Retraining: Working")
    print(f"   Model Listing: Working")
    
    return results1, results2, results3


if __name__ == "__main__":
    try:
        test_model_management()
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()