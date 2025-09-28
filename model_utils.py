"""
Model Management Utilities
Command-line utilities for managing trading models
"""

import argparse
import sys
from model_management import ModelManager, ModelPredictionService
from enhanced_strategy import EnhancedTradingStrategy
import datetime as dt


def list_models():
    """List all available models"""
    manager = ModelManager()
    models = manager.list_available_models()
    
    if not models:
        print("No trained models found.")
        return
    
    print("\n📊 Available Trading Models")
    print("=" * 60)
    
    for symbol, info in models.items():
        print(f"🔹 {symbol}")
        print(f"   Version: {info['version']}")
        print(f"   Created: {info['created_at']}")
        print(f"   Model Type: {info['model_type']}")
        print(f"   Features: {info['features']}")
        print(f"   Training Accuracy: {info['training_accuracy']}")
        print()


def model_info(symbol):
    """Show detailed information about a specific model"""
    manager = ModelManager()
    
    if not manager.model_exists(symbol):
        print(f"❌ No model found for symbol: {symbol}")
        return
    
    info = manager.get_model_info(symbol)
    history = manager.get_model_performance_history(symbol)
    
    print(f"\n📈 Model Information for {symbol}")
    print("=" * 50)
    print(f"Current Version: {info.get('version', 'Unknown')}")
    print(f"Model Type: {info.get('model_type', 'Unknown')}")
    print(f"Created: {info.get('created_at', 'Unknown')}")
    print(f"Training Accuracy: {info.get('performance', {}).get('training_accuracy', 'N/A'):.3f}")
    print(f"Features Count: {info.get('features', {}).get('count', 'Unknown')}")
    print(f"Training Samples: {info.get('training_info', {}).get('training_samples', 'Unknown')}")
    
    if info.get('features', {}).get('names'):
        print(f"\nFeatures Used:")
        for i, feature in enumerate(info['features']['names'][:10], 1):
            print(f"  {i:2d}. {feature}")
        if len(info['features']['names']) > 10:
            print(f"  ... and {len(info['features']['names']) - 10} more features")
    
    print(f"\nPerformance History:")
    print(f"Total Versions: {history['total_versions']}")
    
    for version_info in history['history'][-5:]:  # Show last 5 versions
        version = version_info.get('version', 'Unknown')
        accuracy = version_info.get('training_accuracy', 'N/A')
        timestamp = version_info.get('timestamp', 'Unknown')
        print(f"  v{version}: {accuracy:.3f} accuracy ({timestamp})")


def cleanup_models(symbol=None, keep_versions=5):
    """Clean up old model versions"""
    manager = ModelManager()
    
    if symbol:
        if not manager.model_exists(symbol):
            print(f"❌ No models found for symbol: {symbol}")
            return
        
        print(f"🧹 Cleaning up old versions for {symbol} (keeping {keep_versions} versions)")
        manager.cleanup_old_models(symbol, keep_versions)
        print("✅ Cleanup completed")
    else:
        models = manager.list_available_models()
        print(f"🧹 Cleaning up all models (keeping {keep_versions} versions each)")
        
        for symbol in models.keys():
            manager.cleanup_old_models(symbol, keep_versions)
            
        print("✅ Cleanup completed for all symbols")


def retrain_model(symbol, months_back=6):
    """Retrain a model for a specific symbol"""
    print(f"🔄 Retraining model for {symbol}")
    
    strategy = EnhancedTradingStrategy()
    results = strategy.run_complete_simulation(symbol, months_back, force_retrain=True)
    
    print(f"\n✅ Retraining completed for {symbol}")
    print(f"   New training accuracy: {results['performance']['training_accuracy']:.3f}")
    print(f"   Strategy performance: {results['performance']['strategy_return']:.1%}")


def train_new_model(symbol, months_back=6):
    """Train a new model for a symbol"""
    manager = ModelManager()
    
    if manager.model_exists(symbol):
        response = input(f"Model for {symbol} already exists. Retrain? (y/N): ")
        if response.lower() != 'y':
            print("Training cancelled")
            return
    
    print(f"🎯 Training new model for {symbol}")
    
    strategy = EnhancedTradingStrategy()
    results = strategy.run_complete_simulation(symbol, months_back, force_retrain=True)
    
    print(f"\n✅ Training completed for {symbol}")
    print(f"   Training accuracy: {results['performance']['training_accuracy']:.3f}")
    print(f"   Strategy performance: {results['performance']['strategy_return']:.1%}")


def test_prediction(symbol, sample_size=10):
    """Test prediction service for a symbol"""
    manager = ModelManager()
    
    if not manager.model_exists(symbol):
        print(f"❌ No model found for symbol: {symbol}")
        return
    
    print(f"🧪 Testing prediction service for {symbol}")
    
    # Get some sample data for testing
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=30)
    
    from market_indicators import get_enhanced_features
    
    try:
        # Get features for testing
        features = get_enhanced_features(symbol, start_date, end_date, train=False)
        
        if len(features) < sample_size:
            sample_size = len(features)
            
        test_features = features[-sample_size:]
        
        # Test prediction service
        service = ModelPredictionService(manager)
        predictions, pred_info = service.predict(symbol, test_features)
        summary = service.get_prediction_summary(symbol, predictions)
        
        print(f"✅ Prediction test successful!")
        print(f"   Model Version: {pred_info.get('model_version', 'Unknown')}")
        print(f"   Predictions: {len(predictions)}")
        print(f"   BUY signals: {summary['buy_signals']} ({summary['buy_percentage']:.1f}%)")
        print(f"   SELL signals: {summary['sell_signals']}")
        print(f"   HOLD signals: {summary['hold_signals']}")
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")


def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(description="Trading Model Management Utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List models command
    subparsers.add_parser('list', help='List all available models')
    
    # Model info command
    info_parser = subparsers.add_parser('info', help='Show detailed model information')
    info_parser.add_argument('symbol', help='Trading symbol (e.g., TSLA)')
    
    # Train new model command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('symbol', help='Trading symbol (e.g., TSLA)')
    train_parser.add_argument('--months', type=int, default=6, help='Months of data for simulation (default: 6)')
    
    # Retrain model command
    retrain_parser = subparsers.add_parser('retrain', help='Retrain existing model')
    retrain_parser.add_argument('symbol', help='Trading symbol (e.g., TSLA)')
    retrain_parser.add_argument('--months', type=int, default=6, help='Months of data for simulation (default: 6)')
    
    # Test prediction command
    test_parser = subparsers.add_parser('test', help='Test prediction service')
    test_parser.add_argument('symbol', help='Trading symbol (e.g., TSLA)')
    test_parser.add_argument('--samples', type=int, default=10, help='Number of samples to test (default: 10)')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old model versions')
    cleanup_parser.add_argument('--symbol', help='Specific symbol to clean up (optional)')
    cleanup_parser.add_argument('--keep', type=int, default=5, help='Number of versions to keep (default: 5)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'list':
            list_models()
        elif args.command == 'info':
            model_info(args.symbol.upper())
        elif args.command == 'train':
            train_new_model(args.symbol.upper(), args.months)
        elif args.command == 'retrain':
            retrain_model(args.symbol.upper(), args.months)
        elif args.command == 'test':
            test_prediction(args.symbol.upper(), args.samples)
        elif args.command == 'cleanup':
            symbol = args.symbol.upper() if args.symbol else None
            cleanup_models(symbol, args.keep)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()