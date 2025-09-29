"""
Simplified Main Entry Point for ML Trading System
Uses EnhancedTradingSystemOrchestrator for all operations
"""

import os
import sys
import traceback
import webbrowser
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the production orchestrator
from src.enhanced_orchestrator import ProductionTradingOrchestrator


def main():
    """Enhanced main entry point for the trading system"""
    print("=" * 60)
    print("Production ML Trading System - Modular Architecture")
    print("=" * 60)
    
    try:
        # Initialize the production trading system
        trading_system = ProductionTradingOrchestrator()
        
        # Display system status
        print("\nSystem Status:")
        status = trading_system.get_system_status()
        print(f"  Architecture: {status['architecture']}")
        print(f"  Available Models: {status['runtime_state']['available_models']}")
        print(f"  Starting Capital: ${status['configuration']['starting_capital']:,.0f}")
        print(f"  Commission Rate: {status['configuration']['commission_rate']:.1%}")
        
        # Display modular layers
        print("\nProduction Layers:")
        layers = status.get('layers', {})
        for layer_name, layer_components in layers.items():
            print(f"  {layer_name}:")
            for comp_name, comp_type in layer_components.items():
                print(f"    {comp_name}: {comp_type}")
        
        # Example operations
        print("\n" + "=" * 40)
        print("Running Example Operations")
        print("=" * 40)
        
        # Store chart paths for display later
        chart_paths = []
        
        # 1. Create trading charts for multiple symbols
        symbols_to_chart = ["AAPL", "TSLA", "NVDA"]
        print("\n1. Creating Trading Charts...")
        for symbol in symbols_to_chart:
            print(f"  📊 Creating chart for {symbol}...")
            chart_path = trading_system.create_trading_chart(symbol)
            if chart_path and not chart_path.startswith("Error"):
                chart_paths.append((symbol, chart_path))
                print(f"    ✅ Chart created: {os.path.basename(chart_path)}")
            else:
                print(f"    ❌ Failed to create chart for {symbol}")
        
        # 2. Run production data pipeline (using enhanced components only)
        print("\n2. Running Production Data Pipeline...")
        symbol = "AAPL"
        pipeline_results = trading_system.run_production_data_pipeline(symbol, days=180)
        if 'error' not in pipeline_results:
            print(f"  📊 Data Quality: {pipeline_results['data_quality']['processed_records']} records")
            mc = pipeline_results['market_context']
            print(f"  � SPY Correlation: {mc['spy_correlation']:.3f}")
            print(f"  � Market Regime: {mc['market_regime']}")
            print(f"  📊 Volatility Regime: {mc['volatility_regime']}")
            print(f"  🔢 ML Features: {pipeline_results['data_quality']['feature_count']}")
        else:
            print(f"  ❌ Pipeline failed: {pipeline_results['error']}")
        
        # 3. Run enhanced market context analysis
        print("\n3. Running Enhanced Market Context Analysis...")
        market_analysis = trading_system.analyze_market_context(symbol)
        if 'error' not in market_analysis:
            mc = market_analysis.get('market_context', {})
            print(f"  📈 SPY Correlation: {mc.get('spy_correlation', 0):.3f}")
            print(f"  📊 Market Regime: {mc.get('market_regime', 'Unknown')}")
            print(f"  🔢 Enhanced Features: {market_analysis.get('enhanced_features', {}).get('feature_count', 0)}")
        
        # 4. Train ML model
        print("\n4. Training ML Model...")
        symbol = "AAPL"  # Define symbol for subsequent operations
        ml_results = trading_system.train_ml_model(symbol)
        if ml_results.get('success', False):
            print(f"  🤖 Model trained successfully")
            print(f"  📊 Train Accuracy: {ml_results.get('train_accuracy', 0):.3f}")
            print(f"  🎯 Test Accuracy: {ml_results.get('test_accuracy', 0):.3f}")
        
        # 5. Run comprehensive backtest
        print("\n5. Running Comprehensive Backtest...")
        backtest_results = trading_system.run_comprehensive_backtest(symbol)
        if 'error' not in backtest_results:
            perf = backtest_results.get('performance', {})
            print(f"  💰 Total Return: {perf.get('total_return', 0):.1%}")
            print(f"  📈 Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
            print(f"  📉 Max Drawdown: {perf.get('max_drawdown', 0):.1%}")
        
        # 6. Display charts in browser
        if chart_paths:
            print("\n" + "=" * 40)
            print("📈 Opening Trading Charts in Browser")
            print("=" * 40)
            
            for symbol, chart_path in chart_paths:
                try:
                    full_path = os.path.abspath(chart_path)
                    print(f"🌐 Opening {symbol} chart: {os.path.basename(chart_path)}")
                    webbrowser.open(f"file:///{full_path}")
                    time.sleep(1)  # Small delay between opening charts
                except Exception as e:
                    print(f"❌ Could not open chart for {symbol}: {e}")
            
            print(f"\n📊 Successfully opened {len(chart_paths)} chart(s) in your browser!")
        else:
            print("\n⚠️ No charts were generated to display.")
        
        print("\n" + "=" * 60)
        print("Production Trading System Operations Completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)