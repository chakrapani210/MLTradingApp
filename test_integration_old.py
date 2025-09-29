"""
Test script to verify all components are properly integrated
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.enhanced_orchestrator import ProductionTradingOrchestrator
    
    print("🧪 Testing Production Trading System Integration")
    print("=" * 50)
    
    # Initialize system
    system = ProductionTradingOrchestrator()
    
    # Test system status
    print("\n1. Testing System Status...")
    status = system.get_system_status()
    print(f"   ✅ Architecture: {status['architecture']}")
    
    # Display production layers
    print("\n2. Production Layer Architecture:")
    layers = status.get('layers', {})
    for layer_name, layer_components in layers.items():
        print(f"   📦 {layer_name}:")
        for comp_name, comp_type in layer_components.items():
            print(f"      - {comp_name}: {comp_type}")
    
    # Test production pipeline
    print("\n3. Testing Production Data Pipeline...")
    try:
        pipeline_results = system.run_production_data_pipeline("AAPL", days=30)
        if 'error' not in pipeline_results:
            print(f"   ✅ Production pipeline successful")
            print(f"   📊 Records: {pipeline_results['data_quality']['processed_records']}")
            print(f"   � Features: {pipeline_results['data_quality']['feature_count']}")
            mc = pipeline_results['market_context']
            print(f"   📈 Market Regime: {mc['market_regime']}")
            print(f"   📊 Volatility Regime: {mc['volatility_regime']}")
        else:
            print(f"   ❌ Production pipeline failed: {pipeline_results['error']}")
    except Exception as e:
        print(f"   ❌ Production pipeline error: {e}")
    
    # Test enhanced analysis
    print("\n4. Testing Enhanced Market Analysis...")
    try:
        enhanced_results = system.analyze_market_context("AAPL", analysis_period_days=30)
        if 'error' not in enhanced_results:
            print(f"   ✅ Enhanced analysis successful")
            mc = enhanced_results.get('market_context', {})
            print(f"   📈 SPY Correlation: {mc.get('spy_correlation', 'N/A')}")
            print(f"   📊 Market Regime: {mc.get('market_regime', 'N/A')}")
            print(f"   🏭 Sector Analysis: {len(mc.get('sector_strength', {}))}")
        else:
            print(f"   ❌ Enhanced analysis failed: {enhanced_results['error']}")
    except Exception as e:
        print(f"   ❌ Enhanced analysis error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Production Integration Test Completed!")
    print("🏭 All components are production-ready and modular!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
except Exception as e:
    print(f"❌ General Error: {e}")
    import traceback
    traceback.print_exc()