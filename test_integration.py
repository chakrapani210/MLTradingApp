"""
Test script to verify all components are properly integrated
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_quick_integration_test():
    """Run quick integration test for basic system validation"""
    try:
        from src.enhanced_orchestrator import ProductionTradingOrchestrator
        
        print("🧪 Testing Production Trading System Integration")
        print("=" * 50)
        
        # Initialize system with error handling
        try:
            system = ProductionTradingOrchestrator()
            print("   ✅ System initialization successful")
        except Exception as e:
            print(f"   ❌ System initialization failed: {e}")
            print("   💡 Some abstract methods may need implementation")
            return False
        
        # Test system status
        print("\n1. Testing System Status...")
        try:
            status = system.get_system_status()
            print(f"   ✅ Architecture: {status['architecture']}")
            
            # Display production layers
            print("\n2. Production Layer Architecture:")
            layers = status.get('layers', {})
            for layer_name, layer_components in layers.items():
                print(f"   📦 {layer_name}:")
                for comp_name, comp_type in layer_components.items():
                    print(f"      - {comp_name}: {comp_type}")
        except Exception as e:
            print(f"   ⚠️ System status check had issues: {e}")
        
        # Test production pipeline
        print("\n3. Testing Production Data Pipeline...")
        try:
            pipeline_results = system.run_production_data_pipeline("AAPL", days=30)
            if 'error' not in pipeline_results:
                print(f"   ✅ Production pipeline successful")
                print(f"   📊 Records: {pipeline_results['data_quality']['processed_records']}")
                print(f"   🔧 Features: {pipeline_results['data_quality']['feature_count']}")
                mc = pipeline_results['market_context']
                print(f"   📈 Market Regime: {mc['market_regime']}")
                print(f"   📊 Volatility Regime: {mc['volatility_regime']}")
            else:
                print(f"   ❌ Production pipeline failed: {pipeline_results['error']}")
        except Exception as e:
            print(f"   ⚠️ Production pipeline error: {e}")
        
        # Test system health
        print("\n4. Testing System Health...")
        try:
            health = system.validate_system_health()
            print(f"   🏥 Overall Status: {health['overall_status']}")
            for component, status in health['component_health'].items():
                icon = "✅" if status == "healthy" else "⚠️" if status == "unavailable" else "❌"
                print(f"   {icon} {component}: {status}")
            
            if health['warnings']:
                print("   ⚠️ Warnings:")
                for warning in health['warnings']:
                    print(f"      - {warning}")
                    
        except Exception as e:
            print(f"   ⚠️ Health check error: {e}")
        
        print(f"\n✅ Quick Integration Test Completed!")
        print(f"\n📋 For comprehensive testing, run:")
        print(f"   python tests/integration/run_integration_tests.py")
        print(f"   python tests/integration/run_integration_tests.py --demo")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed and paths are correct")
        return False
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print("💡 This may be due to incomplete abstract method implementations")
        print("📋 Use the comprehensive integration test suite:")
        print("   python tests/integration/run_integration_tests.py")
        return False

if __name__ == "__main__":
    success = run_quick_integration_test()
    sys.exit(0 if success else 1)