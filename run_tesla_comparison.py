#!/usr/bin/env python3
"""
Tesla Model Comparison Runner
Quick script to run the comprehensive Tesla model comparison
"""

import sys
import os
import datetime as dt

# Add the tests directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.join(current_dir, 'tests')
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

def run_tesla_comparison():
    """Run Tesla model comparison test"""
    print("🚀 Starting Tesla Model Comparison...")
    print(f"📅 Date: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        from test_tesla_model_comparison import run_tesla_model_comparison
        
        # Run the comparison
        success = run_tesla_model_comparison()
        
        if success:
            print("\n" + "="*80)
            print("🎉 TESLA MODEL COMPARISON COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("📊 Key Outcomes:")
            print("   ✅ LightGBM and XGBoost models trained and compared")
            print("   ✅ Performance metrics analyzed")
            print("   ✅ Feature importance evaluated") 
            print("   ✅ Backtesting comparison performed")
            print("   ✅ Comprehensive report generated")
            print("="*80)
            return True
        else:
            print("\n❌ Some tests failed during the comparison.")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("   - LightGBM: pip install lightgbm")
        print("   - XGBoost: pip install xgboost")
        return False
    except Exception as e:
        print(f"❌ Error during Tesla comparison: {e}")
        return False

if __name__ == "__main__":
    success = run_tesla_comparison()
    sys.exit(0 if success else 1)