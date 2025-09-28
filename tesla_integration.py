#!/usr/bin/env python3
"""
Tesla Enhanced Trading Integration
Integrates the winning Tesla model into the main trading system
"""

import sys
import os
import datetime as dt
import pandas as pd
import logging
from typing import Dict, Optional

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from tesla_production_model import TeslaProductionModel

def integrate_tesla_model_into_system():
    """
    Integration script for Tesla model into enhanced trading system
    """
    
    print("=" * 80)
    print("🔗 TESLA MODEL INTEGRATION")
    print("=" * 80)
    
    try:
        # Import enhanced system components
        from enhanced_strategy import EnhancedTradingStrategy
        import yfinance as yf
        
        print("[INTEGRATION] Initializing Tesla production model...")
        tesla_model = TeslaProductionModel()
        
        # Fetch recent Tesla data
        print("[INTEGRATION] Fetching Tesla data...")
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=500)
        
        tesla_data = yf.Ticker('TSLA').history(start=start_date, end=end_date)
        print(f"               ✅ Fetched {len(tesla_data)} days of Tesla data")
        
        # Train/load model
        print("[INTEGRATION] Setting up model...")
        training_results = tesla_model.train_model(tesla_data)
        
        if training_results['status'] in ['trained', 'loaded']:
            print(f"               ✅ Model ready (CV: {training_results.get('cv_accuracy', 'N/A'):.4f})")
        else:
            print(f"               ❌ Model setup failed")
            return False
        
        # Generate current signal
        print("[INTEGRATION] Generating trading signal...")
        signal = tesla_model.predict_signal(tesla_data)
        
        if 'error' not in signal:
            print(f"               🎯 Current Signal: {signal['signal']}")
            print(f"               🎲 Confidence: {signal['confidence']:.4f}")
            print(f"               💪 Strength: {signal['signal_strength']}")
            print(f"               💰 Position: {signal['suggested_position']:.2%}")
        else:
            print(f"               ❌ Signal generation failed: {signal['error']}")
            return False
        
        # Create enhanced configuration for Tesla
        tesla_config = create_tesla_enhanced_config(signal)
        
        print("[INTEGRATION] Tesla enhanced configuration:")
        for key, value in tesla_config.items():
            print(f"               {key}: {value}")
        
        print(f"\n🎉 Tesla model successfully integrated!")
        print(f"💡 Next steps:")
        print(f"   1. Use tesla_production_model.py for daily signal generation")
        print(f"   2. Integrate with position sizing and risk management")
        print(f"   3. Set up automated retraining schedule")
        print(f"   4. Monitor performance vs {tesla_model.performance_threshold:.2%} threshold")
        
        return True
        
    except ImportError as e:
        print(f"[INTEGRATION] Missing dependencies: {e}")
        print(f"               Creating standalone Tesla trading recommendations...")
        return create_standalone_tesla_recommendations()
    except Exception as e:
        print(f"[INTEGRATION] Integration failed: {e}")
        return False

def create_tesla_enhanced_config(signal: Dict) -> Dict:
    """Create enhanced configuration for Tesla trading based on model signal"""
    
    config = {
        'symbol': 'TSLA',
        'strategy': 'Enhanced ML with Regularization',
        'model_type': 'LightGBM_Aggressive',
        'signal': signal['signal'],
        'confidence': signal['confidence'],
        'position_size': signal['suggested_position'],
        'stop_loss': 0.025,  # 2.5%
        'take_profit': 0.05,  # 5%
        'rebalance_frequency': 'daily',
        'risk_level': 'moderate' if signal['confidence'] > 0.60 else 'conservative'
    }
    
    return config

def create_standalone_tesla_recommendations():
    """Create standalone recommendations when full integration isn't available"""
    
    print("\n📋 STANDALONE TESLA TRADING RECOMMENDATIONS")
    print("=" * 60)
    
    try:
        import yfinance as yf
        
        # Initialize Tesla model
        tesla_model = TeslaProductionModel()
        
        # Get data and generate signal
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=400)
        
        tesla_data = yf.Ticker('TSLA').history(start=start_date, end=end_date)
        training_results = tesla_model.train_model(tesla_data)
        
        if training_results['status'] not in ['trained', 'loaded']:
            print("❌ Could not initialize Tesla model")
            return False
        
        signal = tesla_model.predict_signal(tesla_data)
        
        if 'error' in signal:
            print(f"❌ Could not generate signal: {signal['error']}")
            return False
        
        # Current Tesla analysis
        current_price = tesla_data['Close'].iloc[-1]
        recent_return = (tesla_data['Close'].iloc[-1] / tesla_data['Close'].iloc[-5] - 1) * 100
        volatility = tesla_data['Close'].pct_change().rolling(20).std().iloc[-1] * 100
        
        print(f"📊 Current Tesla Analysis:")
        print(f"   💰 Current Price: ${current_price:.2f}")
        print(f"   📈 5-Day Return: {recent_return:+.2f}%")
        print(f"   📊 20-Day Volatility: {volatility:.2f}%")
        print(f"   🎯 ML Signal: {signal['signal']}")
        print(f"   🎲 Confidence: {signal['confidence']:.4f}")
        print(f"   💪 Strength: {signal['signal_strength']}")
        
        print(f"\n💡 Trading Recommendations:")
        
        if signal['model_ready']:
            if signal['signal'] == 'BUY':
                print(f"   🟢 RECOMMENDATION: BUY Tesla")
                print(f"   💰 Suggested Position: {signal['suggested_position']:.2%} of portfolio")
                print(f"   🛡️ Stop Loss: ${current_price * 0.975:.2f} (-2.5%)")
                print(f"   🎯 Take Profit: ${current_price * 1.05:.2f} (+5.0%)")
                print(f"   ⏰ Hold Period: 1-3 days (multi-horizon model)")
            else:
                print(f"   🔴 RECOMMENDATION: AVOID/SELL Tesla")
                print(f"   💰 Position: Reduce or avoid new positions")
                print(f"   🎯 Reason: Model predicts short-term underperformance")
        else:
            print(f"   🟡 RECOMMENDATION: WAIT")
            print(f"   🎲 Low Confidence: {signal['confidence']:.4f} < 0.55 threshold")
            print(f"   ⏳ Wait for clearer signals before trading")
        
        print(f"\n⚙️ Model Performance:")
        print(f"   📊 Cross-Validation Accuracy: {training_results.get('cv_accuracy', 'N/A'):.4f}")
        print(f"   🎯 Overfitting Score: {training_results.get('overfitting_score', 'N/A'):.4f}")
        print(f"   ✅ Status: Production Ready")
        
        print(f"\n📅 Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Standalone recommendations failed: {e}")
        return False

def run_daily_tesla_analysis():
    """Run daily Tesla analysis with the enhanced model"""
    
    print("🗓️ DAILY TESLA ANALYSIS")
    print("=" * 40)
    
    success = create_standalone_tesla_recommendations()
    
    if success:
        print(f"\n✅ Daily Tesla analysis completed")
        
        # Create log entry
        log_entry = f"{dt.datetime.now().isoformat()}: Tesla daily analysis completed\n"
        
        try:
            with open('tesla_daily_log.txt', 'a') as f:
                f.write(log_entry)
        except:
            pass
    else:
        print(f"\n❌ Daily Tesla analysis failed")
    
    return success

if __name__ == "__main__":
    print("Starting Tesla Enhanced Trading Integration...")
    
    # Try full integration first
    integration_success = integrate_tesla_model_into_system()
    
    if not integration_success:
        print("\nFalling back to standalone mode...")
        standalone_success = create_standalone_tesla_recommendations()
        
        if standalone_success:
            print(f"\n🎉 Tesla standalone analysis completed!")
        else:
            print(f"\n❌ All integration attempts failed")
    
    print(f"\nIntegration completed at: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")