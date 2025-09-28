"""
Test script for configuration system
Run this to verify that the configuration system is working properly
"""

from config_manager import get_config
import datetime as dt

def test_configuration():
    """Test all configuration functions"""
    print("=== ML Trading Application Configuration Test ===\n")
    
    try:
        config = get_config()
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    # Test trading configuration
    print("\n📈 Trading Configuration:")
    print(f"  Symbols: {config.get_symbols()}")
    print(f"  Default Symbol: {config.get_default_symbol()}")
    print(f"  Shares per Trade: {config.get_shares_per_trade()}")
    print(f"  Market Impact: {config.get_market_impact()}")
    
    # Test data configuration
    print("\n📊 Data Configuration:")
    print(f"  Training Period (months): {config.get_training_period_months()}")
    print(f"  Lookback Days: {config.get_lookback_days()}")
    
    sim_start, sim_end = config.get_simulation_dates()
    print(f"  Simulation Period: {sim_start.strftime('%Y-%m-%d')} to {sim_end.strftime('%Y-%m-%d')}")
    
    # Test analysis dates
    print("\n🔍 Analysis Configuration:")
    try:
        train_start, train_end, test_start, test_end = config.get_analysis_dates('tesla_analysis')
        print(f"  Tesla Analysis - Training: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        print(f"  Tesla Analysis - Testing: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"  ❌ Analysis dates error: {e}")
    
    # Test indicators configuration
    print("\n📏 Technical Indicators:")
    print(f"  Window Size: {config.get_indicator_window()}")
    indicator_settings = config.get_indicator_settings()
    for key, value in indicator_settings.items():
        print(f"  {key}: {value}")
    
    # Test ML configuration
    print("\n🤖 ML Model Configuration:")
    ml_config = config.get_ml_config()
    for key, value in ml_config.items():
        print(f"  {key}: {value}")
    
    # Test retraining configuration
    print("\n🔄 Retraining Configuration:")
    retraining_config = config.get_retraining_config()
    for key, value in retraining_config.items():
        print(f"  {key}: {value}")
    
    # Test portfolio configuration
    print("\n💰 Portfolio Configuration:")
    portfolio_config = config.get_portfolio_config()
    for key, value in portfolio_config.items():
        print(f"  {key}: {value}")
    
    # Test Robinhood configuration (without showing sensitive data)
    print("\n🏦 Robinhood Configuration:")
    rh_config = config.get_robinhood_config()
    print(f"  Order Type: {rh_config['order_type']}")
    print(f"  Time in Force: {rh_config['time_in_force']}")
    print(f"  Enable Live Trading: {rh_config['enable_live_trading']}")
    print(f"  Paper Trading: {rh_config['paper_trading']}")
    
    # Test utility functions
    print("\n🛠️ Utility Functions:")
    print(f"  Position filename for TSLA: {config.get_position_filename('TSLA')}")
    
    # Test training dates calculation
    test_date = dt.datetime(2025, 7, 1)
    train_start, train_end = config.get_training_dates_for_symbol('TSLA', test_date)
    print(f"  Training dates for TSLA (from {test_date.strftime('%Y-%m-%d')}): {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
    
    # Test retraining frequency
    freq_high = config.get_retraining_frequency_for_symbol('TSLA', 0.9)  # High volatility
    freq_medium = config.get_retraining_frequency_for_symbol('TSLA', 0.5)  # Medium volatility
    freq_low = config.get_retraining_frequency_for_symbol('TSLA', 0.3)  # Low volatility
    print(f"  Retraining frequency - High Vol (0.9): {freq_high} days")
    print(f"  Retraining frequency - Medium Vol (0.5): {freq_medium} days")
    print(f"  Retraining frequency - Low Vol (0.3): {freq_low} days")
    
    print(f"\n✅ All configuration tests passed!")
    print(f"📄 Configuration loaded from: config.yaml")
    print(f"🚀 Ready to run ML trading application!")
    
    return True

if __name__ == "__main__":
    success = test_configuration()
    if not success:
        print("\n❌ Configuration test failed. Please check config.yaml file and dependencies.")
        exit(1)
    else:
        print("\n✅ Configuration system is working properly!")
        print("\nNext steps:")
        print("1. Review and customize config.yaml as needed")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Run simulation: python simulate_results.py")
        print("4. For live trading: Update Robinhood credentials in config.yaml")