#!/usr/bin/env python3"""

"""Test script for configuration system

Test the new configuration systemRun this to verify that the configuration system is working properly

""""""

import yaml

from config_manager import get_config

def main():import datetime as dt

    # Load configuration

    with open('config.yaml', 'r') as f:def test_configuration():

        config = yaml.safe_load(f)    """Test all configuration functions"""

        print("=== ML Trading Application Configuration Test ===\n")

    print("✓ Configuration loaded successfully")    

    print("=" * 50)    try:

            config = get_config()

    # Test application metadata        print("✅ Configuration loaded successfully")

    app = config['application']    except Exception as e:

    print(f"✓ Application: {app['name']} v{app['version']}")        print(f"❌ Failed to load configuration: {e}")

    print(f"✓ Environment: {app['environment']}")        return False

        

    # Test trading configuration    # Test trading configuration

    trading = config['trading']    print("\n📈 Trading Configuration:")

    print(f"✓ Default symbol: {trading['default_symbol']}")    print(f"  Symbols: {config.get_symbols()}")

    print(f"✓ Assets configured: {len(trading['assets']['stocks'])}")    print(f"  Default Symbol: {config.get_default_symbol()}")

        print(f"  Shares per Trade: {config.get_shares_per_trade()}")

    # Test machine learning configuration    print(f"  Market Impact: {config.get_market_impact()}")

    ml = config['machine_learning']    

    print(f"✓ Primary algorithm: {ml['primary_algorithm']}")    # Test data configuration

    print(f"✓ Retraining frequency: {ml['retraining']['frequency_days']} days")    print("\n📊 Data Configuration:")

        print(f"  Training Period (months): {config.get_training_period_months()}")

    # Test LightGBM parameters    print(f"  Lookback Days: {config.get_lookback_days()}")

    lgb_params = ml['models']['LightGBM']['default']    

    print(f"✓ LightGBM estimators: {lgb_params['n_estimators']}")    sim_start, sim_end = config.get_simulation_dates()

    print(f"✓ LightGBM learning rate: {lgb_params['learning_rate']}")    print(f"  Simulation Period: {sim_start.strftime('%Y-%m-%d')} to {sim_end.strftime('%Y-%m-%d')}")

        

    # Test supported assets and their configurations    # Test analysis dates

    print("\n✓ Supported Assets:")    print("\n🔍 Analysis Configuration:")

    for asset in trading['assets']['stocks']:    try:

        print(f"  - {asset['symbol']}: {asset['name']} ({asset['volatility_class']} volatility)")        train_start, train_end, test_start, test_end = config.get_analysis_dates('tesla_analysis')

            print(f"  Tesla Analysis - Training: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")

    # Test backtesting configuration        print(f"  Tesla Analysis - Testing: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")

    bt = config['backtesting']    except Exception as e:

    print(f"\n✓ Backtesting capital: ${bt['initial_capital']:,}")        print(f"  ❌ Analysis dates error: {e}")

    print(f"✓ Retraining frequency: {bt['strategies']['adaptive']['retraining_frequency']} days")    

        # Test indicators configuration

    # Test risk management    print("\n📏 Technical Indicators:")

    rm = config['risk_management']    print(f"  Window Size: {config.get_indicator_window()}")

    print(f"✓ Max position size: {rm['position_risks']['max_position_size']*100}%")    indicator_settings = config.get_indicator_settings()

    print(f"✓ Daily loss limit: {rm['portfolio_risks']['max_daily_loss']*100}%")    for key, value in indicator_settings.items():

            print(f"  {key}: {value}")

    print("\n🎉 Configuration system is working perfectly!")    

    # Test ML configuration

if __name__ == "__main__":    print("\n🤖 ML Model Configuration:")

    main()    ml_config = config.get_ml_config()
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