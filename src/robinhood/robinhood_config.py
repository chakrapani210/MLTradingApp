"""
Robinhood Trading Configuration
"""

# Robinhood Account Configuration
ROBINHOOD_CONFIG = {
    # Account credentials (set via environment variables or user input)
    "username": "",  # Set via input or environment variable
    "password": "",  # Set via input or environment variable
    "device_token": "",  # Optional device token for 2FA
    
    # Trading configuration
    "enable_day_trading": True,  # Enable if account qualifies for day trading
    "challenge_type": "sms",  # 'sms' or 'email' for 2FA
    "api_timeout": 30,  # API request timeout in seconds
    
    # Risk management
    "max_order_value": 50000.0,  # Maximum single order value
    "min_order_value": 1.0,      # Minimum single order value
    "max_position_percent": 20.0,  # Maximum position as % of account
    "max_daily_trades": 10,      # Maximum trades per day
    
    # Data management
    "persistence_file": "robinhood_account_state.json",
    "track_portfolio_snapshots": True,
    "backup_frequency_hours": 24,
    
    # Demo/Testing
    "demo_mode": True,  # Set to False for real trading
    "demo_starting_cash": 25000.0,
}

# Trading symbols configuration
TRADING_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
    'NVDA', 'META', 'NFLX', 'CRM', 'ADBE'
]

# ML Model configuration
ML_CONFIG = {
    "retrain_frequency_days": 7,
    "min_confidence_threshold": 0.6,
    "use_ensemble_models": True,
    "feature_importance_threshold": 0.05
}

# Risk management rules
RISK_RULES = {
    "max_portfolio_beta": 1.5,
    "max_sector_concentration": 0.3,
    "stop_loss_percent": 0.05,
    "take_profit_percent": 0.15,
    "trailing_stop_percent": 0.03
}