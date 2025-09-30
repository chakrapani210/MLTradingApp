# Robinhood Trading Application

This folder contains the real-time trading application for Robinhood brokerage accounts.

## Features

✅ **Real Trading Integration**: Connect to Robinhood brokerage accounts  
✅ **ML-Powered Decisions**: Uses the same ML models as paper trading  
✅ **Risk Management**: Comprehensive position sizing and risk controls  
✅ **Transaction Tracking**: Full audit trail for charts and analysis  
✅ **Performance Analytics**: Real-time portfolio monitoring  

## Quick Start

### Interactive Mode
```bash
cd src/robinhood
python run_robinhood.py
```

### Direct Mode
```bash
python robinhood_app.py production 3
```

## Trading Modes

1. **Simple Mode**: Manual orders with real market prices
2. **Basic Mode**: ML model validation with simple logic  
3. **Production Mode**: Full automated trading with comprehensive analysis

## Important Disclaimers

⚠️ **REAL MONEY TRADING**: This application executes actual trades with real money  
⚠️ **RISK WARNING**: Trading involves substantial risk of loss  
⚠️ **DEMO MODE**: Current implementation is in demo mode for safety  
⚠️ **NOT FINANCIAL ADVICE**: This software is for educational/research purposes  

## Configuration

Edit `robinhood_config.py` to customize:
- Account settings
- Risk management rules
- Trading parameters
- ML model settings

## Architecture

The Robinhood app reuses the common `TradingAppBase` class and implements:
- `RobinhoodTradingAccount`: Real trading account interface
- `RobinhoodTradingApp`: Application with interactive menu
- Transaction tracking and portfolio snapshots
- Same ML framework as paper trading

## Safety Features

- Order value limits (min/max)
- Position size restrictions  
- Real-time market data validation
- Comprehensive transaction logging
- Account state persistence
- User confirmation for trades

## Development Notes

Currently in **DEMO MODE** - simulates Robinhood API calls for safety.  
To enable real trading, implement actual Robinhood API integration in `RobinhoodTradingAccount.connect()`.

## Files

- `robinhood_app.py`: Main application
- `robinhood_trading.py`: Trading account implementation  
- `run_robinhood.py`: Launcher script
- `robinhood_config.py`: Configuration settings
- `README.md`: This documentation