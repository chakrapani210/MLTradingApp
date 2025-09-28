# Configuration Guide for ML Trading Application

This guide explains how to customize the `config.yaml` file for your trading needs.

## Quick Start

1. **Test the configuration**:
   ```bash
   python test_config.py
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run simulation**:
   ```bash
   python simulate_results.py
   ```

## Configuration Sections

### 🏪 Trading Configuration

```yaml
trading:
  symbols: ["TSLA", "AAPL", "NVDA"]  # Stocks to trade
  default_symbol: "TSLA"             # Default for single-stock analysis
  shares_per_trade: 20               # Number of shares per transaction
  market_impact: 0.005               # 0.5% market impact assumption
```

**Customization Tips**:
- Add your favorite stocks to `symbols`
- Adjust `shares_per_trade` based on your capital
- `market_impact` affects signal generation (higher = more conservative)

### 📅 Data Configuration

```yaml
data:
  training_period_months: 12  # Optimal based on our analysis
  simulation:
    start_date: "2025-07-01"
    end_date: "2025-12-31"
```

**Customization Tips**:
- `training_period_months`: 6-12 for volatile stocks, 12-24 for stable stocks
- Update `simulation` dates for different backtesting periods
- Add custom analysis periods in the `analysis` section

### 🤖 ML Model Configuration

```yaml
ml_model:
  algorithm: "DecisionTree"     # Options: DecisionTree, RandomForest
  max_depth: 5                  # Prevent overfitting
  random_state: 42              # For reproducible results
  
  retraining:
    default_frequency: 10       # Bi-weekly retraining (optimal)
    high_volatility_frequency: 5    # Weekly for volatile stocks
    medium_volatility_frequency: 10 # Bi-weekly for medium volatility  
    low_volatility_frequency: 22    # Monthly for stable stocks
```

**Customization Tips**:
- Try `RandomForest` for more complex patterns
- Increase `max_depth` if underfitting (but risk overfitting)
- Adjust retraining frequencies based on your trading style

### 💰 Portfolio Configuration

```yaml
portfolio:
  starting_value: 3000    # Starting portfolio value for simulations
  commission: 0.0         # Commission per trade (most brokers now 0)
  impact: 0.005          # Market impact cost
```

**Customization Tips**:
- Set `starting_value` to match your actual capital
- Update `commission` if your broker charges fees
- Adjust `impact` based on stock liquidity

### 🏦 Robinhood Configuration

```yaml
robinhood:
  username: "your_username@gmail.com"
  password: "your_password" 
  qr_code: "your_2fa_code"
  
  enable_live_trading: false  # SAFETY: Keep false until ready
  paper_trading: true         # Use paper trading for testing
```

**⚠️ SECURITY WARNING**:
- Never commit real credentials to version control
- Use environment variables for production:
  ```bash
  export RH_USERNAME="your_username"
  export RH_PASSWORD="your_password"
  export RH_QR_CODE="your_qr_code"
  ```

### 📊 Technical Indicators

```yaml
indicators:
  window: 5                    # 5-day window (optimal for daily trading)
  use_sma: true               # Simple Moving Average
  use_bollinger_bands: true   # Bollinger Bands
  use_momentum: true          # Price momentum
  use_volatility: true        # Rolling volatility
  use_macd: true              # MACD indicator
  
  # Drop correlated indicators
  drop_upper_bb: true         # Drop upper Bollinger Band
  drop_lower_bb: true         # Drop lower Bollinger Band
  keep_bb_value: true         # Keep normalized BB position
```

**Customization Tips**:
- `window: 3-5` for day trading, `10-20` for swing trading
- Disable indicators that don't work for your stocks
- Keep `drop_upper_bb` and `drop_lower_bb` true to avoid multicollinearity

## Stock-Specific Configurations

### High Volatility Stocks (TSLA, NVDA, Crypto)
```yaml
# Recommended settings
training_period_months: 6
retraining_frequency: 5  # Weekly
window: 3-5
```

### Medium Volatility Stocks (AAPL, MSFT, GOOGL)  
```yaml
# Recommended settings  
training_period_months: 12
retraining_frequency: 10  # Bi-weekly
window: 5-7
```

### Low Volatility Stocks (Utilities, Dividends)
```yaml
# Recommended settings
training_period_months: 18-24
retraining_frequency: 22  # Monthly  
window: 10-15
```

## Performance Optimization

### For Better Accuracy
- Increase `training_period_months` (but not beyond 24)
- Use `RandomForest` algorithm
- Enable more technical indicators
- Decrease retraining frequency for stable stocks

### For Faster Execution
- Decrease `training_period_months` to 6-9
- Use `DecisionTree` algorithm  
- Disable unused indicators
- Increase retraining frequency only when needed

### For Lower Risk
- Increase `market_impact` to 0.01 (1%)
- Use longer `window` sizes
- Enable `paper_trading` mode
- Set conservative position sizes

## Testing Your Configuration

1. **Run configuration test**:
   ```bash
   python test_config.py
   ```

2. **Backtest with your settings**:
   ```bash
   python simulate_results.py
   ```

3. **Check performance metrics**:
   - Cumulative Return > 10% annually
   - Sharpe Ratio > 1.0 (good), > 2.0 (excellent)
   - Maximum drawdown < 20%

4. **Paper trade before going live**:
   - Set `enable_live_trading: false`
   - Set `paper_trading: true`  
   - Monitor for 1-2 weeks

## Common Issues and Solutions

### "Configuration file not found"
- Ensure `config.yaml` is in the same directory as the Python scripts
- Check file permissions

### "YAML parsing error"
- Validate YAML syntax at yamllint.com
- Check indentation (use spaces, not tabs)
- Ensure proper quote usage

### Poor backtest performance  
- Try different `training_period_months` values
- Adjust `market_impact` threshold
- Enable/disable different technical indicators
- Check if stock has sufficient historical data

### Memory/performance issues
- Reduce `training_period_months`
- Use fewer symbols
- Disable complex indicators
- Use `DecisionTree` instead of `RandomForest`

## Environment Variables (Production)

For production deployment, use environment variables instead of hardcoded credentials:

```bash
# Set environment variables
export ML_TRADING_RH_USERNAME="your_username"
export ML_TRADING_RH_PASSWORD="your_password"  
export ML_TRADING_RH_QR="your_qr_code"
export ML_TRADING_LIVE_ENABLED="false"
```

Then update your config.yaml:
```yaml
robinhood:
  username: "${ML_TRADING_RH_USERNAME}"
  password: "${ML_TRADING_RH_PASSWORD}"
  qr_code: "${ML_TRADING_RH_QR}"
  enable_live_trading: "${ML_TRADING_LIVE_ENABLED}"
```

## Support

If you encounter issues:
1. Run `python test_config.py` to validate configuration
2. Check the console output for specific error messages
3. Review this guide for common solutions
4. Ensure all dependencies are installed: `pip install -r requirements.txt`