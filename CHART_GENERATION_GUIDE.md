# Enhanced Trading Strategy with Interactive Charts

This project provides an automated way to run enhanced trading strategy simulations and generate professional interactive charts using existing code components.

## Quick Start

### Option 1: Simple Python Script (Recommended)

```bash
# Run with default settings (AAPL, 6 months)
python run_simulation_with_charts.py

# Run with custom symbols and timeframe
python run_simulation_with_charts.py --symbols AAPL TSLA NVDA --months 12
```

### Option 2: Python Import

```python
from run_simulation_with_charts import run_simulation_with_charts

# Run simulation and generate charts
results = run_simulation_with_charts(['AAPL', 'TSLA'], months_back=6)

# Access results
print(f"Charts generated: {len(results['chart_files'])}")
print(f"Dashboard: {results['dashboard']}")
```

### Option 3: Enhanced Test Suite

```bash
# Run comprehensive test with charts
python tests/test_enhanced_strategy.py

# Run full test suite
python tests/test_enhanced_strategy.py --comprehensive
```

## What Gets Generated

### 1. Interactive TradingView-Style Charts
- **Location**: `tests/results/{symbol}_enhanced_chart.html`
- **Features**:
  - Professional candlestick charts
  - Technical indicators (SMA, RSI, Bollinger Bands)
  - Trading signals and pattern markers
  - Golden cross and death cross indicators
  - Short-term pattern highlights
  - Order size and trade reasoning
  - Interactive hover details

### 2. Master Dashboard
- **Location**: `tests/results/enhanced_trading_dashboard.html`
- **Features**:
  - Combined view of all symbols
  - Performance comparison
  - Navigation between charts
  - Professional styling

### 3. Performance Data
- Strategy return vs buy-and-hold
- Sharpe ratio and risk metrics
- Trade details with reasoning
- Pattern analysis results

## Key Features Integrated

### Professional Pattern Detection
- **Golden Cross/Death Cross**: TA-Lib powered SMA crossover detection
- **Short-term Patterns**: RSI, Bollinger Bands, Stochastic, Candlesticks
- **Signal Enhancement**: ML predictions enhanced with pattern confirmation
- **Pattern Integration**: Death cross and short-term patterns now actively used in decisions

### Enhanced Trading Strategy
- **Machine Learning**: Decision tree model with 85.9% accuracy
- **Market Context**: Beta analysis, correlation with SPY/QQQ
- **Intelligent Order Sizing**: Volatility-adjusted position sizing
- **Risk Management**: Pattern strength thresholds and confirmation

### Interactive Visualization
- **Reused Existing Code**: Leverages `create_enhanced_tradingview_charts.py`
- **Professional Charts**: TradingView-style interface
- **Mobile Friendly**: Responsive design
- **Export Ready**: HTML files for easy sharing

## Example Output

```
ENHANCED TRADING STRATEGY SIMULATION + CHARTS
============================================================
Symbols: ['AAPL']
Period: 6 months back

STEP 1: RUNNING TRADING SIMULATIONS
----------------------------------------
Processing AAPL...
  Strategy Return: 42.5%
  Outperformance: +28.8%
  Trades: 17

STEP 2: GENERATING INTERACTIVE CHARTS
----------------------------------------
Creating Enhanced TradingView Charts...
[ENHANCEMENT] Enhanced 4 signals using pattern analysis

RESULTS SUMMARY
========================================
AAPL:
  Chart: tests/results/aapl_enhanced_chart.html (77.4 KB)
  Return: 42.5%
  Sharpe: 0.927

Dashboard: tests/results/enhanced_trading_dashboard.html (5.8 KB)

SUCCESS! Simulation and charts completed.
TIP: Open the HTML files in your browser to view interactive charts
```

## Code Reuse Architecture

The implementation efficiently reuses existing components:

1. **Enhanced Strategy** (`enhanced_strategy.py`)
   - Runs complete simulation with all patterns
   - Provides performance metrics and trade details

2. **Chart Generator** (`create_enhanced_tradingview_charts.py`)
   - Creates professional interactive charts
   - Handles OHLC data and technical indicators

3. **Pattern Detection** (`indicators.py`)
   - TA-Lib powered golden cross detection
   - Professional short-term pattern analysis
   - Combined signal generation

4. **Test Framework** (`tests/test_enhanced_strategy.py`)
   - Comprehensive testing and validation
   - Chart generation integration

## File Structure

```
├── run_simulation_with_charts.py     # Simple standalone script
├── enhanced_strategy.py              # Main trading strategy
├── create_enhanced_tradingview_charts.py  # Chart generation
├── indicators.py                     # Pattern detection
├── tests/
│   ├── test_enhanced_strategy.py     # Enhanced test suite
│   └── results/                      # Generated charts
│       ├── {symbol}_enhanced_chart.html
│       └── enhanced_trading_dashboard.html
```

## Requirements

- Python 3.7+
- Required packages: pandas, numpy, plotly, yfinance, scikit-learn, TA-Lib
- Install with: `pip install -r requirements.txt`

## Usage Tips

1. **View Charts**: Open HTML files in any modern web browser
2. **Multiple Symbols**: Use space-separated symbol list
3. **Timeframes**: Adjust months_back for different analysis periods
4. **Performance**: Longer timeframes provide more training data
5. **Patterns**: All pattern functions are now integrated and actively used

The system now provides a complete end-to-end solution for running enhanced trading simulations with professional visualization!