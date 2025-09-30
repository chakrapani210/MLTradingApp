# Paper Trading Demo System

## Single Unified Demo Application

After consolidating redundant code, we now have **one** comprehensive demo application:

### 🎯 `paper_trading_demo.py` - The Only Demo You Need

**Features:**
- **3 Modes in 1 Application**: Simple, Basic, Production
- **Real Market Data**: No simulated data, uses actual prices and ML models
- **Production Integration**: Leverages existing `ProductionTradingOrchestrator`
- **Flexible**: Command-line arguments for different scenarios

### Usage Examples

```bash
# Production mode (full analysis, default)
python paper_trading_demo.py

# Simple mode (quick manual orders)
python paper_trading_demo.py simple

# Basic mode (ML validation with simple logic)
python paper_trading_demo.py basic 5

# Production mode with specific symbol count
python paper_trading_demo.py production 2

# Help
python paper_trading_demo.py help
```

### Demo Modes

1. **Simple Mode** (`simple`)
   - Manual orders with real market prices
   - Quick functionality test
   - Fixed position sizes (10 shares)

2. **Basic Mode** (`basic`)
   - ML model validation
   - Simple decision logic based on model accuracy
   - Good for model testing

3. **Production Mode** (`production`) - **Default**
   - Full market analysis pipeline
   - ML models + market context + technical analysis
   - Risk-based position sizing
   - Real trading decisions with reasoning

### What Was Consolidated

**Removed Redundant Files:**
- ~~`run_automated_demo.py`~~ 
- ~~`run_paper_trading_demo.py`~~
- ~~`run_production_paper_demo.py`~~
- `run_simple_demo.py` - Now redirects to unified demo

**Benefits:**
- ✅ **DRY Principle**: No duplicate code
- ✅ **Single Maintenance Point**: One file to update
- ✅ **Consistent Features**: All modes use same infrastructure
- ✅ **Real Data**: No simulated data anywhere
- ✅ **Production Integration**: Leverages existing sophisticated orchestrator

### Architecture

```
paper_trading_demo.py
├── UnifiedPaperTradingDemo
│   ├── ProductionTradingOrchestrator (reused!)
│   ├── PaperTradingAccount
│   └── Real market data providers
├── Mode: Simple
│   └── Manual orders with real prices
├── Mode: Basic  
│   └── ML model validation
└── Mode: Production
    ├── Full market analysis
    ├── ML predictions
    ├── Technical indicators
    └── Risk management
```

### Key Features

- **Real Market Data**: Uses `YFinanceProvider` for actual current prices
- **Production ML Models**: Leverages 8 pre-trained models (AAPL, NVDA, TSLA, etc.)
- **Market Context**: SPY correlation, market regime analysis, beta calculations
- **Technical Analysis**: RSI, MACD, Bollinger Bands with TA-Lib
- **Risk Management**: Position sizing based on confidence and available capital
- **Paper Trading**: Safe simulation with real market logic

### Sample Output

```
🎯 UNIFIED PAPER TRADING DEMO
📊 Symbols: AAPL, NVDA, TSLA, MSFT, GOOGL
💰 Capital: $100,000.00
🚀 Using ProductionTradingOrchestrator

🏭 PRODUCTION MODE: Full Analysis (3 symbols)
==================================================
🔍 FULL ANALYSIS: NVDA
==================================================

🤖 ML Model Analysis
   Model: 0.50 test accuracy

📊 Market Context Analysis
   SPY Correlation: 0.603
   Market Regime: bull
   SPY Beta: 1.633

🎯 FINAL DECISION: BUY (confidence: 0.67)
   Reasoning: Strong ML model (0.50); Bull market + correlation (0.60); High beta leverage (1.63)

📋 NVDA: BUY 50 shares @ $182.03
   Position: $9,101.59 (8.0% of cash)
   Reasoning: Strong ML model; Bull market + correlation; High beta leverage
✅ Executed: 50 @ $182.03 = $9,101.59
```

This consolidation eliminates redundancy while providing all the functionality you need in a single, well-structured application!