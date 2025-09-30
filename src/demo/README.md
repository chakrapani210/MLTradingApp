# Paper Trading Demo

## ⚠️ IMPORTANT: Demo Consolidation

The paper trading demo has been **consolidated** into a single unified application. 

## Usage

# Paper Trading Demo

## 🚀 Interactive Launch Menu

The demo now launches with an **interactive menu** on startup!

### From Demo Folder (Recommended)
```bash
cd demo
python paper_trading_demo.py     # Launches interactive menu
```

## 📋 Interactive Menu Options

When you launch the demo without arguments, you'll see:

```
🎯 PAPER TRADING DEMO - INTERACTIVE MENU
═══════════════════════════════════════════════════════════════

Please select an option:

1️⃣  Account Status
    📊 View account summary
    📈 Check current positions
    💰 See portfolio performance
    📋 Review trade history

2️⃣  Start Trading
    🤖 Initiate automated trading
    📈 Run ML analysis
    📋 Generate trading signals
    ⚡ Execute trades

3️⃣  Help & Modes
    ❓ Show available modes
    📖 View usage examples

4️⃣  Exit
    👋 Close the application
```

### 1️⃣ Account Status Features

- **Account Balance**: Cash, Portfolio Value, Total Equity
- **Position Details**: Holdings with P&L and percentages  
- **Portfolio Summary**: Position count and total returns
- **Real-time Data**: Current market values

**Sample Output:**
```
💰 ACCOUNT BALANCE
----------------------------------------
Cash Available:    $91,196.31
Portfolio Value:   $8,803.69
Total Equity:      $100,000.00
Cash Allocation:   91.2%
Portfolio Alloc:   8.8%

📈 CURRENT POSITIONS
----------------------------------------
  AAPL   │     10 shares │ Avg: $ 254.68 │ Value:   $2,547 │ 📈 $    0 (+0.0%)
  NVDA   │     10 shares │ Avg: $ 182.03 │ Value:   $1,820 │ 📈 $    0 (+0.0%)
  TSLA   │     10 shares │ Avg: $ 443.65 │ Value:   $4,437 │ 📈 $    0 (+0.0%)
```

### 2️⃣ Trading Menu Options

When you select "Start Trading", you get sub-options:

1. **Simple Trading**: Manual orders with real prices
2. **Basic ML Trading**: ML model validation  
3. **Production Trading**: Full analysis pipeline
4. **Custom Parameters**: Choose mode + symbol count
5. **Back to Main Menu**

## 🎯 Direct Mode Usage (No Menu)

You can still run modes directly:

```bash
cd demo
python paper_trading_demo.py simple      # Simple mode
python paper_trading_demo.py basic 5     # Basic mode, 5 symbols  
python paper_trading_demo.py production  # Production mode
python paper_trading_demo.py help        # Show help
```

## Demo Modes

1. **Production Mode** (default)
   - Full ML analysis with market context
   - Real market data and prices
   - Risk-based position sizing
   - Complete trading pipeline

2. **Simple Mode** 
   - Quick manual orders with real prices
   - Basic functionality testing
   - Fixed position sizes

3. **Basic Mode**
   - ML model validation
   - Simple decision logic
   - Good for model testing

## Features

✅ **Single Unified Demo**: No more duplicate files  
✅ **Real Market Data**: Uses actual prices and ML models  
✅ **Production Integration**: Leverages existing orchestrator  
✅ **Multiple Modes**: Simple, Basic, Production in one app  
✅ **Risk Management**: Proper position sizing and validation  

## Configuration

The demo uses `paper_trading_config.yaml` for paper trading specific settings, but the main logic is in the unified demo application.

## Architecture

```
demo/
├── run_demo.py               # Launcher (optional convenience)
├── paper_trading_demo.py     # Main unified demo application
├── paper_trading_config.yaml # Demo-specific config
└── README.md                # This file
```

All demo functionality is consolidated into the `paper_trading_demo.py` file in the demo directory.