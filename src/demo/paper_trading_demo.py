"""
Unified Paper Trading Demo
Single demo application with multiple modes - no redundant code
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Optional

# Add project root to path (now we're in src/demo, so go up two levels)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config_manager import get_trading_symbols
from src.enhanced_orchestrator import ProductionTradingOrchestrator
from src.trading.paper_trading import PaperTradingAccount, PaperTradingConfig
from src.trading.trading_app_base import TradingAppBase
from src.interfaces.trading_strategy import Order, OrderType, OrderSide


class UnifiedPaperTradingDemo(TradingAppBase):
    """
    Unified Paper Trading Demo Application
    Uses the common TradingAppBase for consistency with Robinhood app
    """
    
    def __init__(self):
        """Initialize the unified paper trading demo"""
        # Paper trading configuration
        self.paper_config = PaperTradingConfig(
            initial_cash=100000.0,
            commission_rate=0.0,
            slippage_rate=0.001
        )
        
        # Initialize base class
        super().__init__(starting_capital=100000.0, commission_rate=0.0)
        
        # Initialize paper account
        self.account = PaperTradingAccount(self.paper_config)
        
        print("🎯 UNIFIED PAPER TRADING DEMO")
        print(f"📊 Symbols: {', '.join(self.symbols)}")
        print(f"💰 Capital: ${self.paper_config.initial_cash:,.2f}")
        print("🚀 Using ProductionTradingOrchestrator")
    
    async def connect_account(self) -> bool:
        """Connect to paper trading account"""
        return await self.account.connect()
    
    async def disconnect_account(self) -> bool:
        """Disconnect from paper trading account"""
        return await self.account.disconnect()
    
    async def run_mode(self, mode: str = "production", symbol_limit: int = 3):
        """
        Run demo in specified mode (delegates to base class)
        """
        await self.run_trading_mode(mode, symbol_limit)
    
    async def _run_simple_mode(self):
        """Simple mode: Manual orders with real prices"""
        print("\n📋 SIMPLE MODE: Manual Orders with Real Market Prices")
        print("-" * 60)
        
        test_symbols = ['AAPL', 'NVDA', 'TSLA']
        
        for symbol in test_symbols:
            try:
                # Get real current price
                current_price = await self.get_real_price(symbol)
                if not current_price:
                    print(f"   ❌ Could not get price for {symbol}")
                    continue
                
                print(f"\n📈 {symbol} @ ${current_price:.2f}")
                
                # Simple buy order
                quantity = 10
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    metadata={'mode': 'simple', 'real_price': current_price}
                )
                
                print(f"   📋 Placing BUY order: {quantity} shares")
                execution = await self.account.place_order(order)
                
                if execution.is_complete:
                    cost = execution.filled_quantity * execution.avg_fill_price
                    print(f"   ✅ Executed: {execution.filled_quantity} @ ${execution.avg_fill_price:.2f} (${cost:,.2f})")
                else:
                    print(f"   ❌ Failed: {execution.error_message}")
            
            except Exception as e:
                print(f"   ❌ Error with {symbol}: {e}")
    
    async def _run_basic_mode(self, symbol_limit: int):
        """Basic mode: ML model validation with simple signals"""
        print(f"\n🤖 BASIC MODE: ML Model Analysis ({symbol_limit} symbols)")
        print("-" * 60)
        
        decisions = []
        
        for symbol in self.symbols[:symbol_limit]:
            print(f"\n📈 Analyzing {symbol}...")
            
            try:
                # Check ML model
                ml_results = self.production_orchestrator.train_ml_model(symbol, force_retrain=False)
                
                # Simple decision logic
                decision = self._make_basic_decision(symbol, ml_results)
                
                if decision['action'] != 'HOLD':
                    decisions.append(decision)
                
                print(f"   🎯 Decision: {decision['action']} (confidence: {decision['confidence']:.2f})")
                print(f"   📊 ML Model: {ml_results.get('test_accuracy', 'N/A'):.2f} accuracy")
                
            except Exception as e:
                print(f"   ❌ Analysis failed: {e}")
        
        # Execute decisions
        if decisions:
            print(f"\n🎯 Executing {len(decisions)} decisions...")
            for decision in decisions:
                await self._execute_basic_decision(decision)
        else:
            print("\n💤 No trading signals generated")
    
    async def _run_production_mode(self, symbol_limit: int):
        """Production mode: Full analysis with real data"""
        print(f"\n🏭 PRODUCTION MODE: Full Analysis ({symbol_limit} symbols)")
        print("-" * 60)
        
        decisions = []
        
        for symbol in self.symbols[:symbol_limit]:
            print(f"\n{'='*50}")
            print(f"🔍 FULL ANALYSIS: {symbol}")
            print(f"{'='*50}")
            
            try:
                # 1. ML Model Analysis
                print("\n🤖 ML Model Analysis")
                ml_results = self.production_orchestrator.train_ml_model(symbol, force_retrain=False)
                print(f"   Model: {ml_results.get('test_accuracy', 0):.2f} test accuracy")
                
                # 2. Market Context Analysis
                print("\n📊 Market Context Analysis")
                market_analysis = self.production_orchestrator.analyze_market_context(symbol, analysis_period_days=180)
                
                if 'market_context' in market_analysis:
                    ctx = market_analysis['market_context']
                    print(f"   SPY Correlation: {ctx['spy_correlation']:.3f}")
                    print(f"   Market Regime: {ctx['market_regime']}")
                    print(f"   SPY Beta: {ctx['spy_beta']:.3f}")
                
                # 3. Generate production decision using orchestrator
                decision = await self._make_production_decision_via_orchestrator(symbol)
                
                if decision['action'] != 'HOLD':
                    decisions.append(decision)
                
                print(f"\n🎯 FINAL DECISION: {decision['action']} (confidence: {decision['confidence']:.2f})")
                print(f"   Reasoning: {decision['reasoning']}")
                
            except Exception as e:
                print(f"   ❌ Analysis failed: {e}")
        
        # Execute production decisions
        if decisions:
            print(f"\n🎯 EXECUTING {len(decisions)} PRODUCTION DECISIONS")
            print("-" * 60)
            for decision in decisions:
                await self._execute_production_decision(decision)
        else:
            print("\n💤 No trading signals generated")
    
    def _make_basic_decision(self, symbol: str, ml_results: dict) -> dict:
        """Make basic trading decision based on ML model only"""
        confidence = 0.5
        action = 'HOLD'
        
        if ml_results.get('success', False):
            test_acc = ml_results.get('test_accuracy', 0.5)
            train_acc = ml_results.get('train_accuracy', 0.5)
            
            # Simple logic: if model is decent, lean towards buy in general market
            avg_acc = (test_acc + train_acc) / 2
            if avg_acc > 0.6:
                action = 'BUY'
                confidence = avg_acc
            elif avg_acc < 0.4:
                action = 'SELL'
                confidence = 1 - avg_acc
        
        return {
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'reasoning': f"ML model avg accuracy: {confidence:.2f}"
        }
    
    async def _make_production_decision_via_orchestrator(self, symbol: str) -> dict:
        """Make production trading decision using orchestrator's signal generation"""
        try:
            # Use orchestrator's comprehensive signal generation
            signal_result = await self.orchestrator.generate_trading_signal(
                symbol=symbol,
                include_market_context=True,
                apply_production_adjustments=True
            )
            
            if not signal_result.get('success', False):
                return {
                    'symbol': symbol,
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'reasoning': signal_result.get('reasoning', 'Signal generation failed')
                }
            
            return {
                'symbol': symbol,
                'action': signal_result['action'],
                'confidence': signal_result['confidence'],
                'reasoning': signal_result['reasoning'],
                'market_context': signal_result.get('market_context'),
                'orchestrator_signal': True
            }
            
        except Exception as e:
            print(f"❌ Orchestrator signal generation failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'confidence': 0.0,
                'reasoning': f"Orchestrator error: {str(e)}"
            }
    
    async def _execute_basic_decision(self, decision: dict):
        """Execute basic trading decision"""
        symbol = decision['symbol']
        action = decision['action']
        
        # Simple position sizing
        quantity = 10  # Fixed quantity for basic mode
        
        if action == 'BUY':
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
                metadata={'mode': 'basic', 'confidence': decision['confidence']}
            )
            
            print(f"   📋 {symbol}: BUY {quantity} shares")
            execution = await self.account.place_order(order)
            
            if execution.is_complete:
                print(f"   ✅ Executed: ${execution.avg_fill_price:.2f} per share")
            else:
                print(f"   ❌ Failed: {execution.error_message}")
    
    async def _execute_production_decision(self, decision: dict):
        """Execute production trading decision with real pricing"""
        symbol = decision['symbol']
        action = decision['action']
        confidence = decision['confidence']
        
        # Get real current price
        current_price = await self._get_real_price(symbol)
        if not current_price:
            print(f"   ❌ Could not get price for {symbol}")
            return
        
        # Production position sizing
        balance = await self.account.get_account_balance()
        max_position_pct = 0.12 * confidence
        position_value = balance.cash * max_position_pct
        quantity = max(int(position_value / current_price), 1)
        
        # Cap at 20% of available cash
        max_affordable = int(balance.cash * 0.2 / current_price)
        quantity = min(quantity, max_affordable)
        
        if action == 'BUY':
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
                metadata={
                    'mode': 'production',
                    'confidence': confidence,
                    'reasoning': decision['reasoning'],
                    'real_price': current_price
                }
            )
            
            print(f"   📋 {symbol}: BUY {quantity} shares @ ${current_price:.2f}")
            print(f"       Position: ${quantity * current_price:,.2f} ({max_position_pct:.1%} of cash)")
            print(f"       Reasoning: {decision['reasoning']}")
            
            execution = await self.account.place_order(order)
            
            if execution.is_complete:
                total_cost = execution.filled_quantity * execution.avg_fill_price
                print(f"   ✅ Executed: {execution.filled_quantity} @ ${execution.avg_fill_price:.2f} = ${total_cost:,.2f}")
            else:
                print(f"   ❌ Failed: {execution.error_message}")
    
    async def _get_real_price(self, symbol: str) -> Optional[float]:
        """Get real current market price"""
        try:
            # Try current price first
            price = self.production_orchestrator.data_provider.get_current_price(symbol)
            if price and price > 0:
                return float(price)
            
            # Fallback to recent historical data
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=1)
            recent_data = self.production_orchestrator.data_provider.get_historical_data(
                symbol=symbol,
                start_date=start_date.to_pydatetime(),
                end_date=end_date.to_pydatetime()
            )
            
            if not recent_data.empty:
                close_col = 'Close' if 'Close' in recent_data.columns else 'close'
                return float(recent_data[close_col].iloc[-1])
            
            return None
            
        except Exception as e:
            print(f"   ⚠️ Price lookup failed for {symbol}: {e}")
            return None
    
    async def _display_account_status(self):
        """Display current account status"""
        balance = await self.account.get_account_balance()
        portfolio = await self.account.get_portfolio_summary()
        
        print(f"\n💰 ACCOUNT STATUS")
        print("-" * 30)
        print(f"Cash: ${balance.cash:,.2f}")
        print(f"Portfolio: ${balance.portfolio_value:,.2f}")
        print(f"Total: ${balance.total_equity:,.2f}")
        
        if portfolio.positions:
            print(f"\nPositions:")
            for pos in portfolio.positions:
                if not pos.is_flat:
                    pnl_color = "📈" if pos.unrealized_pnl >= 0 else "📉"
                    print(f"  {pos.symbol}: {pos.quantity} @ ${pos.avg_price:.2f} "
                          f"({pnl_color} ${pos.unrealized_pnl:,.2f})")
        else:
            print(f"\nPositions: None")


def show_help():
    """Show usage help"""
    print("""
🎯 UNIFIED PAPER TRADING DEMO

Usage: python paper_trading_demo.py [mode] [symbols]

Modes:
  simple     - Manual orders with real prices (quick test)
  basic      - ML model validation with simple logic
  production - Full analysis with real market data (default)

Examples:
  python paper_trading_demo.py                    # Interactive menu
  python paper_trading_demo.py simple             # Simple mode
  python paper_trading_demo.py basic 5            # Basic mode, 5 symbols
  python paper_trading_demo.py production 2       # Production mode, 2 symbols

Features:
  ✅ Real market data and prices
  ✅ Production ML model integration
  ✅ Market context analysis
  ✅ Risk-based position sizing
  ✅ No simulated data
""")


def show_interactive_menu():
    """Show interactive menu for demo options"""
    print("""
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

3️⃣  Reset Account
    🔄 Reset to initial state
    💰 Restore starting cash
    🗑️  Clear all positions

4️⃣  Help & Modes
    ❓ Show available modes
    📖 View usage examples

5️⃣  Exit
    👋 Close the application

═══════════════════════════════════════════════════════════════
""")


async def handle_account_reset(demo):
    """Handle account reset with confirmation"""
    print("\n" + "="*80)
    print("🔄 ACCOUNT RESET")
    print("="*80)
    
    # Show current state first
    await demo.account.connect()
    balance = await demo.account.get_account_balance()
    portfolio = await demo.account.get_portfolio_summary()
    
    print(f"\n📊 CURRENT ACCOUNT STATE:")
    print(f"   Cash: ${balance.cash:,.2f}")
    print(f"   Portfolio: ${balance.portfolio_value:,.2f}")
    print(f"   Total: ${balance.total_equity:,.2f}")
    print(f"   Positions: {len([p for p in portfolio.positions if not p.is_flat])}")
    
    await demo.account.disconnect()
    
    # Confirmation
    print(f"\n⚠️  WARNING: This will reset your paper trading account!")
    print(f"   • All positions will be closed")
    print(f"   • Trade history will be cleared") 
    print(f"   • Cash will be reset to ${demo.paper_config.initial_cash:,.2f}")
    print(f"   • Persistence file will be deleted")
    
    try:
        confirm = input(f"\n❓ Are you sure you want to reset? (type 'RESET' to confirm): ").strip()
        
        if confirm == 'RESET':
            demo.account.reset_account()
            print(f"\n✅ Account has been reset successfully!")
            print(f"💰 Starting fresh with ${demo.paper_config.initial_cash:,.2f}")
        else:
            print(f"\n❌ Reset cancelled")
            
    except (EOFError, KeyboardInterrupt):
        print(f"\n❌ Reset cancelled by user")


async def handle_account_status(demo):
    """Handle account status display"""
    print("\n" + "="*80)
    print("📊 ACCOUNT STATUS & PORTFOLIO SUMMARY")
    print("="*80)
    
    # Connect to get latest data
    await demo.account.connect()
    
    # Get account balance
    balance = await demo.account.get_account_balance()
    portfolio = await demo.account.get_portfolio_summary()
    
    # Display detailed account information
    print(f"\n💰 ACCOUNT BALANCE")
    print("-" * 40)
    print(f"Cash Available:    ${balance.cash:,.2f}")
    print(f"Portfolio Value:   ${balance.portfolio_value:,.2f}")
    print(f"Total Equity:      ${balance.total_equity:,.2f}")
    
    if balance.portfolio_value > 0:
        cash_pct = (balance.cash / balance.total_equity) * 100
        portfolio_pct = (balance.portfolio_value / balance.total_equity) * 100
        print(f"Cash Allocation:   {cash_pct:.1f}%")
        print(f"Portfolio Alloc:   {portfolio_pct:.1f}%")
    
    # Display positions
    print(f"\n📈 CURRENT POSITIONS")
    print("-" * 40)
    
    if portfolio.positions and any(not pos.is_flat for pos in portfolio.positions):
        total_pnl = 0
        position_count = 0
        
        for pos in portfolio.positions:
            if not pos.is_flat:
                position_count += 1
                pnl_color = "📈" if pos.unrealized_pnl >= 0 else "📉"
                pnl_pct = (pos.unrealized_pnl / (pos.quantity * pos.avg_price)) * 100 if pos.quantity * pos.avg_price > 0 else 0
                market_value = pos.quantity * pos.avg_price  # Simplified - would need current price for real market value
                
                print(f"  {pos.symbol:6} │ {pos.quantity:>6} shares │ Avg: ${pos.avg_price:>7.2f} │ "
                      f"Value: ${market_value:>8,.0f} │ {pnl_color} ${pos.unrealized_pnl:>7,.0f} ({pnl_pct:+.1f}%)")
                total_pnl += pos.unrealized_pnl
        
        print("-" * 40)
        print(f"📊 Portfolio Summary: {position_count} positions")
        pnl_color = "📈" if total_pnl >= 0 else "📉"
        print(f"🎯 Total P&L: {pnl_color} ${total_pnl:,.2f}")
        
        if balance.portfolio_value > 0:
            total_return_pct = (total_pnl / (balance.portfolio_value - total_pnl)) * 100
            print(f"📊 Total Return: {total_return_pct:+.2f}%")
    else:
        print("  No positions currently held")
        print("  💡 Use 'Start Trading' to begin building your portfolio")
    
    # Show performance summary
    performance = demo.account.get_performance_summary()
    print(f"\n📊 PERFORMANCE METRICS")
    print("-" * 40)
    print(f"Account Age:       {performance.get('trading_days', 0)} days")
    print(f"Total Transactions: {performance.get('total_transactions', 0)}")
    print(f"Buy Orders:        {performance.get('buy_transactions', 0)}")
    print(f"Sell Orders:       {performance.get('sell_transactions', 0)}")
    print(f"Total Fees:        ${performance.get('total_fees_paid', 0):.2f}")
    print(f"Total Return:      ${performance.get('total_return', 0):+,.2f} ({performance.get('total_return_pct', 0):+.2f}%)")
    
    # Show recent transactions
    recent_transactions = demo.account.get_transactions()[-5:]  # Last 5 transactions
    if recent_transactions:
        print(f"\n📋 RECENT TRANSACTIONS (Last 5)")
        print("-" * 40)
        for txn in recent_transactions:
            txn_time = txn.timestamp.strftime("%m/%d %H:%M")
            symbol_str = f"{txn.symbol:6}" if txn.symbol else "Account"
            amount_str = f"${abs(txn.amount):>8,.0f}"
            print(f"  {txn_time} │ {txn.transaction_type.value.upper():4} │ {symbol_str} │ {amount_str} │ {txn.description}")
    
    await demo.account.disconnect()
    
    print(f"\n✅ Account status retrieved successfully!")


async def handle_trading_menu(demo):
    """Handle trading mode selection"""
    print("\n" + "="*80)
    print("🤖 TRADING MODE SELECTION")
    print("="*80)
    
    print("""
Select trading mode:

1️⃣  Simple Trading
    📋 Manual orders with real market prices
    🎯 Fixed position sizes (good for testing)
    ⚡ Quick execution

2️⃣  Basic ML Trading  
    🤖 ML model validation with simple logic
    📊 Model accuracy-based decisions
    📈 Moderate sophistication

3️⃣  Production Trading
    🏭 Full analysis pipeline
    🤖 ML models + market context + technical analysis
    💼 Risk-based position sizing
    🎯 Production-ready logic

4️⃣  Custom Parameters
    ⚙️  Choose mode with custom symbol count

5️⃣  Back to Main Menu
    ⬅️  Return to main options

""")
    
    while True:
        try:
            choice = input("👉 Select trading mode (1-5): ").strip()
            
            if choice == '1':
                print("🚀 Starting Simple Trading Mode...")
                await demo.run_mode("simple", 3)
                break
            elif choice == '2':
                print("🚀 Starting Basic ML Trading Mode...")
                await demo.run_mode("basic", 3)
                break
            elif choice == '3':
                print("🚀 Starting Production Trading Mode...")
                await demo.run_mode("production", 3)
                break
            elif choice == '4':
                print("\n⚙️  Custom Parameters:")
                mode_choice = input("   Mode (simple/basic/production): ").strip().lower()
                if mode_choice in ['simple', 'basic', 'production']:
                    try:
                        symbol_count = int(input("   Number of symbols (1-10): ").strip())
                        if 1 <= symbol_count <= 10:
                            print(f"🚀 Starting {mode_choice.title()} Mode with {symbol_count} symbols...")
                            await demo.run_mode(mode_choice, symbol_count)
                            break
                        else:
                            print("   ❌ Symbol count must be between 1-10")
                    except ValueError:
                        print("   ❌ Invalid number")
                else:
                    print("   ❌ Invalid mode")
            elif choice == '5':
                return  # Back to main menu
            else:
                print("❌ Invalid choice. Please select 1-5.")
                
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Returning to main menu...")
            return


async def interactive_mode():
    """Run interactive demo mode"""
    demo = UnifiedPaperTradingDemo()
    
    print("🎯 UNIFIED PAPER TRADING DEMO")
    print(f"📊 Symbols: {', '.join(demo.symbols)}")
    print(f"💰 Capital: ${demo.paper_config.initial_cash:,.2f}")
    print("🚀 Using ProductionTradingOrchestrator")
    
    while True:
        show_interactive_menu()
        
        try:
            choice = input("👉 Please select an option (1-5): ").strip()
            
            if choice == '1':
                await handle_account_status(demo)
                input("\n📝 Press Enter to continue...")
                
            elif choice == '2':
                await handle_trading_menu(demo)
                input("\n📝 Press Enter to continue...")
                
            elif choice == '3':
                await handle_account_reset(demo)
                input("\n📝 Press Enter to continue...")
                
            elif choice == '4':
                show_help()
                input("\n📝 Press Enter to continue...")
                
            elif choice == '5':
                print("\n👋 Thank you for using Paper Trading Demo!")
                print("💡 Happy trading! 🚀")
                break
                
            else:
                print("\n❌ Invalid choice. Please select 1, 2, 3, 4, or 5.")
                input("📝 Press Enter to continue...")
                
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Demo interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("🔄 Returning to main menu...")
            input("📝 Press Enter to continue...")
async def main():
    """Main entry point"""
    # Parse arguments
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if args and args[0] in ['-h', '--help', 'help']:
        show_help()
        return
    
    # If no arguments provided, run interactive mode
    if not args:
        await interactive_mode()
        return
    
    # Parse mode and symbol limit for direct mode
    mode = args[0] if args else 'production'
    symbol_limit = int(args[1]) if len(args) > 1 and args[1].isdigit() else 3
    
    if mode not in ['simple', 'basic', 'production']:
        print(f"❌ Invalid mode: {mode}")
        show_help()
        return
    
    # Run demo directly
    demo = UnifiedPaperTradingDemo()
    await demo.run_mode(mode, symbol_limit)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
