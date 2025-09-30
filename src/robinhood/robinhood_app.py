"""
Robinhood Trading Application
Real-time trading application using Robinhood brokerage account
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.trading_app_base import TradingAppBase
from src.robinhood.robinhood_trading import RobinhoodTradingAccount, RobinhoodConfig


class RobinhoodTradingApp(TradingAppBase):
    """
    Robinhood Trading Application
    Real-time trading with Robinhood brokerage account
    """
    
    def __init__(self, username: str = "", password: str = "", starting_capital: float = 25000.0):
        """Initialize Robinhood trading application"""
        super().__init__(starting_capital=starting_capital, commission_rate=0.0)  # Robinhood = $0 commission
        
        # Robinhood specific configuration
        self.robinhood_config = RobinhoodConfig(
            username=username,
            password=password,
            device_token="",  # Would be set from config file or user input
            challenge_type="sms",
            enable_day_trading=True,  # Enable if account qualifies
            persistence_file="robinhood_account_state.json"
        )
        
        # Initialize Robinhood account
        self.account = RobinhoodTradingAccount(self.robinhood_config)
        
        print(f"🎯 ROBINHOOD TRADING APPLICATION")
        print(f"👤 Username: {username}")
        print(f"💰 Expected Capital: ${starting_capital:,.2f}")
        print("🚀 Ready for Real Trading")
    
    async def connect_account(self) -> bool:
        """Connect to Robinhood account"""
        return await self.account.connect()
    
    async def disconnect_account(self) -> bool:
        """Disconnect from Robinhood account"""
        return await self.account.disconnect()
    
    async def run_interactive_mode(self):
        """Run interactive trading mode with menu"""
        while True:
            self.show_interactive_menu()
            choice = input("\n🔷 Enter your choice (1-5): ").strip()
            
            if choice == "1":
                await self.handle_account_status()
            elif choice == "2":
                await self.handle_start_trading()
            elif choice == "3":
                await self.handle_account_settings()
            elif choice == "4":
                self.show_help()
            elif choice == "5":
                print("\n👋 Exiting Robinhood Trading Application...")
                break
            else:
                print("\n❌ Invalid choice. Please select 1-5.")
            
            input("\n📝 Press Enter to continue...")
    
    def show_interactive_menu(self):
        """Show interactive menu for Robinhood trading"""
        print("""
🎯 ROBINHOOD TRADING APPLICATION - INTERACTIVE MENU
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
    ⚡ Execute real trades

3️⃣  Account Settings
    ⚙️  View/Update configuration
    🔐 Security settings
    📱 Device management

4️⃣  Help & Information
    ❓ Show available modes
    📖 View usage examples
    ⚠️  Risk disclaimers

5️⃣  Exit
    👋 Close the application

═══════════════════════════════════════════════════════════════
""")
    
    async def handle_account_status(self):
        """Handle account status display"""
        print("\n" + "="*80)
        print("📊 ROBINHOOD ACCOUNT STATUS")
        print("="*80)
        
        if not await self.connect_account():
            print("❌ Failed to connect to Robinhood account")
            return
        
        try:
            # Display comprehensive account status
            await self.display_account_status()
            
            # Show performance summary
            performance = await self.get_account_performance_summary()
            
            print(f"\n📈 PERFORMANCE SUMMARY")
            print("-" * 40)
            print(f"Account Age: {performance.get('trading_days', 0)} days")
            print(f"Total Transactions: {performance.get('total_transactions', 0)}")
            print(f"Total Return: ${performance.get('total_return', 0):,.2f}")
            print(f"Return %: {performance.get('total_return_pct', 0):+.2f}%")
            print(f"Total Fees: ${performance.get('total_fees_paid', 0):,.2f}")
            
            # Show recent transactions
            if hasattr(self.account, 'get_transactions'):
                recent_transactions = self.account.get_transactions()[-5:]  # Last 5 transactions
                if recent_transactions:
                    print(f"\n💰 RECENT TRANSACTIONS")
                    print("-" * 40)
                    for txn in recent_transactions:
                        print(f"{txn.timestamp.strftime('%m/%d %H:%M')} | "
                              f"{txn.transaction_type.value.upper():4s} | "
                              f"{txn.symbol or 'CASH':6s} | "
                              f"${txn.amount:+8,.2f}")
            
        except Exception as e:
            print(f"❌ Error retrieving account status: {e}")
        finally:
            await self.disconnect_account()
    
    async def handle_start_trading(self):
        """Handle start trading workflow"""
        print("\n" + "="*80)
        print("🚀 START ROBINHOOD TRADING")
        print("="*80)
        
        # Trading mode selection
        print("\nSelect trading mode:")
        print("1. Simple Mode    - Manual orders with real prices")
        print("2. Basic Mode     - ML model validation")
        print("3. Production Mode - Full analysis + automated trading")
        
        mode_choice = input("\nEnter mode (1-3) [3]: ").strip() or "3"
        mode_map = {"1": "simple", "2": "basic", "3": "production"}
        mode = mode_map.get(mode_choice, "production")
        
        # Symbol limit
        symbol_limit = input(f"\nNumber of symbols to analyze [3]: ").strip() or "3"
        try:
            symbol_limit = int(symbol_limit)
        except ValueError:
            symbol_limit = 3
        
        # Risk confirmation for real trading
        print(f"\n⚠️  RISK WARNING:")
        print(f"You are about to execute REAL TRADES on your Robinhood account!")
        print(f"Mode: {mode.upper()}")
        print(f"Symbols: {symbol_limit}")
        print(f"This will use real money and execute actual trades.")
        
        confirm = input(f"\nType 'CONFIRM' to proceed with real trading: ").strip()
        if confirm != "CONFIRM":
            print("❌ Trading cancelled by user")
            return
        
        try:
            # Run the selected trading mode
            await self.run_trading_mode(mode, symbol_limit)
            
        except Exception as e:
            print(f"❌ Trading execution failed: {e}")
    
    async def handle_account_settings(self):
        """Handle account settings and configuration"""
        print("\n" + "="*80)
        print("⚙️  ROBINHOOD ACCOUNT SETTINGS")
        print("="*80)
        
        print(f"\n📋 Current Configuration:")
        print(f"Username: {self.robinhood_config.username}")
        print(f"Challenge Type: {self.robinhood_config.challenge_type}")
        print(f"Day Trading: {'Enabled' if self.robinhood_config.enable_day_trading else 'Disabled'}")
        print(f"Max Order Value: ${self.robinhood_config.max_order_value:,.2f}")
        print(f"Min Order Value: ${self.robinhood_config.min_order_value:,.2f}")
        print(f"API Timeout: {self.robinhood_config.api_timeout}s")
        
        print(f"\n🔐 Security Information:")
        print(f"Device Token: {'Set' if self.robinhood_config.device_token else 'Not Set'}")
        print(f"Password: {'Set' if self.robinhood_config.password else 'Not Set'}")
        
        print(f"\n💾 Data Management:")
        print(f"Persistence File: {self.robinhood_config.persistence_file}")
        print(f"Portfolio Tracking: {'Enabled' if self.robinhood_config.track_portfolio_snapshots else 'Disabled'}")
        
        # Settings modification options
        print(f"\nSettings Options:")
        print("1. Update credentials")
        print("2. Toggle day trading")
        print("3. Update order limits")
        print("4. Back to main menu")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            await self._update_credentials()
        elif choice == "2":
            self.robinhood_config.enable_day_trading = not self.robinhood_config.enable_day_trading
            print(f"Day trading {'enabled' if self.robinhood_config.enable_day_trading else 'disabled'}")
        elif choice == "3":
            await self._update_order_limits()
        else:
            return
    
    async def _update_credentials(self):
        """Update Robinhood credentials"""
        print("\n🔐 Update Credentials")
        print("⚠️  Warning: Credentials are stored in plain text for demo purposes")
        
        new_username = input(f"Username [{self.robinhood_config.username}]: ").strip()
        if new_username:
            self.robinhood_config.username = new_username
        
        new_password = input("Password (leave blank to keep current): ").strip()
        if new_password:
            self.robinhood_config.password = new_password
        
        challenge_type = input(f"Challenge type (sms/email) [{self.robinhood_config.challenge_type}]: ").strip()
        if challenge_type in ['sms', 'email']:
            self.robinhood_config.challenge_type = challenge_type
        
        print("✅ Credentials updated")
    
    async def _update_order_limits(self):
        """Update order limits"""
        print("\n💰 Update Order Limits")
        
        try:
            min_val = input(f"Minimum order value [{self.robinhood_config.min_order_value}]: ").strip()
            if min_val:
                self.robinhood_config.min_order_value = float(min_val)
            
            max_val = input(f"Maximum order value [{self.robinhood_config.max_order_value}]: ").strip()
            if max_val:
                self.robinhood_config.max_order_value = float(max_val)
            
            print("✅ Order limits updated")
            
        except ValueError:
            print("❌ Invalid input. Order limits not changed.")
    
    def show_help(self):
        """Show Robinhood trading help"""
        print("""
🎯 ROBINHOOD TRADING APPLICATION HELP

OVERVIEW:
This application provides automated trading capabilities using your Robinhood
brokerage account. It integrates machine learning models with real market data
to generate and execute trading signals.

TRADING MODES:
• Simple Mode: Execute manual orders with real market prices
• Basic Mode: Use ML models to validate trading decisions
• Production Mode: Full automated trading with comprehensive analysis

FEATURES:
✅ Real-time market data integration
✅ Machine learning model predictions
✅ Market context analysis
✅ Risk-based position sizing
✅ Transaction tracking and analytics
✅ Portfolio performance monitoring

IMPORTANT DISCLAIMERS:
⚠️  This software executes REAL TRADES with REAL MONEY
⚠️  Past performance does not guarantee future results
⚠️  Trading involves substantial risk of loss
⚠️  Only trade with money you can afford to lose
⚠️  Robinhood API integration is in DEMO MODE

SAFETY FEATURES:
• Order value limits (min/max)
• Position size restrictions
• Real-time validation
• Transaction logging
• Account state persistence

For support or questions, please refer to the documentation.
""")


async def main():
    """Main entry point for Robinhood trading application"""
    print("🎯 ROBINHOOD TRADING APPLICATION")
    print("="*80)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ["help", "--help", "-h"]:
            RobinhoodTradingApp().show_help()
            return
        
        # Direct mode execution
        username = input("Robinhood Username: ").strip()
        if not username:
            print("❌ Username required")
            return
        
        app = RobinhoodTradingApp(username=username)
        
        mode = sys.argv[1] if sys.argv[1] in ["simple", "basic", "production"] else "production"
        symbol_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        
        print(f"\n⚠️  RISK WARNING:")
        print(f"You are about to execute REAL TRADES on your Robinhood account!")
        confirm = input(f"Type 'CONFIRM' to proceed: ").strip()
        if confirm != "CONFIRM":
            print("❌ Trading cancelled")
            return
        
        await app.run_trading_mode(mode, symbol_limit)
    else:
        # Interactive mode
        username = input("Robinhood Username (or 'demo' for demo mode): ").strip()
        if not username:
            print("❌ Username required")
            return
        
        app = RobinhoodTradingApp(username=username)
        await app.run_interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())