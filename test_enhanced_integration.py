"""
Test Enhanced ML Trading Strategy Integration
Tests the integration of EnhancedMLTradingStrategy into TradingAppBase
"""

import os
import sys
import asyncio
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.trading.trading_app_base import TradingAppBase
from src.interfaces.real_trading import TradingAccountInterface, AccountBalance, OrderExecution, AccountType, PortfolioSummary, Trade, OrderStatus
from src.interfaces.trading_strategy import Order, OrderType, OrderSide, Position
import datetime as dt


class TestTradingAccount(TradingAccountInterface):
    """Test implementation of trading account for testing"""
    
    def __init__(self, starting_balance: float = 10000.0):
        super().__init__(AccountType.PAPER)
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.positions = {}
        self.orders = []
        self.trades = []
    
    async def connect(self) -> bool:
        """Connect to test account"""
        print("🔗 Connected to Test Trading Account")
        self._is_connected = True
        return True
    
    async def disconnect(self) -> bool:
        """Disconnect from test account"""
        print("🔌 Disconnected from Test Trading Account")
        self._is_connected = False
        return True
    
    async def get_account_balance(self) -> AccountBalance:
        """Get current account balance"""
        return AccountBalance(
            cash=self.current_balance,
            buying_power=self.current_balance * 2,  # 2:1 margin
            portfolio_value=0.0,  # No positions yet
            day_trade_buying_power=self.current_balance * 4,  # 4:1 day trading
            unsettled_funds=0.0,
            timestamp=pd.Timestamp.now()
        )
    
    async def get_portfolio_summary(self) -> PortfolioSummary:
        """Get portfolio summary"""
        positions = [Position(symbol=sym, quantity=qty, avg_price=100.0, current_price=100.0) 
                    for sym, qty in self.positions.items()]
        return PortfolioSummary(
            total_equity=self.current_balance,
            cash_balance=self.current_balance,
            positions_value=0.0,
            day_pnl=0.0,
            total_pnl=0.0,
            positions=positions,
            timestamp=pd.Timestamp.now()
        )
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        if symbol in self.positions:
            return Position(
                symbol=symbol,
                quantity=self.positions[symbol],
                avg_price=100.0,
                current_price=100.0
            )
        return None
    
    async def get_all_positions(self) -> List[Position]:
        """Get all positions"""
        return [Position(symbol=sym, quantity=qty, avg_price=100.0, current_price=100.0) 
                for sym, qty in self.positions.items()]
    
    async def place_order(self, order: Order) -> OrderExecution:
        """Place a test order"""
        # Simulate execution
        execution = OrderExecution(
            order_id=f"TEST_{len(self.orders)}",
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            remaining_quantity=0,
            avg_fill_price=150.0,  # Mock price
            commission=1.0,
            timestamp=pd.Timestamp.now(),
            error_message=None
        )
        
        # Update balance for BUY orders
        if order.side == OrderSide.BUY:
            cost = execution.filled_quantity * execution.avg_fill_price + execution.commission
            self.current_balance -= cost
            print(f"💰 Balance after order: ${self.current_balance:,.2f}")
        
        self.orders.append(order)
        return execution
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        return True
    
    async def get_order_status(self, order_id: str) -> OrderExecution:
        """Get order status"""
        return OrderExecution(
            order_id=order_id,
            status=OrderStatus.FILLED,
            filled_quantity=10,
            remaining_quantity=0,
            avg_fill_price=150.0,
            commission=1.0,
            timestamp=pd.Timestamp.now()
        )
    
    async def get_trade_history(
        self, 
        symbol: Optional[str] = None,
        start_date: Optional[dt.datetime] = None,
        end_date: Optional[dt.datetime] = None
    ) -> List[Trade]:
        """Get trade history"""
        return self.trades
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for symbol"""
        return {
            'symbol': symbol,
            'price': 150.0,
            'bid': 149.95,
            'ask': 150.05,
            'volume': 1000000
        }
    
    def get_account_type(self) -> AccountType:
        """Get account type"""
        return AccountType.PAPER


class TestTradingApp(TradingAppBase):
    """Test implementation of trading app"""
    
    def __init__(self):
        super().__init__(starting_capital=10000.0, commission_rate=0.0)
        self.test_account = TestTradingAccount(10000.0)
        self.account = self.test_account
    
    async def connect_account(self) -> bool:
        """Connect to test account"""
        return await self.test_account.connect()
    
    async def disconnect_account(self) -> bool:
        """Disconnect from test account"""
        return await self.test_account.disconnect()
    
    async def display_account_status(self):
        """Display test account status"""
        balance = await self.account.get_account_balance()
        print(f"\n💼 TEST ACCOUNT STATUS")
        print(f"   Cash: ${balance.cash:,.2f}")
        print(f"   Buying Power: ${balance.buying_power:,.2f}")
        print(f"   Orders Placed: {len(self.test_account.orders)}")


async def test_enhanced_integration():
    """Test the enhanced ML trading strategy integration"""
    
    print("=" * 80)
    print("🧪 TESTING ENHANCED ML TRADING STRATEGY INTEGRATION")
    print("=" * 80)
    
    # Initialize test app
    print("\n1. Initializing Test Trading App...")
    app = TestTradingApp()
    
    # Test connection
    print("\n2. Testing Account Connection...")
    connected = await app.connect_account()
    if connected:
        print("   ✅ Successfully connected to test account")
    else:
        print("   ❌ Failed to connect to test account")
        return
    
    # Display initial status
    await app.display_account_status()
    
    # Test enhanced decision making
    print("\n3. Testing Enhanced Decision Making...")
    
    test_symbol = "AAPL"
    print(f"\n   Testing symbol: {test_symbol}")
    
    # Mock ML results
    mock_ml_results = {
        'success': True,
        'test_accuracy': 0.75,
        'train_accuracy': 0.80,
        'model_version': 'v1',
        'feature_count': 7
    }
    
    # Test enhanced decision using orchestrator
    try:
        print(f"\n   🧠 Making enhanced decision via orchestrator...")
        
        # The enhanced decision now uses orchestrator internally
        decision = await app.make_enhanced_decision(test_symbol, mock_ml_results)
        
        print(f"   Decision Result:")
        print(f"     Symbol: {decision['symbol']}")
        print(f"     Action: {decision['action']}")
        print(f"     Confidence: {decision['confidence']:.3f}")
        print(f"     Reasoning: {decision['reasoning']}")
        
        if 'signal_type' in decision:
            print(f"     Signal Type: {decision['signal_type']}")
        
        if 'strength' in decision:
            print(f"     Strength: {decision['strength']:.3f}")
        
    except Exception as e:
        print(f"   ❌ Enhanced decision failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test production decision with market context using orchestrator
    print(f"\n4. Testing Production Decision via Orchestrator...")
    
    # Since orchestrator handles market analysis internally, we just need minimal mock data
    mock_market_analysis = {}
    
    try:
        print(f"   🏭 Making production decision...")
        production_decision = await app.make_production_decision(
            test_symbol, mock_ml_results, mock_market_analysis
        )
        
        print(f"   Production Decision Result:")
        print(f"     Symbol: {production_decision['symbol']}")
        print(f"     Action: {production_decision['action']}")
        print(f"     Confidence: {production_decision['confidence']:.3f}")
        print(f"     Market Boost: {production_decision.get('market_context_boost', 1.0):.2f}x")
        print(f"     Reasoning: {production_decision['reasoning']}")
        
    except Exception as e:
        print(f"   ❌ Production decision failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test order execution if we have a BUY decision
    if 'production_decision' in locals() and production_decision['action'] == 'BUY':
        print(f"\n5. Testing Order Execution...")
        try:
            await app.execute_production_decision(production_decision)
            print(f"   ✅ Order execution completed")
        except Exception as e:
            print(f"   ❌ Order execution failed: {e}")
    
    # Final status
    print(f"\n6. Final Account Status...")
    await app.display_account_status()
    
    # Disconnect
    print(f"\n7. Disconnecting...")
    await app.disconnect_account()
    
    print(f"\n" + "=" * 80)
    print("🎉 ENHANCED ML TRADING STRATEGY INTEGRATION TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_enhanced_integration())