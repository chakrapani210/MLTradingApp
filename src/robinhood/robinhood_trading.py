"""
Robinhood Trading Account Implementation
Real trading with Robinhood API
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from decimal import Decimal
import datetime as dt
import json
import os
from enum import Enum

from ..interfaces.real_trading import (
    TradingAccountInterface, 
    AccountType, 
    OrderStatus, 
    Trade, 
    OrderExecution, 
    AccountBalance, 
    PortfolioSummary
)
from ..interfaces.trading_strategy import Order, Position, OrderType, OrderSide
from ..data.providers import YFinanceProvider
from ..trading.paper_trading import Transaction, PortfolioSnapshot, TransactionType


@dataclass
class RobinhoodConfig:
    """Configuration for Robinhood trading"""
    username: str = ""
    password: str = ""
    device_token: str = ""
    challenge_type: str = "sms"  # 'sms' or 'email'
    api_timeout: int = 30
    max_order_value: float = 50000.0
    min_order_value: float = 1.0
    enable_day_trading: bool = False
    persistence_file: str = "robinhood_account_state.json"
    track_portfolio_snapshots: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'username': self.username,
            'device_token': self.device_token,
            'challenge_type': self.challenge_type,
            'api_timeout': self.api_timeout,
            'max_order_value': self.max_order_value,
            'min_order_value': self.min_order_value,
            'enable_day_trading': self.enable_day_trading,
            'persistence_file': self.persistence_file,
            'track_portfolio_snapshots': self.track_portfolio_snapshots
        }


class RobinhoodTradingAccount(TradingAccountInterface):
    """
    Robinhood trading account implementation
    Real trading with Robinhood API
    """
    
    def __init__(self, config: RobinhoodConfig = None):
        super().__init__(AccountType.REAL)
        self.config = config or RobinhoodConfig()
        
        # Account state
        self._cash = 0.0
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, OrderExecution] = {}
        self._trades: List[Trade] = []
        
        # Enhanced tracking for charts and analysis
        self._transactions: List[Transaction] = []
        self._portfolio_snapshots: List[PortfolioSnapshot] = []
        self._last_snapshot_time: Optional[pd.Timestamp] = None
        
        # Market data provider for validation
        self._market_data_provider = YFinanceProvider()
        
        # Robinhood API client (placeholder - would need actual Robinhood API)
        self._rh_client = None
        
        # State tracking
        self._account_created_at = pd.Timestamp.now()
        self._last_update = pd.Timestamp.now()
        
        # Load persisted state if available
        self._load_account_state()
        
        print(f"[ROBINHOOD_ACCOUNT] Initialized (persistence: {os.path.exists(self.config.persistence_file)})")
        print(f"[ROBINHOOD_ACCOUNT] Tracking: {len(self._transactions)} transactions, {len(self._portfolio_snapshots)} snapshots")
        print("⚠️  [ROBINHOOD_ACCOUNT] Demo Mode - Not connected to real Robinhood API")
    
    async def connect(self) -> bool:
        """Connect to Robinhood API"""
        try:
            # In a real implementation, this would connect to Robinhood API
            # For now, we'll simulate the connection
            
            print(f"[ROBINHOOD_ACCOUNT] Connecting to Robinhood API...")
            print(f"[ROBINHOOD_ACCOUNT] Username: {self.config.username}")
            print(f"[ROBINHOOD_ACCOUNT] ⚠️  DEMO MODE: Simulating Robinhood connection")
            
            # Simulate authentication process
            await asyncio.sleep(1)  # Simulate API call delay
            
            # Initialize with demo data
            await self._initialize_demo_account()
            
            self._is_connected = True
            print(f"[ROBINHOOD_ACCOUNT] ✅ Connected successfully (Demo Mode)")
            return True
            
        except Exception as e:
            print(f"[ROBINHOOD_ACCOUNT] ❌ Connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Robinhood API"""
        # Take final snapshot
        if self.config.track_portfolio_snapshots:
            self._take_portfolio_snapshot("Session ended")
        
        # Save account state before disconnecting
        self._save_account_state()
        self._is_connected = False
        print(f"[ROBINHOOD_ACCOUNT] Disconnected and state saved")
        return True
    
    async def _initialize_demo_account(self):
        """Initialize demo account with realistic data"""
        # In real implementation, this would fetch actual account data from Robinhood
        if not self._cash:  # Only initialize if not loaded from persistence
            self._cash = 25000.0  # Realistic starting amount for day trading
            
            # Add some demo positions
            demo_positions = {
                'AAPL': Position(
                    symbol='AAPL',
                    quantity=50,
                    avg_price=175.00,
                    market_value=50 * 175.00,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    timestamp=pd.Timestamp.now()
                ),
                'TSLA': Position(
                    symbol='TSLA',
                    quantity=25,
                    avg_price=240.00,
                    market_value=25 * 240.00,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    timestamp=pd.Timestamp.now()
                )
            }
            
            for symbol, position in demo_positions.items():
                self._positions[symbol] = position
            
            print(f"[ROBINHOOD_ACCOUNT] Demo account initialized with ${self._cash:,.2f} cash")
    
    def _take_portfolio_snapshot(self, description: str = "Transaction snapshot"):
        """Take a portfolio value snapshot for tracking over time"""
        try:
            current_time = pd.Timestamp.now()
            
            # Calculate current portfolio metrics
            portfolio_value = sum(pos.market_value for pos in self._positions.values())
            total_equity = self._cash + portfolio_value
            positions_count = len([p for p in self._positions.values() if not p.is_flat])
            
            # Calculate P&L (for Robinhood, we'd get this from API)
            initial_value = 25000.0  # Demo starting value
            total_pnl = total_equity - initial_value
            
            # Day P&L (simplified as unrealized P&L for now)
            day_pnl = sum(pos.unrealized_pnl for pos in self._positions.values())
            
            # Create snapshot
            snapshot = PortfolioSnapshot(
                timestamp=current_time,
                cash_balance=self._cash,
                portfolio_value=portfolio_value,
                total_equity=total_equity,
                positions_count=positions_count,
                day_pnl=day_pnl,
                total_pnl=total_pnl
            )
            
            self._portfolio_snapshots.append(snapshot)
            self._last_snapshot_time = current_time
            
            print(f"[PORTFOLIO_SNAPSHOT] {description}: ${total_equity:,.2f} total equity, {positions_count} positions")
            
        except Exception as e:
            print(f"[PORTFOLIO_SNAPSHOT] Failed to take snapshot: {e}")
    
    def _record_transaction(self, transaction_type: TransactionType, symbol: Optional[str], 
                          quantity: float, price: float, fees: float, description: str):
        """Record a detailed transaction for analysis and charting"""
        try:
            # Calculate amounts
            amount = abs(quantity * price) + fees
            if transaction_type in [TransactionType.SELL, TransactionType.WITHDRAWAL]:
                amount = -amount
            
            # Get current portfolio state
            portfolio_value = sum(pos.market_value for pos in self._positions.values())
            total_equity = self._cash + portfolio_value
            
            # Create transaction record
            transaction = Transaction(
                transaction_id=str(uuid.uuid4()),
                timestamp=pd.Timestamp.now(),
                transaction_type=transaction_type,
                symbol=symbol,
                quantity=quantity,
                price=price,
                amount=amount,
                fees=fees,
                description=description,
                account_balance_after=self._cash,
                portfolio_value_after=portfolio_value,
                total_equity_after=total_equity
            )
            
            self._transactions.append(transaction)
            
            print(f"[TRANSACTION] {transaction_type.value.upper()}: {symbol or 'Account'} "
                  f"${abs(amount):,.2f} - {description}")
            
            # Take snapshot after recording transaction
            self._take_portfolio_snapshot(f"Transaction: {transaction_type.value} {symbol or 'Account'}")
            
        except Exception as e:
            print(f"[TRANSACTION] Failed to record transaction: {e}")
    
    def _save_account_state(self):
        """Save current account state to persistence file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config.persistence_file), exist_ok=True)
            
            # Prepare state data
            state_data = {
                'config': self.config.to_dict(),
                'cash': float(self._cash),
                'account_created_at': self._account_created_at.isoformat(),
                'last_update': pd.Timestamp.now().isoformat(),
                'positions': {},
                'trades': [],
                'transactions': [],
                'portfolio_snapshots': [],
                'order_count': len(self._orders)
            }
            
            # Serialize positions
            for symbol, position in self._positions.items():
                if not position.is_flat:
                    # Get current market price for Robinhood positions
                    current_price = position.avg_price
                    try:
                        current_data = self._market_data_provider.get_current_price(symbol)
                        if current_data and current_data > 0:
                            current_price = float(current_data)
                    except:
                        pass
                    
                    state_data['positions'][symbol] = {
                        'symbol': position.symbol,
                        'quantity': float(position.quantity),
                        'avg_price': float(position.avg_price),
                        'market_value': float(position.market_value),
                        'unrealized_pnl': float(position.unrealized_pnl),
                        'realized_pnl': float(position.realized_pnl),
                        'current_price': current_price,
                        'timestamp': position.timestamp.isoformat() if hasattr(position, 'timestamp') else pd.Timestamp.now().isoformat()
                    }
            
            # Serialize recent trades (last 100)
            for trade in self._trades[-100:]:
                state_data['trades'].append({
                    'trade_id': trade.trade_id,
                    'order_id': trade.order_id,
                    'symbol': trade.symbol,
                    'side': trade.side.value,
                    'quantity': float(trade.quantity),
                    'price': float(trade.price),
                    'timestamp': trade.timestamp.isoformat(),
                    'commission': float(trade.commission) if trade.commission else 0.0
                })
            
            # Serialize transactions (last 500)
            for transaction in self._transactions[-500:]:
                state_data['transactions'].append(transaction.to_dict())
            
            # Serialize portfolio snapshots (last 100)
            for snapshot in self._portfolio_snapshots[-100:]:
                state_data['portfolio_snapshots'].append(snapshot.to_dict())
            
            # Write to file
            with open(self.config.persistence_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            print(f"[ROBINHOOD_ACCOUNT] State saved to {self.config.persistence_file}")
            
        except Exception as e:
            print(f"[ROBINHOOD_ACCOUNT] Failed to save state: {e}")
    
    def _load_account_state(self):
        """Load account state from persistence file"""
        try:
            if not os.path.exists(self.config.persistence_file):
                print(f"[ROBINHOOD_ACCOUNT] No existing state file found, starting fresh")
                return
            
            with open(self.config.persistence_file, 'r') as f:
                state_data = json.load(f)
            
            # Load basic account data
            self._cash = state_data.get('cash', 0.0)
            self._account_created_at = pd.Timestamp(state_data.get('account_created_at', pd.Timestamp.now().isoformat()))
            self._last_update = pd.Timestamp(state_data.get('last_update', pd.Timestamp.now().isoformat()))
            
            # Load positions
            self._positions = {}
            for symbol, pos_data in state_data.get('positions', {}).items():
                position = Position(
                    symbol=pos_data['symbol'],
                    quantity=pos_data['quantity'],
                    avg_price=pos_data['avg_price'],
                    market_value=pos_data['market_value'],
                    unrealized_pnl=pos_data['unrealized_pnl'],
                    realized_pnl=pos_data['realized_pnl'],
                    timestamp=pd.Timestamp(pos_data.get('timestamp', pd.Timestamp.now().isoformat()))
                )
                self._positions[symbol] = position
            
            # Load recent trades
            self._trades = []
            for trade_data in state_data.get('trades', []):
                trade = Trade(
                    trade_id=trade_data['trade_id'],
                    order_id=trade_data['order_id'],
                    symbol=trade_data['symbol'],
                    side=OrderSide(trade_data['side']),
                    quantity=trade_data['quantity'],
                    price=trade_data['price'],
                    timestamp=pd.Timestamp(trade_data['timestamp']),
                    commission=trade_data.get('commission', 0.0)
                )
                self._trades.append(trade)
            
            # Load transactions
            self._transactions = []
            for transaction_data in state_data.get('transactions', []):
                transaction = Transaction.from_dict(transaction_data)
                self._transactions.append(transaction)
            
            # Load portfolio snapshots
            self._portfolio_snapshots = []
            for snapshot_data in state_data.get('portfolio_snapshots', []):
                snapshot = PortfolioSnapshot.from_dict(snapshot_data)
                self._portfolio_snapshots.append(snapshot)
            
            # Set last snapshot time
            if self._portfolio_snapshots:
                self._last_snapshot_time = self._portfolio_snapshots[-1].timestamp
            
            print(f"[ROBINHOOD_ACCOUNT] State loaded from {self.config.persistence_file}")
            print(f"[ROBINHOOD_ACCOUNT] Loaded: ${self._cash:,.2f} cash, {len(self._positions)} positions, "
                  f"{len(self._trades)} trades, {len(self._transactions)} transactions, "
                  f"{len(self._portfolio_snapshots)} snapshots")
            
        except Exception as e:
            print(f"[ROBINHOOD_ACCOUNT] Failed to load state: {e}")
            print(f"[ROBINHOOD_ACCOUNT] Starting with fresh account")
            # Reset to defaults on load failure
            self._cash = 0.0
            self._positions = {}
            self._trades = []
            self._transactions = []
            self._portfolio_snapshots = []
            self._last_snapshot_time = None
    
    async def get_account_balance(self) -> AccountBalance:
        """Get current account balance"""
        # In real implementation, this would call Robinhood API
        await self.update_positions_market_value()
        
        portfolio_value = sum(pos.market_value for pos in self._positions.values())
        total_equity = self._cash + portfolio_value
        
        # Robinhood specific: Calculate day trade buying power
        day_trade_buying_power = self._cash * 4 if self.config.enable_day_trading else self._cash * 2
        
        return AccountBalance(
            cash=self._cash,
            buying_power=self._cash * 2,  # Standard margin
            portfolio_value=portfolio_value,
            day_trade_buying_power=day_trade_buying_power,
            unsettled_funds=0.0,  # Would get from Robinhood API
            timestamp=pd.Timestamp.now()
        )
    
    async def get_portfolio_summary(self) -> PortfolioSummary:
        """Get portfolio summary"""
        await self.update_positions_market_value()
        
        positions = list(self._positions.values())
        positions_value = sum(pos.market_value for pos in positions)
        total_equity = self._cash + positions_value
        
        # Calculate P&L
        total_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in positions)
        day_pnl = sum(pos.unrealized_pnl for pos in positions)
        
        return PortfolioSummary(
            total_equity=total_equity,
            cash_balance=self._cash,
            positions_value=positions_value,
            day_pnl=day_pnl,
            total_pnl=total_pnl,
            positions=positions,
            timestamp=pd.Timestamp.now()
        )
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        return self._positions.get(symbol)
    
    async def get_all_positions(self) -> List[Position]:
        """Get all positions"""
        return list(self._positions.values())
    
    async def place_order(self, order: Order) -> OrderExecution:
        """Place trading order via Robinhood API"""
        try:
            order_id = str(uuid.uuid4())
            timestamp = pd.Timestamp.now()
            
            print(f"[ROBINHOOD_ORDER] Placing {order.side.value} order for {order.quantity} {order.symbol}")
            print(f"[ROBINHOOD_ORDER] ⚠️  DEMO MODE: Simulating Robinhood order execution")
            
            # In real implementation, this would call Robinhood API
            # For demo, we'll simulate with market data
            market_data = await self.get_market_data(order.symbol)
            if not market_data:
                return OrderExecution(
                    order_id=order_id,
                    status=OrderStatus.REJECTED,
                    error_message=f"Unable to get market data for {order.symbol}",
                    timestamp=timestamp
                )
            
            current_price = market_data['price']
            
            # Simulate order execution
            if order.order_type == OrderType.MARKET:
                execution_price = current_price  # Robinhood typically gets good fills
            elif order.order_type == OrderType.LIMIT:
                if order.price is None:
                    return OrderExecution(
                        order_id=order_id,
                        status=OrderStatus.REJECTED,
                        error_message="Limit price required for limit order",
                        timestamp=timestamp
                    )
                execution_price = order.price
            else:
                return OrderExecution(
                    order_id=order_id,
                    status=OrderStatus.REJECTED,
                    error_message=f"Order type {order.order_type} not supported in demo",
                    timestamp=timestamp
                )
            
            # Robinhood commission (typically $0)
            commission = 0.0
            
            # Validate order
            if not self._validate_order(order, execution_price, commission):
                return OrderExecution(
                    order_id=order_id,
                    status=OrderStatus.REJECTED,
                    error_message="Insufficient funds or invalid order",
                    timestamp=timestamp
                )
            
            # Execute the trade
            trade = Trade(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                timestamp=timestamp,
                order_id=order_id,
                trade_id=str(uuid.uuid4()),
                commission=commission
            )
            
            # Update positions and cash
            self._execute_trade(trade)
            
            # Create execution result
            execution = OrderExecution(
                order_id=order_id,
                status=OrderStatus.FILLED,
                filled_quantity=order.quantity,
                remaining_quantity=0,
                avg_fill_price=execution_price,
                commission=commission,
                timestamp=timestamp,
                trades=[trade]
            )
            
            self._orders[order_id] = execution
            self._trades.append(trade)
            
            # Record detailed transaction
            transaction_type = TransactionType.BUY if order.side == OrderSide.BUY else TransactionType.SELL
            fees = execution.commission or 0.0
            description = f"Robinhood {order.order_type.value.title()} order execution"
            
            self._record_transaction(
                transaction_type=transaction_type,
                symbol=order.symbol,
                quantity=execution.filled_quantity,
                price=execution_price,
                fees=fees,
                description=description
            )
            
            # Auto-save state after each trade
            self._save_account_state()
            
            print(f"[ROBINHOOD_ORDER] ✅ Executed {order.side.value} {order.quantity} {order.symbol} @ ${execution_price:.2f}")
            
            return execution
            
        except Exception as e:
            print(f"[ROBINHOOD_ORDER] ❌ Order execution failed: {e}")
            return OrderExecution(
                order_id=str(uuid.uuid4()),
                status=OrderStatus.REJECTED,
                error_message=str(e),
                timestamp=pd.Timestamp.now()
            )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        # In real implementation, this would call Robinhood API
        if order_id in self._orders:
            execution = self._orders[order_id]
            if execution.status == OrderStatus.PENDING:
                execution.status = OrderStatus.CANCELLED
                execution.timestamp = pd.Timestamp.now()
                print(f"[ROBINHOOD_ORDER] Cancelled order {order_id}")
                return True
        return False
    
    async def get_order_status(self, order_id: str) -> OrderExecution:
        """Get order status"""
        return self._orders.get(order_id)
    
    async def get_trade_history(
        self, 
        symbol: Optional[str] = None,
        start_date: Optional[dt.datetime] = None,
        end_date: Optional[dt.datetime] = None
    ) -> List[Trade]:
        """Get trade history"""
        trades = self._trades.copy()
        
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        if start_date:
            trades = [t for t in trades if t.timestamp >= pd.Timestamp(start_date)]
        
        if end_date:
            trades = [t for t in trades if t.timestamp <= pd.Timestamp(end_date)]
        
        return trades
    
    def get_transactions(
        self, 
        start_date: Optional[dt.datetime] = None,
        end_date: Optional[dt.datetime] = None,
        symbol: Optional[str] = None,
        transaction_type: Optional[TransactionType] = None
    ) -> List[Transaction]:
        """Get filtered transaction history"""
        transactions = self._transactions.copy()
        
        if start_date:
            transactions = [t for t in transactions if t.timestamp >= pd.Timestamp(start_date)]
        
        if end_date:
            transactions = [t for t in transactions if t.timestamp <= pd.Timestamp(end_date)]
        
        if symbol:
            transactions = [t for t in transactions if t.symbol == symbol]
        
        if transaction_type:
            transactions = [t for t in transactions if t.transaction_type == transaction_type]
        
        return transactions
    
    def get_portfolio_snapshots(
        self,
        start_date: Optional[dt.datetime] = None,
        end_date: Optional[dt.datetime] = None
    ) -> List[PortfolioSnapshot]:
        """Get portfolio value snapshots for charting"""
        snapshots = self._portfolio_snapshots.copy()
        
        if start_date:
            snapshots = [s for s in snapshots if s.timestamp >= pd.Timestamp(start_date)]
        
        if end_date:
            snapshots = [s for s in snapshots if s.timestamp <= pd.Timestamp(end_date)]
        
        return snapshots
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for analysis"""
        if not self._transactions:
            return {
                'total_transactions': 0,
                'total_equity': self._cash,
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'positions_count': 0,
                'snapshots_count': 0
            }
        
        # Calculate performance metrics
        current_portfolio_value = sum(pos.market_value for pos in self._positions.values())
        current_total_equity = self._cash + current_portfolio_value
        initial_value = 25000.0  # Demo starting value
        total_return = current_total_equity - initial_value
        total_return_pct = (total_return / initial_value) * 100 if initial_value > 0 else 0.0
        
        # Transaction statistics
        buy_transactions = len([t for t in self._transactions if t.transaction_type == TransactionType.BUY])
        sell_transactions = len([t for t in self._transactions if t.transaction_type == TransactionType.SELL])
        total_fees = sum(t.fees for t in self._transactions)
        
        return {
            'account_created': self._account_created_at.isoformat(),
            'last_update': self._last_update.isoformat(),
            'initial_cash': initial_value,
            'current_cash': self._cash,
            'current_portfolio_value': current_portfolio_value,
            'current_total_equity': current_total_equity,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_transactions': len(self._transactions),
            'buy_transactions': buy_transactions,
            'sell_transactions': sell_transactions,
            'total_fees_paid': total_fees,
            'positions_count': len([p for p in self._positions.values() if not p.is_flat]),
            'snapshots_count': len(self._portfolio_snapshots),
            'trading_days': (pd.Timestamp.now() - self._account_created_at).days
        }
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for symbol"""
        try:
            # Use the market data provider for validation
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=2)
            
            data = self._market_data_provider.get_historical_data(
                symbol=symbol,
                start_date=start_date.to_pydatetime(),
                end_date=end_date.to_pydatetime()
            )
            
            if data.empty:
                return None
            
            latest = data.iloc[-1]
            
            # Handle both 'Close' and 'close' column names
            close_col = 'Close' if 'Close' in data.columns else 'close'
            open_col = 'Open' if 'Open' in data.columns else 'open'
            high_col = 'High' if 'High' in data.columns else 'high'
            low_col = 'Low' if 'Low' in data.columns else 'low'
            volume_col = 'Volume' if 'Volume' in data.columns else 'volume'
            
            return {
                'symbol': symbol,
                'price': float(latest[close_col]),
                'open': float(latest[open_col]),
                'high': float(latest[high_col]),
                'low': float(latest[low_col]),
                'volume': int(latest[volume_col]),
                'timestamp': latest.name
            }
            
        except Exception as e:
            print(f"[ROBINHOOD_ACCOUNT] Error getting market data for {symbol}: {e}")
            return None
    
    def _validate_order(self, order: Order, price: float, commission: float) -> bool:
        """Validate order against account constraints"""
        order_value = order.quantity * price + commission
        
        # Check minimum/maximum order value
        if order_value < self.config.min_order_value:
            return False
        if order_value > self.config.max_order_value:
            return False
        
        # Check buying power for buy orders
        if order.side == OrderSide.BUY:
            if order_value > self._cash:
                return False
        
        # Check position for sell orders
        elif order.side == OrderSide.SELL:
            current_position = self._positions.get(order.symbol)
            if not current_position or current_position.quantity < order.quantity:
                return False
        
        return True
    
    def _execute_trade(self, trade: Trade):
        """Execute trade and update positions/cash"""
        # Update cash
        if trade.side == OrderSide.BUY:
            self._cash -= (trade.quantity * trade.price + trade.commission)
        else:
            self._cash += (trade.quantity * trade.price - trade.commission)
        
        # Update position
        symbol = trade.symbol
        current_position = self._positions.get(symbol)
        
        if current_position is None:
            # Create new position
            if trade.side == OrderSide.BUY:
                quantity = trade.quantity
                avg_price = trade.price
            else:
                quantity = -trade.quantity
                avg_price = trade.price
            
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=avg_price,
                market_value=quantity * trade.price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                timestamp=trade.timestamp
            )
        else:
            # Update existing position
            if trade.side == OrderSide.BUY:
                new_quantity = current_position.quantity + trade.quantity
                if current_position.quantity >= 0:
                    # Adding to long position
                    total_cost = (current_position.quantity * current_position.avg_price + 
                                trade.quantity * trade.price)
                    new_avg_price = total_cost / new_quantity if new_quantity > 0 else 0
                else:
                    # Covering short position
                    realized_pnl = trade.quantity * (current_position.avg_price - trade.price)
                    current_position.realized_pnl += realized_pnl
                    new_avg_price = current_position.avg_price
            else:
                new_quantity = current_position.quantity - trade.quantity
                if current_position.quantity > 0:
                    # Selling long position
                    realized_pnl = trade.quantity * (trade.price - current_position.avg_price)
                    current_position.realized_pnl += realized_pnl
                    new_avg_price = current_position.avg_price
                else:
                    # Adding to short position
                    total_cost = (abs(current_position.quantity) * current_position.avg_price + 
                                trade.quantity * trade.price)
                    new_avg_price = total_cost / abs(new_quantity) if new_quantity < 0 else 0
            
            current_position.quantity = new_quantity
            current_position.avg_price = new_avg_price
            current_position.timestamp = trade.timestamp
            
            # Update market value
            current_position.market_value = new_quantity * trade.price
            
            # Remove position if quantity is zero
            if new_quantity == 0:
                del self._positions[symbol]
    
    async def update_positions_market_value(self):
        """Update market values of all positions with current prices"""
        for symbol, position in self._positions.items():
            market_data = await self.get_market_data(symbol)
            if market_data:
                current_price = market_data['price']
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = position.quantity * (current_price - position.avg_price)
                position.timestamp = pd.Timestamp.now()
        
        self._last_update = pd.Timestamp.now()