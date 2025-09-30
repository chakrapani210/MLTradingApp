"""
Paper Trading Implementation
Simulates real trading with virtual money and positions
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


class TransactionType(Enum):
    """Types of account transactions"""
    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    FEE = "fee"
    INTEREST = "interest"


@dataclass
class Transaction:
    """
    Detailed transaction record for portfolio tracking and analysis
    """
    transaction_id: str
    timestamp: pd.Timestamp
    transaction_type: TransactionType
    symbol: Optional[str]  # None for account-level transactions
    quantity: float
    price: float
    amount: float  # Total transaction amount (quantity * price + fees)
    fees: float
    description: str
    account_balance_after: float
    portfolio_value_after: float
    total_equity_after: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary for serialization"""
        return {
            'transaction_id': self.transaction_id,
            'timestamp': self.timestamp.isoformat(),
            'transaction_type': self.transaction_type.value,
            'symbol': self.symbol,
            'quantity': float(self.quantity),
            'price': float(self.price),
            'amount': float(self.amount),
            'fees': float(self.fees),
            'description': self.description,
            'account_balance_after': float(self.account_balance_after),
            'portfolio_value_after': float(self.portfolio_value_after),
            'total_equity_after': float(self.total_equity_after)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create transaction from dictionary"""
        return cls(
            transaction_id=data['transaction_id'],
            timestamp=pd.Timestamp(data['timestamp']),
            transaction_type=TransactionType(data['transaction_type']),
            symbol=data.get('symbol'),
            quantity=data['quantity'],
            price=data['price'],
            amount=data['amount'],
            fees=data['fees'],
            description=data['description'],
            account_balance_after=data['account_balance_after'],
            portfolio_value_after=data['portfolio_value_after'],
            total_equity_after=data['total_equity_after']
        )


@dataclass
class PortfolioSnapshot:
    """
    Portfolio value snapshot for tracking performance over time
    """
    timestamp: pd.Timestamp
    cash_balance: float
    portfolio_value: float
    total_equity: float
    positions_count: int
    day_pnl: float
    total_pnl: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cash_balance': float(self.cash_balance),
            'portfolio_value': float(self.portfolio_value),
            'total_equity': float(self.total_equity),
            'positions_count': int(self.positions_count),
            'day_pnl': float(self.day_pnl),
            'total_pnl': float(self.total_pnl)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioSnapshot':
        """Create snapshot from dictionary"""
        return cls(
            timestamp=pd.Timestamp(data['timestamp']),
            cash_balance=data['cash_balance'],
            portfolio_value=data['portfolio_value'],
            total_equity=data['total_equity'],
            positions_count=data['positions_count'],
            day_pnl=data['day_pnl'],
            total_pnl=data['total_pnl']
        )


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading"""
    initial_cash: float = 100000.0
    commission_per_trade: float = 0.0  # Commission per trade
    commission_rate: float = 0.001  # 0.1% commission rate
    slippage_rate: float = 0.0005  # 0.05% slippage
    min_order_value: float = 1.0
    max_order_value: float = 50000.0
    allow_fractional_shares: bool = False
    enable_after_hours: bool = False
    # File to save account state (relative path will be created automatically)
    persistence_file: str = os.path.join("data", "paper_account_state.json")
    track_portfolio_snapshots: bool = True  # Enable portfolio value tracking
    # Note: snapshots are taken on every transaction, not time-based
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'initial_cash': self.initial_cash,
            'commission_per_trade': self.commission_per_trade,
            'commission_rate': self.commission_rate,
            'slippage_rate': self.slippage_rate,
            'min_order_value': self.min_order_value,
            'max_order_value': self.max_order_value,
            'allow_fractional_shares': self.allow_fractional_shares,
            'enable_after_hours': self.enable_after_hours,
            'persistence_file': self.persistence_file,
            'track_portfolio_snapshots': self.track_portfolio_snapshots
        }


class PaperTradingAccount(TradingAccountInterface):
    """
    Paper trading account implementation
    Simulates real trading with virtual money
    """
    
    def __init__(self, config: PaperTradingConfig = None):
        super().__init__(AccountType.PAPER)
        self.config = config or PaperTradingConfig()
        
        # Account state (will be loaded from persistence if exists)
        self._cash = self.config.initial_cash
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, OrderExecution] = {}
        self._trades: List[Trade] = []
        
        # Enhanced tracking for charts and analysis
        self._transactions: List[Transaction] = []  # Detailed transaction history
        self._portfolio_snapshots: List[PortfolioSnapshot] = []  # Value tracking over time
        self._last_snapshot_time: Optional[pd.Timestamp] = None
        
        # Market data provider
        self._market_data_provider = YFinanceProvider()
        
        # State tracking
        self._account_created_at = pd.Timestamp.now()
        self._last_update = pd.Timestamp.now()
        
        # Load persisted state if available
        self._load_account_state()
        
        # Take initial snapshot if enabled
        if self.config.track_portfolio_snapshots:
            self._take_portfolio_snapshot("Account initialized")
        
        print(f"[PAPER_ACCOUNT] Initialized with ${self._cash:,.2f} (persistence: {os.path.exists(self.config.persistence_file)})")
        print(f"[PAPER_ACCOUNT] Tracking: {len(self._transactions)} transactions, {len(self._portfolio_snapshots)} snapshots")
    
    async def connect(self) -> bool:
        """Connect to paper trading platform"""
        try:
            self._is_connected = True
            print(f"[PAPER_ACCOUNT] Connected successfully")
            return True
        except Exception as e:
            print(f"[PAPER_ACCOUNT] Connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from paper trading platform"""
        # Take final snapshot
        if self.config.track_portfolio_snapshots:
            self._take_portfolio_snapshot("Session ended")
        
        # Save account state before disconnecting
        self._save_account_state()
        self._is_connected = False
        print(f"[PAPER_ACCOUNT] Disconnected and state saved")
        return True
    
    def _take_portfolio_snapshot(self, description: str = "Transaction snapshot"):
        """Take a portfolio value snapshot for tracking over time"""
        try:
            current_time = pd.Timestamp.now()
            
            # Take snapshot on every call - no time-based filtering
            # This ensures we capture portfolio state after every significant transaction
            
            # Calculate current portfolio metrics
            portfolio_value = sum(pos.market_value for pos in self._positions.values())
            total_equity = self._cash + portfolio_value
            positions_count = len([p for p in self._positions.values() if not p.is_flat])
            
            # Calculate P&L
            initial_value = self.config.initial_cash
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
            
            # Take snapshot after recording transaction (for all transaction types)
            # This ensures we have portfolio state captured at every significant event
            self._take_portfolio_snapshot(f"Transaction: {transaction_type.value} {symbol or 'Account'}")
            
        except Exception as e:
            print(f"[TRANSACTION] Failed to record transaction: {e}")
    
    def _save_account_state(self):
        """Save current account state to persistence file"""
        try:
            # Ensure directory exists (support bare filename case)
            persistence_dir = os.path.dirname(self.config.persistence_file)
            if persistence_dir and not os.path.exists(persistence_dir):
                os.makedirs(persistence_dir, exist_ok=True)
            
            # Prepare state data
            state_data = {
                'config': self.config.to_dict(),
                'cash': float(self._cash),
                'account_created_at': self._account_created_at.isoformat(),
                'last_update': pd.Timestamp.now().isoformat(),
                'positions': {},
                'trades': [],
                'transactions': [],  # Enhanced transaction tracking
                'portfolio_snapshots': [],  # Portfolio value over time
                'order_count': len(self._orders)
            }
            
            # Serialize positions
            for symbol, position in self._positions.items():
                if not position.is_flat:
                    # Calculate current market price (simplified as avg_price for now)
                    current_price = position.avg_price
                    try:
                        # Try to get real current price
                        current_data = self._market_data_provider.get_current_price(symbol)
                        if current_data and current_data > 0:
                            current_price = float(current_data)
                    except:
                        pass  # Use avg_price as fallback
                    
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
            
            # Serialize transactions (last 500 for comprehensive history)
            for transaction in self._transactions[-500:]:
                state_data['transactions'].append(transaction.to_dict())
            
            # Serialize portfolio snapshots (last 100 for charting)
            for snapshot in self._portfolio_snapshots[-100:]:
                state_data['portfolio_snapshots'].append(snapshot.to_dict())
            
            # Write to file
            with open(self.config.persistence_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            print(f"[PAPER_ACCOUNT] State saved to {self.config.persistence_file}")
            
        except Exception as e:
            print(f"[PAPER_ACCOUNT] Failed to save state: {e}")
    
    def _load_account_state(self):
        """Load account state from persistence file"""
        try:
            # Ensure directory exists before attempting to load
            persistence_dir = os.path.dirname(self.config.persistence_file)
            if persistence_dir and not os.path.exists(persistence_dir):
                os.makedirs(persistence_dir, exist_ok=True)

            if not os.path.exists(self.config.persistence_file):
                print(f"[PAPER_ACCOUNT] No existing state file found, starting fresh")
                return
            
            with open(self.config.persistence_file, 'r') as f:
                state_data = json.load(f)
            
            # Load basic account data
            self._cash = state_data.get('cash', self.config.initial_cash)
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
            
            print(f"[PAPER_ACCOUNT] State loaded from {self.config.persistence_file}")
            print(f"[PAPER_ACCOUNT] Loaded: ${self._cash:,.2f} cash, {len(self._positions)} positions, "
                  f"{len(self._trades)} trades, {len(self._transactions)} transactions, "
                  f"{len(self._portfolio_snapshots)} snapshots")
            
        except Exception as e:
            print(f"[PAPER_ACCOUNT] Failed to load state: {e}")
            print(f"[PAPER_ACCOUNT] Starting with fresh account")
            # Reset to defaults on load failure
            self._cash = self.config.initial_cash
            self._positions = {}
            self._trades = []
            self._transactions = []
            self._portfolio_snapshots = []
            self._last_snapshot_time = None
    
    async def get_account_balance(self) -> AccountBalance:
        """Get current account balance"""
        portfolio_value = sum(pos.market_value for pos in self._positions.values())
        total_equity = self._cash + portfolio_value
        
        return AccountBalance(
            cash=self._cash,
            buying_power=self._cash * 2,  # Simulate 2:1 margin
            portfolio_value=portfolio_value,
            day_trade_buying_power=self._cash * 4,  # Simulate 4:1 day trading
            unsettled_funds=0.0,
            timestamp=pd.Timestamp.now()
        )
    
    async def get_portfolio_summary(self) -> PortfolioSummary:
        """Get portfolio summary"""
        positions = list(self._positions.values())
        positions_value = sum(pos.market_value for pos in positions)
        total_equity = self._cash + positions_value
        
        # Calculate P&L
        total_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in positions)
        day_pnl = sum(pos.unrealized_pnl for pos in positions)  # Simplified
        
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
        """Place trading order in paper account"""
        try:
            order_id = str(uuid.uuid4())
            timestamp = pd.Timestamp.now()
            
            print(f"[PAPER_ORDER] Placing {order.side.value} order for {order.quantity} {order.symbol}")
            
            # Get current market price
            market_data = await self.get_market_data(order.symbol)
            if not market_data:
                return OrderExecution(
                    order_id=order_id,
                    status=OrderStatus.REJECTED,
                    error_message=f"Unable to get market data for {order.symbol}",
                    timestamp=timestamp
                )
            
            current_price = market_data['price']
            
            # Determine execution price based on order type
            if order.order_type == OrderType.MARKET:
                execution_price = self._apply_slippage(current_price, order.side)
            elif order.order_type == OrderType.LIMIT:
                if order.price is None:
                    return OrderExecution(
                        order_id=order_id,
                        status=OrderStatus.REJECTED,
                        error_message="Limit price required for limit order",
                        timestamp=timestamp
                    )
                # For paper trading, assume limit orders fill immediately if price is favorable
                if (order.side == OrderSide.BUY and order.price >= current_price) or \
                   (order.side == OrderSide.SELL and order.price <= current_price):
                    execution_price = order.price
                else:
                    # Order doesn't fill immediately
                    execution = OrderExecution(
                        order_id=order_id,
                        status=OrderStatus.PENDING,
                        remaining_quantity=order.quantity,
                        timestamp=timestamp
                    )
                    self._orders[order_id] = execution
                    return execution
            else:
                return OrderExecution(
                    order_id=order_id,
                    status=OrderStatus.REJECTED,
                    error_message=f"Order type {order.order_type} not supported",
                    timestamp=timestamp
                )
            
            # Calculate commission
            commission = self._calculate_commission(order.quantity, execution_price)
            
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
            
            # Record detailed transaction for analysis
            transaction_type = TransactionType.BUY if order.side == OrderSide.BUY else TransactionType.SELL
            fees = execution.commission or 0.0
            description = f"{order.order_type.value.title()} order execution"
            
            self._record_transaction(
                transaction_type=transaction_type,
                symbol=order.symbol,
                quantity=execution.filled_quantity,
                price=execution_price,
                fees=fees,
                description=description
            )
            
            # Take snapshot after every trade execution for precise tracking
            self._take_portfolio_snapshot(f"After {transaction_type.value}: {order.symbol}")
            
            # Auto-save state after each trade
            self._save_account_state()
            
            print(f"[PAPER_ORDER] ✅ Executed {order.side.value} {order.quantity} {order.symbol} @ ${execution_price:.2f}")
            
            return execution
            
        except Exception as e:
            print(f"[PAPER_ORDER] ❌ Order execution failed: {e}")
            return OrderExecution(
                order_id=str(uuid.uuid4()),
                status=OrderStatus.REJECTED,
                error_message=str(e),
                timestamp=pd.Timestamp.now()
            )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        if order_id in self._orders:
            execution = self._orders[order_id]
            if execution.status == OrderStatus.PENDING:
                execution.status = OrderStatus.CANCELLED
                execution.timestamp = pd.Timestamp.now()
                print(f"[PAPER_ORDER] Cancelled order {order_id}")
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
            # Return a full key set even when no transactions yet, to keep UI templates simple
            return {
                'account_created': self._account_created_at.isoformat(),
                'last_update': self._last_update.isoformat(),
                'initial_cash': self.config.initial_cash,
                'current_cash': self._cash,
                'current_portfolio_value': 0.0,
                'current_total_equity': self._cash,
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'total_transactions': 0,
                'buy_transactions': 0,
                'sell_transactions': 0,
                'total_fees_paid': 0.0,
                'positions_count': 0,
                'snapshots_count': len(self._portfolio_snapshots),
                'trading_days': 0
            }
        
        # Calculate performance metrics
        current_portfolio_value = sum(pos.market_value for pos in self._positions.values())
        current_total_equity = self._cash + current_portfolio_value
        initial_value = self.config.initial_cash
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
    
    def reset_account(self):
        """Reset paper trading account to initial state"""
        print(f"[PAPER_ACCOUNT] Resetting account to initial state")
        self._cash = self.config.initial_cash
        self._positions = {}
        self._orders = {}
        self._trades = []
        self._transactions = []  # Clear transaction history
        self._portfolio_snapshots = []  # Clear snapshots
        self._last_snapshot_time = None
        self._account_created_at = pd.Timestamp.now()
        self._last_update = pd.Timestamp.now()
        
        # Record reset transaction
        self._record_transaction(
            transaction_type=TransactionType.DEPOSIT,
            symbol=None,
            quantity=0,
            price=0,
            fees=0,
            description=f"Account reset to initial cash: ${self.config.initial_cash:,.2f}"
        )
        
        # Take initial snapshot
        if self.config.track_portfolio_snapshots:
            self._take_portfolio_snapshot("Account reset")
        
        # Remove persistence file
        if os.path.exists(self.config.persistence_file):
            os.remove(self.config.persistence_file)
            print(f"[PAPER_ACCOUNT] Removed persistence file: {self.config.persistence_file}")
        
        print(f"[PAPER_ACCOUNT] Account reset complete - ${self.config.initial_cash:,.2f} starting cash")
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for symbol"""
        try:
            # Use the existing data provider
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
            print(f"[PAPER_ACCOUNT] Error getting market data for {symbol}: {e}")
            return None
    
    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply slippage to market orders"""
        slippage = price * self.config.slippage_rate
        if side == OrderSide.BUY:
            return price + slippage
        else:
            return price - slippage
    
    def _calculate_commission(self, quantity: int, price: float) -> float:
        """Calculate commission for trade"""
        if self.config.commission_per_trade > 0:
            return self.config.commission_per_trade
        else:
            return quantity * price * self.config.commission_rate
    
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
            
            # Update market value (will be updated with real-time prices)
            current_position.market_value = new_quantity * trade.price
            
            # Remove position if quantity is zero
            if new_quantity == 0:
                del self._positions[symbol]
    
    def save_state(self, filepath: str):
        """Save paper trading state to file"""
        state = {
            'config': self.config.to_dict(),
            'cash': self._cash,
            'positions': {symbol: {
                'symbol': pos.symbol,
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl,
                'timestamp': pos.timestamp.isoformat()
            } for symbol, pos in self._positions.items()},
            'trades': [trade.to_dict() for trade in self._trades],
            'account_created_at': self._account_created_at.isoformat(),
            'last_update': pd.Timestamp.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"[PAPER_ACCOUNT] State saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load paper trading state from file"""
        if not os.path.exists(filepath):
            print(f"[PAPER_ACCOUNT] State file {filepath} not found")
            return
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self._cash = state['cash']
        
        # Restore positions
        self._positions = {}
        for symbol, pos_data in state['positions'].items():
            self._positions[symbol] = Position(
                symbol=pos_data['symbol'],
                quantity=pos_data['quantity'],
                avg_price=pos_data['avg_price'],
                market_value=pos_data['market_value'],
                unrealized_pnl=pos_data['unrealized_pnl'],
                realized_pnl=pos_data['realized_pnl'],
                timestamp=pd.Timestamp(pos_data['timestamp'])
            )
        
        # Restore trades
        self._trades = []
        for trade_data in state['trades']:
            self._trades.append(Trade(
                symbol=trade_data['symbol'],
                side=OrderSide(trade_data['side']),
                quantity=trade_data['quantity'],
                price=trade_data['price'],
                timestamp=pd.Timestamp(trade_data['timestamp']),
                order_id=trade_data['order_id'],
                trade_id=trade_data['trade_id'],
                commission=trade_data['commission'],
                metadata=trade_data.get('metadata', {})
            ))
        
        self._account_created_at = pd.Timestamp(state['account_created_at'])
        print(f"[PAPER_ACCOUNT] State loaded from {filepath}")
    
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