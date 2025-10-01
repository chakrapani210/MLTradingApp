"""
Trading Application Base Class
Common functionality for both Paper Trading and Real Trading applications
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime

from src.trading.enhanced_strategies import EnhancedMLTradingStrategy

from src.enhanced_orchestrator import ProductionTradingOrchestrator
from src.interfaces.real_trading import TradingAccountInterface
from src.interfaces.trading_strategy import Order, OrderType, OrderSide, TradingSignal
from src.interfaces.signal_generator import SignalType
from src.data.providers import YFinanceProvider
from config_manager import get_trading_symbols


class TradingAppBase(ABC):
    """
    Abstract base class for trading applications
    Provides common functionality for paper trading and real trading
    """
    
    def __init__(self, starting_capital: float = 100000.0, commission_rate: float = 0.0):
        """Initialize the trading application base"""
        self.starting_capital = starting_capital
        self.commission_rate = commission_rate
        
        # Initialize production orchestrator (reuse existing!)
        self.production_orchestrator = ProductionTradingOrchestrator(
            starting_capital=starting_capital,
            commission_rate=commission_rate,
            slippage_rate=0.001
        )
        
        # Configuration
        self.symbols = get_trading_symbols()
        
        # Account interface - to be set by concrete implementations
        self.account: Optional[TradingAccountInterface] = None
        
        # Initialize enhanced trading strategies
        self.data_provider = YFinanceProvider()
        self.trading_strategies: Dict[str, EnhancedMLTradingStrategy] = {}  # Will contain EnhancedMLTradingStrategy instances
        self._initialize_trading_strategies()
        
        print(f"🎯 TRADING APPLICATION INITIALIZED")
        print(f"📊 Symbols: {', '.join(self.symbols)}")
        print(f"💰 Capital: ${starting_capital:,.2f}")
        print("🚀 Using ProductionTradingOrchestrator & EnhancedMLTradingStrategy")
        print(f"🧠 Enhanced Strategies: {len(self.trading_strategies)} symbols")
    
    @abstractmethod
    async def connect_account(self) -> bool:
        """Connect to trading account - implemented by subclasses"""
        pass
    
    @abstractmethod
    async def disconnect_account(self) -> bool:
        """Disconnect from trading account - implemented by subclasses"""
        pass
    
    def _initialize_trading_strategies(self):
        """Initialize EnhancedMLTradingStrategy for each symbol using ProductionTradingOrchestrator"""
        
        success_count = 0
        for symbol in self.symbols:
            try:
                # Use the orchestrator's create_enhanced_strategy method
                strategy = self.production_orchestrator.create_enhanced_strategy(
                    symbol=symbol,
                    order_sizing_strategy="percentage",
                    portfolio_pct=0.05,  # 5% of portfolio per position
                    golden_cross_enabled=True,  # Now that signal generators are fixed
                    short_term_patterns_enabled=True,  # Now that signal generators are fixed
                    # Enable technical indicators now that they work
                    rsi_enabled=True,
                    macd_enabled=True,
                    bb_enabled=True,
                    sma_crossover_enabled=True,
                    ema_enabled=True,
                    volume_analysis_enabled=True
                )
                self.trading_strategies[symbol] = strategy
                success_count += 1
                print(f"   ✅ Strategy initialized for {symbol}")
            except Exception as e:
                print(f"   ⚠️  Failed to initialize strategy for {symbol}: {e}")
                # Create a minimal fallback entry
                self.trading_strategies[symbol] = None
        
        print(f"   📊 Successfully initialized {success_count}/{len(self.symbols)} strategies")

    async def run_trading_mode(self, mode: str = "production", symbol_limit: int = 3):
        """
        Run trading in specified mode
        
        Args:
            mode: 'simple', 'basic', or 'production'
            symbol_limit: Number of symbols to analyze
        """
        print(f"\n{'='*80}")
        print(f"🚀 TRADING APPLICATION - {mode.upper()} MODE")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Connect to account
        if not await self.connect_account():
            print("❌ Failed to connect to trading account")
            return
        
        await self.display_account_status()
        
        if mode == "simple":
            await self._run_simple_mode()
        elif mode == "basic":
            await self._run_basic_mode(symbol_limit)
        elif mode == "production":
            await self._run_production_mode(symbol_limit)
        else:
            print(f"❌ Unknown mode: {mode}")
            return
        
        # Final status
        await self.display_account_status()
        await self.disconnect_account()
        print(f"\n✅ {mode.upper()} trading completed!")
    
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
                
                # Simple buy order
                quantity = 10
                print(f"   📋 {symbol}: BUY {quantity} @ ${current_price:.2f}")
                
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    metadata={'mode': 'simple', 'real_price': current_price}
                )
                
                execution = await self.account.place_order(order)
                
                if execution.is_complete:
                    total_cost = execution.filled_quantity * execution.avg_fill_price
                    print(f"   ✅ Executed: {execution.filled_quantity} @ ${execution.avg_fill_price:.2f} = ${total_cost:,.2f}")
                else:
                    print(f"   ❌ Failed: {execution.error_message}")
                    
            except Exception as e:
                print(f"   ❌ Error processing {symbol}: {e}")
    
    async def _run_basic_mode(self, symbol_limit: int):
        """Basic mode: ML model validation with simple logic"""
        print(f"\n🏭 BASIC MODE: ML Model Analysis ({symbol_limit} symbols)")
        print("-" * 60)
        
        decisions = []
        test_symbols = self.symbols[:symbol_limit]
        
        for symbol in test_symbols:
            try:
                print(f"\n{'='*50}")
                print(f"🔍 BASIC ANALYSIS: {symbol}")
                print(f"{'='*50}")
                
                # ML Model Analysis with Enhanced Strategy
                ml_results = await self.production_orchestrator.analyze_symbol_ml(symbol)
                decision = await self.make_enhanced_decision(symbol, ml_results)
                
                print(f"\n🎯 DECISION: {decision['action']} (confidence: {decision['confidence']:.2f})")
                print(f"   Reasoning: {decision['reasoning']}")
                
                if decision['action'] != 'HOLD':
                    decisions.append(decision)
                    
            except Exception as e:
                print(f"   ❌ Analysis failed: {e}")
        
        # Execute basic decisions
        if decisions:
            print(f"\n🎯 EXECUTING {len(decisions)} BASIC DECISIONS")
            print("-" * 60)
            for decision in decisions:
                await self.execute_basic_decision(decision)
        else:
            print("\n💤 No trading signals generated")
    
    async def _run_production_mode(self, symbol_limit: int):
        """Production mode: Full analysis with real market data"""
        print(f"\n🏭 PRODUCTION MODE: Full Analysis ({symbol_limit} symbols)")
        print("-" * 60)
        
        decisions = []
        test_symbols = self.symbols[:symbol_limit]
        
        for symbol in test_symbols:
            try:
                print(f"\n{'='*50}")
                print(f"🔍 FULL ANALYSIS: {symbol}")
                print(f"{'='*50}")
                
                # Full Production Analysis with Enhanced Strategy
                print(f"\n🤖 ML Model Analysis")
                ml_results = await self.trading_strategies[symbol].generate_signal(symbol)
                
                print(f"\n📊 Market Context Analysis")
                market_analysis = await self.production_orchestrator.analyze_symbol_market_context(symbol)
                
                # Make production decision with enhanced strategy
                decision = await self.make_production_decision(symbol, ml_results, market_analysis)
                
                print(f"\n🎯 FINAL DECISION: {decision['action']} (confidence: {decision['confidence']:.2f})")
                print(f"   Reasoning: {decision['reasoning']}")
                
                if decision['action'] != 'HOLD':
                    decisions.append(decision)
                    
            except Exception as e:
                print(f"   ❌ Analysis failed: {e}")
        
        # Execute production decisions
        if decisions:
            print(f"\n🎯 EXECUTING {len(decisions)} PRODUCTION DECISIONS")
            print("-" * 60)
            for decision in decisions:
                await self.execute_production_decision(decision)
        else:
            print("\n💤 No trading signals generated")
    
    async def make_enhanced_decision(self, symbol: str, ml_results: dict) -> dict:
        """Make enhanced trading decision using orchestrator's signal generation"""
        try:
            # Use orchestrator's comprehensive signal generation
            signal_result = await self.production_orchestrator.generate_trading_signal(
                symbol=symbol,
                include_market_context=False,  # Enhanced mode doesn't include market context
                apply_production_adjustments=False
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
                'strength': signal_result['strength'],
                'signal_type': signal_result['signal_type'],
                'reasoning': signal_result['reasoning'],
                'raw_signal': signal_result
            }
                
        except Exception as e:
            print(f"❌ Enhanced decision failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'confidence': 0.0,
                'reasoning': f"Orchestrator signal generation error: {str(e)}"
            }
    
    async def make_production_decision(self, symbol: str, ml_results: dict, market_analysis: dict) -> dict:
        """Make production trading decision using orchestrator's comprehensive signal generation"""
        try:
            # Use orchestrator's comprehensive signal generation with full market context
            signal_result = await self.production_orchestrator.generate_trading_signal(
                symbol=symbol,
                include_market_context=True,
                apply_production_adjustments=True
            )
            
            if not signal_result.get('success', False):
                return {
                    'symbol': symbol,
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'reasoning': signal_result.get('reasoning', 'Production signal generation failed')
                }
            
            return {
                'symbol': symbol,
                'action': signal_result['action'],
                'confidence': signal_result['confidence'],
                'strength': signal_result['strength'],
                'signal_type': signal_result['signal_type'],
                'reasoning': signal_result['reasoning'],
                'raw_signal': signal_result,
                'market_context': signal_result.get('market_context'),
                'market_adjustments': signal_result.get('metadata', {}).get('market_adjustments_applied', False)
            }
            
        except Exception as e:
            print(f"❌ Production decision failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'confidence': 0.0,
                'reasoning': f"Orchestrator production signal error: {str(e)}"
            }
    
    async def execute_basic_decision(self, decision: dict):
        """Execute enhanced trading decision with smart position sizing"""
        symbol = decision['symbol']
        action = decision['action']
        confidence = decision['confidence']
        
        # Get current price
        current_price = await self.get_real_price(symbol)
        if not current_price:
            print(f"   ❌ Could not get price for {symbol}")
            return
        
        # Use enhanced strategy for order sizing if available
        try:
            if symbol in self.trading_strategies and 'raw_signal' in decision:
                strategy = self.trading_strategies[symbol]
                raw_signal = decision['raw_signal']
                
                # Get recent data for order sizing
                end_date = pd.Timestamp.now()
                start_date = end_date - pd.Timedelta(days=30)
                price_data = self.data_provider.get_historical_data(
                    symbol, start_date.to_pydatetime(), end_date.to_pydatetime()
                )
                
                if price_data is not None and len(price_data) > 0:
                    # Convert to Series for order size calculation
                    price_series = price_data['Close'] if 'Close' in price_data.columns else price_data['close']
                    optimal_quantity = abs(strategy.order_size_manager.calculate_order_size(
                        symbol, raw_signal, price_series
                    ))
                    quantity = max(optimal_quantity, 1)
                else:
                    quantity = max(int(10 * confidence), 1)  # Fallback
            else:
                quantity = max(int(10 * confidence), 1)  # Simple fallback
        except Exception as e:
            print(f"   ⚠️  Order sizing failed, using fallback: {e}")
            quantity = max(int(10 * confidence), 1)
        
        if action == 'BUY':
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
                metadata={
                    'mode': 'enhanced_basic',
                    'confidence': confidence,
                    'signal_type': decision.get('signal_type', 'UNKNOWN'),
                    'strength': decision.get('strength', confidence),
                    'reasoning': decision['reasoning']
                }
            )
            
            print(f"   📋 {symbol}: BUY {quantity} shares @ ${current_price:.2f}")
            print(f"       Value: ${quantity * current_price:,.2f}")
            print(f"       Signal: {decision.get('signal_type', 'ENHANCED')} (confidence: {confidence:.2f})")
            
            execution = await self.account.place_order(order)
            
            if execution.is_complete:
                total_cost = execution.filled_quantity * execution.avg_fill_price
                print(f"   ✅ Executed: {execution.filled_quantity} @ ${execution.avg_fill_price:.2f} = ${total_cost:,.2f}")
            else:
                print(f"   ❌ Failed: {execution.error_message}")
    
    async def execute_production_decision(self, decision: dict):
        """Execute production trading decision with enhanced strategy and dynamic sizing"""
        symbol = decision['symbol']
        action = decision['action']
        confidence = decision['confidence']
        
        # Get real current price
        current_price = await self.get_real_price(symbol)
        if not current_price:
            print(f"   ❌ Could not get price for {symbol}")
            return
        
        # Enhanced position sizing using strategy if available
        try:
            if symbol in self.trading_strategies and 'raw_signal' in decision:
                strategy = self.trading_strategies[symbol]
                raw_signal = decision['raw_signal']
                
                # Get recent data for order sizing
                end_date = pd.Timestamp.now()
                start_date = end_date - pd.Timedelta(days=60)
                price_data = self.data_provider.get_historical_data(
                    symbol, start_date.to_pydatetime(), end_date.to_pydatetime()
                )
                
                if price_data is not None and len(price_data) > 0:
                    price_series = price_data['Close'] if 'Close' in price_data.columns else price_data['close']
                    optimal_quantity = abs(strategy.order_size_manager.calculate_order_size(
                        symbol, raw_signal, price_series
                    ))
                    
                    # Apply market context boost
                    market_boost = decision.get('market_context_boost', 1.0)
                    quantity = max(int(optimal_quantity * market_boost), 1)
                else:
                    # Fallback to percentage-based sizing
                    balance = await self.account.get_account_balance()
                    max_position_pct = 0.05 * confidence * decision.get('market_context_boost', 1.0)
                    position_value = balance.cash * max_position_pct
                    quantity = max(int(position_value / current_price), 1)
            else:
                # Fallback to percentage-based sizing
                balance = await self.account.get_account_balance()
                max_position_pct = 0.05 * confidence * decision.get('market_context_boost', 1.0)
                position_value = balance.cash * max_position_pct
                quantity = max(int(position_value / current_price), 1)
        except Exception as e:
            print(f"   ⚠️  Enhanced sizing failed, using fallback: {e}")
            # Simple fallback
            balance = await self.account.get_account_balance()
            max_position_pct = 0.05 * confidence
            position_value = balance.cash * max_position_pct
            quantity = max(int(position_value / current_price), 1)
        
        # Safety cap: max 10% of available cash
        try:
            balance = await self.account.get_account_balance()
            max_affordable = int(balance.cash * 0.1 / current_price)
            quantity = min(quantity, max_affordable)
        except:
            pass
        
        if action == 'BUY':
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
                metadata={
                    'mode': 'enhanced_production',
                    'confidence': confidence,
                    'signal_type': decision.get('signal_type', 'PRODUCTION'),
                    'strength': decision.get('strength', confidence),
                    'market_context_boost': decision.get('market_context_boost', 1.0),
                    'reasoning': decision['reasoning'],
                    'real_price': current_price
                }
            )
            
            position_value = quantity * current_price
            print(f"   📋 {symbol}: BUY {quantity} shares @ ${current_price:.2f}")
            print(f"       Position Value: ${position_value:,.2f}")
            print(f"       Signal: {decision.get('signal_type', 'PRODUCTION')} (confidence: {confidence:.2f})")
            print(f"       Context Boost: {decision.get('market_context_boost', 1.0):.2f}x")
            print(f"       Reasoning: {decision['reasoning']}")
            
            execution = await self.account.place_order(order)
            
            if execution.is_complete:
                total_cost = execution.filled_quantity * execution.avg_fill_price
                print(f"   ✅ Executed: {execution.filled_quantity} @ ${execution.avg_fill_price:.2f} = ${total_cost:,.2f}")
            else:
                print(f"   ❌ Failed: {execution.error_message}")
    
    async def get_real_price(self, symbol: str) -> Optional[float]:
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
    
    def _make_fallback_decision(self, symbol: str, ml_results: dict) -> dict:
        """Fallback decision making using simple ML results"""
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
            'reasoning': f"Fallback ML decision (avg accuracy: {confidence:.2f})"
        }
    
    async def display_account_status(self):
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
    
    async def get_account_performance_summary(self) -> Dict[str, Any]:
        """Get account performance summary"""
        if hasattr(self.account, 'get_performance_summary'):
            return self.account.get_performance_summary()
        
        # Fallback basic performance calculation
        balance = await self.account.get_account_balance()
        return {
            'current_total_equity': balance.total_equity,
            'total_return': balance.total_equity - self.starting_capital,
            'total_return_pct': ((balance.total_equity - self.starting_capital) / self.starting_capital) * 100
        }
    
    def show_help(self):
        """Show usage help"""
        print(f"""
🎯 TRADING APPLICATION

Usage: python trading_app.py [mode] [symbols]

Modes:
  simple     - Manual orders with real prices (quick test)
  basic      - ML model validation with simple logic
  production - Full analysis with real market data (default)

Examples:
  python trading_app.py                    # Interactive menu
  python trading_app.py simple             # Simple mode
  python trading_app.py basic 5            # Basic mode, 5 symbols
  python trading_app.py production 2       # Production mode, 2 symbols

Features:
  ✅ Real market data and prices
  ✅ Production ML model integration
  ✅ Market context analysis
  ✅ Risk-based position sizing
  ✅ No simulated data
""")

    async def refresh_account_prices(self, verbose: bool = False) -> Dict[str, Any]:
        """Refresh market prices for all current positions and update unrealized PnL.

        This method pulls latest prices (using get_real_price with fallback) and recalculates:
          - position.market_value
          - position.unrealized_pnl (market_value - avg_price * quantity)
        It then forces any performance summary recalculation by returning updated summary.
        (Persistence will capture new snapshot on next order or explicit save if implemented.)
        """
        if not self.account:
            raise RuntimeError("Account not connected")
        positions = await self.account.get_all_positions()
        symbols_to_update = [p.symbol for p in positions if not p.is_flat]
        updated = 0

        # Batch fetch logic: attempt to pull small historical window once per symbol using data_provider
        price_cache: Dict[str, float] = {}
        for sym in symbols_to_update:
            try:
                price = self.production_orchestrator.data_provider.get_current_price(sym)
                if (not price or price <= 0):
                    # fallback 1-day historical close
                    end_date = pd.Timestamp.now()
                    start_date = end_date - pd.Timedelta(days=1)
                    hist = self.production_orchestrator.data_provider.get_historical_data(
                        symbol=sym,
                        start_date=start_date.to_pydatetime(),
                        end_date=end_date.to_pydatetime()
                    )
                    if hist is not None and not hist.empty:
                        close_col = 'Close' if 'Close' in hist.columns else 'close'
                        price = float(hist[close_col].iloc[-1])
                if price and price > 0:
                    price_cache[sym] = float(price)
            except Exception as e:
                if verbose:
                    print(f"[REFRESH] Failed to batch fetch {sym}: {e}")

        for pos in positions:
            if pos.is_flat:
                continue
            price = price_cache.get(pos.symbol)
            if price is None:
                # Final fallback via account provider if available
                if hasattr(self.account, 'get_market_data'):
                    try:
                        md = await self.account.get_market_data(pos.symbol)  # type: ignore[attr-defined]
                        price = md.get('price') if md else None
                    except Exception:
                        price = None
            if price is None or price <= 0:
                continue
            old_mv = pos.market_value
            pos.market_value = price * pos.quantity
            pos.unrealized_pnl = pos.market_value - (pos.avg_price * pos.quantity)
            updated += 1
            if verbose:
                print(f"[REFRESH] {pos.symbol}: {price:.2f} mv {old_mv:.2f}->{pos.market_value:.2f} pnl {pos.unrealized_pnl:.2f}")

        # Persist snapshot immediately if account supports snapshot/persistence
        if hasattr(self.account, '_take_portfolio_snapshot'):
            try:
                # type: ignore[attr-defined]
                self.account._take_portfolio_snapshot("Manual refresh prices")  # noqa: SLF001
            except Exception as e:
                if verbose:
                    print(f"[REFRESH] Snapshot failed: {e}")
        if hasattr(self.account, '_save_account_state'):
            try:
                # type: ignore[attr-defined]
                self.account._save_account_state()  # noqa: SLF001
            except Exception as e:
                if verbose:
                    print(f"[REFRESH] Save failed: {e}")
        # Return current performance summary (leveraging account's method if present)
        if hasattr(self.account, 'get_performance_summary'):
            return self.account.get_performance_summary()
        return await self.get_account_performance_summary()

    async def run_scheduled_auto_trading(
        self,
        allocation_fraction: float = 0.05,
        symbol_limit: int | None = None,
        rule: str = "production_signals",
        auto_disconnect: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Scheduled auto trading using production signal generation.

        Updated Design:
          - Uses ProductionTradingOrchestrator.generate_trading_signal() per symbol
          - Supports BUY & SELL actions
          - BUY sizing: leverage enhanced strategy order sizing when available; fallback to allocation_fraction * confidence of cash
          - SELL sizing: if confidence >= 0.6 sell full position else sell proportional (confidence fraction, >=1 share)
          - Skips HOLD / failed signals gracefully

        Args:
            allocation_fraction: Base portfolio cash fraction for new BUY positions (0 < f <= 1)
            symbol_limit: Optional limit to number of symbols processed
            rule: Rule label (defaults to 'production_signals')
            auto_disconnect: Disconnect account after completion
            verbose: Emit detailed logs

        Returns:
            dict summary with orders, cash deltas, errors
        """
        if allocation_fraction <= 0 or allocation_fraction > 1:
            raise ValueError("allocation_fraction must be between 0 and 1")

        # Ensure account is connected
        if not self.account:
            if verbose:
                print("[AUTO] Account not initialized - connecting...")
            if not await self.connect_account():
                raise RuntimeError("Failed to connect trading account for scheduled auto trading")

        start_balance = await self.account.get_account_balance()
        start_cash = start_balance.cash
        placed_orders: List[Dict[str, Any]] = []
        errors: List[str] = []

        symbols = self.symbols if symbol_limit is None else self.symbols[:symbol_limit]
        run_started = datetime.now()

        if verbose:
            print(f"\n=== SCHEDULED AUTO TRADING ({rule}) @ {run_started.strftime('%Y-%m-%d %H:%M:%S')} ===")
            print(f"Allocation fraction per new symbol: {allocation_fraction:.2%}")
            print(f"Symbols: {', '.join(symbols)}")

        for symbol in symbols:
            try:
                if verbose:
                    print(f"\n[SYMBOL] {symbol}")
                # Generate production signal with market context & adjustments
                signal_result = await self.production_orchestrator.generate_trading_signal(
                    symbol=symbol,
                    include_market_context=True,
                    apply_production_adjustments=True
                )

                if not signal_result.get('success', False):
                    if verbose:
                        print(f"  ❌ Signal generation failed: {signal_result.get('reasoning', 'unknown')} ")
                    continue

                action = signal_result.get('action', 'HOLD')
                confidence = float(signal_result.get('confidence', 0.0) or 0.0)
                if action == 'HOLD' or confidence <= 0:
                    if verbose:
                        print(f"  ℹ️  HOLD (confidence {confidence:.2f}) - skipping")
                    continue

                # Get current / real price
                price = await self.get_real_price(symbol)
                if not price and hasattr(self.account, 'get_market_data'):
                    try:
                        market_data = await self.account.get_market_data(symbol)  # type: ignore[attr-defined]
                        price = market_data.get('price') if market_data else None
                    except Exception:
                        price = None
                if not price or price <= 0:
                    if verbose:
                        print("  ❌ Could not resolve price - skipping")
                    continue

                position = await self.account.get_position(symbol)

                # Determine quantity
                quantity = 0
                try:
                    strategy = self.trading_strategies.get(symbol)
                    if strategy is not None:
                        # Historical data for sizing similar to execute_production_decision
                        end_date = pd.Timestamp.now()
                        start_date = end_date - pd.Timedelta(days=60)
                        price_data = self.data_provider.get_historical_data(
                            symbol, start_date.to_pydatetime(), end_date.to_pydatetime()
                        )
                        if price_data is not None and len(price_data) > 0:
                            series = price_data['Close'] if 'Close' in price_data.columns else price_data['close']
                            raw_signal = signal_result  # pass entire dict
                            base_qty = abs(strategy.order_size_manager.calculate_order_size(symbol, raw_signal, series))
                            quantity = int(max(base_qty, 1))
                    if quantity == 0:
                        # Fallback sizing
                        balance = await self.account.get_account_balance()
                        if action == 'BUY':
                            position_value = balance.cash * allocation_fraction * min(confidence, 1.0)
                            quantity = max(int(position_value / price), 1)
                        elif action == 'SELL' and position and not position.is_flat:
                            qty_conf = int(max(int(position.quantity * max(confidence, 0.1)), 1))
                            quantity = min(qty_conf, int(position.quantity))
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Sizing fallback due to error: {e}")
                    balance = await self.account.get_account_balance()
                    if action == 'BUY':
                        position_value = balance.cash * allocation_fraction * max(confidence, 0.1)
                        quantity = max(int(position_value / price), 1)
                    elif action == 'SELL' and position and not position.is_flat:
                        quantity = max(int(position.quantity * confidence), 1)

                # SELL-specific adjustments
                if action == 'SELL':
                    if not position or position.is_flat:
                        if verbose:
                            print("  ℹ️  No existing position to SELL - skipping")
                        continue
                    quantity = min(quantity, int(position.quantity))
                    if quantity < 1:
                        if verbose:
                            print("  ℹ️  Computed sell quantity <1 - skipping")
                        continue

                # BUY-specific pre-check: avoid duplicate open if already have position and rule disallows scaling
                if action == 'BUY' and position and not position.is_flat:
                    # For now skip scaling; future: allow pyramiding logic
                    if verbose:
                        print("  ℹ️  Position already open - skipping additional BUY (scaling disabled)")
                    continue

                side = OrderSide.BUY if action == 'BUY' else OrderSide.SELL
                order = Order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    metadata={
                        'mode': 'scheduled_auto',
                        'rule': rule,
                        'allocation_fraction': allocation_fraction,
                        'confidence': confidence,
                        'action': action,
                        'signal_type': signal_result.get('signal_type'),
                        'strength': signal_result.get('strength'),
                        'reasoning': signal_result.get('reasoning'),
                        'real_price': price
                    }
                )

                execution = await self.account.place_order(order)
                if execution.is_complete:
                    placed_orders.append({
                        'symbol': symbol,
                        'action': action,
                        'quantity': execution.filled_quantity,
                        'price': execution.avg_fill_price,
                        'confidence': confidence
                    })
                    if verbose:
                        print(f"  ✅ {action} {execution.filled_quantity} @ {execution.avg_fill_price:.2f} (conf {confidence:.2f})")
                else:
                    msg = f"Order failed for {symbol}: {execution.error_message}"
                    errors.append(msg)
                    if verbose:
                        print(f"  ❌ {msg}")

            except Exception as e:
                msg = f"Error processing {symbol}: {e}"
                errors.append(msg)
                if verbose:
                    print(f"  ❌ {msg}")

        end_balance = await self.account.get_account_balance()
        summary = {
            'rule': rule,
            'allocation_fraction': allocation_fraction,
            'started_at': run_started.isoformat(),
            'ended_at': datetime.now().isoformat(),
            'orders_placed_count': len(placed_orders),
            'orders': placed_orders,
            'start_cash': start_cash,
            'end_cash': end_balance.cash,
            'cash_used': start_cash - end_balance.cash,
            'errors': errors
        }

        if verbose:
            print("\n=== AUTO RUN SUMMARY ===")
            print(f"Orders placed: {summary['orders_placed_count']}")
            for o in placed_orders:
                print(f"  {o['symbol']}: {o['quantity']} @ {o['price']:.2f}")
            print(f"Cash used: ${summary['cash_used']:,.2f}  Remaining: ${summary['end_cash']:,.2f}")
            if errors:
                print(f"Errors: {len(errors)}")
        
        if auto_disconnect:
            await self.disconnect_account()

        return summary