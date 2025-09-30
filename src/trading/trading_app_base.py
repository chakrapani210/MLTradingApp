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