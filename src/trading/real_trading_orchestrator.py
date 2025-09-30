"""
Real Trading Orchestrator
Orchestrates ML-based trading with real or paper accounts
"""

import asyncio
import sys
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config_manager import get_trading_symbols, get_default_symbol, get_config_manager
from src.interfaces.real_trading import TradingAccountInterface, AccountType
from src.interfaces.trading_strategy import Order, OrderType, OrderSide
from src.trading.paper_trading import PaperTradingAccount, PaperTradingConfig
from src.models.enhanced_model_management import EnhancedModelManager
from src.data.providers import YFinanceProvider


@dataclass
class TradingDecision:
    """Trading decision with ML analysis"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    signal_strength: float
    model_prediction: Optional[str] = None
    technical_signals: Dict[str, Any] = None
    position_size: int = 0
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: str = ""
    timestamp: pd.Timestamp = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = pd.Timestamp.now()


class MLTradingOrchestrator:
    """
    Orchestrates ML-based trading decisions with paper/real accounts
    """
    
    def __init__(self, account: TradingAccountInterface, use_ml_models: bool = True):
        self.account = account
        self.use_ml_models = use_ml_models
        
        # Initialize components
        self.config_manager = get_config_manager()
        self.symbols = get_trading_symbols()
        self.default_symbol = get_default_symbol()
        
        # ML and data components
        self.model_manager = EnhancedModelManager() if use_ml_models else None
        self.data_provider = YFinanceProvider()
        
        # Trading state
        self.active_decisions: Dict[str, TradingDecision] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
        print(f"[ML_ORCHESTRATOR] Initialized for {account.account_type.value} trading")
        print(f"[ML_ORCHESTRATOR] Configured symbols: {', '.join(self.symbols)}")
        print(f"[ML_ORCHESTRATOR] ML Models enabled: {use_ml_models}")
    
    async def start_trading_session(self):
        """Start automated trading session"""
        print("\\n🚀 STARTING ML TRADING SESSION")
        print("=" * 60)
        
        # Connect to account
        if not await self.account.connect():
            print("❌ Failed to connect to trading account")
            return False
        
        # Display account status
        await self._display_account_status()
        
        # Run initial analysis
        print("\\n📊 Running initial market analysis...")
        decisions = await self.analyze_all_symbols()
        
        # Execute trading decisions
        if decisions:
            print(f"\\n🎯 Executing {len(decisions)} trading decisions...")
            await self.execute_trading_decisions(decisions)
        else:
            print("\\n💤 No trading signals generated")
        
        # Display final status
        await self._display_account_status()
        
        print("\\n✅ Trading session completed")
        return True
    
    async def analyze_all_symbols(self) -> List[TradingDecision]:
        """Analyze all configured symbols and generate trading decisions"""
        decisions = []
        
        for symbol in self.symbols:
            print(f"\\n📈 Analyzing {symbol}...")
            try:
                decision = await self.analyze_symbol(symbol)
                if decision and decision.action != 'HOLD':
                    decisions.append(decision)
                    print(f"   ✅ {symbol}: {decision.action} (confidence: {decision.confidence:.2f})")
                else:
                    print(f"   💤 {symbol}: HOLD")
            except Exception as e:
                print(f"   ❌ {symbol}: Analysis failed - {e}")
        
        return decisions
    
    async def analyze_symbol(self, symbol: str) -> Optional[TradingDecision]:
        """Analyze individual symbol and generate trading decision"""
        try:
            # Get market data
            market_data = await self._get_market_data(symbol)
            if market_data is None or market_data.empty:
                return None
            
            # Get technical analysis signals
            technical_signals = await self._get_technical_signals(symbol, market_data)
            
            # Get ML model prediction if available
            ml_prediction = None
            ml_confidence = 0.0
            
            if self.use_ml_models and self.model_manager:
                try:
                    ml_prediction, ml_confidence = await self._get_ml_prediction(symbol, market_data)
                except Exception as e:
                    print(f"   ⚠️ ML prediction failed for {symbol}: {e}")
            
            # Combine signals to make trading decision
            decision = self._make_trading_decision(
                symbol=symbol,
                market_data=market_data,
                technical_signals=technical_signals,
                ml_prediction=ml_prediction,
                ml_confidence=ml_confidence
            )
            
            return decision
            
        except Exception as e:
            import traceback
            print(f"   ❌ Error analyzing {symbol}: {e}")
            print(f"   📋 Traceback: {traceback.format_exc()}")
            return None
    
    async def execute_trading_decisions(self, decisions: List[TradingDecision]):
        """Execute trading decisions"""
        for decision in decisions:
            try:
                await self._execute_single_decision(decision)
            except Exception as e:
                print(f"❌ Failed to execute decision for {decision.symbol}: {e}")
    
    async def _execute_single_decision(self, decision: TradingDecision):
        """Execute single trading decision"""
        symbol = decision.symbol
        
        # Get current position
        current_position = await self.account.get_position(symbol)
        current_quantity = current_position.quantity if current_position else 0
        
        # Calculate order quantity
        order_quantity = self._calculate_order_quantity(decision, current_quantity)
        
        if order_quantity == 0:
            print(f"   💤 {symbol}: No action needed (current position: {current_quantity})")
            return
        
        # Determine order side
        order_side = OrderSide.BUY if order_quantity > 0 else OrderSide.SELL
        order_quantity = abs(order_quantity)
        
        # Create order
        order = Order(
            symbol=symbol,
            side=order_side,
            quantity=order_quantity,
            order_type=OrderType.MARKET,
            metadata={
                'decision_confidence': decision.confidence,
                'signal_strength': decision.signal_strength,
                'reasoning': decision.reasoning,
                'ml_prediction': decision.model_prediction,
                'timestamp': decision.timestamp.isoformat()
            }
        )
        
        # Place order
        print(f"   📋 Placing {order_side.value} order: {order_quantity} shares of {symbol}")
        execution = await self.account.place_order(order)
        
        if execution.is_complete:
            print(f"   ✅ Order executed: {execution.filled_quantity} shares @ ${execution.avg_fill_price:.2f}")
            
            # Record trade
            self.trade_history.append({
                'timestamp': execution.timestamp,
                'symbol': symbol,
                'action': order_side.value,
                'quantity': execution.filled_quantity,
                'price': execution.avg_fill_price,
                'commission': execution.commission,
                'decision': decision.__dict__
            })
        else:
            print(f"   ❌ Order failed: {execution.error_message}")
    
    async def _get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data for symbol"""
        try:
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=90)
            
            # Convert to datetime objects for YFinanceProvider
            start_dt = start_date.to_pydatetime()
            end_dt = end_date.to_pydatetime()
            
            print(f"   📥 Fetching data for {symbol} from {start_dt.date()} to {end_dt.date()}")
            
            data = self.data_provider.get_historical_data(
                symbol=symbol,
                start_date=start_dt,
                end_date=end_dt
            )
            
            if data is not None and not data.empty:
                print(f"   ✅ Got {len(data)} data points for {symbol}")
                return data
            else:
                print(f"   ❌ No data received for {symbol}")
                return None
            
        except Exception as e:
            print(f"   ❌ Error getting market data for {symbol}: {e}")
            return None
    
    async def _get_technical_signals(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Get technical analysis signals"""
        try:
            # Get technical analysis configuration
            tech_config = self.config_manager.config.get('technical_analysis', {})
            
            # Generate technical signals using centralized config
            signals = {}
            
            # RSI signals
            rsi_config = tech_config.get('signal_generators', {}).get('rsi', {})
            if rsi_config:
                rsi_period = rsi_config.get('period', 14)
                if len(market_data) >= rsi_period:
                    # Handle both 'Close' and 'close' column names
                    close_col = 'Close' if 'Close' in market_data.columns else 'close'
                    if close_col in market_data.columns:
                        rsi_values = self._calculate_rsi(market_data[close_col], rsi_period)
                        current_rsi = rsi_values.iloc[-1] if not rsi_values.empty else 50
                        
                        signals['rsi'] = {
                            'value': current_rsi,
                            'signal': 'BUY' if current_rsi < rsi_config.get('oversold_threshold', 30) else
                                     'SELL' if current_rsi > rsi_config.get('overbought_threshold', 70) else 'HOLD',
                            'strength': abs(current_rsi - 50) / 50
                        }
            
            # MACD signals
            macd_config = tech_config.get('signal_generators', {}).get('macd', {})
            if macd_config:
                fast_period = macd_config.get('fast_period', 12)
                slow_period = macd_config.get('slow_period', 26)
                signal_period = macd_config.get('signal_period', 9)
                
                if len(market_data) >= slow_period:
                    close_col = 'Close' if 'Close' in market_data.columns else 'close'
                    if close_col in market_data.columns:
                        macd_line, macd_signal, macd_hist = self._calculate_macd(
                            market_data[close_col], fast_period, slow_period, signal_period
                        )
                        
                        current_macd = macd_line.iloc[-1] if not macd_line.empty else 0
                        current_signal = macd_signal.iloc[-1] if not macd_signal.empty else 0
                        
                        signals['macd'] = {
                            'macd': current_macd,
                            'signal': current_signal,
                            'histogram': macd_hist.iloc[-1] if not macd_hist.empty else 0,
                            'signal_type': 'BUY' if current_macd > current_signal else 'SELL',
                            'strength': abs(current_macd - current_signal) / max(abs(current_macd), abs(current_signal), 0.001)
                        }
            
            # SMA crossover signals
            sma_config = tech_config.get('signal_generators', {}).get('sma_crossover', {})
            if sma_config:
                short_period = sma_config.get('short_period', 20)
                long_period = sma_config.get('long_period', 50)
                
                if len(market_data) >= long_period:
                    close_col = 'Close' if 'Close' in market_data.columns else 'close'
                    if close_col in market_data.columns:
                        short_sma = market_data[close_col].rolling(window=short_period).mean()
                        long_sma = market_data[close_col].rolling(window=long_period).mean()
                        
                        current_short = short_sma.iloc[-1] if not short_sma.empty else 0
                        current_long = long_sma.iloc[-1] if not long_sma.empty else 0
                        
                        signals['sma_crossover'] = {
                            'short_sma': current_short,
                            'long_sma': current_long,
                            'signal': 'BUY' if current_short > current_long else 'SELL',
                            'strength': abs(current_short - current_long) / max(current_long, 0.001)
                        }
            
            return signals
            
        except Exception as e:
            print(f"Error generating technical signals for {symbol}: {e}")
            return {}
    
    async def _get_ml_prediction(self, symbol: str, market_data: pd.DataFrame) -> tuple:
        """Get ML model prediction"""
        try:
            if not self.model_manager.model_exists(symbol):
                return None, 0.0
            
            # This would normally use the trained model
            # For now, simulate ML prediction
            import random
            predictions = ['BUY', 'SELL', 'HOLD']
            prediction = random.choice(predictions)
            confidence = random.uniform(0.6, 0.95)
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Error getting ML prediction for {symbol}: {e}")
            return None, 0.0
    
    def _make_trading_decision(
        self, 
        symbol: str, 
        market_data: pd.DataFrame,
        technical_signals: Dict[str, Any],
        ml_prediction: Optional[str],
        ml_confidence: float
    ) -> TradingDecision:
        """Combine all signals to make trading decision"""
        
        # Initialize decision variables
        signal_votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        signal_strengths = []
        reasoning_parts = []
        
        # Technical signals voting
        for signal_name, signal_data in technical_signals.items():
            print(f"   🔍 {signal_name}: {signal_data}")  # Debug output
            if isinstance(signal_data, dict) and 'signal' in signal_data:
                vote = str(signal_data['signal'])  # Ensure it's a string
                strength = float(signal_data.get('strength', 0.5))  # Ensure it's a float
                
                # Only accept valid vote types
                if vote in signal_votes:
                    signal_votes[vote] += strength
                    signal_strengths.append(strength)
                    reasoning_parts.append(f"{signal_name}: {vote} ({strength:.2f})")
                else:
                    # If invalid vote, treat as HOLD
                    signal_votes['HOLD'] += strength
                    signal_strengths.append(strength)
                    reasoning_parts.append(f"{signal_name}: HOLD (invalid: {vote})")
            else:
                print(f"   ⚠️ Invalid signal data for {signal_name}: {signal_data}")
        
        # ML model voting (weighted higher)
        if ml_prediction and ml_confidence > 0.6:
            ml_weight = 2.0  # Weight ML prediction higher
            signal_votes[ml_prediction] += ml_confidence * ml_weight
            signal_strengths.append(ml_confidence)
            reasoning_parts.append(f"ML: {ml_prediction} ({ml_confidence:.2f})")
        
        # Determine final decision
        max_vote = max(signal_votes.values())
        final_action = max(signal_votes, key=signal_votes.get)
        
        # Only trade if confidence is high enough
        confidence_threshold = 0.6
        overall_confidence = max_vote / sum(signal_votes.values()) if sum(signal_votes.values()) > 0 else 0
        
        if overall_confidence < confidence_threshold:
            final_action = 'HOLD'
        
        # Calculate position size based on confidence
        position_size = self._calculate_position_size(symbol, overall_confidence)
        
        return TradingDecision(
            symbol=symbol,
            action=final_action,
            confidence=overall_confidence,
            signal_strength=np.mean(signal_strengths) if signal_strengths else 0,
            model_prediction=ml_prediction,
            technical_signals=technical_signals,
            position_size=position_size,
            reasoning="; ".join(reasoning_parts)
        )
    
    def _calculate_position_size(self, symbol: str, confidence: float) -> int:
        """Calculate position size based on confidence and account size"""
        try:
            # For simplicity, use a fixed calculation based on confidence
            # In production, this would use real account balance
            base_position_value = 10000.0  # $10k base position
            confidence_multiplier = confidence  # Scale by confidence
            
            # Get current price (simplified - would use real market price)
            current_price = 100.0  # Placeholder price
            
            # Calculate shares
            position_value = base_position_value * confidence_multiplier
            shares = int(position_value / current_price)
            
            return max(shares, 0)
            
        except Exception as e:
            print(f"Error calculating position size for {symbol}: {e}")
            return 10  # Default small position
    
    def _calculate_order_quantity(self, decision: TradingDecision, current_quantity: int) -> int:
        """Calculate order quantity based on decision and current position"""
        target_quantity = 0
        
        if decision.action == 'BUY':
            target_quantity = decision.position_size
        elif decision.action == 'SELL':
            target_quantity = 0  # Close position
        else:  # HOLD
            target_quantity = current_quantity
        
        return target_quantity - current_quantity
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> tuple:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    async def _display_account_status(self):
        """Display current account status"""
        try:
            balance = await self.account.get_account_balance()
            portfolio = await self.account.get_portfolio_summary()
            
            print(f"\\n💰 ACCOUNT STATUS ({self.account.account_type.value.upper()})")
            print("-" * 40)
            print(f"Cash Balance: ${balance.cash:,.2f}")
            print(f"Portfolio Value: ${balance.portfolio_value:,.2f}")
            print(f"Total Equity: ${balance.total_equity:,.2f}")
            print(f"Buying Power: ${balance.buying_power:,.2f}")
            
            if portfolio.positions:
                print(f"\\n📊 POSITIONS ({len(portfolio.positions)})")
                for pos in portfolio.positions:
                    if not pos.is_flat:
                        pnl_color = "📈" if pos.unrealized_pnl >= 0 else "📉"
                        print(f"  {pos.symbol}: {pos.quantity} shares @ ${pos.avg_price:.2f} "
                              f"(Market: ${pos.market_value:,.2f}, P&L: {pnl_color} ${pos.unrealized_pnl:,.2f})")
            else:
                print("\\n📊 POSITIONS: None")
                
        except Exception as e:
            print(f"Error displaying account status: {e}")
    
    async def stop_trading_session(self):
        """Stop trading session and cleanup"""
        await self.account.disconnect()
        print("\\n🛑 Trading session stopped")