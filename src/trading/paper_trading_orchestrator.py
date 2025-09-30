"""
Paper Trading Integration with Production Orchestrator
Integrates paper trading functionality with the existing ProductionTradingOrchestrator
"""

import asyncio
import sys
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config_manager import get_trading_symbols, get_config_manager
from src.enhanced_orchestrator import ProductionTradingOrchestrator
from src.trading.paper_trading import PaperTradingAccount, PaperTradingConfig
from src.interfaces.trading_strategy import Order, OrderType, OrderSide


@dataclass
class PaperTradingDecision:
    """Paper trading decision with production ML analysis"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    signal_strength: float
    reasoning: str = ""
    position_size: int = 0
    timestamp: pd.Timestamp = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = pd.Timestamp.now()


class PaperTradingOrchestrator:
    """
    Paper Trading Integration with Production Orchestrator
    Uses the existing ProductionTradingOrchestrator for all ML and analysis features
    """
    
    def __init__(self, paper_config: PaperTradingConfig):
        self.paper_config = paper_config
        self.paper_account = PaperTradingAccount(paper_config)
        
        # Initialize the production orchestrator with all its features
        self.production_orchestrator = ProductionTradingOrchestrator(
            starting_capital=paper_config.initial_cash,
            commission_rate=paper_config.commission_rate,
            slippage_rate=paper_config.slippage_rate
        )
        
        # Configuration
        self.config_manager = get_config_manager()
        self.symbols = get_trading_symbols()
        
        # Trading state
        self.active_decisions: Dict[str, PaperTradingDecision] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
        print(f"[PAPER_ORCHESTRATOR] Initialized with production features")
        print(f"[PAPER_ORCHESTRATOR] Configured symbols: {', '.join(self.symbols)}")
        print(f"[PAPER_ORCHESTRATOR] Starting capital: ${paper_config.initial_cash:,.2f}")
    
    async def start_trading_session(self):
        """Start automated paper trading session using production orchestrator"""
        print("\\n🚀 STARTING PAPER TRADING SESSION (PRODUCTION-READY)")
        print("=" * 70)
        
        # Connect to paper account
        if not await self.paper_account.connect():
            print("❌ Failed to connect to paper trading account")
            return False
        
        # Display account status
        await self._display_account_status()
        
        # Run analysis and trading for each symbol
        print("\\n📊 Running production ML analysis for all symbols...")
        decisions = await self.analyze_all_symbols()
        
        # Execute trading decisions
        if decisions:
            print(f"\\n🎯 Executing {len(decisions)} trading decisions...")
            await self.execute_trading_decisions(decisions)
        else:
            print("\\n💤 No trading signals generated")
        
        # Display final status
        await self._display_account_status()
        
        print("\\n✅ Paper trading session completed")
        return True
    
    async def analyze_all_symbols(self) -> List[PaperTradingDecision]:
        """Analyze all symbols using production orchestrator features"""
        decisions = []
        
        for symbol in self.symbols:
            print(f"\\n📈 Analyzing {symbol} with production features...")
            try:
                decision = await self.analyze_symbol_with_production_features(symbol)
                if decision and decision.action != 'HOLD':
                    decisions.append(decision)
                    print(f"   ✅ {symbol}: {decision.action} (confidence: {decision.confidence:.2f})")
                else:
                    print(f"   💤 {symbol}: HOLD")
            except Exception as e:
                print(f"   ❌ {symbol}: Analysis failed - {e}")
        
        return decisions
    
    async def analyze_symbol_with_production_features(self, symbol: str) -> Optional[PaperTradingDecision]:
        """Analyze symbol using production orchestrator's comprehensive features"""
        try:
            # 1. Train/validate ML model using production orchestrator
            print(f"   🤖 Checking ML model for {symbol}...")
            ml_results = self.production_orchestrator.train_ml_model(
                symbol=symbol,
                algorithm='RandomForest',
                force_retrain=False
            )
            
            # 2. Run market context analysis
            print(f"   📊 Analyzing market context for {symbol}...")
            market_analysis = self.production_orchestrator.analyze_market_context(
                symbol=symbol,
                analysis_period_days=180
            )
            
            # 3. Create enhanced strategy
            print(f"   🎯 Creating enhanced strategy for {symbol}...")
            strategy = self.production_orchestrator.create_enhanced_strategy(
                symbol=symbol,
                order_sizing_strategy="percentage",
                golden_cross_enabled=True,
                short_term_patterns_enabled=True
            )
            
            # 4. Generate trading signals using production features
            decision = await self._generate_trading_decision(
                symbol=symbol,
                ml_results=ml_results,
                market_analysis=market_analysis,
                strategy=strategy
            )
            
            return decision
            
        except Exception as e:
            print(f"   ❌ Error in production analysis for {symbol}: {e}")
            return None
    
    async def _generate_trading_decision(
        self, 
        symbol: str,
        ml_results: Dict[str, Any],
        market_analysis: Dict[str, Any],
        strategy: Any
    ) -> PaperTradingDecision:
        """Generate trading decision using production analysis results"""
        
        # Initialize decision variables
        signal_votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        reasoning_parts = []
        
        # ML Model Signal (if available)
        if ml_results.get('success', False):
            ml_confidence = ml_results.get('test_accuracy', 0.6)
            if ml_confidence > 0.3:  # Use model if it has some predictive power
                # Get real ML prediction using the production model
                try:
                    ml_prediction = await self._get_real_ml_prediction(symbol, ml_results)
                    if ml_prediction:
                        ml_signal = ml_prediction['prediction']
                        ml_pred_confidence = ml_prediction['confidence']
                        ml_weight = 2.0 * ml_pred_confidence  # Weight by prediction confidence
                        signal_votes[ml_signal] += ml_weight
                        reasoning_parts.append(f"ML: {ml_signal} (model_acc: {ml_confidence:.2f}, pred_conf: {ml_pred_confidence:.2f})")
                except Exception as e:
                    print(f"   ⚠️ ML prediction failed for {symbol}: {e}")
                    # Fallback to model accuracy based signal
                    if ml_confidence > 0.5:
                        # Use market context to guide ML signal direction
                        if 'market_context' in market_analysis:
                            market_regime = market_analysis['market_context'].get('market_regime', 'neutral')
                            if market_regime in ['bull', 'strong_bull']:
                                ml_signal = 'BUY'
                            elif market_regime in ['bear', 'strong_bear']:
                                ml_signal = 'SELL'
                            else:
                                ml_signal = 'HOLD'
                        else:
                            ml_signal = 'HOLD'
                        
                        ml_weight = ml_confidence * 1.5
                        signal_votes[ml_signal] += ml_weight
                        reasoning_parts.append(f"ML: {ml_signal} (model_acc: {ml_confidence:.2f})")
        
        # Market Context Signal
        if 'market_context' in market_analysis:
            market_ctx = market_analysis['market_context']
            spy_correlation = market_ctx.get('spy_correlation', 0)
            market_regime = market_ctx.get('market_regime', 'neutral')
            
            # Strong positive correlation with SPY in bullish market
            if spy_correlation > 0.7 and market_regime in ['bullish', 'strong_bullish']:
                signal_votes['BUY'] += 1.5
                reasoning_parts.append(f"Market: BUY (SPY corr: {spy_correlation:.2f}, regime: {market_regime})")
            elif spy_correlation < -0.3 or market_regime in ['bearish', 'strong_bearish']:
                signal_votes['SELL'] += 1.0
                reasoning_parts.append(f"Market: SELL (SPY corr: {spy_correlation:.2f}, regime: {market_regime})")
            else:
                signal_votes['HOLD'] += 0.5
                reasoning_parts.append(f"Market: HOLD (SPY corr: {spy_correlation:.2f}, regime: {market_regime})")
        
        # Technical Analysis from Strategy
        # Get real technical signals using production data
        tech_signals = await self._get_real_technical_signals(symbol)
        for signal_name, signal_info in tech_signals.items():
            vote = signal_info['signal']
            strength = signal_info['strength']
            signal_votes[vote] += strength
            reasoning_parts.append(f"{signal_name}: {vote} ({strength:.2f})")
        
        # Determine final decision
        max_vote = max(signal_votes.values())
        final_action = max(signal_votes, key=signal_votes.get)
        
        # Calculate overall confidence
        total_votes = sum(signal_votes.values())
        overall_confidence = max_vote / total_votes if total_votes > 0 else 0
        
        # Only trade if confidence is high enough
        confidence_threshold = 0.6
        if overall_confidence < confidence_threshold:
            final_action = 'HOLD'
        
        # Calculate position size using real market data
        position_size = await self._calculate_real_position_size(symbol, overall_confidence)
        
        return PaperTradingDecision(
            symbol=symbol,
            action=final_action,
            confidence=overall_confidence,
            signal_strength=max_vote,
            reasoning="; ".join(reasoning_parts),
            position_size=position_size
        )
    
    async def _get_real_technical_signals(self, symbol: str) -> Dict[str, Dict]:
        """Get real technical analysis signals using production data providers"""
        try:
            # Get real market data for technical analysis
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=90)
            
            market_data = self.production_orchestrator.data_provider.get_historical_data(
                symbol=symbol,
                start_date=start_date.to_pydatetime(),
                end_date=end_date.to_pydatetime()
            )
            
            if market_data.empty or len(market_data) < 50:
                print(f"   ⚠️ Insufficient data for technical analysis of {symbol}")
                return {}
            
            # Use production technical indicator calculator
            calculator = self.production_orchestrator.indicator_calculator
            
            signals = {}
            
            # RSI Analysis
            try:
                rsi_values = calculator.calculate_rsi(market_data, period=14)
                if not rsi_values.empty:
                    current_rsi = rsi_values.iloc[-1]
                    if current_rsi < 30:
                        rsi_signal = 'BUY'
                        rsi_strength = (30 - current_rsi) / 30
                    elif current_rsi > 70:
                        rsi_signal = 'SELL'
                        rsi_strength = (current_rsi - 70) / 30
                    else:
                        rsi_signal = 'HOLD'
                        rsi_strength = abs(current_rsi - 50) / 50
                    
                    signals['RSI'] = {
                        'signal': rsi_signal,
                        'strength': float(rsi_strength),
                        'value': float(current_rsi)
                    }
            except Exception as e:
                print(f"   ⚠️ RSI calculation failed for {symbol}: {e}")
            
            # MACD Analysis
            try:
                macd_data = calculator.calculate_macd(market_data, fast=12, slow=26, signal=9)
                if not macd_data.empty and len(macd_data) > 1:
                    current_macd = macd_data['macd'].iloc[-1]
                    current_signal = macd_data['signal'].iloc[-1]
                    previous_macd = macd_data['macd'].iloc[-2]
                    previous_signal = macd_data['signal'].iloc[-2]
                    
                    # MACD crossover detection
                    if current_macd > current_signal and previous_macd <= previous_signal:
                        macd_signal = 'BUY'
                        macd_strength = min(abs(current_macd - current_signal) / abs(current_signal), 1.0)
                    elif current_macd < current_signal and previous_macd >= previous_signal:
                        macd_signal = 'SELL'
                        macd_strength = min(abs(current_macd - current_signal) / abs(current_signal), 1.0)
                    else:
                        macd_signal = 'HOLD'
                        macd_strength = abs(current_macd - current_signal) / max(abs(current_macd), abs(current_signal), 0.001)
                    
                    signals['MACD'] = {
                        'signal': macd_signal,
                        'strength': float(macd_strength),
                        'macd': float(current_macd),
                        'signal_line': float(current_signal)
                    }
            except Exception as e:
                print(f"   ⚠️ MACD calculation failed for {symbol}: {e}")
            
            # SMA/EMA Analysis (Golden Cross)
            try:
                sma_20 = calculator.calculate_sma(market_data, period=20)
                sma_50 = calculator.calculate_sma(market_data, period=50)
                
                if not sma_20.empty and not sma_50.empty and len(sma_20) > 1 and len(sma_50) > 1:
                    current_sma20 = sma_20.iloc[-1]
                    current_sma50 = sma_50.iloc[-1]
                    previous_sma20 = sma_20.iloc[-2]
                    previous_sma50 = sma_50.iloc[-2]
                    
                    # Golden Cross detection
                    if current_sma20 > current_sma50 and previous_sma20 <= previous_sma50:
                        sma_signal = 'BUY'
                        sma_strength = min((current_sma20 - current_sma50) / current_sma50, 0.1) * 10
                    elif current_sma20 < current_sma50 and previous_sma20 >= previous_sma50:
                        sma_signal = 'SELL'
                        sma_strength = min((current_sma50 - current_sma20) / current_sma50, 0.1) * 10
                    else:
                        sma_signal = 'BUY' if current_sma20 > current_sma50 else 'SELL'
                        sma_strength = abs(current_sma20 - current_sma50) / current_sma50
                    
                    signals['Golden_Cross'] = {
                        'signal': sma_signal,
                        'strength': float(min(sma_strength, 1.0)),
                        'sma_20': float(current_sma20),
                        'sma_50': float(current_sma50)
                    }
            except Exception as e:
                print(f"   ⚠️ SMA calculation failed for {symbol}: {e}")
            
            # Bollinger Bands Analysis
            try:
                bb_data = calculator.calculate_bollinger_bands(market_data, period=20, std_dev=2)
                if not bb_data.empty:
                    close_price = market_data['Close'].iloc[-1] if 'Close' in market_data.columns else market_data['close'].iloc[-1]
                    upper_band = bb_data['upper'].iloc[-1]
                    lower_band = bb_data['lower'].iloc[-1]
                    middle_band = bb_data['middle'].iloc[-1]
                    
                    # Bollinger Band signals
                    if close_price <= lower_band:
                        bb_signal = 'BUY'
                        bb_strength = (lower_band - close_price) / (upper_band - lower_band)
                    elif close_price >= upper_band:
                        bb_signal = 'SELL'
                        bb_strength = (close_price - upper_band) / (upper_band - lower_band)
                    else:
                        bb_signal = 'HOLD'
                        bb_strength = abs(close_price - middle_band) / (upper_band - lower_band)
                    
                    signals['Bollinger_Bands'] = {
                        'signal': bb_signal,
                        'strength': float(min(bb_strength, 1.0)),
                        'price': float(close_price),
                        'upper': float(upper_band),
                        'lower': float(lower_band)
                    }
            except Exception as e:
                print(f"   ⚠️ Bollinger Bands calculation failed for {symbol}: {e}")
            
            return signals
            
        except Exception as e:
            print(f"   ❌ Technical analysis failed for {symbol}: {e}")
            return {}
    
    async def _get_real_ml_prediction(self, symbol: str, ml_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get real ML prediction using the production model"""
        try:
            # Check if model exists and is usable
            if not self.production_orchestrator.model_manager.model_exists(symbol):
                return None
            
            # Get recent market data for prediction
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=60)  # 60 days for feature calculation
            
            market_data = self.production_orchestrator.data_provider.get_historical_data(
                symbol=symbol,
                start_date=start_date.to_pydatetime(),
                end_date=end_date.to_pydatetime()
            )
            
            if market_data.empty or len(market_data) < 20:
                return None
            
            # Use production feature engineer to create features
            enhanced_features = self.production_orchestrator.feature_engineer.create_enhanced_features(
                symbol=symbol,
                start_date=start_date.to_pydatetime(),
                end_date=end_date.to_pydatetime(),
                include_market_context=True
            )
            
            if enhanced_features.features.empty:
                return None
            
            # Get the latest feature vector
            latest_features = enhanced_features.features.iloc[-1:].values
            
            # Load the actual model and make prediction
            model_info = self.production_orchestrator.model_manager.get_latest_model_info(symbol)
            if model_info:
                # Use model manager's prediction capabilities
                try:
                    # This would use the actual trained model
                    prediction_proba = self.production_orchestrator.model_manager.predict_proba(
                        symbol=symbol,
                        features=latest_features
                    )
                    
                    if prediction_proba is not None and len(prediction_proba) > 0:
                        # Convert probabilities to prediction and confidence
                        max_prob_idx = np.argmax(prediction_proba[0])
                        max_prob = prediction_proba[0][max_prob_idx]
                        
                        # Map index to signal (assuming 0=SELL, 1=HOLD, 2=BUY)
                        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
                        prediction = signal_map.get(max_prob_idx, 'HOLD')
                        
                        return {
                            'prediction': prediction,
                            'confidence': float(max_prob),
                            'probabilities': prediction_proba[0].tolist()
                        }
                
                except Exception as e:
                    print(f"   ⚠️ Model prediction failed, using feature-based heuristic: {e}")
                    # Fallback: Use enhanced features to make a heuristic prediction
                    return self._make_feature_based_prediction(enhanced_features)
            
            return None
            
        except Exception as e:
            print(f"   ❌ ML prediction error for {symbol}: {e}")
            return None
    
    def _make_feature_based_prediction(self, enhanced_features) -> Dict[str, Any]:
        """Make prediction based on enhanced features when model fails"""
        try:
            # Use the target labels distribution as a signal
            if hasattr(enhanced_features, 'target_labels') and len(enhanced_features.target_labels) > 0:
                # Look at recent target labels (last 10 samples)
                recent_labels = enhanced_features.target_labels[-10:]
                
                buy_count = np.sum(recent_labels == 1) if len(recent_labels) > 0 else 0
                sell_count = np.sum(recent_labels == -1) if len(recent_labels) > 0 else 0
                hold_count = np.sum(recent_labels == 0) if len(recent_labels) > 0 else 0
                
                total_count = len(recent_labels)
                
                if total_count > 0:
                    buy_prob = buy_count / total_count
                    sell_prob = sell_count / total_count
                    hold_prob = hold_count / total_count
                    
                    # Determine prediction based on highest probability
                    if buy_prob > sell_prob and buy_prob > hold_prob:
                        return {
                            'prediction': 'BUY',
                            'confidence': float(buy_prob),
                            'probabilities': [sell_prob, hold_prob, buy_prob]
                        }
                    elif sell_prob > hold_prob:
                        return {
                            'prediction': 'SELL',
                            'confidence': float(sell_prob),
                            'probabilities': [sell_prob, hold_prob, buy_prob]
                        }
                    else:
                        return {
                            'prediction': 'HOLD',
                            'confidence': float(hold_prob),
                            'probabilities': [sell_prob, hold_prob, buy_prob]
                        }
            
            # Default fallback
            return {
                'prediction': 'HOLD',
                'confidence': 0.33,
                'probabilities': [0.33, 0.33, 0.33]
            }
            
        except Exception as e:
            print(f"   ⚠️ Feature-based prediction failed: {e}")
            return {
                'prediction': 'HOLD',
                'confidence': 0.33,
                'probabilities': [0.33, 0.33, 0.33]
            }
        """Calculate position size based on confidence and production risk management"""
        try:
            # Use production orchestrator's risk management principles
            base_position_value = self.paper_config.initial_cash * 0.1  # 10% of capital
            confidence_multiplier = confidence
            
            # Simple position sizing (in production, this would use advanced algorithms)
            position_value = base_position_value * confidence_multiplier
            estimated_price = 100.0  # Placeholder - would get real price
            shares = int(position_value / estimated_price)
            
            return max(shares, 1)
            
        except Exception as e:
            print(f"Error calculating position size for {symbol}: {e}")
            return 10  # Default small position
    
    async def execute_trading_decisions(self, decisions: List[PaperTradingDecision]):
        """Execute trading decisions using paper account"""
        for decision in decisions:
            try:
                await self._execute_single_decision(decision)
            except Exception as e:
                print(f"❌ Failed to execute decision for {decision.symbol}: {e}")
    
    async def _execute_single_decision(self, decision: PaperTradingDecision):
        """Execute single trading decision"""
        symbol = decision.symbol
        
        # Get current position
        current_position = await self.paper_account.get_position(symbol)
        current_quantity = current_position.quantity if current_position else 0
        
        # Calculate order quantity
        if decision.action == 'BUY':
            order_quantity = decision.position_size
        elif decision.action == 'SELL':
            order_quantity = -current_quantity if current_quantity > 0 else 0
        else:
            order_quantity = 0
        
        if order_quantity == 0:
            print(f"   💤 {symbol}: No action needed")
            return
        
        # Determine order side and quantity
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
                'production_analysis': True,
                'timestamp': decision.timestamp.isoformat()
            }
        )
        
        # Place order
        print(f"   📋 Placing {order_side.value} order: {order_quantity} shares of {symbol}")
        execution = await self.paper_account.place_order(order)
        
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
                'decision': decision.__dict__,
                'production_features': True
            })
        else:
            print(f"   ❌ Order failed: {execution.error_message}")
    
    async def _display_account_status(self):
        """Display current account status"""
        try:
            balance = await self.paper_account.get_account_balance()
            portfolio = await self.paper_account.get_portfolio_summary()
            
            print(f"\\n💰 ACCOUNT STATUS (PAPER TRADING)")
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
    
    async def _calculate_real_position_size(self, symbol: str, confidence: float) -> int:
        """Calculate position size using real market data and production risk management"""
        try:
            # Get real current price
            current_price = await self._get_real_current_price(symbol)
            if not current_price:
                return 10  # Default fallback
            
            # Get account balance
            balance = await self.paper_account.get_account_balance()
            available_cash = balance.cash
            
            # Use production-style risk management
            max_position_pct = 0.15  # Maximum 15% of portfolio per position
            confidence_multiplier = confidence  # Scale by confidence
            
            # Calculate position value based on available cash and confidence
            max_position_value = available_cash * max_position_pct * confidence_multiplier
            
            # Calculate shares based on real current price
            shares = int(max_position_value / current_price)
            
            # Minimum and maximum bounds
            min_shares = 1
            max_shares = int(available_cash * 0.2 / current_price)  # Max 20% of cash
            
            return max(min(shares, max_shares), min_shares)
            
        except Exception as e:
            print(f"   ⚠️ Position size calculation failed for {symbol}: {e}")
            return 10  # Default small position
    
    async def _get_real_current_price(self, symbol: str) -> Optional[float]:
        """Get real current market price"""
        try:
            # Get current price from production data provider
            current_price = self.production_orchestrator.data_provider.get_current_price(symbol)
            if current_price and current_price > 0:
                return float(current_price)
            
            # Fallback: get latest price from recent historical data
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=1)
            
            recent_data = self.production_orchestrator.data_provider.get_historical_data(
                symbol=symbol,
                start_date=start_date.to_pydatetime(),
                end_date=end_date.to_pydatetime()
            )
            
            if not recent_data.empty:
                close_col = 'Close' if 'Close' in recent_data.columns else 'close'
                latest_price = recent_data[close_col].iloc[-1]
                return float(latest_price)
            
            return None
            
        except Exception as e:
            print(f"   ⚠️ Could not get real price for {symbol}: {e}")
            return None
    
    async def stop_trading_session(self):
        """Stop trading session and cleanup"""
        await self.paper_account.disconnect()
        self.production_orchestrator.cleanup_resources()
        print("\\n🛑 Trading session stopped")
    
    def get_production_system_status(self) -> Dict[str, Any]:
        """Get status from production orchestrator"""
        return self.production_orchestrator.get_system_status()
    
    def get_production_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from production orchestrator"""
        return self.production_orchestrator.get_performance_metrics()