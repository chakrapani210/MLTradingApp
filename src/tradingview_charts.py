"""
TradingView-Style Chart Generator
=================================
Creates comprehensive trading charts with candlesticks, indicators, signals, orders, and portfolio tracking
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Any
import os
from dataclasses import dataclass
from enum import Enum

# Import the main TradingSignal from the interfaces module
from .interfaces.signal_generator import TradingSignal, SignalType as MainSignalType


class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    timestamp: datetime
    symbol: str
    order_type: OrderType
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    size_usd: float
    status: str = "filled"
    reason: str = ""


@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    daily_pnl: float
    total_pnl: float
    daily_return: float
    total_return: float


class TechnicalIndicators:
    """Calculate various technical indicators for charting"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0.0)  # Use 0.0 instead of 0
        loss = -delta.where(delta < 0, 0.0)  # Use 0.0 instead of 0
        
        # Use rolling mean with min_periods to handle edge cases
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        exp1 = data.ewm(span=fast).mean()
        exp2 = data.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def volume_profile(prices: pd.Series, volumes: pd.Series, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Volume Profile"""
        if len(prices) == 0 or len(volumes) == 0:
            return np.array([]), np.array([])
        
        price_min, price_max = float(prices.min()), float(prices.max())
        
        # Handle edge case where all prices are the same
        if price_min == price_max:
            price_levels = np.array([price_min])
            volume_profile = np.array([float(volumes.sum())])
            return price_levels, volume_profile
        
        price_bins = np.linspace(price_min, price_max, bins + 1)
        volume_profile = np.zeros(bins)
        
        for i in range(len(prices)):
            bin_idx = np.digitize(float(prices.iloc[i]), price_bins) - 1
            bin_idx = max(0, min(bins - 1, bin_idx))
            volume_profile[bin_idx] += float(volumes.iloc[i])
        
        price_levels = (price_bins[:-1] + price_bins[1:]) / 2
        return price_levels, volume_profile


class TradingViewChartGenerator:
    """
    Generate comprehensive TradingView-style charts with all trading information
    """
    
    def __init__(self, results_dir: str = "results"):
        """Initialize the chart generator"""
        self.results_dir = results_dir or "results"  # Handle None case
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Color scheme (TradingView dark theme inspired)
        self.colors = {
            'background': '#131722',
            'grid': '#363c4e',
            'text': '#d1d4dc',
            'green': '#26a69a',  # Bullish candles
            'red': '#ef5350',    # Bearish candles
            'blue': '#2196f3',   # Indicators
            'orange': '#ff9800', # Signals
            'purple': '#9c27b0', # Orders
            'yellow': '#ffeb3b'  # Highlights
        }
    
    def get_market_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch market data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            # Return empty DataFrame if fetch fails
            return pd.DataFrame()

    def _convert_real_signals_to_chart_format(self, real_signals: List, data: pd.DataFrame) -> List[TradingSignal]:
        """Convert real trading signals to chart format using the main TradingSignal structure"""
        chart_signals = []
        
        for signal in real_signals:
            try:
                # Skip HOLD signals for cleaner charts
                if hasattr(signal, 'signal_type') and signal.signal_type == MainSignalType.HOLD:
                    continue
                
                # Find the price at the signal timestamp
                signal_date = signal.timestamp
                closest_idx = data.index.get_indexer([signal_date], method='nearest')[0]
                price = data['Close'].iloc[closest_idx] if closest_idx >= 0 else 0.0
                
                # Create chart-compatible metadata
                chart_metadata = signal.metadata.copy() if signal.metadata else {}
                chart_metadata.update({
                    'chart_price': price,
                    'chart_reason': f"Confidence: {signal.confidence:.3f}, Strength: {signal.strength:.3f}",
                    'chart_display_type': signal.signal_type.name.lower()  # 'buy', 'sell', 'hold'
                })
                
                # Use the existing TradingSignal structure but add chart-specific metadata
                chart_signal = TradingSignal(
                    symbol=signal.symbol,
                    timestamp=signal.timestamp,
                    signal_type=signal.signal_type,
                    confidence=signal.confidence,
                    strength=signal.strength,
                    source=signal.source,
                    metadata=chart_metadata
                )
                chart_signals.append(chart_signal)
                
            except Exception as e:
                print(f"Warning: Failed to convert signal {signal}: {e}")
                continue
        
        print(f"   Converted {len(chart_signals)} real signals for chart display")
        return chart_signals
    
    def create_comprehensive_chart(self, symbol: str, period: str = "6mo", 
                                 real_signals: Optional[List] = None,
                                 backtest_results: Optional[Dict] = None) -> str:
        """
        Create a comprehensive TradingView-style chart with all components
        
        Args:
            symbol: Trading symbol
            period: Time period for chart data
            real_signals: Optional list of real trading signals to display
            backtest_results: Optional backtest results with performance data
        
        Returns:
            Path to the saved chart file
        """
        # Get market data
        print(f"📊 Fetching market data for {symbol}...")
        data = self.get_market_data(symbol, period)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Calculate technical indicators
        print("🔧 Calculating technical indicators...")
        sma_20 = TechnicalIndicators.sma(data['Close'], 20)
        sma_50 = TechnicalIndicators.sma(data['Close'], 50)
        ema_12 = TechnicalIndicators.ema(data['Close'], 12)
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(data['Close'])
        rsi = TechnicalIndicators.rsi(data['Close'])
        macd_line, signal_line, histogram = TechnicalIndicators.macd(data['Close'])
        
        # Generate trading data
        print("🎯 Generating trading signals...")
        if real_signals is not None:
            print(f"   Using {len(real_signals)} real trading signals")
            signals = self._convert_real_signals_to_chart_format(real_signals, data)
        else:
            print("   No real signals provided - chart will show price data only")
            signals = []
        
        # Use real backtest data if available
        if backtest_results is not None:
            print("   Using real backtest results for orders and portfolio")
            orders = self._extract_orders_from_backtest(backtest_results)
            portfolio_snapshots = self._extract_portfolio_from_backtest(backtest_results, data)
        else:
            print("   No backtest results provided - chart will show basic portfolio tracking")
            orders = []
            portfolio_snapshots = []
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            column_widths=[0.8, 0.2],
            specs=[
                [{"secondary_y": True}, {"rowspan": 4}],  # Main price chart + Volume profile
                [{"secondary_y": False}, None],           # RSI
                [{"secondary_y": False}, None],           # MACD  
                [{"secondary_y": False}, None]            # Portfolio
            ],
            subplot_titles=[
                f'{symbol} - Price & Volume', 'Volume Profile',
                'RSI (14)', '',
                'MACD', '',
                'Portfolio Value', ''
            ],
            vertical_spacing=0.05,
            horizontal_spacing=0.05
        )
        
        # 1. MAIN PRICE CHART (Candlesticks)
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=symbol,
                increasing_line_color=self.colors['green'],
                decreasing_line_color=self.colors['red'],
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 2. MOVING AVERAGES
        fig.add_trace(
            go.Scatter(
                x=data.index, y=sma_20, 
                name='SMA 20', 
                line=dict(color=self.colors['blue'], width=1),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index, y=sma_50, 
                name='SMA 50', 
                line=dict(color=self.colors['orange'], width=1),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index, y=ema_12, 
                name='EMA 12', 
                line=dict(color=self.colors['purple'], width=1, dash='dot'),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # 3. BOLLINGER BANDS
        fig.add_trace(
            go.Scatter(
                x=data.index, y=bb_upper,
                name='BB Upper',
                line=dict(color=self.colors['yellow'], width=1),
                opacity=0.3
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index, y=bb_lower,
                name='BB Lower',
                line=dict(color=self.colors['yellow'], width=1),
                fill='tonexty',
                fillcolor='rgba(255, 235, 59, 0.1)',
                opacity=0.3
            ),
            row=1, col=1
        )
        
        # 4. VOLUME (Secondary Y-axis)
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=self.colors['blue'],
                opacity=0.3,
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # 5. TRADING SIGNALS
        buy_signals = []
        sell_signals = []
        
        for s in signals:
            # Handle both TradingSignal objects and dictionaries
            if hasattr(s, 'signal_type'):
                # TradingSignal object
                if s.signal_type == MainSignalType.BUY:
                    buy_signals.append(s)
                elif s.signal_type == MainSignalType.SELL:
                    sell_signals.append(s)
            elif isinstance(s, dict):
                # Dictionary format (legacy support)
                signal_type = s.get('signal_type', '')
                if 'BUY' in str(signal_type) or signal_type == 1:
                    buy_signals.append(s)
                elif 'SELL' in str(signal_type) or signal_type == -1:
                    sell_signals.append(s)
        
        if buy_signals:
            # Extract timestamps and prices safely
            buy_timestamps = []
            buy_prices = []
            buy_texts = []
            
            for s in buy_signals:
                if hasattr(s, 'timestamp'):
                    buy_timestamps.append(s.timestamp)
                    buy_prices.append(s.metadata.get('chart_price', 0.0))
                    buy_texts.append(f"Buy: {s.confidence:.1%}<br>{s.metadata.get('chart_reason', '')}")
                elif isinstance(s, dict):
                    buy_timestamps.append(s.get('timestamp', pd.Timestamp.now()))
                    buy_prices.append(s.get('price', 0.0))
                    buy_texts.append(f"Buy: {s.get('confidence', 0.5):.1%}<br>{s.get('reason', '')}")
            
            fig.add_trace(
                go.Scatter(
                    x=buy_timestamps,
                    y=buy_prices,
                    mode='markers',
                    name='Buy Signals',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color=self.colors['green'],
                        line=dict(width=2, color='white')
                    ),
                    text=buy_texts,
                    textposition="top center"
                ),
                row=1, col=1
            )
        
        if sell_signals:
            # Extract timestamps and prices safely
            sell_timestamps = []
            sell_prices = []
            sell_texts = []
            
            for s in sell_signals:
                if hasattr(s, 'timestamp'):
                    sell_timestamps.append(s.timestamp)
                    sell_prices.append(s.metadata.get('chart_price', 0.0))
                    sell_texts.append(f"Sell: {s.confidence:.1%}<br>{s.metadata.get('chart_reason', '')}")
                elif isinstance(s, dict):
                    sell_timestamps.append(s.get('timestamp', pd.Timestamp.now()))
                    sell_prices.append(s.get('price', 0.0))
                    sell_texts.append(f"Sell: {s.get('confidence', 0.5):.1%}<br>{s.get('reason', '')}")
            
            fig.add_trace(
                go.Scatter(
                    x=sell_timestamps,
                    y=sell_prices,
                    mode='markers',
                    name='Sell Signals',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color=self.colors['red'],
                        line=dict(width=2, color='white')
                    ),
                    text=sell_texts,
                    textposition="bottom center"
                ),
                row=1, col=1
            )
        
        # 6. ORDERS (with size information)
        buy_orders = [o for o in orders if o.side == 'buy']
        sell_orders = [o for o in orders if o.side == 'sell']
        
        if buy_orders:
            fig.add_trace(
                go.Scatter(
                    x=[o.timestamp for o in buy_orders],
                    y=[o.price for o in buy_orders],
                    mode='markers+text',
                    name='💰 Buy Orders',
                    marker=dict(
                        symbol='triangle-up',
                        size=[min(25, max(12, o.size_usd / 3000)) for o in buy_orders],  # Larger, more visible
                        color=self.colors['green'],
                        opacity=0.9,  # More opaque
                        line=dict(width=2, color='white')
                    ),
                    text=[f"BUY<br>${o.size_usd:,.0f}" for o in buy_orders],
                    textposition="top center",
                    textfont=dict(size=8, color='white'),
                    hovertext=[f"Buy Order: ${o.size_usd:,.0f}<br>{o.quantity:.1f} shares<br>{o.reason}" for o in buy_orders],
                    hoverinfo='text'
                ),
                row=1, col=1
            )
        
        if sell_orders:
            fig.add_trace(
                go.Scatter(
                    x=[o.timestamp for o in sell_orders],
                    y=[o.price for o in sell_orders],
                    mode='markers+text',
                    name='💸 Sell Orders',
                    marker=dict(
                        symbol='triangle-down',
                        size=[min(25, max(12, o.size_usd / 3000)) for o in sell_orders],  # Larger, more visible
                        color=self.colors['red'],
                        opacity=0.9,  # More opaque
                        line=dict(width=2, color='white')
                    ),
                    text=[f"SELL<br>${o.size_usd:,.0f}" for o in sell_orders],
                    textposition="bottom center",
                    textfont=dict(size=8, color='white'),
                    hovertext=[f"Sell Order: ${o.size_usd:,.0f}<br>{o.quantity:.1f} shares<br>{o.reason}" for o in sell_orders],
                    hoverinfo='text'
                ),
                row=1, col=1
            )
        
        # 7. VOLUME PROFILE (Right side)
        price_levels, volume_profile = TechnicalIndicators.volume_profile(data['Close'], data['Volume'])
        fig.add_trace(
            go.Scatter(
                x=volume_profile,
                y=price_levels,
                mode='lines',
                name='Volume Profile',
                fill='tozeroy',
                line=dict(color=self.colors['blue'], width=2),
                fillcolor='rgba(33, 150, 243, 0.3)'
            ),
            row=1, col=2
        )
        
        # 8. RSI INDICATOR
        fig.add_trace(
            go.Scatter(
                x=data.index, y=rsi,
                name='RSI',
                line=dict(color=self.colors['purple'], width=2)
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color=self.colors['red'], opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=self.colors['green'], opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color=self.colors['text'], opacity=0.3, row=2, col=1)
        
        # 9. MACD INDICATOR
        fig.add_trace(
            go.Scatter(
                x=data.index, y=macd_line,
                name='MACD',
                line=dict(color=self.colors['blue'], width=2)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index, y=signal_line,
                name='Signal',
                line=dict(color=self.colors['orange'], width=2)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=data.index, y=histogram,
                name='Histogram',
                marker_color=self.colors['green'],
                opacity=0.6
            ),
            row=3, col=1
        )
        
        # 10. PORTFOLIO VALUE
        if portfolio_snapshots:
            portfolio_dates = [p.timestamp for p in portfolio_snapshots]
            portfolio_values = [p.total_value for p in portfolio_snapshots]
            
            fig.add_trace(
                go.Scatter(
                    x=portfolio_dates,
                    y=portfolio_values,
                    name='Portfolio Value',
                    line=dict(color=self.colors['yellow'], width=3),
                    fill='tonexty'
                ),
                row=4, col=1
            )
            
            # Add portfolio annotations
            final_value = portfolio_values[-1]
            initial_value = portfolio_values[0]
            total_return = (final_value / initial_value - 1) * 100
            
            fig.add_annotation(
                x=portfolio_dates[-1],
                y=final_value,
                text=f"${final_value:,.0f}<br>({total_return:+.1f}%)",
                showarrow=True,
                arrowhead=2,
                arrowcolor=self.colors['yellow'],
                bgcolor=self.colors['background'],
                bordercolor=self.colors['yellow'],
                borderwidth=1,
                row=4, col=1
            )
        
        # UPDATE LAYOUT
        fig.update_layout(
            title=dict(
                text=f'🚀 {symbol} Trading Analysis - TradingView Style Dashboard',
                x=0.5,
                y=0.98,  # Move title higher to create space
                font=dict(size=20, color=self.colors['text'])
            ),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.95,  # Position legend below title
                xanchor="center",
                x=0.5,
                bgcolor='rgba(19, 23, 34, 0.9)',
                bordercolor=self.colors['grid'],
                borderwidth=1,
                font=dict(size=10)  # Smaller font for compact legend
            ),
            height=1200,
            width=1600,
            margin=dict(t=120, b=60, l=60, r=60)  # Add top margin for title/legend
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=self.colors['grid'],
            showline=True,
            linewidth=1,
            linecolor=self.colors['grid']
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=self.colors['grid'],
            showline=True,
            linewidth=1,
            linecolor=self.colors['grid']
        )
        
        # Hide volume y-axis labels on secondary axis
        fig.update_yaxes(secondary_y=True, showticklabels=False, row=1, col=1)
        
        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trading_chart_{symbol}_{timestamp}.html"
        filepath = os.path.join(self.results_dir, filename)
        
        fig.write_html(filepath)
        print(f"📈 Chart saved: {filepath}")
        
        # Also save as PNG if kaleido is available
        # Temporarily disabled PNG generation due to hanging issues
        # try:
        #     png_filepath = filepath.replace('.html', '.png')
        #     fig.write_image(png_filepath, width=1600, height=1200)
        #     print(f"📸 PNG chart saved: {png_filepath}")
        # except Exception as e:
        #     print(f"⚠️ Could not save PNG (install kaleido for PNG export): {e}")
        print("📸 PNG generation disabled to prevent hanging")
        
        return filepath

    def _extract_orders_from_backtest(self, backtest_results: Dict) -> List[Order]:
        """Extract orders from backtest results"""
        orders = []
        if 'trades' in backtest_results:
            for trade in backtest_results['trades']:
                # Convert backtest trade to Order format
                order = Order(
                    timestamp=trade.get('entry_date', datetime.now()),
                    symbol=trade.get('symbol', 'UNKNOWN'),
                    action=trade.get('action', 'BUY'),
                    quantity=trade.get('quantity', 100),
                    price=trade.get('entry_price', 0.0),
                    order_type='MARKET'
                )
                orders.append(order)
        return orders
    
    def _extract_portfolio_from_backtest(self, backtest_results: Dict, data: pd.DataFrame) -> List[Dict]:
        """Extract portfolio snapshots from backtest results"""
        portfolio_snapshots = []
        if 'portfolio_history' in backtest_results:
            for snapshot in backtest_results['portfolio_history']:
                portfolio_snapshots.append({
                    'date': snapshot.get('date', datetime.now()),
                    'value': snapshot.get('total_value', 100000),
                    'cash': snapshot.get('cash', 100000),
                    'positions': snapshot.get('positions', 0)
                })
        else:
            # Generate basic portfolio performance if no detailed history
            performance = backtest_results.get('performance', {})
            initial_value = 100000
            final_return = performance.get('total_return', 0.0)
            final_value = initial_value * (1 + final_return)
            
            # Create simple linear progression
            for i, date in enumerate(data.index[-30:]):  # Last 30 days
                progress = i / 29.0 if len(data.index[-30:]) > 1 else 1.0
                current_value = initial_value + (final_value - initial_value) * progress
                portfolio_snapshots.append({
                    'date': date,
                    'value': current_value,
                    'cash': current_value * 0.1,  # Assume 10% cash
                    'positions': current_value * 0.9
                })
        
        return portfolio_snapshots


if __name__ == "__main__":
    # Example usage: python tradingview_charts.py
    print("TradingView Chart Generator")
    print("Import this module to use chart generation functionality")
    print("Example: chart_generator = TradingViewChartGenerator()")
    print("         chart_path = chart_generator.create_comprehensive_chart('AAPL')")