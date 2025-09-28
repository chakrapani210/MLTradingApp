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
class TradingSignal:
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    confidence: float
    price: float
    source: str
    reason: str = ""


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
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
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
        price_min, price_max = prices.min(), prices.max()
        price_bins = np.linspace(price_min, price_max, bins + 1)
        volume_profile = np.zeros(bins)
        
        for i in range(len(prices)):
            bin_idx = np.digitize(prices.iloc[i], price_bins) - 1
            bin_idx = max(0, min(bins - 1, bin_idx))
            volume_profile[bin_idx] += volumes.iloc[i]
        
        price_levels = (price_bins[:-1] + price_bins[1:]) / 2
        return price_levels, volume_profile


class TradingViewChartGenerator:
    """
    Generate comprehensive TradingView-style charts with all trading information
    """
    
    def __init__(self, results_dir: str = "results"):
        """Initialize the chart generator"""
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
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
            # Return sample data if fetch fails
            return self._generate_sample_data(symbol)
    
    def _generate_sample_data(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """Generate sample market data for demonstration"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate realistic price data
        np.random.seed(42)  # For reproducible results
        price = 150  # Starting price
        prices = []
        volumes = []
        
        for i in range(days):
            # Random walk with slight upward bias
            change = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
            price = price * (1 + change)
            prices.append(price)
            
            # Generate volume (higher on price moves)
            base_volume = 1000000
            volume_multiplier = 1 + abs(change) * 10
            volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 2.0))
            volumes.append(volume)
        
        # Create OHLC data
        data = pd.DataFrame(index=dates)
        data['Close'] = prices
        
        # Generate OHLC from close prices
        data['Open'] = data['Close'].shift(1).fillna(prices[0])
        data['High'] = np.maximum(data['Open'], data['Close']) * np.random.uniform(1.000, 1.020, len(data))
        data['Low'] = np.minimum(data['Open'], data['Close']) * np.random.uniform(0.980, 1.000, len(data))
        data['Volume'] = volumes
        
        return data
    
    def generate_sample_signals(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """Generate sample trading signals for demonstration"""
        signals = []
        
        # Calculate RSI for signal generation
        rsi = TechnicalIndicators.rsi(data['Close'])
        
        for i in range(50, len(data), 10):  # Every 10 days starting from day 50
            timestamp = data.index[i]
            price = data['Close'].iloc[i]
            rsi_value = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50
            
            # Generate signals based on RSI
            if rsi_value < 30:  # Oversold
                signal = TradingSignal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    confidence=min(0.9, (30 - rsi_value) / 30 + 0.5),
                    price=price,
                    source="RSI_Strategy",
                    reason=f"RSI oversold: {rsi_value:.1f}"
                )
                signals.append(signal)
            elif rsi_value > 70:  # Overbought
                signal = TradingSignal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=min(0.9, (rsi_value - 70) / 30 + 0.5),
                    price=price,
                    source="RSI_Strategy",
                    reason=f"RSI overbought: {rsi_value:.1f}"
                )
                signals.append(signal)
        
        return signals
    
    def generate_sample_orders(self, signals: List[TradingSignal], portfolio_value: float = 100000) -> List[Order]:
        """Generate sample orders based on signals"""
        orders = []
        position_size_pct = 0.08  # 8% of portfolio per trade
        
        for signal in signals:
            if signal.confidence > 0.5:  # Lower threshold to generate more orders
                side = 'buy' if signal.signal_type == SignalType.BUY else 'sell'
                size_usd = portfolio_value * position_size_pct * signal.confidence
                quantity = size_usd / signal.price
                
                order = Order(
                    timestamp=signal.timestamp,
                    symbol=signal.symbol,
                    order_type=OrderType.MARKET,
                    side=side,
                    quantity=quantity,
                    price=signal.price,
                    size_usd=size_usd,
                    reason=f"Signal: {signal.reason}"
                )
                orders.append(order)
        
        return orders
    
    def generate_sample_portfolio(self, data: pd.DataFrame, orders: List[Order], 
                                 initial_value: float = 100000) -> List[PortfolioSnapshot]:
        """Generate sample portfolio performance"""
        portfolio_snapshots = []
        
        current_value = initial_value
        total_return = 0.0
        
        # Sample every 10 days
        for i in range(0, len(data), 10):
            timestamp = data.index[i]
            
            # Simulate portfolio growth (based on market performance)
            if i > 0:
                market_return = (data['Close'].iloc[i] / data['Close'].iloc[i-10] - 1)
                portfolio_return = market_return * 0.8  # Slightly underperform market
                daily_return = portfolio_return / 10  # Average daily return
                
                new_value = current_value * (1 + portfolio_return)
                daily_pnl = new_value - current_value
                current_value = new_value
                total_return = (current_value / initial_value - 1) * 100
            else:
                daily_return = 0.0
                daily_pnl = 0.0
            
            # Simulate cash vs positions (assume 80% invested)
            positions_value = current_value * 0.8
            cash = current_value * 0.2
            
            snapshot = PortfolioSnapshot(
                timestamp=timestamp,
                total_value=current_value,
                cash=cash,
                positions_value=positions_value,
                daily_pnl=daily_pnl,
                total_pnl=current_value - initial_value,
                daily_return=daily_return,
                total_return=total_return
            )
            portfolio_snapshots.append(snapshot)
        
        return portfolio_snapshots
    
    def create_comprehensive_chart(self, symbol: str, period: str = "6mo") -> str:
        """
        Create a comprehensive TradingView-style chart with all components
        
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
        
        # Generate sample trading data
        print("🎯 Generating trading signals...")
        signals = self.generate_sample_signals(data, symbol)
        orders = self.generate_sample_orders(signals)
        portfolio_snapshots = self.generate_sample_portfolio(data, orders)
        
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
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        if buy_signals:
            fig.add_trace(
                go.Scatter(
                    x=[s.timestamp for s in buy_signals],
                    y=[s.price for s in buy_signals],
                    mode='markers',
                    name='Buy Signals',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color=self.colors['green'],
                        line=dict(width=2, color='white')
                    ),
                    text=[f"Buy: {s.confidence:.1%}<br>{s.reason}" for s in buy_signals],
                    textposition="top center"
                ),
                row=1, col=1
            )
        
        if sell_signals:
            fig.add_trace(
                go.Scatter(
                    x=[s.timestamp for s in sell_signals],
                    y=[s.price for s in sell_signals],
                    mode='markers',
                    name='Sell Signals',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color=self.colors['red'],
                        line=dict(width=2, color='white')
                    ),
                    text=[f"Sell: {s.confidence:.1%}<br>{s.reason}" for s in sell_signals],
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
        try:
            png_filepath = filepath.replace('.html', '.png')
            fig.write_image(png_filepath, width=1600, height=1200)
            print(f"📸 PNG chart saved: {png_filepath}")
        except Exception as e:
            print(f"⚠️ Could not save PNG (install kaleido for PNG export): {e}")
        
        return filepath


# Example usage function
def create_sample_charts(symbols: List[str] = None):
    """Create sample charts for demonstration"""
    if symbols is None:
        symbols = ['AAPL', 'TSLA', 'MSFT']
    
    chart_generator = TradingViewChartGenerator()
    
    for symbol in symbols:
        try:
            print(f"\n🎨 Creating comprehensive chart for {symbol}...")
            filepath = chart_generator.create_comprehensive_chart(symbol)
            print(f"✅ Chart created successfully: {filepath}")
        except Exception as e:
            print(f"❌ Error creating chart for {symbol}: {e}")


if __name__ == "__main__":
    # Create sample charts
    create_sample_charts(['AAPL'])