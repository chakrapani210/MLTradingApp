"""
Enhanced TradingView-Style Chart Generator
Creates interactive fullscreen charts with configurable indicators, order sizes, and decision reasons
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_indicators import get_market_data
from enhanced_strategy import EnhancedTradingStrategy
from config_manager import get_config
import datetime as dt

def create_enhanced_tradingview_charts_for_symbols(symbols=['NVDA', 'TSLA', 'AAPL']):
    """Create enhanced TradingView charts for multiple symbols with order details"""
    print("Creating Enhanced TradingView Charts with Order Details...")
    
    # Get configuration
    config = get_config()
    trading_config = config.config.get('trading', {})
    sma_short = trading_config.get('indicators', {}).get('sma_short', 50)
    sma_long = trading_config.get('indicators', {}).get('sma_long', 200)
    rsi_period = trading_config.get('indicators', {}).get('rsi_period', 14)
    
    print(f"Using SMA {sma_short}/{sma_long} configuration")
    
    results = {}
    
    for symbol in symbols:
        print(f"\\nProcessing {symbol}...")
        result = create_symbol_chart_with_trades(symbol, config, sma_short, sma_long, rsi_period)
        results[symbol] = result
    
    # Create master dashboard
    create_master_dashboard(results, sma_short, sma_long)
    
    return results

def create_symbol_chart_with_trades(symbol, config, sma_short, sma_long, rsi_period):
    """Create detailed chart for a single symbol with trade information"""
    
    # Define test period
    end_date = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - dt.timedelta(days=365)
    train_end = start_date + dt.timedelta(days=183)
    
    # Get market data with OHLC for candlestick charts - FULL PERIOD for complete chart
    stock_data = get_market_data([symbol], start_date, end_date, include_ohlc=True)[symbol]
    
    # Also get close-only data for strategy simulation (legacy compatibility)
    close_data = get_market_data([symbol], train_end, end_date, include_ohlc=False)[symbol]
    
    # Run enhanced trading strategy to get trade details
    strategy = EnhancedTradingStrategy(config)
    
    # Calculate months between start and end date
    months_back = int((end_date - start_date).days / 30)
    
    performance_result = strategy.run_complete_simulation(symbol, months_back=months_back)
    trade_details = getattr(strategy, 'latest_trade_details', [])
    
    print(f"Found {len(trade_details)} trades for {symbol}")
    
    # Add configurable technical indicators using close price
    close_price = stock_data['Close'] if 'Close' in stock_data.columns else stock_data.iloc[:, 0]
    stock_data[f'SMA_{sma_short}'] = close_price.rolling(window=sma_short).mean()
    stock_data[f'SMA_{sma_long}'] = close_price.rolling(window=sma_long).mean()
    
    # Calculate RSI with configurable period
    delta = close_price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    
    # Create the chart
    fig = create_interactive_chart(stock_data, symbol, sma_short, sma_long, trade_details, performance_result)
    
    # Save individual chart
    filename = f"tests/results/{symbol.lower()}_enhanced_chart.html"
    fig.write_html(filename, include_plotlyjs='cdn')
    print(f"{symbol} Enhanced Chart saved: {filename}")
    
    return {
        'symbol': symbol,
        'filename': filename,
        'trades': len(trade_details),
        'performance': performance_result,
        'chart': fig
    }

def create_interactive_chart(stock_data, symbol, sma_short, sma_long, trade_details, performance):
    """Create the interactive Plotly chart with proper OHLC data and fullscreen layout"""
    
    # Create subplots with fullscreen layout
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(f'{symbol} Stock Price & Trading Signals', 'Volume', 'RSI'),
        vertical_spacing=0.03,
        row_heights=[0.70, 0.20, 0.10],
        shared_xaxes=True
    )
    
    # Main price chart (Candlestick) with proper OHLC data
    if all(col in stock_data.columns for col in ['Open', 'High', 'Low', 'Close']):
        # Use actual OHLC data
        fig.add_trace(
            go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name=f'{symbol} Price',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444',
                increasing_fillcolor='#00ff88',
                decreasing_fillcolor='#ff4444'
            ),
            row=1, col=1
        )
        close_price = stock_data['Close']
    else:
        # Fallback to line chart if OHLC not available
        close_price = stock_data.iloc[:, 0]
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=close_price,
                mode='lines',
                name=f'{symbol} Price',
                line=dict(color='#00aaff', width=2)
            ),
            row=1, col=1
        )
    
    # Add SMA lines using close price
    if f'SMA_{sma_short}' in stock_data.columns:
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data[f'SMA_{sma_short}'],
                mode='lines',
                name=f'SMA {sma_short}',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
    
    if f'SMA_{sma_long}' in stock_data.columns:
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data[f'SMA_{sma_long}'],
                mode='lines',
                name=f'SMA {sma_long}',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
    
    # Add trade markers with enhanced information
    if trade_details:
        buy_trades = [t for t in trade_details if t['action'] == 'BUY']
        sell_trades = [t for t in trade_details if t['action'] == 'SELL']
        
        if buy_trades:
            buy_dates = [t['date'] for t in buy_trades]
            buy_prices = [t['price'] for t in buy_trades]
            buy_text = [
                f"🔵 BUY {t['shares']} shares at ${t['price']:.2f}<br>"
                f"💰 Total: ${t['value']:.2f}<br>"
                f"📋 Reason: {t['reason'][:100]}..." if len(t['reason']) > 100 else f"📋 Reason: {t['reason']}"
                for t in buy_trades
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='green',
                        line=dict(width=2, color='white')
                    ),
                    name='Buy Orders',
                    text=buy_text,
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )
        
        if sell_trades:
            sell_dates = [t['date'] for t in sell_trades]
            sell_prices = [t['price'] for t in sell_trades]
            sell_text = [
                f"🔴 SELL {t['shares']} shares at ${t['price']:.2f}<br>"
                f"💰 Total: ${t['value']:.2f}<br>"
                f"📋 Reason: {t['reason'][:100]}..." if len(t['reason']) > 100 else f"📋 Reason: {t['reason']}"
                for t in sell_trades
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='red',
                        line=dict(width=2, color='white')
                    ),
                    name='Sell Orders',
                    text=sell_text,
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Volume chart with real data
    if 'Volume' in stock_data.columns and not stock_data['Volume'].isna().all():
        # Use real volume data
        volume_data = stock_data['Volume']
        # Color bars based on price direction
        close_prices = stock_data['Close'] if 'Close' in stock_data.columns else stock_data.iloc[:, 0]
        price_changes = close_prices.diff()
        colors = ['#00ff88' if change >= 0 else '#ff4444' for change in price_changes]
    else:
        # Fallback to simulated volume data
        volume_data = [1000000 + np.random.randint(-200000, 200000) for _ in range(len(stock_data))]
        colors = ['#00ff88' if i % 2 == 0 else '#ff4444' for i in range(len(volume_data))]
    
    fig.add_trace(
        go.Bar(
            x=stock_data.index,
            y=volume_data,
            name='Volume',
            marker_color=colors,
            opacity=0.7,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # RSI chart
    if 'RSI' in stock_data.columns:
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=3, col=1
        )
        
        # Add RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", row=3, col=1)
    
    # Update layout for fullscreen display
    fig.update_layout(
        title=f'{symbol} Enhanced Trading Analysis - Fullscreen View<br><sub>SMA {sma_short}/{sma_long} | Performance: {performance.get("strategy_return", 0):.1%}</sub>',
        template='plotly_dark',
        height=900,  # Fullscreen height
        width=None,  # Auto width to fill container
        showlegend=True,
        hovermode='x unified',
        margin=dict(l=80, r=80, t=120, b=80),  # Generous margins for fullscreen
        font=dict(size=13),  # Larger font for readability
        title_font_size=18,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Customize axes
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    return fig

def create_master_dashboard(results, sma_short, sma_long):
    """Create a master dashboard with fullscreen styling"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced Trading Strategy Dashboard - Fullscreen</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1e3c72, #2a5298);
                color: white;
                min-height: 100vh;
                overflow-x: auto;
            }}
            .fullscreen-container {{
                padding: 40px;
                max-width: 1600px;
                margin: 0 auto;
                min-height: 100vh;
            }}
            .header {{
                text-align: center;
                margin-bottom: 50px;
            }}
            .header h1 {{
                font-size: 3em;
                margin-bottom: 15px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            }}
            .header h2 {{
                font-size: 1.8em;
                margin-bottom: 20px;
                color: #ffd700;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 30px;
                margin-bottom: 50px;
            }}
            .stat-card {{
                background: rgba(255, 255, 255, 0.15);
                border-radius: 20px;
                padding: 40px;
                backdrop-filter: blur(15px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.3s ease;
                text-align: center;
            }}
            .stat-card:hover {{
                transform: translateY(-8px);
                box-shadow: 0 15px 40px rgba(0,0,0,0.4);
                background: rgba(255, 255, 255, 0.2);
            }}
            .symbol-title {{
                font-size: 2.2em;
                font-weight: bold;
                margin-bottom: 20px;
            }}
            .performance {{
                font-size: 3.5em;
                font-weight: bold;
                margin: 20px 0;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            }}
            .positive {{ color: #00ff88; }}
            .negative {{ color: #ff4444; }}
            .neutral {{ color: #ffaa00; }}
            .stat-details {{
                font-size: 1.2em;
                line-height: 1.8;
                margin-top: 15px;
            }}
            .chart-links {{
                text-align: center;
                margin-top: 50px;
            }}
            .chart-links h3 {{
                font-size: 2em;
                margin-bottom: 30px;
                color: #ffd700;
            }}
            .chart-link {{
                display: inline-block;
                margin: 20px;
                padding: 25px 50px;
                background: rgba(255, 255, 255, 0.2);
                color: white;
                text-decoration: none;
                border-radius: 15px;
                font-size: 1.3em;
                font-weight: 600;
                transition: all 0.3s ease;
                border: 1px solid rgba(255, 255, 255, 0.1);
                min-width: 250px;
            }}
            .chart-link:hover {{
                background: rgba(255, 255, 255, 0.35);
                transform: translateY(-5px) scale(1.05);
                box-shadow: 0 12px 35px rgba(0,0,0,0.3);
            }}
            .fullscreen-note {{
                text-align: center;
                margin-top: 50px;
                padding: 30px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                font-size: 1.1em;
                line-height: 1.6;
            }}
        </style>
    </head>
    <body>
        <div class="fullscreen-container">
            <div class="header">
                <h1>🚀 Enhanced Trading Dashboard</h1>
                <h2>📈 SMA {sma_short}/{sma_long} Configuration - Fullscreen Experience</h2>
                <p style="font-size: 1.2em;">Interactive charts optimized for fullscreen viewing with detailed order analysis</p>
            </div>
            
            <div class="stats-grid">
    """
    
    for symbol_data in results.values():
        symbol = symbol_data['symbol']
        performance = symbol_data['performance']
        trades = symbol_data['trades']
        
        strategy_return = performance.get('strategy_return', 0)
        performance_class = 'positive' if strategy_return > 0 else 'negative' if strategy_return < 0 else 'neutral'
        
        html_content += f"""
            <div class="stat-card">
                <div class="symbol-title">{symbol}</div>
                <div class="performance {performance_class}">{strategy_return:.1%}</div>
                <div class="stat-details">
                    <div>📊 Total Trades: {trades}</div>
                    <div>📈 Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}</div>
                    <div>💰 vs Buy-Hold: {performance.get('outperformance', 0):.1%}</div>
                    <div>🎯 Max Drawdown: {performance.get('max_drawdown', 0):.1%}</div>
                </div>
            </div>
        """
    
    html_content += """
            </div>
            
            <div class="chart-links">
                <h3>📋 Fullscreen Interactive Charts</h3>
    """
    
    for symbol_data in results.values():
        symbol = symbol_data['symbol']
        filename = symbol_data['filename'].split('/')[-1]  # Just the filename
        html_content += f"""
            <a href="{filename}" class="chart-link" target="_blank">
                📈 {symbol} Fullscreen Analysis
            </a>
        """
    
    html_content += f"""
            </div>
            
            <div class="fullscreen-note">
                <h3>🔍 Fullscreen Chart Features</h3>
                <p>💡 <strong>Order Details:</strong> Hover over trade markers to see exact order sizes and AI decision reasoning</p>
                <p>📊 <strong>Technical Analysis:</strong> SMA {sma_short}/{sma_long} indicators with RSI and volume analysis</p>
                <p>🖥️ <strong>Optimized Display:</strong> 900px height charts designed for fullscreen viewing experience</p>
                <p>⚡ <strong>Interactive Elements:</strong> Zoom, pan, and hover for detailed trade information</p>
                <p>🎯 <strong>Performance Metrics:</strong> Real-time comparison with buy-and-hold strategies</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    dashboard_file = "tests/results/enhanced_trading_dashboard.html"
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Enhanced Fullscreen Master Dashboard created: {dashboard_file}")
    return dashboard_file

# Legacy function for backward compatibility
def create_simple_tradingview_chart(symbol='TSLA'):
    """Legacy function - redirects to enhanced version"""
    results = create_enhanced_tradingview_charts_for_symbols([symbol])
    return results[symbol]['filename']

def create_complete_dashboard():
    """Create complete dashboard with fullscreen features"""
    print("Creating Complete Enhanced Fullscreen Dashboard...")
    
    # Create charts for all symbols
    results = create_enhanced_tradingview_charts_for_symbols(['NVDA', 'TSLA', 'AAPL'])
    
    print("\\nEnhanced Fullscreen Dashboard Creation Complete!")
    print("Features included:")
    print("   - Configurable SMA indicators from config.yaml")
    print("   - Order sizes and AI decision reasons")
    print("   - Interactive trade markers with hover details")
    print("   - Performance comparisons vs buy-hold")
    print("   - 900px height for fullscreen display")
    print("   - Professional fullscreen styling")
    print("   - Enhanced responsive design")
    
    return results

if __name__ == "__main__":
    create_complete_dashboard()
    print("\\nFullscreen dashboard ready! Open the HTML files for the ultimate trading analysis experience!")