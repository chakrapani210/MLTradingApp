"""
Enhanced Strategy Test Framework
Comprehensive testing for the ML trading strategy with visualization
"""

import unittest
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import mplfinance as mpf
from typing import Dict, Any, List, Tuple
import warnings
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_strategy import EnhancedTradingStrategy
from model_management import ModelManager
from market_indicators import get_market_data
from config_manager import get_config
from create_enhanced_tradingview_charts import create_symbol_chart_with_trades, create_enhanced_tradingview_charts_for_symbols

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TestEnhancedStrategy(unittest.TestCase):
    """
    Comprehensive test suite for Enhanced Trading Strategy
    
    Tests the strategy with 1-year historical data:
    - First 6 months: Training period
    - Next 6 months: Testing period
    - Generates comprehensive performance charts and analysis
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_symbols = ["TSLA", "AAPL", "NVDA"]
        cls.results = {}
        
        # Calculate test periods (last 1 year)
        cls.end_date = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cls.start_date = cls.end_date - dt.timedelta(days=365)
        cls.train_end = cls.start_date + dt.timedelta(days=183)  # 6 months training
        
        print(f"Test Period Setup:")
        print(f"  Full Period: {cls.start_date.strftime('%Y-%m-%d')} to {cls.end_date.strftime('%Y-%m-%d')}")
        print(f"  Training: {cls.start_date.strftime('%Y-%m-%d')} to {cls.train_end.strftime('%Y-%m-%d')}")
        print(f"  Testing: {cls.train_end.strftime('%Y-%m-%d')} to {cls.end_date.strftime('%Y-%m-%d')}")
        
        # Create results directory
        cls.results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(cls.results_dir, exist_ok=True)
    
    def setUp(self):
        """Set up each test"""
        self.strategy = EnhancedTradingStrategy()
    
    def test_single_symbol_performance(self):
        """Test strategy performance on a single symbol (TSLA)"""
        print(f"\n{'='*60}")
        print(f"TESTING SINGLE SYMBOL PERFORMANCE: TSLA")
        print(f"{'='*60}")
        
        symbol = "TSLA"
        
        # Step 1: Train model with first 6 months
        print(f"\n[TRAIN] Training model with first 6 months data...")
        X_train, y_train = self.strategy.prepare_training_data(
            symbol, self.start_date, self.train_end
        )
        self.strategy.symbol = symbol
        self.strategy.train_model(X_train, y_train, save_model=True)
        
        # Step 2: Test with next 6 months
        print(f"\n[TEST] Testing with next 6 months data...")
        predictions, signal_analysis = self.strategy.generate_predictions(
            symbol, self.train_end, self.end_date
        )
        
        # Step 3: Run backtest
        print(f"\n[BACKTEST] Running comprehensive backtest...")
        performance = self.strategy.backtest_strategy(
            symbol, self.train_end, self.end_date, predictions
        )
        
        # Step 4: Analyze results
        feature_analysis = self.strategy.analyze_feature_importance()
        
        # Compile results
        test_results = {
            'symbol': symbol,
            'dates': {
                'train_start': self.start_date,
                'train_end': self.train_end,
                'test_start': self.train_end,
                'test_end': self.end_date
            },
            'performance': performance,
            'signal_analysis': signal_analysis,
            'feature_analysis': feature_analysis,
            'predictions': predictions
        }
        
        self.results[symbol] = test_results
        
        # Assertions for test validation
        self.assertIsNotNone(performance)
        self.assertIn('strategy_return', performance)
        self.assertIn('buy_hold_return', performance)
        self.assertGreater(len(predictions), 0)
        
        print(f"\n[RESULTS] Test completed successfully:")
        print(f"  Strategy Return: {performance.get('strategy_return', 0):.1%}")
        print(f"  Buy-Hold Return: {performance.get('buy_hold_return', 0):.1%}")
        print(f"  Outperformance: {performance.get('outperformance', 0):.1%}")
        print(f"  Sharpe Ratio: {performance.get('sharpe_ratio', 0):.3f}")
    
    def test_multiple_symbols_comparison(self):
        """Test strategy performance across multiple symbols"""
        print(f"\n{'='*60}")
        print(f"TESTING MULTIPLE SYMBOLS COMPARISON")
        print(f"{'='*60}")
        
        for symbol in self.test_symbols:
            if symbol in self.results:
                print(f"\n[SKIP] {symbol} already tested")
                continue
                
            print(f"\n[TEST] Testing {symbol}...")
            
            try:
                # Create fresh strategy instance for each symbol
                strategy = EnhancedTradingStrategy()
                
                # Train and test
                X_train, y_train = strategy.prepare_training_data(
                    symbol, self.start_date, self.train_end
                )
                strategy.symbol = symbol
                strategy.train_model(X_train, y_train, save_model=True)
                
                predictions, signal_analysis = strategy.generate_predictions(
                    symbol, self.train_end, self.end_date
                )
                
                performance = strategy.backtest_strategy(
                    symbol, self.train_end, self.end_date, predictions
                )
                
                feature_analysis = strategy.analyze_feature_importance()
                
                # Store results
                self.results[symbol] = {
                    'symbol': symbol,
                    'performance': performance,
                    'signal_analysis': signal_analysis,
                    'feature_analysis': feature_analysis,
                    'predictions': predictions
                }
                
                print(f"  [OK] {symbol}: {performance.get('strategy_return', 0):.1%} return")
                
            except Exception as e:
                print(f"  [ERROR] {symbol}: {e}")
                self.results[symbol] = None
        
        # Validate we have results for comparison
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        self.assertGreater(len(valid_results), 0, "Should have at least one valid result")
    
    def test_strategy_robustness(self):
        """Test strategy robustness across different market conditions"""
        print(f"\n{'='*60}")
        print(f"TESTING STRATEGY ROBUSTNESS")
        print(f"{'='*60}")
        
        # Test with different time periods within the year
        periods = [
            ("Q1-Q2", self.start_date, self.start_date + dt.timedelta(days=183)),
            ("Q2-Q3", self.start_date + dt.timedelta(days=91), self.start_date + dt.timedelta(days=274)),
            ("Q3-Q4", self.start_date + dt.timedelta(days=183), self.end_date)
        ]
        
        symbol = "TSLA"  # Use TSLA for robustness testing
        robustness_results = {}
        
        for period_name, period_start, period_end in periods:
            print(f"\n[ROBUSTNESS] Testing period: {period_name}")
            print(f"  Period: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
            
            try:
                # Split period into train/test
                period_mid = period_start + dt.timedelta(days=91)  # ~3 months each
                
                strategy = EnhancedTradingStrategy()
                
                # Train on first half, test on second half
                X_train, y_train = strategy.prepare_training_data(
                    symbol, period_start, period_mid
                )
                strategy.symbol = symbol
                strategy.train_model(X_train, y_train, save_model=False)  # Don't overwrite saved models
                
                predictions, _ = strategy.generate_predictions(
                    symbol, period_mid, period_end
                )
                
                performance = strategy.backtest_strategy(
                    symbol, period_mid, period_end, predictions
                )
                
                robustness_results[period_name] = {
                    'period': f"{period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}",
                    'strategy_return': performance.get('strategy_return', 0),
                    'sharpe_ratio': performance.get('sharpe_ratio', 0),
                    'total_trades': len([p for p in predictions if p != 0])
                }
                
                print(f"    [OK] Return: {performance.get('strategy_return', 0):.1%}")
                print(f"    [OK] Sharpe: {performance.get('sharpe_ratio', 0):.3f}")
                
            except Exception as e:
                print(f"    [ERROR] {e}")
                robustness_results[period_name] = None
        
        # Store robustness results
        self.results['robustness'] = robustness_results
        
        # Validate robustness
        valid_periods = [k for k, v in robustness_results.items() if v is not None]
        self.assertGreater(len(valid_periods), 0, "Should have at least one valid robustness test")
    
    def create_performance_charts(self):
        """Create comprehensive performance visualization charts"""
        print(f"\n{'='*60}")
        print(f"CREATING PERFORMANCE VISUALIZATION CHARTS")
        print(f"{'='*60}")
        
        # Filter valid results
        valid_results = {k: v for k, v in self.results.items() if v is not None and k != 'robustness'}
        
        if not valid_results:
            print("[WARNING] No valid results to chart")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Strategy Performance Comparison
        ax1 = plt.subplot(3, 3, 1)
        self._plot_strategy_comparison(valid_results, ax1)
        
        # 2. Returns Distribution
        ax2 = plt.subplot(3, 3, 2)
        self._plot_returns_distribution(valid_results, ax2)
        
        # 3. Risk-Return Scatter
        ax3 = plt.subplot(3, 3, 3)
        self._plot_risk_return_scatter(valid_results, ax3)
        
        # 4. Feature Importance (for first symbol)
        ax4 = plt.subplot(3, 3, 4)
        first_symbol = list(valid_results.keys())[0]
        self._plot_feature_importance(valid_results[first_symbol], ax4)
        
        # 5. Signal Analysis
        ax5 = plt.subplot(3, 3, 5)
        self._plot_signal_analysis(valid_results, ax5)
        
        # 6. Portfolio Value Evolution (if available)
        ax6 = plt.subplot(3, 3, 6)
        self._plot_portfolio_evolution(valid_results, ax6)
        
        # 7. Robustness Analysis
        ax7 = plt.subplot(3, 3, 7)
        if 'robustness' in self.results:
            self._plot_robustness_analysis(self.results['robustness'], ax7)
        
        # 8. Order Sizing Analysis
        ax8 = plt.subplot(3, 3, 8)
        self._plot_order_sizing_analysis(valid_results, ax8)
        
        # 9. Market Context Analysis
        ax9 = plt.subplot(3, 3, 9)
        self._plot_market_context(valid_results, ax9)
        
        plt.tight_layout()
        
        # Save the chart
        chart_path = os.path.join(self.results_dir, 'enhanced_strategy_performance_analysis.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"[CHART] Performance analysis saved to: {chart_path}")
        
        # Close the figure to free memory
        plt.close()
        
        return chart_path
    
    def create_tradingview_style_charts(self):
        """Create TradingView-style interactive charts using existing enhanced chart generator"""
        print(f"\n{'='*60}")
        print(f"CREATING ENHANCED TRADINGVIEW CHARTS")
        print(f"{'='*60}")
        
        try:
            # Get symbols that have been tested
            valid_symbols = [symbol for symbol, result in self.results.items() 
                           if result is not None and symbol != 'robustness']
            
            if not valid_symbols:
                print("[WARNING] No valid test results found. Running quick simulation...")
                # Run a quick simulation on AAPL if no results exist
                valid_symbols = ['AAPL']
                self.test_single_symbol_performance()
            
            print(f"[CHARTS] Generating enhanced charts for symbols: {valid_symbols}")
            
            # Use the existing enhanced chart generator
            chart_results = create_enhanced_tradingview_charts_for_symbols(valid_symbols)
            
            # Extract chart paths
            chart_files = []
            for symbol, chart_info in chart_results.items():
                if 'filename' in chart_info:
                    chart_files.append(chart_info['filename'])
                    print(f"    ✅ {symbol}: {chart_info['filename']}")
                    
                    # Print performance summary
                    if 'performance' in chart_info:
                        perf = chart_info['performance']
                        if isinstance(perf, dict) and 'performance_summary' in perf:
                            summary = perf['performance_summary']
                            print(f"       Performance: {summary.get('strategy_return', 0):.1f}% return")
                            print(f"       Trades: {chart_info.get('trades', 0)}")
            
            # Find dashboard file
            dashboard_files = [f for f in os.listdir('tests/results') if 'dashboard' in f.lower()]
            dashboard_path = f"tests/results/{dashboard_files[0]}" if dashboard_files else None
            
            if dashboard_path:
                print(f"[DASHBOARD] Interactive dashboard: {dashboard_path}")
            
            print(f"\n🎯 CHART GENERATION COMPLETE!")
            print(f"   📊 Individual Charts: {len(chart_files)}")
            print(f"   📈 Dashboard: {'Yes' if dashboard_path else 'No'}")
            print(f"   📁 Location: tests/results/")
            print(f"\n💡 TIP: Open the HTML files in your browser to view interactive charts!")
            
            return dashboard_path if dashboard_path else chart_files[0] if chart_files else None
            
        except Exception as e:
            print(f"[ERROR] Chart generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_simulation_with_charts(self, symbols=['AAPL'], months_back=12):
        """
        Simple method to run simulation and immediately generate charts
        
        Args:
            symbols: List of symbols to test (default: ['AAPL'])
            months_back: Number of months of data to use (default: 12)
        
        Returns:
            dict: Results with chart paths and performance data
        """
        print(f"\n🚀 RUNNING ENHANCED SIMULATION WITH AUTOMATIC CHARTS")
        print(f"{'='*65}")
        print(f"Symbols: {symbols}")
        print(f"Period: {months_back} months back")
        
        results = {}
        
        for symbol in symbols:
            print(f"\n📈 Processing {symbol}...")
            
            try:
                # Initialize fresh strategy for each symbol
                strategy = EnhancedTradingStrategy()
                
                # Run complete simulation
                simulation_result = strategy.run_complete_simulation(symbol, months_back=months_back)
                
                # Store results
                results[symbol] = {
                    'simulation': simulation_result,
                    'strategy': strategy,
                    'performance_summary': simulation_result.get('performance_summary', {}),
                    'trade_details': getattr(strategy, 'latest_trade_details', [])
                }
                
                # Print quick summary
                if 'performance_summary' in simulation_result:
                    perf = simulation_result['performance_summary']
                    print(f"   ✅ Strategy Return: {perf.get('strategy_return', 0):.1f}%")
                    print(f"   ✅ Outperformance: {perf.get('outperformance', 0):.1f}%")
                    print(f"   ✅ Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}")
                
            except Exception as e:
                print(f"   ❌ Error processing {symbol}: {e}")
                results[symbol] = {'error': str(e)}
        
        # Generate charts using existing enhanced chart generator
        print(f"\n📊 GENERATING INTERACTIVE CHARTS...")
        chart_results = create_enhanced_tradingview_charts_for_symbols(symbols)
        
        # Combine results
        final_results = {
            'simulation_results': results,
            'chart_results': chart_results,
            'chart_files': []
        }
        
        # Extract chart file paths and display info
        print(f"\n🎯 RESULTS SUMMARY:")
        print(f"{'='*50}")
        
        for symbol in symbols:
            if symbol in chart_results and 'filename' in chart_results[symbol]:
                chart_file = chart_results[symbol]['filename']
                final_results['chart_files'].append(chart_file)
                
                print(f"📈 {symbol}:")
                print(f"   Chart: {chart_file}")
                
                if symbol in results and 'performance_summary' in results[symbol]:
                    perf = results[symbol]['performance_summary']
                    print(f"   Return: {perf.get('strategy_return', 0):.1f}%")
                    print(f"   Trades: {len(results[symbol].get('trade_details', []))}")
        
        # Check for dashboard
        dashboard_files = [f for f in os.listdir('tests/results') if 'dashboard' in f.lower()]
        if dashboard_files:
            dashboard_path = f"tests/results/{dashboard_files[0]}"
            final_results['dashboard'] = dashboard_path
            print(f"\n📊 Master Dashboard: {dashboard_path}")
        
        print(f"\n💡 TIP: Open the HTML files in your browser to view interactive charts!")
        print(f"📁 Charts saved to: tests/results/")
        
        return final_results
    
    def _create_tradingview_chart(self, symbol: str, stock_data: pd.DataFrame, result: Dict) -> str:
        """Create a single TradingView-style chart for a symbol"""
        
        # Prepare OHLCV data
        df = stock_data.copy()
        df.index = pd.to_datetime(df.index)
        
        # Calculate technical indicators for visualization
        df = self._add_technical_indicators(df)
        
        # Get predictions and orders
        predictions = result.get('predictions', [])
        performance = result.get('performance', {})
        
        # Create subplot structure
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=(
                f'{symbol} - Candlestick Chart with Signals',
                'Volume & Order Sizing',
                'Technical Indicators',
                'Portfolio Value Evolution'
            ),
            row_width=[0.4, 0.2, 0.2, 0.2],
            specs=[[{"secondary_y": True}],
                   [{"secondary_y": True}],
                   [{"secondary_y": True}],
                   [{"secondary_y": False}]]
        )
        
        # 1. Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df.iloc[:, 1] if len(df.columns) > 1 else df.iloc[:, 0],
                high=df.iloc[:, 2] if len(df.columns) > 2 else df.iloc[:, 0],
                low=df.iloc[:, 3] if len(df.columns) > 3 else df.iloc[:, 0],
                close=df.iloc[:, 0],
                name=f'{symbol} Price',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Add configurable moving averages
        config = get_config()
        trading_config = config.config.get('trading', {})
        sma_short = trading_config.get('indicators', {}).get('sma_short', 50)
        sma_long = trading_config.get('indicators', {}).get('sma_long', 200)
        
        if f'SMA_{sma_short}' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[f'SMA_{sma_short}'],
                    mode='lines',
                    name=f'SMA {sma_short}',
                    line=dict(color='orange', width=2)
                ),
                row=1, col=1
            )
        
        if f'SMA_{sma_long}' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[f'SMA_{sma_long}'],
                    mode='lines',
                    name=f'SMA {sma_long}',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # 3. Add trading signals
        self._add_trading_signals(fig, df, predictions, symbol)
        
        # 4. Volume chart
        if len(df.columns) > 4:  # If volume data available
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df.iloc[:, 4] if len(df.columns) > 4 else [1000] * len(df),
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.6,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 5. Technical indicators (RSI, MACD)
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=3, col=1
            )
            
            # RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # 6. Portfolio evolution
        portfolio_values = performance.get('portfolio_values')
        if portfolio_values is not None:
            fig.add_trace(
                go.Scatter(
                    x=portfolio_values.index,
                    y=portfolio_values.values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='green', width=3),
                    fill='tonexty'
                ),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - Enhanced Trading Strategy Analysis',
            xaxis_rangeslider_visible=False,
            height=1000,
            template='plotly_dark',
            font=dict(size=12),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_yaxes(title_text="Portfolio ($)", row=4, col=1)
        
        # Save the chart
        chart_path = os.path.join(self.results_dir, f'{symbol}_tradingview_chart.html')
        fig.write_html(chart_path)
        
        return chart_path
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe with configurable values"""
        df_with_indicators = df.copy()
        
        # Get SMA configuration from config
        config = get_config()
        trading_config = config.config.get('trading', {})
        sma_short = trading_config.get('indicators', {}).get('sma_short', 50)
        sma_long = trading_config.get('indicators', {}).get('sma_long', 200)
        rsi_period = trading_config.get('indicators', {}).get('rsi_period', 14)
        
        print(f"[CHARTS] Using configurable SMA {sma_short}/{sma_long}")
        
        # Configurable Simple Moving Averages
        df_with_indicators[f'SMA_{sma_short}'] = df.iloc[:, 0].rolling(window=sma_short).mean()
        df_with_indicators[f'SMA_{sma_long}'] = df.iloc[:, 0].rolling(window=sma_long).mean()
        
        # RSI with configurable period
        delta = df.iloc[:, 0].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df_with_indicators['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands using short SMA
        sma_short_series = df_with_indicators[f'SMA_{sma_short}']
        std_short = df.iloc[:, 0].rolling(window=sma_short).std()
        df_with_indicators['BB_Upper'] = sma_short_series + (std_short * 2)
        df_with_indicators['BB_Lower'] = sma_short_series - (std_short * 2)
        
        return df_with_indicators
    
    def _add_trading_signals(self, fig, df: pd.DataFrame, predictions: List, symbol: str):
        """Add trading signals (buy/sell markers) to the chart"""
        if not predictions or len(predictions) == 0:
            return
        
        # Convert predictions to signals
        buy_dates = []
        buy_prices = []
        sell_dates = []
        sell_prices = []
        
        for i, signal in enumerate(predictions):
            if i < len(df):
                date = df.index[i]
                price = df.iloc[i, 0]  # Close price
                
                if signal == 1:  # BUY signal
                    buy_dates.append(date)
                    buy_prices.append(price)
                elif signal == -1:  # SELL signal
                    sell_dates.append(date)
                    sell_prices.append(price)
        
        # Add BUY signals
        if buy_dates:
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='lime',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='BUY Signal',
                    hovertemplate=f'{symbol} BUY<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add SELL signals
        if sell_dates:
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    name='SELL Signal',
                    hovertemplate=f'{symbol} SELL<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                ),
                row=1, col=1
            )
    
    def _create_trading_dashboard(self, chart_files: List[str]) -> str:
        """Create a comprehensive trading dashboard HTML file"""
        dashboard_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Trading Strategy - TradingView Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: #1a1a1a;
            color: #ffffff;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50, #3498db);
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            color: #ffffff;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .chart-container {{
            margin: 20px;
            background: #2c2c2c;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        .chart-title {{
            font-size: 1.5em;
            color: #3498db;
            margin-bottom: 15px;
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .chart-grid {{
            display: grid;
            gap: 20px;
            margin-top: 20px;
        }}
        .chart-frame {{
            width: 100%;
            height: 1000px;
            border: none;
            border-radius: 8px;
            background: #ffffff;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px;
            padding: 20px;
            background: #2c2c2c;
            border-radius: 10px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card.positive {{
            background: linear-gradient(135deg, #27ae60, #2ecc71);
        }}
        .stat-card.neutral {{
            background: linear-gradient(135deg, #f39c12, #e67e22);
        }}
        .stat-number {{
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .navigation {{
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
        }}
        .nav-button {{
            background: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            margin: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }}
        .nav-button:hover {{
            background: #2980b9;
        }}
        .features-list {{
            background: #2c2c2c;
            margin: 20px;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #3498db;
        }}
        .features-list h3 {{
            color: #3498db;
            margin-bottom: 15px;
        }}
        .features-list ul {{
            list-style: none;
            padding: 0;
        }}
        .features-list li {{
            padding: 8px 0;
            border-bottom: 1px solid #444;
        }}
        .features-list li:last-child {{
            border-bottom: none;
        }}
    </style>
</head>
<body>
    <div class="navigation">
        <button class="nav-button" onclick="window.open('../performance_dashboard.html')">📊 Performance Dashboard</button>
        <button class="nav-button" onclick="scrollToTop()">⬆️ Top</button>
    </div>

    <div class="header">
        <h1>📈 TradingView-Style Interactive Charts</h1>
        <p>Enhanced Trading Strategy • Candlesticks • Indicators • Signals • Orders</p>
        <p>Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}</p>
    </div>

    <div class="summary-stats">
        <div class="stat-card positive">
            <div class="stat-number">{len([k for k, v in self.results.items() if v is not None and k != 'robustness'])}</div>
            <div class="stat-label">Symbols Analyzed</div>
        </div>
        <div class="stat-card neutral">
            <div class="stat-number">6</div>
            <div class="stat-label">Months Training</div>
        </div>
        <div class="stat-card neutral">
            <div class="stat-number">6</div>
            <div class="stat-label">Months Testing</div>
        </div>
        <div class="stat-card positive">
            <div class="stat-number">17</div>
            <div class="stat-label">ML Features</div>
        </div>
    </div>

    <div class="features-list">
        <h3>🎯 Chart Features</h3>
        <ul>
            <li>📊 <strong>Candlestick Charts:</strong> OHLC price data with professional styling</li>
            <li>📈 <strong>Technical Indicators:</strong> Moving averages (SMA 50/200), RSI, Bollinger Bands</li>
            <li>🎯 <strong>Trading Signals:</strong> ML-generated BUY/SELL signals with hover details</li>
            <li>📊 <strong>Volume Analysis:</strong> Trading volume with order sizing correlation</li>
            <li>💰 <strong>Portfolio Evolution:</strong> Real-time portfolio value tracking</li>
            <li>🖱️ <strong>Interactive Controls:</strong> Zoom, pan, hover for detailed information</li>
        </ul>
    </div>

    <div class="chart-grid">
"""
        
        # Add each symbol's chart
        for i, chart_file in enumerate(chart_files):
            if os.path.exists(chart_file):
                chart_name = os.path.basename(chart_file).replace('_tradingview_chart.html', '').upper()
                dashboard_content += f"""
        <div class="chart-container">
            <div class="chart-title">💹 {chart_name} - Complete Trading Analysis</div>
            <iframe class="chart-frame" src="{os.path.basename(chart_file)}"></iframe>
        </div>
"""
        
        dashboard_content += """
    </div>

    <script>
        function scrollToTop() {
            window.scrollTo({top: 0, behavior: 'smooth'});
        }
        
        // Auto-refresh every 5 minutes for live data (if implemented)
        // setInterval(() => location.reload(), 300000);
    </script>
</body>
</html>
"""
        
        # Save dashboard
        dashboard_path = os.path.join(self.results_dir, 'tradingview_dashboard.html')
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_content)
        
        return dashboard_path
    
    def _plot_strategy_comparison(self, results: Dict, ax):
        """Plot strategy vs buy-hold comparison"""
        symbols = list(results.keys())
        strategy_returns = [results[s]['performance'].get('strategy_return', 0) * 100 for s in symbols]
        buyhold_returns = [results[s]['performance'].get('buy_hold_return', 0) * 100 for s in symbols]
        
        x = np.arange(len(symbols))
        width = 0.35
        
        ax.bar(x - width/2, strategy_returns, width, label='Enhanced Strategy', alpha=0.8)
        ax.bar(x + width/2, buyhold_returns, width, label='Buy & Hold', alpha=0.8)
        
        ax.set_xlabel('Symbols')
        ax.set_ylabel('Returns (%)')
        ax.set_title('Strategy vs Buy & Hold Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(symbols)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_returns_distribution(self, results: Dict, ax):
        """Plot returns distribution"""
        strategy_returns = [results[s]['performance'].get('strategy_return', 0) for s in results.keys()]
        
        ax.hist(strategy_returns, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Strategy Returns')
        ax.set_ylabel('Frequency')
        ax.set_title('Strategy Returns Distribution')
        ax.axvline(np.mean(strategy_returns), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(strategy_returns):.1%}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_risk_return_scatter(self, results: Dict, ax):
        """Plot risk-return scatter"""
        returns = [results[s]['performance'].get('strategy_return', 0) * 100 for s in results.keys()]
        sharpe_ratios = [results[s]['performance'].get('sharpe_ratio', 0) for s in results.keys()]
        symbols = list(results.keys())
        
        scatter = ax.scatter(sharpe_ratios, returns, s=100, alpha=0.7)
        
        for i, symbol in enumerate(symbols):
            ax.annotate(symbol, (sharpe_ratios[i], returns[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Sharpe Ratio')
        ax.set_ylabel('Strategy Returns (%)')
        ax.set_title('Risk-Return Analysis')
        ax.grid(True, alpha=0.3)
    
    def _plot_feature_importance(self, result: Dict, ax):
        """Plot feature importance for a symbol"""
        feature_analysis = result.get('feature_analysis', {})
        if not feature_analysis or 'feature_importance' not in feature_analysis:
            ax.text(0.5, 0.5, 'Feature importance\ndata not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance')
            return
        
        features = feature_analysis['feature_importance'][:10]  # Top 10
        if not features:
            ax.text(0.5, 0.5, 'No feature importance\ndata available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        names = [f[0] for f in features]
        importance = [f[1] for f in features]
        
        ax.barh(names, importance)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top 10 Features - {result["symbol"]}')
        ax.grid(True, alpha=0.3)
    
    def _plot_signal_analysis(self, results: Dict, ax):
        """Plot signal distribution analysis"""
        symbols = list(results.keys())
        buy_pcts = [results[s]['signal_analysis'].get('buy_percentage', 0) for s in symbols]
        
        ax.bar(symbols, buy_pcts, alpha=0.7, color='green')
        ax.set_xlabel('Symbols')
        ax.set_ylabel('Buy Signal Percentage (%)')
        ax.set_title('Buy Signal Distribution')
        ax.grid(True, alpha=0.3)
    
    def _plot_portfolio_evolution(self, results: Dict, ax):
        """Plot portfolio value evolution if available"""
        # Get portfolio values for the first symbol (if available)
        first_symbol = list(results.keys())[0]
        performance = results[first_symbol]['performance']
        
        if 'portfolio_values' in performance:
            portfolio_values = performance['portfolio_values']
            ax.plot(portfolio_values.index, portfolio_values.values, 
                   linewidth=2, label='Portfolio Value')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value ($)')
            ax.set_title(f'Portfolio Evolution - {first_symbol}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Portfolio evolution\ndata not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Portfolio Evolution')
    
    def _plot_robustness_analysis(self, robustness_data: Dict, ax):
        """Plot robustness analysis across different periods"""
        valid_periods = {k: v for k, v in robustness_data.items() if v is not None}
        
        if not valid_periods:
            ax.text(0.5, 0.5, 'Robustness analysis\ndata not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Robustness Analysis')
            return
        
        periods = list(valid_periods.keys())
        returns = [valid_periods[p]['strategy_return'] * 100 for p in periods]
        
        ax.bar(periods, returns, alpha=0.7, color='orange')
        ax.set_xlabel('Time Periods')
        ax.set_ylabel('Strategy Returns (%)')
        ax.set_title('Strategy Robustness Across Periods')
        ax.grid(True, alpha=0.3)
    
    def _plot_order_sizing_analysis(self, results: Dict, ax):
        """Plot order sizing analysis"""
        symbols = list(results.keys())
        avg_sizes = []
        
        for symbol in symbols:
            perf = results[symbol]['performance']
            if 'order_sizing' in perf and perf['order_sizing']:
                avg_sizes.append(perf['order_sizing'].get('avg_order_size', 0))
            else:
                avg_sizes.append(0)
        
        ax.bar(symbols, avg_sizes, alpha=0.7, color='purple')
        ax.set_xlabel('Symbols')
        ax.set_ylabel('Average Order Size (shares)')
        ax.set_title('Order Sizing Analysis')
        ax.grid(True, alpha=0.3)
    
    def _plot_market_context(self, results: Dict, ax):
        """Plot market context analysis"""
        symbols = list(results.keys())
        
        # For this example, we'll use beta values if available
        # Otherwise show a placeholder
        ax.text(0.5, 0.5, 'Market Context Analysis\n(Beta vs SPY/QQQ)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Market Context')
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print(f"\n{'='*60}")
        print(f"GENERATING COMPREHENSIVE TEST REPORT")
        print(f"{'='*60}")
        
        report_path = os.path.join(self.results_dir, 'test_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("Enhanced Trading Strategy - Test Report\n")
            f.write("="*50 + "\n\n")
            
            # Test period information
            f.write(f"Test Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n")
            f.write(f"Training Period: {self.start_date.strftime('%Y-%m-%d')} to {self.train_end.strftime('%Y-%m-%d')}\n")
            f.write(f"Testing Period: {self.train_end.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n\n")
            
            # Results summary
            valid_results = {k: v for k, v in self.results.items() if v is not None and k != 'robustness'}
            
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            for symbol, result in valid_results.items():
                perf = result['performance']
                f.write(f"\n{symbol}:\n")
                f.write(f"  Strategy Return: {perf.get('strategy_return', 0):.1%}\n")
                f.write(f"  Buy-Hold Return: {perf.get('buy_hold_return', 0):.1%}\n")
                f.write(f"  Outperformance: {perf.get('outperformance', 0):.1%}\n")
                f.write(f"  Sharpe Ratio: {perf.get('sharpe_ratio', 0):.3f}\n")
                f.write(f"  Total Trades: {result['signal_analysis'].get('total_predictions', 0)}\n")
            
            # Robustness analysis
            if 'robustness' in self.results:
                f.write("\nROBUSTNESS ANALYSIS\n")
                f.write("-" * 30 + "\n")
                for period, data in self.results['robustness'].items():
                    if data:
                        f.write(f"\n{period}:\n")
                        f.write(f"  Period: {data['period']}\n")
                        f.write(f"  Return: {data['strategy_return']:.1%}\n")
                        f.write(f"  Sharpe: {data['sharpe_ratio']:.3f}\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("Report generated on: " + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        print(f"[REPORT] Test report saved to: {report_path}")
        return report_path
    
    def tearDown(self):
        """Clean up after each test"""
        pass
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        print(f"\n{'='*60}")
        print(f"TEST SUITE COMPLETED")
        print(f"{'='*60}")


def run_comprehensive_test():
    """Run the complete test suite with visualization"""
    print("Starting Enhanced Strategy Comprehensive Test Suite...")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests in order
    suite.addTest(TestEnhancedStrategy('test_single_symbol_performance'))
    suite.addTest(TestEnhancedStrategy('test_multiple_symbols_comparison'))
    suite.addTest(TestEnhancedStrategy('test_strategy_robustness'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Create test instance for visualization
    test_instance = TestEnhancedStrategy()
    test_instance.setUpClass()
    test_instance.setUp()  # Initialize strategy
    test_instance.results = {}  # Initialize results dict
    
    # Run individual tests to populate results
    test_instance.setUp()  # Initialize strategy
    test_instance.test_single_symbol_performance()
    test_instance.test_multiple_symbols_comparison()
    test_instance.test_strategy_robustness()
    
    # Generate visualizations and reports
    chart_path = test_instance.create_performance_charts()
    trading_chart_path = test_instance.create_tradingview_style_charts()
    report_path = test_instance.generate_test_report()
    
    print(f"\n🎉 COMPREHENSIVE TEST COMPLETED!")
    print(f"   ✅ Performance Charts: {chart_path}")
    print(f"   ✅ Trading Charts: {trading_chart_path}")
    print(f"   ✅ Report saved to: {report_path}")
    print(f"   ✅ Tests passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   ❌ Tests failed: {len(result.failures) + len(result.errors)}")
    
    return result, chart_path, trading_chart_path, report_path


def run_simulation_with_charts(symbols=['AAPL'], months_back=12):
    """
    EASY-TO-USE FUNCTION: Run enhanced trading simulation and generate charts
    
    This function combines simulation + chart generation in one simple call.
    Perfect for quick testing and visualization.
    
    Args:
        symbols (list): Symbols to test (default: ['AAPL'])
        months_back (int): Months of historical data (default: 12)
    
    Returns:
        dict: Complete results with charts, performance data, and file paths
        
    Example:
        results = run_simulation_with_charts(['AAPL', 'TSLA'], months_back=6)
    """
    print(f"ENHANCED TRADING SIMULATION + CHARTS")
    print(f"{'='*60}")
    
    # Create test instance
    test_instance = TestEnhancedStrategy()
    test_instance.setUpClass()
    test_instance.setUp()
    
    # Run simulation with charts
    results = test_instance.run_simulation_with_charts(symbols, months_back)
    
    return results


if __name__ == "__main__":
    # Option 1: Run comprehensive test suite (original functionality)
    if len(sys.argv) > 1 and sys.argv[1] == '--comprehensive':
        run_comprehensive_test()
    
    # Option 2: Quick simulation with charts (new default)
    else:
        print("Running Quick Simulation with Charts...")
        print("TIP: Use --comprehensive flag for full test suite")
        print()
        
        # Run quick simulation for popular symbols
        results = run_simulation_with_charts(['AAPL', 'TSLA'], months_back=6)
        
        print(f"\nSIMULATION COMPLETE!")
        print(f"Chart files generated: {len(results.get('chart_files', []))}")
        if 'dashboard' in results:
            print(f"Open dashboard: {results['dashboard']}")
        else:
            print(f"Open charts in: tests/results/")