"""
Dashboard Launcher
Quick launcher for all dashboard views
"""

import os
import webbrowser
import time

def launch_dashboards():
    """Launch all dashboard views in the browser"""
    
    results_dir = "tests/results"
    dashboards = [
        ("Enhanced Master Dashboard", "enhanced_trading_dashboard.html"),
        ("NVDA Enhanced Chart", "nvda_enhanced_chart.html"),
        ("TSLA Enhanced Chart", "tsla_enhanced_chart.html"),
        ("AAPL Enhanced Chart", "aapl_enhanced_chart.html")
    ]
    
    print("🚀 Launching Enhanced Trading Strategy Dashboards...")
    print("="*60)
    
    for name, filename in dashboards:
        filepath = os.path.join(results_dir, filename)
        
        if os.path.exists(filepath):
            abs_path = os.path.abspath(filepath)
            url = f"file:///{abs_path.replace(os.sep, '/')}"
            
            print(f"📊 Opening {name}...")
            webbrowser.open(url)
            time.sleep(1)  # Small delay between opens
        else:
            print(f"❌ {name} not found: {filepath}")
    
    print("\n✅ Dashboard launch completed!")
    print("🔍 Check your browser for the opened tabs")
    print("\n📋 Available Enhanced Dashboards:")
    print("  1. Master Dashboard - Performance comparison across all symbols")
    print("  2. NVDA Chart - Detailed NVDA analysis with candlesticks and order details")
    print("  3. TSLA Chart - Detailed TSLA analysis with candlesticks and order details")
    print("  4. AAPL Chart - Detailed AAPL analysis with candlesticks and order details")

if __name__ == "__main__":
    launch_dashboards()