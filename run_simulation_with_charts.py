#!/usr/bin/env python3
"""
Simple script to run enhanced trading strategy simulation and generate charts
Reuses existing chart generation code for professional visualization
"""

import os
import sys
import datetime as dt

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def run_simulation_with_charts(symbols=['AAPL'], months_back=6):
    """
    Run enhanced trading strategy simulation and automatically generate interactive charts
    
    Args:
        symbols (list): Stock symbols to analyze (default: ['AAPL'])
        months_back (int): Number of months of historical data (default: 6)
    
    Returns:
        dict: Results including performance metrics and chart file paths
    """
    print("ENHANCED TRADING STRATEGY SIMULATION + CHARTS")
    print("=" * 60)
    print(f"Symbols: {symbols}")
    print(f"Period: {months_back} months back")
    print()
    
    try:
        # Import required modules
        from enhanced_strategy import EnhancedTradingStrategy
        from create_enhanced_tradingview_charts import create_enhanced_tradingview_charts_for_symbols
        
        results = {}
        
        # Step 1: Run enhanced trading simulations
        print("STEP 1: RUNNING TRADING SIMULATIONS")
        print("-" * 40)
        
        for symbol in symbols:
            print(f"Processing {symbol}...")
            
            try:
                # Initialize strategy
                strategy = EnhancedTradingStrategy()
                
                # Run complete simulation
                simulation_result = strategy.run_complete_simulation(symbol, months_back=months_back)
                
                # Store results
                results[symbol] = {
                    'simulation': simulation_result,
                    'trades': getattr(strategy, 'latest_trade_details', [])
                }
                
                # Print summary
                if 'performance_summary' in simulation_result:
                    perf = simulation_result['performance_summary']
                    print(f"  Strategy Return: {perf.get('strategy_return', 0):.1f}%")
                    print(f"  Outperformance: {perf.get('outperformance', 0):.1f}%")
                    print(f"  Trades: {len(results[symbol]['trades'])}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                results[symbol] = {'error': str(e)}
        
        print()
        
        # Step 2: Generate interactive charts
        print("STEP 2: GENERATING INTERACTIVE CHARTS")
        print("-" * 40)
        
        chart_results = create_enhanced_tradingview_charts_for_symbols(symbols)
        
        # Step 3: Display results
        print()
        print("RESULTS SUMMARY")
        print("=" * 40)
        
        chart_files = []
        dashboard_file = None
        
        for symbol in symbols:
            if symbol in chart_results and 'filename' in chart_results[symbol]:
                chart_file = chart_results[symbol]['filename']
                chart_files.append(chart_file)
                
                # Check if file exists and get size
                if os.path.exists(chart_file):
                    file_size = os.path.getsize(chart_file) / 1024  # KB
                    print(f"{symbol}:")
                    print(f"  Chart: {chart_file} ({file_size:.1f} KB)")
                    
                    # Performance from simulation
                    if symbol in results and 'simulation' in results[symbol]:
                        sim_data = results[symbol]['simulation']
                        if 'performance_summary' in sim_data:
                            perf = sim_data['performance_summary']
                            print(f"  Return: {perf.get('strategy_return', 0):.1f}%")
                            print(f"  Sharpe: {perf.get('sharpe_ratio', 0):.3f}")
                    print()
        
        # Check for dashboard
        results_dir = 'tests/results'
        if os.path.exists(results_dir):
            dashboard_files = [f for f in os.listdir(results_dir) if 'dashboard' in f.lower()]
            if dashboard_files:
                dashboard_file = os.path.join(results_dir, dashboard_files[0])
                file_size = os.path.getsize(dashboard_file) / 1024
                print(f"Dashboard: {dashboard_file} ({file_size:.1f} KB)")
        
        print()
        print("SUCCESS! Simulation and charts completed.")
        print("TIP: Open the HTML files in your browser to view interactive charts")
        print()
        
        return {
            'simulation_results': results,
            'chart_results': chart_results,
            'chart_files': chart_files,
            'dashboard': dashboard_file
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run enhanced trading strategy with charts')
    parser.add_argument('--symbols', nargs='+', default=['AAPL'], 
                        help='Stock symbols to analyze (default: AAPL)')
    parser.add_argument('--months', type=int, default=6,
                        help='Number of months of historical data (default: 6)')
    
    args = parser.parse_args()
    
    # Run simulation with charts
    results = run_simulation_with_charts(args.symbols, args.months)
    
    if results:
        print("Files generated:")
        for chart_file in results.get('chart_files', []):
            print(f"  {chart_file}")
        if results.get('dashboard'):
            print(f"  {results['dashboard']}")


if __name__ == "__main__":
    main()