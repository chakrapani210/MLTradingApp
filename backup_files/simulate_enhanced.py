"""
Enhanced simulation with market context indicators
Demonstrates the power of using SPY and QQQ for better signal generation
"""

import datetime as dt
import pandas as pd 
import numpy as np
import math
from marketsimcode import *
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from config_manager import get_config
from market_indicators import get_enhanced_features, get_market_data


def get_labels(prices, market_impact=None):
    """Generate trading labels based on future returns"""
    config = get_config()
    if market_impact is None:
        market_impact = config.get_market_impact()
    
    labels = np.zeros(prices.shape[0]-3)
    for i in range(prices.shape[0]-3):
        ret = (prices.values[i+3] - prices.values[i])
        if ret / prices.values[i] < (-1*market_impact - 0.02):
            labels[i] = -1 # SELL
        elif ret / prices.values[i] > (market_impact + 0.02):
            labels[i] = 1 # BUY
    return labels


def create_trades_df(prices, symbol, Y_test):
    """Create trades dataframe from ML predictions"""
    trades = pd.DataFrame(index=prices.index) 
    trades[symbol] = 0
    shares = 0
    config = get_config()
    shares_per_trade = config.get_shares_per_trade()
    
    for i in range(prices.shape[0]-3):
        if shares == 0 and Y_test[i] == 1:
            trades.iloc[i,0] = shares_per_trade
            shares = shares_per_trade
        elif shares > 0 and Y_test[i] == -1:
            trades.iloc[i,0] = -shares_per_trade
            shares = 0
    return trades


def compare_strategies():
    """Compare original vs market-enhanced strategies"""
    print("=== COMPARING ORIGINAL VS MARKET-ENHANCED STRATEGIES ===\n")
    
    config = get_config()
    symbol = config.get_default_symbol()
    
    # Get analysis dates
    analysis_type = 'tesla_analysis'
    sd_train, ed_train, sd_test, ed_test = config.get_analysis_dates(analysis_type)
    
    print(f"Analyzing {symbol}")
    print(f"Training: {sd_train.strftime('%Y-%m-%d')} to {ed_train.strftime('%Y-%m-%d')}")
    print(f"Testing: {sd_test.strftime('%Y-%m-%d')} to {ed_test.strftime('%Y-%m-%d')}")
    
    # === ORIGINAL STRATEGY ===
    print(f"\n1. ORIGINAL STRATEGY (Technical Indicators Only)")
    print("-" * 50)
    
    # Get data the original way
    from simulate_results import get_stock_data, get_indicator_data
    
    stock_df_train = get_stock_data(symbol, sd_train, ed_train)
    stock_df_train.columns = [symbol]
    window_size = config.get_indicator_window()
    indicators_train = get_indicator_data(stock_df_train, symbol, window=window_size, train=True)
    labels_train = get_labels(stock_df_train)
    
    stock_df_test = get_stock_data(symbol, sd_test, ed_test)
    stock_df_test.columns = [symbol]
    indicators_test = get_indicator_data(stock_df_test, symbol, window=window_size, train=False)
    
    # Train original model
    ml_config = config.get_ml_config()
    clf_original = DecisionTreeClassifier(
        max_depth=ml_config['max_depth'],
        random_state=ml_config['random_state']
    )
    clf_original.fit(indicators_train, labels_train)
    labels_test_original = clf_original.predict(indicators_test[:-3])
    
    # Calculate original performance
    trades_df_original = create_trades_df(stock_df_test, symbol, labels_test_original)
    orders_original = trades2orders(trades_df_original, symbol)
    
    portfolio_config = config.get_portfolio_config()
    strategy_pval_original = compute_portvals(
        stock_df_test, orders_original,
        start_val=portfolio_config['starting_value'],
        commission=portfolio_config['commission'],
        impact=portfolio_config['impact']
    )
    
    crb_orig, adrb_orig, sddrb_orig, srb_orig = compute_stats(strategy_pval_original)
    
    print(f"Original Strategy Results:")
    print(f"  Features used: {indicators_train.shape[1]}")
    print(f"  Cumulative Return: {crb_orig:.4f} ({crb_orig*100:.1f}%)")
    print(f"  Sharpe Ratio: {srb_orig:.4f}")
    print(f"  Daily Volatility: {sddrb_orig:.4f}")
    
    # === ENHANCED STRATEGY ===
    print(f"\n2. ENHANCED STRATEGY (Technical + Market Context)")
    print("-" * 50)
    
    # Get enhanced features with market context
    enhanced_features_train = get_enhanced_features(
        symbol, sd_train, ed_train, window=window_size, train=True
    )
    enhanced_features_test = get_enhanced_features(
        symbol, sd_test, ed_test, window=window_size, train=False
    )
    
    # Train enhanced model  
    clf_enhanced = DecisionTreeClassifier(
        max_depth=ml_config['max_depth'],
        random_state=ml_config['random_state']
    )
    clf_enhanced.fit(enhanced_features_train, labels_train)
    labels_test_enhanced = clf_enhanced.predict(enhanced_features_test[:-3])
    
    # Calculate enhanced performance
    trades_df_enhanced = create_trades_df(stock_df_test, symbol, labels_test_enhanced)
    orders_enhanced = trades2orders(trades_df_enhanced, symbol)
    
    strategy_pval_enhanced = compute_portvals(
        stock_df_test, orders_enhanced,
        start_val=portfolio_config['starting_value'],
        commission=portfolio_config['commission'],
        impact=portfolio_config['impact']
    )
    
    crb_enh, adrb_enh, sddrb_enh, srb_enh = compute_stats(strategy_pval_enhanced)
    
    print(f"Enhanced Strategy Results:")
    print(f"  Features used: {enhanced_features_train.shape[1]}")
    print(f"  Cumulative Return: {crb_enh:.4f} ({crb_enh*100:.1f}%)")
    print(f"  Sharpe Ratio: {srb_enh:.4f}")
    print(f"  Daily Volatility: {sddrb_enh:.4f}")
    
    # === COMPARISON ===
    print(f"\n3. STRATEGY COMPARISON")
    print("=" * 50)
    
    improvement_return = ((crb_enh - crb_orig) / abs(crb_orig)) * 100
    improvement_sharpe = ((srb_enh - srb_orig) / abs(srb_orig)) * 100
    
    print(f"Performance Improvement:")
    print(f"  Return: {improvement_return:+.1f}%")
    print(f"  Sharpe Ratio: {improvement_sharpe:+.1f}%")
    print(f"  Feature Count: {enhanced_features_train.shape[1]} vs {indicators_train.shape[1]} (+{enhanced_features_train.shape[1] - indicators_train.shape[1]})")
    
    if crb_enh > crb_orig:
        print(f"  ✅ Enhanced strategy OUTPERFORMED original by {abs(improvement_return):.1f}%")
    else:
        print(f"  ❌ Enhanced strategy underperformed by {abs(improvement_return):.1f}%")
    
    if srb_enh > srb_orig:
        print(f"  ✅ Enhanced strategy has BETTER risk-adjusted returns")
    else:
        print(f"  ❌ Enhanced strategy has worse risk-adjusted returns")
    
    # Feature Importance Analysis
    print(f"\n4. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 50)
    
    # Get feature names for enhanced model
    from market_indicators import compute_enhanced_indicators
    data = get_market_data([symbol, 'SPY', 'QQQ'], sd_train, ed_train)
    sample_features = compute_enhanced_indicators(
        data[symbol], data['SPY'], data['QQQ'], window=window_size
    )
    
    # Drop the same features as in training
    if 'Upper_BB' in sample_features.columns:
        sample_features.drop('Upper_BB', axis=1, inplace=True)
    if 'Down_BB' in sample_features.columns:
        sample_features.drop('Down_BB', axis=1, inplace=True)
    
    feature_names = sample_features.columns
    importances = clf_enhanced.feature_importances_
    
    # Sort features by importance
    feature_importance_pairs = list(zip(feature_names, importances))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print("Top Features (by importance):")
    for i, (name, importance) in enumerate(feature_importance_pairs[:10]):
        marker = "🏆" if i < 3 else "📊" if i < 6 else "📈"
        feature_type = "Market" if any(x in name for x in ['SPY', 'QQQ', 'beta', 'market', 'tech']) else "Technical"
        print(f"  {marker} {name}: {importance:.4f} ({feature_type})")
    
    # Count market vs technical features in top 10
    top_features = [pair[0] for pair in feature_importance_pairs[:10]]
    market_features_top = sum(1 for f in top_features if any(x in f for x in ['SPY', 'QQQ', 'beta', 'market', 'tech']))
    technical_features_top = 10 - market_features_top
    
    print(f"\nTop 10 Features Breakdown:")
    print(f"  Market Context Features: {market_features_top}/10")
    print(f"  Technical Features: {technical_features_top}/10")
    
    if market_features_top >= 5:
        print(f"  ✅ Market context features dominate - market indices are highly predictive!")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Normalize both strategies to start at 1.0
    strategy_pval_original_norm = strategy_pval_original / strategy_pval_original.iloc[0]
    strategy_pval_enhanced_norm = strategy_pval_enhanced / strategy_pval_enhanced.iloc[0]
    
    plt.plot(strategy_pval_original_norm, 'b-', label=f'Original Strategy ({crb_orig*100:.1f}%)', linewidth=2)
    plt.plot(strategy_pval_enhanced_norm, 'r-', label=f'Enhanced Strategy ({crb_enh*100:.1f}%)', linewidth=2)
    
    plt.title(f'{symbol} Strategy Comparison: Technical vs Technical+Market Context')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add performance text box
    textstr = f'Enhanced vs Original:\nReturn: {improvement_return:+.1f}%\nSharpe: {improvement_sharpe:+.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.show()
    
    return {
        'original': {
            'return': crb_orig,
            'sharpe': srb_orig,
            'volatility': sddrb_orig,
            'features': indicators_train.shape[1]
        },
        'enhanced': {
            'return': crb_enh,
            'sharpe': srb_enh,
            'volatility': sddrb_enh,
            'features': enhanced_features_train.shape[1]
        },
        'improvement': {
            'return_pct': improvement_return,
            'sharpe_pct': improvement_sharpe
        }
    }


if __name__ == "__main__":
    results = compare_strategies()