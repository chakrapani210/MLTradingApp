"""
Enhanced Market Indicators with Market Context Features
Incorporates SPY and QQQ market indices for improved signal generation
"""

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from indicators import compute_indicators


def get_market_data(symbols, start_date, end_date, include_ohlc=False):
    """
    Get market data for multiple symbols including market indices
    
    Args:
        symbols (list): List of symbols including target stock and market indices
        start_date (datetime): Start date for data
        end_date (datetime): End date for data
        include_ohlc (bool): Whether to include full OHLC data for candlestick charts
    
    Returns:
        dict: Dictionary with stock data for each symbol
    """
    data = {}
    for symbol in symbols:
        stock_df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if include_ohlc and not stock_df.empty:
            # Return full OHLC data for candlestick charts
            if isinstance(stock_df.columns, pd.MultiIndex):
                # Multi-level columns from yfinance
                ohlc_data = pd.DataFrame({
                    'Close': stock_df['Close'].iloc[:, 0] if len(stock_df['Close'].columns) > 0 else stock_df['Close'],
                    'Open': stock_df['Open'].iloc[:, 0] if len(stock_df['Open'].columns) > 0 else stock_df['Open'],
                    'High': stock_df['High'].iloc[:, 0] if len(stock_df['High'].columns) > 0 else stock_df['High'],
                    'Low': stock_df['Low'].iloc[:, 0] if len(stock_df['Low'].columns) > 0 else stock_df['Low'],
                    'Volume': stock_df['Volume'].iloc[:, 0] if len(stock_df['Volume'].columns) > 0 else stock_df['Volume']
                })
            else:
                # Single-level columns
                ohlc_data = stock_df[['Close', 'Open', 'High', 'Low', 'Volume']].copy()
            
            data[symbol] = ohlc_data
        else:
            # Return only close price for legacy compatibility
            if isinstance(stock_df.columns, pd.MultiIndex):
                adj_close = stock_df[('Close', symbol)] if ('Close', symbol) in stock_df.columns else stock_df['Close'].iloc[:, 0]
            else:
                adj_close = stock_df['Close']
            data[symbol] = adj_close.to_frame()
            data[symbol].columns = [symbol]
    
    return data


def compute_market_context_indicators(target_data, spy_data, qqq_data, window=5):
    """
    Compute market context indicators using SPY and QQQ
    
    Args:
        target_data (pd.DataFrame): Target stock price data
        spy_data (pd.DataFrame): SPY price data  
        qqq_data (pd.DataFrame): QQQ price data
        window (int): Window size for calculations
        
    Returns:
        pd.DataFrame: Market context indicators
    """
    target_symbol = target_data.columns[0]
    
    # Align all data to the same index
    combined_data = pd.concat([target_data, spy_data, qqq_data], axis=1, join='inner')
    combined_data = combined_data.ffill()  # Forward fill missing values
    
    target_col = combined_data.columns[0]
    spy_col = 'SPY'
    qqq_col = 'QQQ'
    
    indicators = pd.DataFrame(index=combined_data.index)
    
    # 1. MARKET TREND INDICATORS
    
    # SPY trend (is market going up or down?)
    spy_sma = combined_data[spy_col].rolling(window=window).mean()
    indicators['SPY_trend'] = (combined_data[spy_col] / spy_sma - 1)  # Above/below SMA
    
    # QQQ trend (is tech sector going up or down?)
    qqq_sma = combined_data[qqq_col].rolling(window=window).mean()
    indicators['QQQ_trend'] = (combined_data[qqq_col] / qqq_sma - 1)  # Above/below SMA
    
    # Market momentum
    indicators['SPY_momentum'] = (combined_data[spy_col] / combined_data[spy_col].shift(window) - 1)
    indicators['QQQ_momentum'] = (combined_data[qqq_col] / combined_data[qqq_col].shift(window) - 1)
    
    # 2. RELATIVE STRENGTH INDICATORS
    
    # Stock vs SPY relative strength
    target_returns = combined_data[target_col].pct_change(window)
    spy_returns = combined_data[spy_col].pct_change(window)
    indicators['stock_vs_SPY_strength'] = target_returns - spy_returns
    
    # Stock vs QQQ relative strength  
    qqq_returns = combined_data[qqq_col].pct_change(window)
    indicators['stock_vs_QQQ_strength'] = target_returns - qqq_returns
    
    # QQQ vs SPY (tech vs broad market)
    indicators['QQQ_vs_SPY_strength'] = qqq_returns - spy_returns
    
    # 3. BETA RELATIONSHIPS
    
    # Rolling beta vs SPY
    cov_spy = target_returns.rolling(window=window*4).cov(spy_returns)  # Longer window for beta
    var_spy = spy_returns.rolling(window=window*4).var()
    beta_spy = cov_spy / var_spy
    indicators['beta_SPY'] = beta_spy.fillna(1.0)  # Default beta = 1
    
    # Rolling beta vs QQQ  
    cov_qqq = target_returns.rolling(window=window*4).cov(qqq_returns)
    var_qqq = qqq_returns.rolling(window=window*4).var()
    beta_qqq = cov_qqq / var_qqq
    indicators['beta_QQQ'] = beta_qqq.fillna(1.0)  # Default beta = 1
    
    # 4. MARKET VOLATILITY CONTEXT
    
    # VIX-like indicator using SPY volatility
    spy_volatility = combined_data[spy_col].rolling(window=window).std()
    spy_vol_ma = spy_volatility.rolling(window=window*2).mean()
    indicators['market_volatility_regime'] = spy_volatility / spy_vol_ma - 1  # Above/below average vol
    
    # 5. SECTOR ROTATION SIGNALS
    
    # When QQQ outperforms SPY, tech is in favor
    indicators['tech_outperformance'] = (combined_data[qqq_col].pct_change(window) - 
                                       combined_data[spy_col].pct_change(window))
    
    # 6. MARKET BREADTH (using SPY as proxy)
    
    # Market strength indicator
    spy_high_20 = combined_data[spy_col].rolling(window=window*4).max()
    spy_low_20 = combined_data[spy_col].rolling(window=window*4).min()
    indicators['market_breadth'] = ((combined_data[spy_col] - spy_low_20) / 
                                   (spy_high_20 - spy_low_20))
    
    # Fill any remaining NaN values
    indicators.fillna(0, inplace=True)
    
    return indicators


def compute_enhanced_indicators(target_data, spy_data, qqq_data, window=5):
    """
    Compute both original technical indicators and market context indicators
    
    Args:
        target_data (pd.DataFrame): Target stock data
        spy_data (pd.DataFrame): SPY data
        qqq_data (pd.DataFrame): QQQ data  
        window (int): Window size
        
    Returns:
        pd.DataFrame: Combined technical and market indicators
    """
    target_symbol = target_data.columns[0]
    
    # Get original technical indicators
    tech_indicators = compute_indicators(target_data, target_symbol, window=window)
    
    # Get market context indicators
    market_indicators = compute_market_context_indicators(target_data, spy_data, qqq_data, window)
    
    # Combine both sets of indicators
    combined_indicators = pd.concat([tech_indicators, market_indicators], axis=1, join='inner')
    combined_indicators.fillna(0, inplace=True)
    
    return combined_indicators


def analyze_market_correlation(target_symbol, start_date, end_date, window=5):
    """
    Analyze correlation between target stock and market indices
    
    Args:
        target_symbol (str): Target stock symbol
        start_date (datetime): Analysis start date
        end_date (datetime): Analysis end date
        window (int): Window size for analysis
    """
    print(f"\n=== MARKET CORRELATION ANALYSIS FOR {target_symbol} ===")
    
    # Get data for target stock and market indices
    symbols = [target_symbol, 'SPY', 'QQQ']
    data = get_market_data(symbols, start_date, end_date)
    
    target_data = data[target_symbol]
    spy_data = data['SPY']
    qqq_data = data['QQQ']
    
    # Calculate returns
    target_returns = target_data.pct_change().dropna()
    spy_returns = spy_data.pct_change().dropna()
    qqq_returns = qqq_data.pct_change().dropna()
    
    # Align all returns
    returns_df = pd.concat([target_returns, spy_returns, qqq_returns], axis=1, join='inner')
    returns_df.columns = [target_symbol, 'SPY', 'QQQ']
    
    # Calculate correlations
    correlations = returns_df.corr()
    
    print(f"\nDaily Return Correlations:")
    print(f"  {target_symbol} vs SPY: {correlations.loc[target_symbol, 'SPY']:.3f}")
    print(f"  {target_symbol} vs QQQ: {correlations.loc[target_symbol, 'QQQ']:.3f}")
    print(f"  SPY vs QQQ: {correlations.loc['SPY', 'QQQ']:.3f}")
    
    # Calculate rolling correlations
    rolling_corr_spy = returns_df[target_symbol].rolling(window=30).corr(returns_df['SPY'])
    rolling_corr_qqq = returns_df[target_symbol].rolling(window=30).corr(returns_df['QQQ'])
    
    print(f"\nRolling 30-Day Correlations (Recent):")
    print(f"  {target_symbol} vs SPY: {rolling_corr_spy.iloc[-1]:.3f}")
    print(f"  {target_symbol} vs QQQ: {rolling_corr_qqq.iloc[-1]:.3f}")
    
    # Calculate beta
    cov_spy = returns_df[target_symbol].cov(returns_df['SPY'])
    var_spy = returns_df['SPY'].var()
    beta_spy = cov_spy / var_spy
    
    cov_qqq = returns_df[target_symbol].cov(returns_df['QQQ'])
    var_qqq = returns_df['QQQ'].var()
    beta_qqq = cov_qqq / var_qqq
    
    print(f"\nBeta Calculations:")
    print(f"  {target_symbol} Beta vs SPY: {beta_spy:.3f}")
    print(f"  {target_symbol} Beta vs QQQ: {beta_qqq:.3f}")
    
    if beta_spy > 1.3:
        print(f"  -> {target_symbol} is HIGH BETA vs market (amplifies market moves)")
    elif beta_spy < 0.7:
        print(f"  -> {target_symbol} is LOW BETA vs market (less sensitive to market)")
    else:
        print(f"  -> {target_symbol} has MODERATE BETA vs market")
    
    # Get market indicators
    market_indicators = compute_market_context_indicators(target_data, spy_data, qqq_data, window)
    
    print(f"\nLatest Market Context Indicators:")
    latest = market_indicators.iloc[-1]
    for col, val in latest.items():
        print(f"  {col}: {val:.4f}")
    
    return correlations, beta_spy, beta_qqq, market_indicators


def get_enhanced_features(target_symbol, start_date, end_date, window=5, train=True):
    """
    Get enhanced feature set including market context for ML training
    
    Args:
        target_symbol (str): Target stock symbol
        start_date (datetime): Start date
        end_date (datetime): End date  
        window (int): Window size
        train (bool): Whether this is for training (excludes last 3 days)
        
    Returns:
        numpy.array: Enhanced feature matrix
    """
    # Get data for target stock and market indices
    symbols = [target_symbol, 'SPY', 'QQQ']
    data = get_market_data(symbols, start_date, end_date)
    
    target_data = data[target_symbol]
    spy_data = data['SPY']  
    qqq_data = data['QQQ']
    
    # Get enhanced indicators
    enhanced_indicators = compute_enhanced_indicators(target_data, spy_data, qqq_data, window)
    
    # Prepare for ML
    if train:
        enhanced_indicators = enhanced_indicators[:-3]
    
    # Drop highly correlated indicators (similar to original approach)
    if 'Upper_BB' in enhanced_indicators.columns:
        enhanced_indicators.drop('Upper_BB', axis=1, inplace=True)
    if 'Down_BB' in enhanced_indicators.columns:
        enhanced_indicators.drop('Down_BB', axis=1, inplace=True)
    
    return enhanced_indicators.values


if __name__ == "__main__":
    # Example usage
    target_symbol = "TSLA"
    end_date = dt.datetime(2025, 12, 31)
    start_date = end_date - dt.timedelta(days=365)  # 1 year of data
    
    # Analyze market correlation
    correlations, beta_spy, beta_qqq, market_indicators = analyze_market_correlation(
        target_symbol, start_date, end_date
    )
    
    # Get enhanced features
    features = get_enhanced_features(target_symbol, start_date, end_date, window=5)
    print(f"\nEnhanced feature matrix shape: {features.shape}")
    print(f"Number of features: {features.shape[1]} (vs ~5 in original system)")