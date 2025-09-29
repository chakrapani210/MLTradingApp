#!/usr/bin/env python3
"""
Configuration profiles for signal generators to handle different data availability scenarios
"""

# Conservative profiles for limited data scenarios
SIGNAL_GENERATOR_PROFILES = {
    'default': {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'sma_short': 20,
        'sma_long': 50,
        'ema_periods': [12, 26],
        'volume_sma': 20,
        'volume_obv': 20,
    },
    'limited_data': {
        'rsi_period': 10,          # Reduced from 14
        'macd_fast': 8,            # Reduced from 12
        'macd_slow': 18,           # Reduced from 26
        'macd_signal': 6,          # Reduced from 9
        'bb_period': 15,           # Reduced from 20
        'sma_short': 10,           # Reduced from 20
        'sma_long': 30,            # Reduced from 50
        'ema_periods': [8, 18],    # Reduced from [12, 26]
        'volume_sma': 15,          # Reduced from 20
        'volume_obv': 15,          # Reduced from 20
    },
    'minimal_data': {
        'rsi_period': 7,           # Minimal RSI
        'macd_fast': 5,            # Minimal MACD
        'macd_slow': 12,           # Minimal MACD
        'macd_signal': 5,          # Minimal signal
        'bb_period': 10,           # Minimal BB
        'sma_short': 5,            # Minimal SMA
        'sma_long': 20,            # Minimal long SMA
        'ema_periods': [5, 12],    # Minimal EMA
        'volume_sma': 10,          # Minimal volume
        'volume_obv': 10,          # Minimal OBV
    }
}

def get_signal_config_for_data_length(data_length: int) -> dict:
    """
    Get appropriate signal generator configuration based on available data length
    
    Args:
        data_length: Number of data points available
        
    Returns:
        Dictionary with signal generator configuration
    """
    if data_length >= 60:
        return SIGNAL_GENERATOR_PROFILES['default']
    elif data_length >= 40:
        return SIGNAL_GENERATOR_PROFILES['limited_data']
    else:
        return SIGNAL_GENERATOR_PROFILES['minimal_data']

def get_profile_requirements(profile_name: str = 'default') -> dict:
    """
    Get the data requirements for each signal generator in a profile
    
    Args:
        profile_name: Name of the profile ('default', 'limited_data', 'minimal_data')
        
    Returns:
        Dictionary with minimum data requirements for each generator
    """
    profile = SIGNAL_GENERATOR_PROFILES.get(profile_name, SIGNAL_GENERATOR_PROFILES['default'])
    
    return {
        'RSI': profile['rsi_period'] + 2,
        'MACD': profile['macd_slow'] + profile['macd_signal'] + 2,
        'Bollinger_Bands': profile['bb_period'] + 2,
        'SMA_Crossover': profile['sma_long'] + 5,
        'EMA': max(profile['ema_periods']) + 5,
        'Volume_Analysis': profile['volume_sma'] + profile['volume_obv'] + 5,
    }