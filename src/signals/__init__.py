"""
Signals Module
Concrete implementations of signal generators
"""

from .technical import RSISignalGenerator, MACDSignalGenerator, BollingerBandsSignalGenerator
from .pattern import GoldenCrossSignalGenerator, CandlestickPatternSignalGenerator
from .ml_signals import MLSignalGenerator
from .composite import CompositeSignalGenerator

__all__ = [
    'RSISignalGenerator',
    'MACDSignalGenerator', 
    'BollingerBandsSignalGenerator',
    'GoldenCrossSignalGenerator',
    'CandlestickPatternSignalGenerator',
    'MLSignalGenerator',
    'CompositeSignalGenerator'
]