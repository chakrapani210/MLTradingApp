"""
Signals Module
Concrete implementations of signal generators
"""

from .technical import RSISignalGenerator, MACDSignalGenerator, BollingerBandsSignalGenerator
# from .pattern import GoldenCrossSignalGenerator, CandlestickPatternSignalGenerator  # TODO: Implement
# from .ml_signals import MLSignalGenerator  # TODO: Implement
# from .composite import CompositeSignalGenerator  # TODO: Implement

__all__ = [
    'RSISignalGenerator',
    'MACDSignalGenerator', 
    'BollingerBandsSignalGenerator',
    'GoldenCrossSignalGenerator',
    'CandlestickPatternSignalGenerator',
    'MLSignalGenerator',
    'CompositeSignalGenerator'
]