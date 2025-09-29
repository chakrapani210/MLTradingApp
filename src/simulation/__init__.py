"""
Simulation Package

This package contains comprehensive simulation capabilities for the trading system.
Simulations encompass end-to-end system testing including strategy creation,
ML training, market analysis, backtesting, and performance evaluation.
"""

from .enhanced_simulation import EnhancedTradingSimulator
from .simulation_runner import SimulationRunner

__all__ = [
    'EnhancedTradingSimulator',
    'SimulationRunner'
]