"""
Risk Manager Interface
Abstract base class for risk management in the trading system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from .trading_strategy import Order, Position


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """
    Risk assessment metrics
    """
    var_1d: float  # 1-day Value at Risk
    var_1d_pct: float  # 1-day VaR as percentage
    expected_shortfall: float  # Expected Shortfall (CVaR)
    maximum_drawdown: float  # Maximum drawdown
    volatility: float  # Portfolio volatility
    beta: float  # Portfolio beta vs benchmark
    concentration_risk: float  # Position concentration metric
    leverage: float  # Portfolio leverage
    risk_level: RiskLevel  # Overall risk assessment
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'var_1d': self.var_1d,
            'var_1d_pct': self.var_1d_pct,
            'expected_shortfall': self.expected_shortfall,
            'maximum_drawdown': self.maximum_drawdown,
            'volatility': self.volatility,
            'beta': self.beta,
            'concentration_risk': self.concentration_risk,
            'leverage': self.leverage,
            'risk_level': self.risk_level.value
        }


@dataclass
class RiskLimit:
    """
    Risk limit definition
    """
    name: str
    metric: str  # Which risk metric to check
    threshold: float  # Limit threshold
    action: str  # Action to take if breached
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert limit to dictionary"""
        return {
            'name': self.name,
            'metric': self.metric,
            'threshold': self.threshold,
            'action': self.action,
            'enabled': self.enabled
        }


@dataclass
class RiskAssessment:
    """
    Complete risk assessment result
    """
    timestamp: pd.Timestamp
    metrics: RiskMetrics
    limit_breaches: List[RiskLimit]
    recommendations: List[str]
    approved_orders: List[Order]
    rejected_orders: List[Tuple[Order, str]]  # Order and rejection reason
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics.to_dict(),
            'limit_breaches': [limit.to_dict() for limit in self.limit_breaches],
            'recommendations': self.recommendations,
            'approved_orders': len(self.approved_orders),
            'rejected_orders': len(self.rejected_orders)
        }


class RiskManager(ABC):
    """
    Abstract base class for risk management
    
    Implements Chain of Responsibility Pattern for risk checks
    Follows Single Responsibility Principle for risk assessment
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize risk manager
        
        Args:
            name: Risk manager name
            config: Risk management configuration
        """
        self.name = name
        self.config = config or {}
        self.risk_limits: List[RiskLimit] = []
        self.enabled = self.config.get('enabled', True)
    
    @abstractmethod
    def assess_portfolio_risk(self, positions: Dict[str, Position],
                             portfolio_value: float,
                             market_data: Dict[str, pd.DataFrame]) -> RiskMetrics:
        """
        Assess current portfolio risk
        
        Args:
            positions: Current positions
            portfolio_value: Total portfolio value
            market_data: Market data for risk calculation
            
        Returns:
            Risk metrics
        """
        pass
    
    @abstractmethod
    def validate_orders(self, orders: List[Order],
                       positions: Dict[str, Position],
                       portfolio_value: float,
                       market_data: Dict[str, pd.DataFrame]) -> RiskAssessment:
        """
        Validate orders against risk limits
        
        Args:
            orders: Orders to validate
            positions: Current positions
            portfolio_value: Portfolio value
            market_data: Market data
            
        Returns:
            Risk assessment with approved/rejected orders
        """
        pass
    
    @abstractmethod
    def calculate_position_size_limit(self, symbol: str,
                                    current_price: float,
                                    portfolio_value: float,
                                    positions: Dict[str, Position]) -> int:
        """
        Calculate maximum allowed position size
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            portfolio_value: Portfolio value
            positions: Current positions
            
        Returns:
            Maximum position size in shares
        """
        pass
    
    @abstractmethod
    def check_concentration_risk(self, positions: Dict[str, Position],
                               portfolio_value: float) -> float:
        """
        Check portfolio concentration risk
        
        Args:
            positions: Current positions
            portfolio_value: Total portfolio value
            
        Returns:
            Concentration risk score (0.0 to 1.0)
        """
        pass
    
    def add_risk_limit(self, limit: RiskLimit) -> None:
        """Add a risk limit"""
        self.risk_limits.append(limit)
    
    def remove_risk_limit(self, limit_name: str) -> bool:
        """Remove a risk limit by name"""
        for i, limit in enumerate(self.risk_limits):
            if limit.name == limit_name:
                del self.risk_limits[i]
                return True
        return False
    
    def get_risk_limits(self) -> List[RiskLimit]:
        """Get all risk limits"""
        return self.risk_limits.copy()
    
    def is_enabled(self) -> bool:
        """Check if risk manager is enabled"""
        return self.enabled
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable risk manager"""
        self.enabled = enabled
    
    def get_name(self) -> str:
        """Get risk manager name"""
        return self.name