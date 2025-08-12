"""
Base strategy interface and registry for options strategies.

This module defines the base Strategy class that all option strategies must implement,
as well as the StrategyRegistry for managing and retrieving available strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar

import pandas as pd
import numpy as np

from app.core.data_providers import OptionsChain


class StrategyType(Enum):
    """Classification of option strategy types."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class OptionType(Enum):
    """Option contract type."""
    CALL = "call"
    PUT = "put"


class Direction(Enum):
    """Strategy directional bias."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class TradeAction(Enum):
    """Action to take on an option contract."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class OptionLeg:
    """Represents a single leg in an options strategy."""
    option_type: OptionType
    strike: float
    expiration_date: str  # ISO format date string
    action: TradeAction
    quantity: int = 1
    delta: Optional[float] = None
    price: Optional[float] = None
    
    @property
    def is_long(self) -> bool:
        """Return True if this is a long option position."""
        return self.action == TradeAction.BUY
    
    @property
    def is_short(self) -> bool:
        """Return True if this is a short option position."""
        return self.action == TradeAction.SELL


@dataclass
class StrategyCandidate:
    """Represents a complete strategy setup with all legs and risk metrics."""
    symbol: str
    strategy_name: str
    legs: List[OptionLeg] = field(default_factory=list)
    
    dte_short: Optional[int] = None
    dte_long: Optional[int] = None
    
    estimated_price: float = 0.0
    probability_of_profit: float = 0.0
    expected_return: float = 0.0
    max_loss: float = 0.0
    max_profit: float = 0.0
    buying_power_effect: float = 0.0
    
    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    # Liquidity metrics
    avg_spread_pct: float = 0.0
    avg_open_interest: float = 0.0
    
    # Additional info
    scenario_table: Optional[pd.DataFrame] = None
    notes: str = ""
    composite_score: float = 0.0
    
    @property
    def width(self) -> Optional[float]:
        """Calculate width between strikes if applicable."""
        if len(self.legs) < 2:
            return None
        
        strikes = sorted([leg.strike for leg in self.legs])
        return strikes[-1] - strikes[0]


class Strategy(ABC):
    """Base class for all options strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the strategy."""
        pass
    
    @property
    @abstractmethod
    def type(self) -> StrategyType:
        """Type classification of the strategy."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of the strategy."""
        pass
    
    @property
    def directions(self) -> Set[Direction]:
        """
        Set of directions this strategy can be used for.
        Default implementation returns all directions.
        """
        return {Direction.BULLISH, Direction.BEARISH, Direction.NEUTRAL}
    
    @abstractmethod
    def generate_candidates(
        self,
        symbol: str,
        spot_price: float,
        iv_surface: pd.DataFrame,
        dte_range: List[int],
        delta_targets: Dict[str, List[float]],
        constraints: Dict[str, Any]
    ) -> List[StrategyCandidate]:
        """
        Generate strategy candidates based on market data and constraints.
        
        Args:
            symbol: Ticker symbol
            spot_price: Current spot price
            iv_surface: DataFrame with IVs indexed by DTE and delta
            dte_range: List of DTEs to consider
            delta_targets: Dict mapping leg names to delta targets
            constraints: Additional constraints for strategy generation
            
        Returns:
            List of strategy candidate objects
        """
        pass
    
    def calculate_probability_of_profit(
        self,
        candidate: StrategyCandidate,
        spot_price: float,
        iv: float,
        days_to_expiration: int,
        risk_free_rate: float = 0.05
    ) -> float:
        """
        Calculate probability of profit for a strategy candidate.
        
        Args:
            candidate: Strategy candidate
            spot_price: Current spot price
            iv: Implied volatility
            days_to_expiration: Days to expiration
            risk_free_rate: Annual risk-free rate as decimal
            
        Returns:
            Probability of profit as a percentage
        """
        # Default implementation using simple approximation
        # This should be overridden in specific strategy implementations
        # for more accurate calculations
        
        # Convert to years for Black-Scholes
        t = days_to_expiration / 365.0
        
        # Generate price points for the scenario analysis
        std_dev = spot_price * iv * np.sqrt(t)
        price_points = np.linspace(
            spot_price - 3 * std_dev,
            spot_price + 3 * std_dev,
            100
        )
        
        # Calculate profit at each price point
        profits = np.array([
            self._calculate_profit_at_price(candidate, price, days_to_expiration)
            for price in price_points
        ])
        
        # Probability density function for lognormal distribution
        drift = risk_free_rate - 0.5 * iv * iv
        sigma = iv
        
        def pdf(x: float) -> float:
            """Lognormal probability density function."""
            if x <= 0:
                return 0.0
            return (1.0 / (x * sigma * np.sqrt(2 * np.pi * t))) * np.exp(
                -((np.log(x / spot_price) - drift * t) ** 2) / (2 * sigma * sigma * t)
            )
        
        # Calculate probabilities for each price point
        probs = np.array([pdf(p) for p in price_points])
        probs = probs / np.sum(probs)  # Normalize
        
        # Calculate probability of profit
        pop = np.sum(probs[profits > 0])
        
        return float(pop * 100)  # Convert to percentage
    
    def _calculate_profit_at_price(
        self,
        candidate: StrategyCandidate,
        price_at_expiration: float,
        days_to_expiration: int
    ) -> float:
        """
        Calculate the profit of a strategy at a given underlying price at expiration.
        
        Args:
            candidate: Strategy candidate
            price_at_expiration: Underlying price at expiration
            days_to_expiration: Days to expiration
            
        Returns:
            Profit amount
        """
        # Default implementation assumes European options
        # This should be overridden in specific strategy implementations
        # for more accurate calculations
        
        total_value = 0.0
        
        for leg in candidate.legs:
            # Calculate intrinsic value at expiration
            if leg.option_type == OptionType.CALL:
                intrinsic = max(0, price_at_expiration - leg.strike)
            else:  # PUT
                intrinsic = max(0, leg.strike - price_at_expiration)
            
            # Multiply by quantity and action direction
            if leg.action == TradeAction.BUY:
                total_value += intrinsic * leg.quantity
            else:  # SELL
                total_value -= intrinsic * leg.quantity
        
        # Subtract initial premium
        profit = total_value - candidate.estimated_price
        
        return profit
    
    def calculate_expected_return(
        self, 
        candidate: StrategyCandidate,
        spot_price: float,
        iv: float,
        days_to_expiration: int,
        risk_free_rate: float = 0.05,
        scenarios: Optional[List[Tuple[str, float]]] = None
    ) -> Tuple[float, pd.DataFrame]:
        """
        Calculate expected return and scenario table for a strategy.
        
        Args:
            candidate: Strategy candidate
            spot_price: Current spot price
            iv: Implied volatility
            days_to_expiration: Days to expiration
            risk_free_rate: Annual risk-free rate as decimal
            scenarios: Optional list of (scenario_name, price_move) tuples
            
        Returns:
            Tuple of (expected_return, scenario_table)
        """
        # Convert to years for Black-Scholes
        t = days_to_expiration / 365.0
        
        # Define default scenarios if none provided
        if scenarios is None:
            std_dev = spot_price * iv * np.sqrt(t)
            scenarios = [
                ("Large Down (-2σ)", spot_price - 2 * std_dev),
                ("Down (-1σ)", spot_price - std_dev),
                ("Unchanged", spot_price),
                ("Up (+1σ)", spot_price + std_dev),
                ("Large Up (+2σ)", spot_price + 2 * std_dev),
            ]
        
        # Generate scenario table
        results = []
        for scenario_name, price in scenarios:
            profit = self._calculate_profit_at_price(
                candidate, price, days_to_expiration
            )
            roi = profit / candidate.buying_power_effect * 100 if candidate.buying_power_effect != 0 else 0
            results.append({
                "Scenario": scenario_name,
                "Price": price,
                "P/L": profit,
                "ROI %": roi
            })
        
        scenario_table = pd.DataFrame(results)
        
        # Calculate expected return using scenario probabilities
        # This is a simplified calculation - more sophisticated models would be used
        # for a real production system
        if len(scenarios) == 5:  # Standard -2σ to +2σ
            # Approximate probabilities for a normal distribution
            probabilities = [0.05, 0.25, 0.40, 0.25, 0.05]
            expected_return = sum(
                p * r["ROI %"] for p, r in zip(probabilities, results)
            )
        else:
            # Equal weighting if not standard scenarios
            expected_return = scenario_table["ROI %"].mean()
        
        return expected_return, scenario_table


class StrategyRegistry:
    """Registry of available options strategies."""
    
    _strategies: Dict[str, Type[Strategy]] = {}
    
    @classmethod
    def register(cls, strategy_class: Type[Strategy]) -> Type[Strategy]:
        """
        Register a strategy class.
        
        Can be used as a decorator:
        @StrategyRegistry.register
        class MyStrategy(Strategy):
            ...
            
        Args:
            strategy_class: Strategy class to register
            
        Returns:
            The registered strategy class (for decorator use)
        """
        instance = strategy_class()
        cls._strategies[instance.name] = strategy_class
        return strategy_class
    
    @classmethod
    def get_strategy(cls, name: str) -> Type[Strategy]:
        """
        Get a strategy class by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy class
            
        Raises:
            ValueError: If strategy not found
        """
        if name not in cls._strategies:
            raise ValueError(f"Strategy '{name}' not found in registry")
        return cls._strategies[name]
    
    @classmethod
    def get_all_strategies(cls) -> Dict[str, Type[Strategy]]:
        """
        Get all registered strategies.
        
        Returns:
            Dict mapping strategy names to strategy classes
        """
        return cls._strategies.copy()
    
    @classmethod
    def get_strategies_by_type(cls, strategy_type: StrategyType) -> Dict[str, Type[Strategy]]:
        """
        Get all strategies of a specific type.
        
        Args:
            strategy_type: Type of strategies to retrieve
            
        Returns:
            Dict mapping strategy names to strategy classes
        """
        return {
            name: strategy_class
            for name, strategy_class in cls._strategies.items()
            if strategy_class().type == strategy_type
        }


# Type alias for better type hints
StrategyClass = TypeVar('StrategyClass', bound=Type[Strategy])
