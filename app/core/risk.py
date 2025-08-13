"""
Risk evaluation module for options strategies.

This module contains functions for evaluating the risk of options strategies,
including max loss calculations, probability of profit, and scenario analysis.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from app.core.strategies.base import StrategyCandidate
from app.utils.config import get_config


def calculate_max_loss(candidate: StrategyCandidate) -> float:
    """
    Calculate the maximum possible loss for a strategy candidate.
    
    Args:
        candidate: The strategy candidate to evaluate
        
    Returns:
        Maximum loss in dollars (positive number)
    """
    # If the candidate already has max_loss calculated, return it
    if candidate.max_loss != 0:
        return candidate.max_loss
    
    # Otherwise, calculate based on strategy characteristics
    # This is a simplified implementation - in reality, we would need
    # more complex logic specific to each strategy type
    
    # For undefined-risk strategies, estimate using a large price move
    # In a real system, this would use strategy-specific calculations
    return candidate.buying_power_effect


def calculate_max_loss_percentage(
    candidate: StrategyCandidate, 
    account_equity: float
) -> float:
    """
    Calculate maximum loss as a percentage of account equity.
    
    Args:
        candidate: The strategy candidate to evaluate
        account_equity: Total account equity
        
    Returns:
        Maximum loss as a percentage of account equity
    """
    max_loss = calculate_max_loss(candidate)
    return (max_loss / account_equity) * 100


def generate_price_scenario_table(
    spot_price: float,
    iv: float,
    days_to_expiration: int,
    price_points: Optional[List[float]] = None,
    include_iv_scenarios: bool = True
) -> pd.DataFrame:
    """
    Generate a price scenario table for a strategy candidate.
    
    Args:
        candidate: The strategy candidate to evaluate
        spot_price: Current spot price
        iv: Implied volatility
        days_to_expiration: Days to expiration
        price_points: Optional list of price points to evaluate
        include_iv_scenarios: Whether to include IV change scenarios
        
    Returns:
        DataFrame with scenario analysis results
    """
    config = get_config()
    risk_conf = config['risk_scenarios']
    price_multipliers = risk_conf['price_move_multipliers']
    iv_changes = risk_conf['iv_changes']

    # Generate price points if not provided
    if price_points is None:
        # Calculate 1 and 2 standard deviation moves
        t = days_to_expiration / 365.0
        std_dev = spot_price * iv * np.sqrt(t)
        
        price_points = [
            spot_price * price_multipliers[0],  # Large down move
            spot_price - 2 * std_dev,  # -2σ
            spot_price - std_dev,  # -1σ
            spot_price,  # Unchanged
            spot_price + std_dev,  # +1σ
            spot_price + 2 * std_dev,  # +2σ
            spot_price * price_multipliers[1],  # Large up move
        ]
    
    # Create base scenario table
    scenarios = []
    
    # Add price scenarios
    for price in price_points:
        scenario = {
            'Scenario': f"${price:.2f}",
            'Move %': ((price / spot_price) - 1) * 100,
            'P/L $': 0,  # Will be filled in by strategy
            'P/L %': 0,  # Will be filled in by strategy
            'IV Change': 0  # No IV change in base scenario
        }
        scenarios.append(scenario)
    
    # Add IV change scenarios if requested
    if include_iv_scenarios:
        for iv_change in iv_changes:
            scenario = {
                'Scenario': f"IV {iv_change:+.2f}",
                'Move %': 0,  # No price change
                'P/L $': 0,  # Will be filled in by strategy
                'P/L %': 0,  # Will be filled in by strategy
                'IV Change': iv_change
            }
            scenarios.append(scenario)
    
    # Create DataFrame
    scenario_table = pd.DataFrame(scenarios)
    
    # Return the table
    # In a real system, we would fill in the P/L columns with strategy-specific logic
    return scenario_table


def calculate_capital_efficiency(candidate: StrategyCandidate) -> float:
    """
    Calculate capital efficiency (theta per unit buying power).
    
    Args:
        candidate: The strategy candidate to evaluate
        
    Returns:
        Capital efficiency score (higher is better)
    """
    if candidate.buying_power_effect <= 0:
        return 0
    
    # If theta is available, use it
    if candidate.theta != 0:
        return candidate.theta / candidate.buying_power_effect * 100
    
    # Otherwise, use expected return as a proxy
    return candidate.expected_return / candidate.buying_power_effect * 100


def calculate_liquidity_penalty(candidate: StrategyCandidate) -> float:
    """
    Calculate a penalty score for liquidity issues.
    
    Args:
        candidate: The strategy candidate to evaluate
        
    Returns:
        Liquidity penalty score (0-100, higher is worse)
    """
    config = get_config()
    penalty_conf = config['liquidity_penalty']
    penalty = 0
    
    # Penalize wide spreads
    for item in penalty_conf['spread_pct']:
        if candidate.avg_spread_pct > item['threshold']:
            penalty += item['penalty']
            break # Apply highest penalty and stop

    # Penalize low open interest
    for item in penalty_conf['open_interest']:
        if candidate.avg_open_interest < item['threshold']:
            penalty += item['penalty']
            break # Apply highest penalty and stop
    
    # Cap at 100
    return min(100, penalty)


def calculate_event_penalty(
    candidate: StrategyCandidate, 
    days_to_event: Optional[int] = None
) -> float:
    """
    Calculate penalty for proximity to earnings or economic events.
    
    Args:
        candidate: The strategy candidate to evaluate
        days_to_event: Days until next significant event
        
    Returns:
        Event risk penalty score (0-100, higher is worse)
    """
    # If no event data or no short leg, no penalty
    if days_to_event is None or candidate.dte_short is None:
        return 0
    
    config = get_config()
    penalty_conf = config['event_penalty']

    # Calculate penalty based on how close the expiration is to the event
    days_buffer = abs(candidate.dte_short - days_to_event)
    
    for item in penalty_conf['days_buffer']:
        if days_buffer <= item['threshold']:
            return item['penalty']

    return 0  # No penalty if well separated from events
