"""
Scoring module for ranking options strategy candidates.

This module contains functions for scoring and ranking options strategy candidates
based on multiple objectives, including expected return, probability of profit,
capital efficiency, and risk factors.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.strategies.base import StrategyCandidate
from app.core.risk import (
    calculate_max_loss_percentage,
    calculate_capital_efficiency,
    calculate_liquidity_penalty,
    calculate_event_penalty
)
from app.utils.config import get_config


def normalize_value(
    value: float,
    min_value: float,
    max_value: float,
    higher_is_better: bool = True
) -> float:
    """
    Normalize a value to a 0-100 scale.
    
    Args:
        value: Value to normalize
        min_value: Minimum value in the range
        max_value: Maximum value in the range
        higher_is_better: Whether higher values are better
        
    Returns:
        Normalized value (0-100)
    """
    # Prevent division by zero
    if min_value == max_value:
        return 50.0
    
    # Clip value to range
    value = max(min_value, min(value, max_value))
    
    # Normalize
    if higher_is_better:
        return ((value - min_value) / (max_value - min_value)) * 100
    else:
        return ((max_value - value) / (max_value - min_value)) * 100


def calculate_composite_score(
    candidate: StrategyCandidate,
    account_equity: float,
    days_to_event: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate a composite score for a strategy candidate.
    
    Args:
        candidate: Strategy candidate to score
        account_equity: Total account equity
        days_to_event: Days until next significant event
        weights: Optional custom weights for scoring components
        
    Returns:
        Composite score (0-100, higher is better)
    """
    # Get default weights from config if not provided
    if weights is None:
        config = get_config()
        weights = config['scoring_weights']
    
    # Calculate individual component scores
    
    # Probability of Profit (higher is better)
    pop_score = candidate.probability_of_profit
    
    # Expected Return (higher is better)
    # Normalize to a reasonable range (e.g., 0% to 30%)
    er_score = normalize_value(
        candidate.expected_return,
        min_value=0,
        max_value=30,
        higher_is_better=True
    )
    
    # Capital Efficiency (higher is better)
    capital_efficiency = calculate_capital_efficiency(candidate)
    ce_score = normalize_value(
        capital_efficiency,
        min_value=0,
        max_value=0.5,  # 0.5% theta per day per BP dollar is excellent
        higher_is_better=True
    )
    
    # Max Loss Percentage (lower is better)
    max_loss_pct = calculate_max_loss_percentage(candidate, account_equity)
    ml_score = normalize_value(
        max_loss_pct,
        min_value=0,
        max_value=10,  # 10% of account is considered high risk
        higher_is_better=False
    )
    
    # Liquidity Penalty (lower is better)
    liquidity_penalty = calculate_liquidity_penalty(candidate)
    lp_score = 100 - liquidity_penalty  # Convert penalty to score
    
    # Event Penalty (lower is better)
    event_penalty = calculate_event_penalty(candidate, days_to_event)
    ep_score = 100 - event_penalty  # Convert penalty to score
    
    # Calculate weighted score
    composite_score = (
        weights['probability_of_profit'] * pop_score +
        weights['expected_return'] * er_score +
        weights['capital_efficiency'] * ce_score +
        weights['max_loss_pct'] * ml_score +
        weights['liquidity_penalty'] * lp_score +
        weights['event_penalty'] * ep_score
    )
    
    # Add the scores to the candidate
    candidate.composite_score = composite_score
    
    return composite_score


def rank_candidates(
    candidates: List[StrategyCandidate],
    account_equity: float,
    days_to_event: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None
) -> List[StrategyCandidate]:
    """
    Score and rank a list of strategy candidates.
    
    Args:
        candidates: List of strategy candidates
        account_equity: Total account equity
        days_to_event: Days until next significant event
        weights: Optional custom weights for scoring components
        
    Returns:
        Sorted list of candidates (best first)
    """
    # Calculate score for each candidate
    for candidate in candidates:
        calculate_composite_score(
            candidate,
            account_equity,
            days_to_event,
            weights
        )
    
    # Sort by score (descending)
    ranked_candidates = sorted(
        candidates,
        key=lambda c: c.composite_score,
        reverse=True
    )
    
    return ranked_candidates


def get_top_candidates(
    candidates: List[StrategyCandidate],
    account_equity: float,
    days_to_event: Optional[int] = None,
    top_n: int = 3,
    include_categories: bool = True
) -> List[StrategyCandidate]:
    """
    Get the top N candidates, optionally including top performers by category.
    
    Args:
        candidates: List of strategy candidates
        account_equity: Total account equity
        days_to_event: Days until next significant event
        top_n: Number of top candidates to return
        include_categories: Whether to include category winners
        
    Returns:
        List of top candidates
    """
    # Rank all candidates by default weights
    ranked = rank_candidates(candidates, account_equity, days_to_event)
    
    if not include_categories:
        # Return top N candidates
        return ranked[:top_n]
    
    # Create custom weightings for different categories
    config = get_config()
    default_weights = config['scoring_weights']
    
    # Profit-maximizing weights (emphasize expected return)
    profit_weights = default_weights.copy()
    profit_weights['expected_return'] = 0.6
    profit_weights['probability_of_profit'] = 0.1
    profit_weights['capital_efficiency'] = 0.1
    profit_weights['max_loss_pct'] = 0.1
    profit_weights['liquidity_penalty'] = 0.05
    profit_weights['event_penalty'] = 0.05
    
    # Probability-maximizing weights (emphasize probability of profit)
    prob_weights = default_weights.copy()
    prob_weights['expected_return'] = 0.1
    prob_weights['probability_of_profit'] = 0.6
    prob_weights['capital_efficiency'] = 0.1
    prob_weights['max_loss_pct'] = 0.1
    prob_weights['liquidity_penalty'] = 0.05
    prob_weights['event_penalty'] = 0.05
    
    # Rank candidates by each category
    profit_ranked = rank_candidates(
        candidates, account_equity, days_to_event, profit_weights
    )
    
    prob_ranked = rank_candidates(
        candidates, account_equity, days_to_event, prob_weights
    )
    
    # Get top candidates from each category
    top_candidates = []
    
    # Add the overall best candidate
    if ranked and len(ranked) > 0:
        top_ranked = ranked[0]
        top_ranked.notes += " (Balanced)"
        top_candidates.append(top_ranked)
    
    # Add the top profit-maximizing candidate (if different)
    if profit_ranked and len(profit_ranked) > 0:
        top_profit = profit_ranked[0]
        if top_profit not in top_candidates:
            top_profit.notes += " (Profit-Max)"
            top_candidates.append(top_profit)
    
    # Add the top probability-maximizing candidate (if different)
    if prob_ranked and len(prob_ranked) > 0:
        top_prob = prob_ranked[0]
        if top_prob not in top_candidates:
            top_prob.notes += " (PoP-Max)"
            top_candidates.append(top_prob)
    
    # If we have less than top_n candidates, fill in with the next best overall
    i = 1
    while len(top_candidates) < top_n and i < len(ranked):
        if ranked[i] not in top_candidates:
            top_candidates.append(ranked[i])
        i += 1
    
    return top_candidates


def generate_pareto_chart_data(
    candidates: List[StrategyCandidate]
) -> pd.DataFrame:
    """
    Generate data for a Pareto chart showing the trade-offs between
    profit, probability, and max loss.
    
    Args:
        candidates: List of strategy candidates
        
    Returns:
        DataFrame with data for the Pareto chart
    """
    # Extract the key metrics from each candidate
    data = []
    for candidate in candidates:
        data.append({
            'Strategy': candidate.strategy_name,
            'DTE': candidate.dte_short or 0,
            'Expected Return (%)': candidate.expected_return,
            'Probability of Profit (%)': candidate.probability_of_profit,
            'Max Loss (%)': candidate.max_loss / candidate.buying_power_effect * 100 if candidate.buying_power_effect else 0,
            'Score': candidate.composite_score,
        })
    
    return pd.DataFrame(data)
