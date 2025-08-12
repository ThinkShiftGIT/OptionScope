"""
Market indicators and volatility metrics for options analysis.

This module implements various technical and volatility indicators used in the OptionScope
options scanner/optimizer, including:
- Average True Range (ATR)
- At-the-Money (ATM) Implied Volatility
- IV Term Structure
- IV Rank
- Liquidity metrics
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from app.core.data_providers import (
    DataProviderType, OptionsChain, PriceDataProvider, 
    OptionsDataProvider, VolatilityDataProvider, get_provider
)


def calculate_atr(symbol: str, period: int = 5) -> float:
    """
    Calculate Average True Range (Wilder's) for a symbol.
    
    Args:
        symbol: Ticker symbol
        period: ATR period (default: 5)
        
    Returns:
        float: ATR value
    """
    # Get price provider and data
    price_provider = get_provider(DataProviderType.PRICE)
    # Get enough data to calculate ATR
    ohlc = price_provider.get_ohlc(symbol, period="1mo", interval="1d")
    
    # Calculate true ranges
    high_low = ohlc['High'] - ohlc['Low']
    high_close_prev = abs(ohlc['High'] - ohlc['Close'].shift(1))
    low_close_prev = abs(ohlc['Low'] - ohlc['Close'].shift(1))
    
    # True Range is the greatest of the three
    true_ranges = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate Wilder's ATR (first value is simple average, subsequent values use smoothing)
    atr = true_ranges.rolling(window=period).apply(
        lambda x: x[0] if len(x) < period else 
        (x.iloc[-1] + (period-1) * x.iloc[0]) / period
    )
    
    # Return the latest ATR value
    return float(atr.iloc[-1])


def get_atm_iv(symbol: str, target_dte: int) -> float:
    """
    Get At-The-Money implied volatility for a specific days-to-expiration.
    Interpolates between expirations if necessary.
    
    Args:
        symbol: Ticker symbol
        target_dte: Target days to expiration
        
    Returns:
        float: ATM implied volatility
    """
    # Get required providers
    price_provider = get_provider(DataProviderType.PRICE)
    options_provider = get_provider(DataProviderType.OPTIONS)
    
    # Get current price
    spot_price = price_provider.get_price(symbol)
    
    # Get expirations
    expirations = options_provider.get_expirations(symbol)
    
    # Calculate DTE for each expiration
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    dte_list = [(exp - today).days for exp in expirations]
    
    # Find the two expirations that bracket our target DTE
    exp_dte_pairs = list(zip(expirations, dte_list))
    exp_dte_pairs.sort(key=lambda x: x[1])  # Sort by DTE
    
    # If we have an exact match, use it
    for exp, dte in exp_dte_pairs:
        if dte == target_dte:
            chain = options_provider.get_options_chain(symbol, exp)
            call_iv, put_iv = chain.get_atm_iv(spot_price)
            # Average call and put IV for a more stable estimate
            return (call_iv + put_iv) / 2
    
    # Otherwise, interpolate between the two closest expirations
    if not exp_dte_pairs:
        raise ValueError(f"No options expirations found for {symbol}")
    
    # Find the two closest DTEs
    if target_dte < exp_dte_pairs[0][1]:
        # Target is before the first expiration, use the first expiration
        closest_exp, closest_dte = exp_dte_pairs[0]
        chain = options_provider.get_options_chain(symbol, closest_exp)
        call_iv, put_iv = chain.get_atm_iv(spot_price)
        return (call_iv + put_iv) / 2
    
    if target_dte > exp_dte_pairs[-1][1]:
        # Target is after the last expiration, use the last expiration
        closest_exp, closest_dte = exp_dte_pairs[-1]
        chain = options_provider.get_options_chain(symbol, closest_exp)
        call_iv, put_iv = chain.get_atm_iv(spot_price)
        return (call_iv + put_iv) / 2
    
    # Find the two expirations that bracket our target DTE
    lower_idx = next(i for i, (_, dte) in enumerate(exp_dte_pairs) if dte > target_dte) - 1
    lower_exp, lower_dte = exp_dte_pairs[lower_idx]
    upper_exp, upper_dte = exp_dte_pairs[lower_idx + 1]
    
    # Get IV for both expirations
    lower_chain = options_provider.get_options_chain(symbol, lower_exp)
    upper_chain = options_provider.get_options_chain(symbol, upper_exp)
    
    lower_call_iv, lower_put_iv = lower_chain.get_atm_iv(spot_price)
    upper_call_iv, upper_put_iv = upper_chain.get_atm_iv(spot_price)
    
    lower_iv = (lower_call_iv + lower_put_iv) / 2
    upper_iv = (upper_call_iv + upper_put_iv) / 2
    
    # Linear interpolation
    weight = (target_dte - lower_dte) / (upper_dte - lower_dte)
    interpolated_iv = lower_iv + weight * (upper_iv - lower_iv)
    
    return interpolated_iv


def calculate_iv_term_structure(
    symbol: str, 
    short_dte: int = 30, 
    long_dte: int = 60
) -> Tuple[float, float, float]:
    """
    Calculate IV term structure between two tenors.
    
    Args:
        symbol: Ticker symbol
        short_dte: Shorter-term DTE
        long_dte: Longer-term DTE
        
    Returns:
        Tuple[float, float, float]: (short_term_iv, long_term_iv, spread)
    """
    short_term_iv = get_atm_iv(symbol, short_dte)
    long_term_iv = get_atm_iv(symbol, long_dte)
    spread = long_term_iv - short_term_iv
    
    return short_term_iv, long_term_iv, spread


def calculate_iv_rank(symbol: str) -> float:
    """
    Calculate IV Rank using volatility index (VIX/VXN) 1-year range.
    
    Args:
        symbol: Ticker symbol
        
    Returns:
        float: IV Rank as a percentage (0-100)
    """
    # Get volatility provider
    volatility_provider = get_provider(DataProviderType.VOLATILITY)
    
    # Map symbol to appropriate volatility index
    vol_indices = {
        'SPY': '^VIX',
        'QQQ': '^VXN'
    }
    
    vol_index = vol_indices.get(symbol)
    if not vol_index:
        raise ValueError(f"No volatility index mapping for {symbol}")
    
    # Get current volatility index value
    current_value = volatility_provider.get_index_value(vol_index)
    
    # Get 1-year historical data
    historical_data = volatility_provider.get_historical_data(vol_index, period="1y")
    
    # Calculate min and max values
    min_value = historical_data['Close'].min()
    max_value = historical_data['Close'].max()
    
    # Calculate IV Rank
    if max_value - min_value == 0:  # Avoid division by zero
        return 50.0
    
    iv_rank = (current_value - min_value) / (max_value - min_value) * 100
    return round(iv_rank, 2)


def calculate_liquidity_metrics(chain: OptionsChain) -> Dict[str, float]:
    """
    Calculate liquidity metrics for an options chain.
    
    Args:
        chain: OptionsChain object
        
    Returns:
        Dict with liquidity metrics:
        - avg_spread_pct: Average bid-ask spread percentage
        - avg_oi: Average open interest
        - liquidity_score: Overall liquidity score (0-100)
    """
    # Combine calls and puts
    all_options = chain.calls + chain.puts
    
    if not all_options:
        return {
            'avg_spread_pct': float('inf'),
            'avg_oi': 0,
            'liquidity_score': 0
        }
    
    # Calculate average spread percentage
    spreads = [opt.spread_pct for opt in all_options if opt.spread_pct < float('inf')]
    avg_spread_pct = sum(spreads) / len(spreads) if spreads else float('inf')
    
    # Calculate average open interest
    oi_values = [opt.open_interest or 0 for opt in all_options]
    avg_oi = sum(oi_values) / len(oi_values)
    
    # Calculate liquidity score
    # Lower spread_pct is better, higher open interest is better
    # Normalize and combine into a score from 0-100
    if avg_spread_pct == float('inf') or avg_spread_pct > 1.0:
        spread_score = 0
    else:
        spread_score = max(0, min(100, 100 * (1 - avg_spread_pct)))
    
    # Normalize open interest score (arbitrary scale, adjust as needed)
    oi_score = min(100, max(0, avg_oi / 100))
    
    # Combined liquidity score (weighted average)
    liquidity_score = 0.7 * spread_score + 0.3 * oi_score
    
    return {
        'avg_spread_pct': avg_spread_pct,
        'avg_oi': avg_oi,
        'liquidity_score': liquidity_score
    }


def get_vix_level(symbol: str) -> float:
    """
    Get current VIX/VXN level for a symbol.
    
    Args:
        symbol: Ticker symbol
        
    Returns:
        float: Current volatility index value
    """
    # Get volatility provider
    volatility_provider = get_provider(DataProviderType.VOLATILITY)
    
    # Map symbol to appropriate volatility index
    vol_indices = {
        'SPY': '^VIX',
        'QQQ': '^VXN'
    }
    
    vol_index = vol_indices.get(symbol)
    if not vol_index:
        raise ValueError(f"No volatility index mapping for {symbol}")
    
    # Get current volatility index value
    return volatility_provider.get_index_value(vol_index)


def get_realized_vol(symbol: str, period_days: int = 30) -> float:
    """
    Calculate realized volatility over a specified period.
    
    Args:
        symbol: Ticker symbol
        period_days: Number of trading days to look back
        
    Returns:
        float: Realized volatility (annualized)
    """
    # Get price provider
    price_provider = get_provider(DataProviderType.PRICE)
    
    # Get daily close prices
    # Add some buffer to account for weekends and holidays
    buffer_days = int(period_days * 1.5)
    ohlc = price_provider.get_ohlc(symbol, period=f"{buffer_days}d", interval="1d")
    
    # Get the last 'period_days' records
    prices = ohlc['Close'].tail(period_days)
    
    # Calculate daily returns
    returns = np.log(prices / prices.shift(1)).dropna()
    
    # Calculate annualized volatility
    daily_std = returns.std()
    annualized_vol = daily_std * np.sqrt(252)  # Annualize using trading days
    
    return float(annualized_vol)


def get_iv_skew(chain: OptionsChain, spot_price: float) -> float:
    """
    Calculate IV skew (difference between downside put IV and upside call IV).
    
    Args:
        chain: OptionsChain object
        spot_price: Current spot price
        
    Returns:
        float: IV skew value (positive means puts are more expensive than calls)
    """
    if not chain.calls or not chain.puts:
        return 0.0
    
    # Find strikes approximately 10% away from spot
    down_strike = spot_price * 0.9
    up_strike = spot_price * 1.1
    
    # Find closest puts to down_strike
    down_put = min(chain.puts, key=lambda x: abs(x.strike - down_strike))
    
    # Find closest calls to up_strike
    up_call = min(chain.calls, key=lambda x: abs(x.strike - up_strike))
    
    # Calculate skew
    skew = down_put.implied_volatility - up_call.implied_volatility
    
    return skew


def check_rule_compliance(symbol: str) -> Dict[str, bool]:
    """
    Check if a symbol meets all the mechanical entry rules.
    
    Args:
        symbol: Ticker symbol
        
    Returns:
        Dict of rule names and their compliance status
    """
    from app.utils.config import get_config
    config = get_config()
    
    # Get current market data
    vol_level = get_vix_level(symbol)
    iv_rank = calculate_iv_rank(symbol)
    atr = calculate_atr(symbol, period=config['mechanical_rules']['atr_width']['period'])
    
    # Get term structure
    short_dte = config['strategy_params']['calendars']['short_dte_range'][1]  # Use upper bound
    long_dte = config['strategy_params']['calendars']['long_dte_range'][0]    # Use lower bound
    _, _, term_spread = calculate_iv_term_structure(symbol, short_dte, long_dte)
    
    # Check rule compliance
    rules = {
        'term_structure': False,
        'iv_rank': False,
        'vix_range': False,
        'atr_width': True,  # Default to True, will be checked per strategy
        'event_window': True  # Default to True, will be checked when evaluating strategies
    }
    
    # Term structure rule
    min_gap = config['mechanical_rules']['term_structure']['min_gap']
    sign = config['mechanical_rules']['term_structure']['sign']
    
    if sign == 'positive':
        rules['term_structure'] = term_spread >= min_gap
    else:
        rules['term_structure'] = term_spread <= -min_gap
    
    # IV Rank rule
    rules['iv_rank'] = (
        config['mechanical_rules']['iv_rank']['min'] <= iv_rank <= 
        config['mechanical_rules']['iv_rank']['max']
    )
    
    # VIX/VXN range rule
    rules['vix_range'] = (
        config['mechanical_rules']['vix_range']['min'] <= vol_level <= 
        config['mechanical_rules']['vix_range']['max']
    )
    
    return rules


def get_iv_surface(
    symbol: str, 
    dte_range: List[int], 
    delta_range: List[float]
) -> pd.DataFrame:
    """
    Generate an implied volatility surface for a range of DTEs and deltas.
    
    Args:
        symbol: Ticker symbol
        dte_range: List of DTEs to include
        delta_range: List of deltas to include
        
    Returns:
        DataFrame with IV values indexed by DTE and delta
    """
    # Get required providers
    price_provider = get_provider(DataProviderType.PRICE)
    options_provider = get_provider(DataProviderType.OPTIONS)
    
    # Get current price
    spot_price = price_provider.get_price(symbol)
    
    # Get expirations
    expirations = options_provider.get_expirations(symbol)
    
    # Calculate DTE for each expiration
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    dte_exp_map = {(exp - today).days: exp for exp in expirations}
    
    # Find closest available DTEs to the requested ones
    available_dtes = sorted(dte_exp_map.keys())
    closest_dtes = [min(available_dtes, key=lambda d: abs(d - dte)) for dte in dte_range]
    
    # Create empty DataFrame for the surface
    index = pd.Index(closest_dtes, name='DTE')
    columns = pd.Index(delta_range, name='Delta')
    surface = pd.DataFrame(index=index, columns=columns)
    
    # Fill the surface
    for dte in closest_dtes:
        exp = dte_exp_map[dte]
        chain = options_provider.get_options_chain(symbol, exp)
        
        # Process calls for positive deltas
        for delta in [d for d in delta_range if d > 0]:
            # Find the option with delta closest to the target
            call = min(
                [c for c in chain.calls if c.delta is not None], 
                key=lambda x: abs(x.delta - delta),
                default=None
            )
            if call:
                surface.loc[dte, delta] = call.implied_volatility
        
        # Process puts for negative deltas
        for delta in [d for d in delta_range if d < 0]:
            # For puts, delta is negative but stored as positive in some data sources
            put = min(
                [p for p in chain.puts if p.delta is not None], 
                key=lambda x: abs(x.delta - abs(delta)),
                default=None
            )
            if put:
                surface.loc[dte, delta] = put.implied_volatility
    
    return surface
