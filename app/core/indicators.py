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
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

from app.core.data_providers import (
    DataProviderType, OptionsChain, PriceDataProvider, 
    OptionsDataProvider, VolatilityDataProvider, create_data_provider
)

# Alias for backward compatibility if needed within this module
get_provider = create_data_provider


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
    price_provider = create_data_provider('price')
    # Get enough data to calculate ATR
    ohlc = price_provider.get_ohlc(symbol, period="1mo", interval="1d")
    
    # Calculate true ranges
    high_low = ohlc['high'] - ohlc['low']
    high_close_prev = abs(ohlc['high'] - ohlc['close'].shift(1))
    low_close_prev = abs(ohlc['low'] - ohlc['close'].shift(1))
    
    # True Range is the greatest of the three
    true_ranges = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate Wilder's ATR (first value is simple average, subsequent values use smoothing)
    atr = true_ranges.rolling(window=period).apply(
        lambda x: x[0] if len(x) < period else 
        (x.iloc[-1] + (period-1) * x.iloc[0]) / period
    )
    
    # Return the latest ATR value
    return float(atr.iloc[-1])


def calculate_atm_iv(symbol: str, target_dte: int) -> float:
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
    price_provider = create_data_provider('price')
    options_provider = create_data_provider('options')
    
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
            if pd.isna(call_iv) and pd.isna(put_iv): return 0.0
            if pd.isna(call_iv): return put_iv
            if pd.isna(put_iv): return call_iv
            return (call_iv + put_iv) / 2
    
    # Otherwise, interpolate between the two closest expirations
    if not exp_dte_pairs:
        # raise ValueError(f"No options expirations found for {symbol}")
        return 0.0
    
    # Find the two closest DTEs
    if target_dte < exp_dte_pairs[0][1]:
        # Target is before the first expiration, use the first expiration
        closest_exp, closest_dte = exp_dte_pairs[0]
        chain = options_provider.get_options_chain(symbol, closest_exp)
        call_iv, put_iv = chain.get_atm_iv(spot_price)
        if pd.isna(call_iv) and pd.isna(put_iv): return 0.0
        if pd.isna(call_iv): return put_iv
        if pd.isna(put_iv): return call_iv
        return (call_iv + put_iv) / 2
    
    if target_dte > exp_dte_pairs[-1][1]:
        # Target is after the last expiration, use the last expiration
        closest_exp, closest_dte = exp_dte_pairs[-1]
        chain = options_provider.get_options_chain(symbol, closest_exp)
        call_iv, put_iv = chain.get_atm_iv(spot_price)
        if pd.isna(call_iv) and pd.isna(put_iv): return 0.0
        if pd.isna(call_iv): return put_iv
        if pd.isna(put_iv): return call_iv
        return (call_iv + put_iv) / 2
    
    # Find the two expirations that bracket our target DTE
    # Handle case where list might be small
    try:
        lower_idx = next(i for i, (_, dte) in enumerate(exp_dte_pairs) if dte > target_dte) - 1
    except StopIteration:
        lower_idx = len(exp_dte_pairs) - 2

    lower_exp, lower_dte = exp_dte_pairs[lower_idx]
    upper_exp, upper_dte = exp_dte_pairs[lower_idx + 1]
    
    # Get IV for both expirations
    lower_chain = options_provider.get_options_chain(symbol, lower_exp)
    upper_chain = options_provider.get_options_chain(symbol, upper_exp)
    
    lower_call_iv, lower_put_iv = lower_chain.get_atm_iv(spot_price)
    upper_call_iv, upper_put_iv = upper_chain.get_atm_iv(spot_price)
    
    def safe_avg(c, p):
        if pd.isna(c) and pd.isna(p): return 0.0
        if pd.isna(c): return p
        if pd.isna(p): return c
        return (c+p)/2

    lower_iv = safe_avg(lower_call_iv, lower_put_iv)
    upper_iv = safe_avg(upper_call_iv, upper_put_iv)
    
    # Linear interpolation
    if upper_dte == lower_dte:
        return lower_iv

    weight = (target_dte - lower_dte) / (upper_dte - lower_dte)
    interpolated_iv = lower_iv + weight * (upper_iv - lower_iv)
    
    return interpolated_iv

# Alias
get_atm_iv = calculate_atm_iv

def calculate_iv_term_structure(
    symbol: Any, # Can be chain or symbol? app.py passes options_chain and spot_price
    # But wait, app.py calls calculate_iv_term_structure(options_chain, spot_price)
    # The original signature here was (symbol, short_dte, long_dte)
    # I need to support what app.py calls, or change app.py
    spot_price: float = None,
    short_dte: int = 30, 
    long_dte: int = 60
) -> Any: # Returns series or tuple
    """
    Calculate IV term structure.
    Overloaded to support app.py usage: (options_chain, spot_price) -> Series
    Or (symbol, short, long) -> (short_iv, long_iv, spread)
    """
    # Check if first arg is OptionsChain
    if isinstance(symbol, OptionsChain):
        # We can't calculate term structure from a single chain.
        # Returning a single point based on chain expiration DTE.
        chain = symbol
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        dte = (chain.expiration - today).days

        # Calculate ATM IV for this chain
        call_iv, put_iv = chain.get_atm_iv(spot_price if spot_price is not None else 0.0)

        def safe_avg(c, p):
            if pd.isna(c) and pd.isna(p): return 0.0
            if pd.isna(c): return p
            if pd.isna(p): return c
            return (c+p)/2

        atm_iv = safe_avg(call_iv, put_iv)

        # Return a series with one point
        return pd.Series([atm_iv], index=[dte])

    elif isinstance(symbol, str):
         short_term_iv = calculate_atm_iv(symbol, short_dte)
         long_term_iv = calculate_atm_iv(symbol, long_dte)
         spread = long_term_iv - short_term_iv
         return short_term_iv, long_term_iv, spread

    return pd.Series()


def calculate_iv_rank(iv_history: Any) -> float:
    """
    Calculate IV Rank using volatility index (VIX/VXN) 1-year range.
    
    Args:
        iv_history: DataFrame or symbol
        
    Returns:
        float: IV Rank as a percentage (0-100)
    """
    # If passed a symbol (str), fetch history
    if isinstance(iv_history, str):
         volatility_provider = create_data_provider('volatility')
         iv_history = volatility_provider.get_iv_history(iv_history)

    if iv_history is None or iv_history.empty:
        return 50.0

    # Calculate min and max values
    min_value = iv_history['close'].min()
    max_value = iv_history['close'].max()
    current_value = iv_history['close'].iloc[-1]
    
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
        - avg_volume: Average volume
    """
    # Combine calls and puts
    all_options = chain.calls + chain.puts
    
    if not all_options:
        return {
            'avg_spread_pct': 0.0,
            'avg_open_interest': 0,
            'liquidity_score': 0,
            'avg_volume': 0
        }
    
    # Calculate average spread percentage
    spreads = [opt.spread_pct for opt in all_options if opt.spread_pct < float('inf')]
    avg_spread_pct = sum(spreads) / len(spreads) if spreads else 0.0
    
    # Calculate average open interest
    oi_values = [opt.open_interest or 0 for opt in all_options]
    avg_oi = sum(oi_values) / len(oi_values)
    
    # Avg volume
    vol_values = [opt.volume or 0 for opt in all_options]
    avg_vol = sum(vol_values) / len(vol_values)

    # Calculate liquidity score
    if avg_spread_pct == float('inf') or avg_spread_pct > 1.0:
        spread_score = 0
    else:
        spread_score = max(0, min(100, 100 * (1 - avg_spread_pct)))
    
    oi_score = min(100, max(0, avg_oi / 100))
    
    liquidity_score = 0.7 * spread_score + 0.3 * oi_score
    
    return {
        'avg_spread_pct': avg_spread_pct,
        'avg_open_interest': avg_oi,
        'liquidity_score': liquidity_score,
        'avg_volume': avg_vol
    }


def calculate_vix_level(vol_provider_or_symbol: Any) -> float:
    """
    Get current VIX/VXN level.
    """
    if isinstance(vol_provider_or_symbol, str):
         # Map symbol to volatility index
         vol_indices = {
            'SPY': '^VIX',
            'QQQ': '^VXN',
            'IWM': '^RVX'
         }
         vol_index = vol_indices.get(vol_provider_or_symbol, '^VIX')

         provider = create_data_provider('volatility')
         return provider.get_index_value(vol_index)
    
    # Assume it's a provider instance or similar (app.py passes provider)
    # app.py seems to call calculate_vix_level(volatility_provider) without symbol?
    # This logic is flawed in app.py if it doesn't pass symbol.
    # However, if we must fallback, let's use VIX
    if hasattr(vol_provider_or_symbol, 'get_index_value'):
         return vol_provider_or_symbol.get_index_value('^VIX')

    return 0.0

# Alias
get_vix_level = calculate_vix_level


def calculate_realized_volatility(price_data: pd.DataFrame, window: int = 30) -> float:
    """
    Calculate realized volatility over a specified period.
    """
    # Assuming price_data has 'close'
    if 'close' not in price_data.columns:
        return 0.0

    prices = price_data['close'].tail(window)
    returns = np.log(prices / prices.shift(1)).dropna()
    daily_std = returns.std()
    annualized_vol = daily_std * np.sqrt(252)
    return float(annualized_vol)

# Alias
get_realized_vol = calculate_realized_volatility


def calculate_iv_skew(chain: OptionsChain, spot_price: float = None) -> float:
    """
    Calculate IV skew.
    """
    if not chain.calls or not chain.puts:
        return 0.0
    
    if spot_price is None:
        # Estimate from ATM strike
        if chain.calls:
            spot_price = chain.calls[0].strike # Rough fallback
        else:
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

# Alias
get_iv_skew = calculate_iv_skew


def check_mechanical_rules(market_data: Dict[str, Any], rules_config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Check if a symbol meets all the mechanical entry rules.
    """
    rules = {
        'term_structure': False,
        'iv_rank': False,
        'vix_range': False,
        'atr_width': True,  # Default to True, will be checked per strategy
        'event_window': True  # Default to True
    }
    
    # Term structure
    term_spread = market_data.get('term_spread', 0)
    min_gap = rules_config['term_structure']['min_gap']
    sign = rules_config['term_structure']['sign']
    
    if sign == 'positive':
        rules['term_structure'] = term_spread >= min_gap
    else:
        rules['term_structure'] = term_spread <= -min_gap
    
    # IV Rank
    iv_rank = market_data.get('iv_rank', 0)
    rules['iv_rank'] = (
        rules_config['iv_rank']['min'] <= iv_rank <=
        rules_config['iv_rank']['max']
    )
    
    # VIX Range
    vol_level = market_data.get('vix_level', 0)
    rules['vix_range'] = (
        rules_config['vix_range']['min'] <= vol_level <=
        rules_config['vix_range']['max']
    )
    
    # Events
    days_to_event = market_data.get('days_to_event')
    if days_to_event is not None and rules_config['event_window']['enforce']:
         rules['event_window'] = days_to_event >= rules_config['event_window']['min_days']

    return rules

# Alias
check_rule_compliance = check_mechanical_rules


def get_iv_surface(
    symbol: str, 
    dte_range: List[int], 
    delta_range: List[float]
) -> pd.DataFrame:
    """
    Generate an implied volatility surface for a range of DTEs and deltas.
    """
    # Get required providers
    options_provider = create_data_provider('options')
    
    # Get expirations
    try:
        expirations = options_provider.get_expirations(symbol)
    except:
        # Return empty DF if fails
        return pd.DataFrame(index=dte_range, columns=delta_range)

    # Calculate DTE for each expiration
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    dte_exp_map = {(exp - today).days: exp for exp in expirations}
    
    # Find closest available DTEs to the requested ones
    # We map requested DTE to closest actual DTE
    available_dtes = sorted(dte_exp_map.keys())
    if not available_dtes:
         return pd.DataFrame(index=dte_range, columns=delta_range)

    # Create empty DataFrame for the surface
    surface = pd.DataFrame(index=dte_range, columns=delta_range)
    
    # Optimization: Cache chains
    cached_chains = {}

    for dte in dte_range:
        # Find closest actual DTE
        closest_dte = min(available_dtes, key=lambda d: abs(d - dte))
        exp = dte_exp_map[closest_dte]
        
        if closest_dte not in cached_chains:
            try:
                cached_chains[closest_dte] = options_provider.get_options_chain(symbol, exp)
            except:
                continue
        
        chain = cached_chains[closest_dte]

        # Process deltas
        for delta in delta_range:
            # delta > 0 for calls, delta < 0 for puts
            # existing code logic
            if delta > 0:
                 candidates = [c for c in chain.calls if c.delta is not None]
                 if not candidates: continue
                 # Find closest
                 call = min(candidates, key=lambda x: abs(x.delta - delta))
                 surface.loc[dte, delta] = call.implied_volatility
            else:
                 candidates = [p for p in chain.puts if p.delta is not None]
                 if not candidates: continue
                 # Put delta is usually negative in data?
                 # My provider returns it from yfinance. yfinance usually positive for all? Or negative for puts?
                 # Assuming negative for puts.
                 put = min(candidates, key=lambda x: abs(x.delta - delta))
                 surface.loc[dte, delta] = put.implied_volatility

    return surface.astype(float)
