"""
Covered Call strategy implementation.

This module implements the Covered Call strategy, a basic income strategy
where the trader owns shares of the underlying and sells call options against them.
"""

from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
from scipy.stats import norm

from app.core.strategies.base import (
    Direction, OptionType, Strategy, StrategyCandidate, 
    StrategyRegistry, StrategyType, TradeAction, OptionLeg
)


@StrategyRegistry.register
class CoveredCallStrategy(Strategy):
    """Covered Call strategy implementation."""
    
    @property
    def name(self) -> str:
        return "Covered Call"
    
    @property
    def type(self) -> StrategyType:
        return StrategyType.BASIC
    
    @property
    def description(self) -> str:
        return (
            "A strategy where you own 100 shares of stock and sell a call option "
            "against those shares. Used to generate income on existing stock positions "
            "and potentially sell shares at a higher price."
        )
    
    @property
    def directions(self) -> Set[Direction]:
        return {Direction.NEUTRAL, Direction.BEARISH}
    
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
        Generate Covered Call candidates.
        
        Args:
            symbol: Ticker symbol
            spot_price: Current spot price
            iv_surface: DataFrame with IVs indexed by DTE and delta
            dte_range: List of DTEs to consider
            delta_targets: Dict mapping leg names to delta targets 
                          (for Covered Call, should contain 'call_delta')
            constraints: Additional constraints for strategy generation
            
        Returns:
            List of Covered Call strategy candidates
        """
        candidates = []
        
        # Get call delta targets
        call_deltas = delta_targets.get('call_delta', [0.20, 0.25, 0.30])
        
        # For each DTE and delta combination, create a candidate
        for dte in dte_range:
            for delta in call_deltas:
                # Skip if this DTE or delta is not in our IV surface
                if dte not in iv_surface.index:
                    continue
                
                # Get IV for this DTE and delta
                iv = self._get_iv_for_delta(iv_surface, dte, delta)
                if pd.isna(iv):
                    continue
                
                # Calculate strike based on delta (approximate)
                t = dte / 365.0  # Convert DTE to years
                
                # Z-score approximation
                try:
                    z_score = norm.ppf(delta) # for Call OTM delta is < 0.5 usually?
                    # If delta is 0.2 (20 delta call), it is OTM. N(d1) = 0.2? No.
                    # N(d1) is delta. OTM call has low delta.
                    # If delta is 0.2, then Z is roughly -0.84.
                    # But for Call Strike > Spot.
                    # Wait, N(d1) is probability of finishing ITM roughly.
                    # If N(d1) = 0.2, then Strike is above Spot.
                    # So Z score should be positive?
                    # d1 = (ln(S/K) + ...) / sigma sqrt(T).
                    # If S < K, ln(S/K) is negative.
                    # So d1 is negative.
                    # N(d1) < 0.5.

                    # My previous logic for CSP was: N(d1) - 1 = -0.2 => N(d1) = 0.8 => d1 > 0 => S > K. Put is OTM. Correct.

                    # For Call: N(d1) = 0.2 => d1 < 0 => S < K. Call is OTM. Correct.
                    # So if delta is 0.2, z_score is negative.
                    # Strike = Spot * exp(-sigma * sqrt(T) * z_score).
                    # Actually d1 = (ln(S/K) + 0.5v^2T) / v sqrt(T)
                    # z_score = d1
                    # z_score * v sqrt(T) = ln(S/K) + ...
                    # ln(K/S) = -z_score * v sqrt(T) ...
                    # K = S * exp(...)
                    pass
                except:
                    pass

                # Approximation: Strike = Spot * (1 + sigma * sqrt(T) * inverse_normal(1-delta))
                # If delta 0.20, we want strike s.t. Prob(ITM) = 0.20.
                # It means it's on the right tail.
                # No, standard delta for calls is 0 to 1. 0.20 delta is OTM.
                # So we want strike > spot.
                # The 'z-score' distance from mean is roughly norm.ppf(1-delta).
                # e.g. delta 0.16 => z approx 1.
                # Strike approx Spot * (1 + iv * sqrt(t))

                z_dist = norm.ppf(1 - delta)
                strike = spot_price * (1 + z_dist * iv * np.sqrt(t))
                
                # Round strike to nearest 0.5 or 1 depending on price level
                if spot_price < 50:
                    strike = round(strike * 2) / 2  # Round to nearest 0.5
                else:
                    strike = round(strike)  # Round to nearest 1
                
                # Calculate option price (rough approximation)
                # In real implementation, we would use actual option chain data
                option_price = self._estimate_option_price(
                    spot_price, strike, iv, t, OptionType.CALL, TradeAction.SELL
                )
                
                # Create option leg for the short call
                option_leg = OptionLeg(
                    option_type=OptionType.CALL,
                    strike=strike,
                    expiration_date=f"DTE_{dte}",  # Placeholder
                    action=TradeAction.SELL,
                    quantity=1,
                    delta=delta,
                    price=option_price
                )
                
                # Create stock leg
                stock_leg = OptionLeg(
                    option_type=OptionType.CALL,  # Not used for stock
                    strike=0,  # Not used for stock
                    expiration_date="",  # Not used for stock
                    action=TradeAction.BUY,  # Long stock
                    quantity=100,  # 100 shares
                    delta=1.0,  # Stock has delta of 1
                    price=spot_price
                )
                
                # Create candidate
                candidate = StrategyCandidate(
                    symbol=symbol,
                    strategy_name=self.name,
                    legs=[stock_leg, option_leg],
                    dte_short=dte,
                    estimated_price=(spot_price * 100) - (option_price * 100),  # Stock cost - option premium
                    buying_power_effect=spot_price * 100,  # Full cost of shares
                )
                
                # Calculate probability of profit
                candidate.probability_of_profit = self.calculate_probability_of_profit(
                    candidate, spot_price, iv, dte
                )
                
                # Calculate expected return
                er, scenario_table = self.calculate_expected_return(
                    candidate, spot_price, iv, dte
                )
                candidate.expected_return = er
                candidate.scenario_table = scenario_table
                
                # Calculate max profit and loss
                candidate.max_profit = ((strike - spot_price) + option_price) * 100  # Strike gain + premium
                candidate.max_loss = (spot_price - option_price) * 100  # Stock could go to zero
                
                # Add notes
                candidate.notes = (
                    f"Covered Call with 100 shares at ${spot_price:.2f} and short call at "
                    f"${strike:.2f} strike ({delta*100:.1f}% delta), {dte} DTE. "
                    f"Call premium: ${option_price:.2f} per share."
                )
                
                candidates.append(candidate)
        
        return candidates
    
    def _get_iv_for_delta(self, surface: pd.DataFrame, dte: int, delta: float) -> float:
        """Find the closest delta in the IV surface."""
        if delta in surface.columns:
            return surface.loc[dte, delta]

        # Find closest available delta
        available_deltas = surface.columns
        closest = min(available_deltas, key=lambda x: abs(x - delta))
        return surface.loc[dte, closest]

    def _estimate_option_price(
        self,
        spot: float,
        strike: float,
        iv: float,
        t: float,
        option_type: OptionType,
        action: TradeAction
    ) -> float:
        """
        Estimate option price using a simple model.
        
        This is a simplified approximation for demonstration purposes.
        In a real system, we would get actual prices from the options chain.
        
        Args:
            spot: Spot price
            strike: Strike price
            iv: Implied volatility
            t: Time to expiration in years
            option_type: Call or Put
            action: Buy or Sell
            
        Returns:
            Estimated option price
        """
        # Simple approximation based on intrinsic + time value
        if option_type == OptionType.CALL:
            intrinsic = max(0, spot - strike)
        else:  # PUT
            intrinsic = max(0, strike - spot)
        
        # Time value approximation (very simplified)
        time_value = spot * iv * np.sqrt(t) * 0.4  # Rough heuristic
        
        # Adjust for deep ITM or OTM
        if option_type == OptionType.CALL:
            moneyness = spot / strike - 1
        else:  # PUT
            moneyness = 1 - spot / strike
        
        # Reduce time value for deep ITM/OTM options
        if abs(moneyness) > 0.1:
            time_value *= max(0, 1 - abs(moneyness) * 3)
        
        price = intrinsic + time_value
        
        # Add a small bid-ask spread
        if action == TradeAction.BUY:
            price *= 1.05  # Pay slightly more when buying
        else:  # SELL
            price *= 0.95  # Receive slightly less when selling
        
        return price
