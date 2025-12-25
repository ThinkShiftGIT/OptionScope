"""
Cash-Secured Put strategy implementation.

This module implements the Cash-Secured Put (CSP) strategy, a basic income strategy
where the trader sells a put option and sets aside enough cash to buy the underlying
at the strike price if assigned.
"""

from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

from app.core.strategies.base import (
    Direction, OptionType, Strategy, StrategyCandidate, 
    StrategyRegistry, StrategyType, TradeAction, OptionLeg
)


@StrategyRegistry.register
class CashSecuredPutStrategy(Strategy):
    """Cash-Secured Put strategy implementation."""
    
    @property
    def name(self) -> str:
        return "Cash-Secured Put"
    
    @property
    def type(self) -> StrategyType:
        return StrategyType.BASIC
    
    @property
    def description(self) -> str:
        return (
            "A strategy where you sell a put option and set aside enough cash to "
            "buy shares at the strike price if assigned. Used to generate income "
            "and potentially acquire shares at a lower price."
        )
    
    @property
    def directions(self) -> Set[Direction]:
        return {Direction.BULLISH, Direction.NEUTRAL}
    
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
        Generate Cash-Secured Put candidates.
        
        Args:
            symbol: Ticker symbol
            spot_price: Current spot price
            iv_surface: DataFrame with IVs indexed by DTE and delta
            dte_range: List of DTEs to consider
            delta_targets: Dict mapping leg names to delta targets 
                          (for CSP, should contain 'put_delta')
            constraints: Additional constraints for strategy generation
            
        Returns:
            List of CSP strategy candidates
        """
        candidates = []
        
        # Get put delta targets
        put_deltas = delta_targets.get('put_delta', [-0.20, -0.25, -0.30])
        
        # For each DTE and delta combination, create a candidate
        for dte in dte_range:
            for delta in put_deltas:
                # Skip if this DTE or delta is not in our IV surface
                if dte not in iv_surface.index:
                    continue
                
                # Get IV for this DTE and delta
                iv = self._get_iv_for_delta(iv_surface, dte, delta)
                if pd.isna(iv):
                    continue
                
                # Calculate strike based on delta (approximate)
                t = dte / 365.0  # Convert DTE to years
                std_dev = spot_price * iv * np.sqrt(t)
                
                # Approximation: for puts, each 0.1 delta ~= 0.25 std_dev away from ATM
                # This is a rough approx, Black-Scholes inversion would be better
                # But for this simulation, we use a simple Z-score approximation
                # Delta -0.5 is ATM. Delta -0.2 is OTM (Strike < Spot).
                # N^-1(0.2) is approx -0.84.
                # So Z score is approx N^-1(|delta|) if we assume simplified model
                # But put delta is N(d1) - 1.
                # Let's stick to the existing simple approximation for now but improve it slightly

                # Z-score approximation
                # For Put delta = N(d1) - 1. So N(d1) = 1 + delta.
                # If delta is -0.2, N(d1) = 0.8. Z ~ 0.84.
                # Strike = Spot * exp(-Z * sigma * sqrt(T)) roughly
                # But here z_score is derived from delta where delta = N(d1) - 1.
                # If delta = -0.2, 1+delta = 0.8. norm.ppf(0.8) ~= 0.84 (positive).
                # Since we want OTM put (Strike < Spot), we need a negative exponent if formula is Spot * exp(Z...).
                # So we should use a negative sign if z_score is positive.
                # Or just realize z_score represents distance from mean. OTM put is "below" mean.

                from scipy.stats import norm
                try:
                    # For Put, N(d1) - 1 = delta  => N(d1) = 1 + delta
                    # d1 = norm.ppf(1 + delta)
                    # d1 approx (ln(S/K) + ...) / sigma*sqrt(t)
                    # if delta is -0.2 (OTM), d1 is > 0 (0.84). This implies S > K.
                    # ln(S/K) > 0 => S/K > 1 => S > K. Correct.
                    # So Strike K = S / exp(d1 * sigma * sqrt(t)) approx?
                    # If K = S * exp(Z), then Z must be negative for OTM put.
                    # Let's just force OTM logic: Strike < Spot.
                    # If z_score comes out positive (0.84), we negate it to get Strike < Spot.

                    z_score = norm.ppf(1 + delta)
                    if z_score > 0:
                        z_score = -z_score
                except:
                    z_score = -1.0

                strike = spot_price * np.exp(z_score * iv * np.sqrt(t))
                
                # Round strike to nearest 0.5 or 1 depending on price level
                if spot_price < 50:
                    strike = round(strike * 2) / 2  # Round to nearest 0.5
                else:
                    strike = round(strike)  # Round to nearest 1
                
                # Calculate option price (rough approximation)
                # In real implementation, we would use actual option chain data
                option_price = self._estimate_option_price(
                    spot_price, strike, iv, t, OptionType.PUT, TradeAction.SELL
                )
                
                # Create option leg
                leg = OptionLeg(
                    option_type=OptionType.PUT,
                    strike=strike,
                    expiration_date=f"DTE_{dte}",  # Placeholder
                    action=TradeAction.SELL,
                    quantity=1,
                    delta=delta,
                    price=option_price
                )
                
                # Create candidate
                candidate = StrategyCandidate(
                    symbol=symbol,
                    strategy_name=self.name,
                    legs=[leg],
                    dte_short=dte,
                    estimated_price=option_price,  # Credit for selling put
                    buying_power_effect=strike * 100,  # Full strike price in cash
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
                candidate.max_profit = option_price * 100  # Premium received
                candidate.max_loss = (strike - option_price) * 100  # Strike minus premium
                
                # Add notes
                candidate.notes = (
                    f"CSP at {strike:.2f} strike ({abs(delta)*100:.1f}% delta), "
                    f"{dte} DTE. Premium: ${option_price:.2f} per share."
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
