"""
Vertical Spread strategy implementation.

This module implements the Vertical Spread strategy (Bull Put Spread / Bear Call Spread),
a defined-risk strategy where the trader sells an option and buys another option
of the same type with a different strike price.
"""

from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from app.core.strategies.base import (
    Direction, OptionType, Strategy, StrategyCandidate,
    StrategyRegistry, StrategyType, TradeAction, OptionLeg
)


@StrategyRegistry.register
class VerticalSpreadStrategy(Strategy):
    """Vertical Spread strategy implementation (Credit Spreads)."""

    @property
    def name(self) -> str:
        return "Vertical Spread"

    @property
    def type(self) -> StrategyType:
        return StrategyType.BASIC

    @property
    def description(self) -> str:
        return (
            "A defined-risk strategy where you sell an option and buy another option "
            "of the same type with a different strike price. Includes Bull Put Spreads "
            "(bullish) and Bear Call Spreads (bearish)."
        )

    @property
    def directions(self) -> Set[Direction]:
        return {Direction.BULLISH, Direction.BEARISH, Direction.NEUTRAL}

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
        Generate Vertical Spread candidates.

        Args:
            symbol: Ticker symbol
            spot_price: Current spot price
            iv_surface: DataFrame with IVs indexed by DTE and delta
            dte_range: List of DTEs to consider
            delta_targets: Dict mapping leg names to delta targets
                          (sell_delta, buy_delta)
            constraints: Additional constraints (e.g., width)

        Returns:
            List of Vertical Spread strategy candidates
        """
        candidates = []

        # Get delta targets
        # For spreads, we need a short leg and a long leg
        sell_deltas = delta_targets.get('sell_delta', [0.20, 0.30])
        buy_deltas = delta_targets.get('buy_delta', [0.05, 0.10, 0.15])

        # Determine valid widths from constraints or default
        min_width = constraints.get('min_width', 1.0)
        max_width = constraints.get('max_width', 20.0)

        # Loop through DTEs
        for dte in dte_range:
            # 1. Bull Put Spreads (Sell Put, Buy Lower Strike Put)
            # Use negative deltas for puts
            for sell_delta in [-d for d in sell_deltas]:
                for buy_delta in [-d for d in buy_deltas]:
                    # Ensure buy delta is smaller (further OTM) than sell delta (absolute value)
                    if abs(buy_delta) >= abs(sell_delta):
                        continue

                    self._create_spread_candidate(
                        candidates, symbol, spot_price, iv_surface, dte,
                        sell_delta, buy_delta, OptionType.PUT,
                        min_width, max_width, "Bull Put Spread"
                    )

            # 2. Bear Call Spreads (Sell Call, Buy Higher Strike Call)
            for sell_delta in sell_deltas:
                for buy_delta in buy_deltas:
                    # Ensure buy delta is smaller (further OTM) than sell delta
                    if buy_delta >= sell_delta:
                        continue

                    self._create_spread_candidate(
                        candidates, symbol, spot_price, iv_surface, dte,
                        sell_delta, buy_delta, OptionType.CALL,
                        min_width, max_width, "Bear Call Spread"
                    )

        return candidates

    def _create_spread_candidate(
        self,
        candidates: List[StrategyCandidate],
        symbol: str,
        spot_price: float,
        iv_surface: pd.DataFrame,
        dte: int,
        sell_delta: float,
        buy_delta: float,
        option_type: OptionType,
        min_width: float,
        max_width: float,
        subtype_name: str
    ):
        """Helper to create and add a spread candidate."""

        # Check if DTE and deltas are in IV surface
        if dte not in iv_surface.index:
            return

        # Get IVs
        sell_iv = self._get_iv_for_delta(iv_surface, dte, sell_delta)
        buy_iv = self._get_iv_for_delta(iv_surface, dte, buy_delta)

        if pd.isna(sell_iv) or pd.isna(buy_iv):
            return

        # Calculate strikes
        t = dte / 365.0

        sell_strike = self._calculate_strike(spot_price, sell_iv, t, sell_delta, option_type)
        buy_strike = self._calculate_strike(spot_price, buy_iv, t, buy_delta, option_type)

        # Validate width
        width = abs(sell_strike - buy_strike)
        if width < min_width or width > max_width:
            return

        # Estimate prices
        sell_price = self._estimate_option_price(
            spot_price, sell_strike, sell_iv, t, option_type, TradeAction.SELL
        )
        buy_price = self._estimate_option_price(
            spot_price, buy_strike, buy_iv, t, option_type, TradeAction.BUY
        )

        net_credit = sell_price - buy_price

        # Only valid if we receive a credit
        if net_credit <= 0:
            return

        # Create legs
        short_leg = OptionLeg(
            option_type=option_type,
            strike=sell_strike,
            expiration_date=f"DTE_{dte}",
            action=TradeAction.SELL,
            quantity=1,
            delta=sell_delta,
            price=sell_price
        )

        long_leg = OptionLeg(
            option_type=option_type,
            strike=buy_strike,
            expiration_date=f"DTE_{dte}",
            action=TradeAction.BUY,
            quantity=1,
            delta=buy_delta,
            price=buy_price
        )

        # Max loss is width - credit (multiplied by 100)
        max_loss = (width - net_credit) * 100

        # Create candidate
        candidate = StrategyCandidate(
            symbol=symbol,
            strategy_name=f"{self.name} ({subtype_name})",
            legs=[short_leg, long_leg],
            dte_short=dte,
            dte_long=dte,
            estimated_price=net_credit,
            buying_power_effect=max_loss, # Buying power reduction is max loss
        )

        # Calculate POP (simplified: 1 - delta of short leg roughly)
        # Or using the base method
        candidate.probability_of_profit = self.calculate_probability_of_profit(
            candidate, spot_price, (sell_iv + buy_iv)/2, dte
        )

        # Calculate expected return
        er, scenario_table = self.calculate_expected_return(
            candidate, spot_price, (sell_iv + buy_iv)/2, dte
        )
        candidate.expected_return = er
        candidate.scenario_table = scenario_table

        candidate.max_profit = net_credit * 100
        candidate.max_loss = max_loss

        candidate.notes = (
            f"{subtype_name}: Sell {sell_strike:.2f} / Buy {buy_strike:.2f} "
            f"({width:.2f} width). Credit: ${net_credit:.2f}. "
            f"Max Risk: ${max_loss:.2f}."
        )

        candidates.append(candidate)

    def _get_iv_for_delta(self, surface: pd.DataFrame, dte: int, delta: float) -> float:
        """Find the closest delta in the IV surface."""
        if delta in surface.columns:
            return surface.loc[dte, delta]

        # Find closest available delta
        available_deltas = surface.columns
        closest = min(available_deltas, key=lambda x: abs(x - delta))
        return surface.loc[dte, closest]

    def _calculate_strike(
        self, spot: float, iv: float, t: float, delta: float, option_type: OptionType
    ) -> float:
        """Calculate strike from delta."""
        std_dev = spot * iv * np.sqrt(t)

        # Approximation
        if option_type == OptionType.CALL:
            # Delta 0.5 is ATM
            # Delta < 0.5 is OTM (Strike > Spot)
            # 0.5 - 0.16 = 0.34 (1 std dev)
            # Simple linear approx for small deviations
            z_score = (0.5 - delta) * 2.5 # Very rough
            strike = spot * (1 + z_score * iv * np.sqrt(t))
        else:
            # Put delta is negative
            # Delta -0.5 is ATM
            # Delta > -0.5 (e.g. -0.3) is OTM (Strike < Spot)
            z_score = (abs(delta) - 0.5) * 2.5
            strike = spot * (1 + z_score * iv * np.sqrt(t))

        # Round logic
        if spot < 50:
            return round(strike * 2) / 2
        else:
            return round(strike)

    def _estimate_option_price(
        self,
        spot: float,
        strike: float,
        iv: float,
        t: float,
        option_type: OptionType,
        action: TradeAction
    ) -> float:
        """Estimate option price."""
        # Reuse the logic from other strategies or make a shared utility
        if option_type == OptionType.CALL:
            intrinsic = max(0, spot - strike)
        else:
            intrinsic = max(0, strike - spot)

        time_value = spot * iv * np.sqrt(t) * 0.4

        if option_type == OptionType.CALL:
            moneyness = spot / strike - 1
        else:
            moneyness = 1 - spot / strike

        if abs(moneyness) > 0.1:
            time_value *= max(0, 1 - abs(moneyness) * 3)

        price = intrinsic + time_value

        if action == TradeAction.BUY:
            price *= 1.05
        else:
            price *= 0.95

        return price
