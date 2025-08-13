"""
Covered Call strategy implementation.

This module implements the Covered Call strategy, a basic income strategy
where the trader owns shares of the underlying and sells call options against them.
"""

from typing import Any, Dict, List, Set
from datetime import datetime

import numpy as np
import pandas as pd

from app.core.strategies.base import (
    Direction, OptionType, Strategy, StrategyCandidate,
    StrategyRegistry, StrategyType, TradeAction, OptionLeg
)
from app.core.data_providers import OptionsChain
from app.utils.config import get_config


@StrategyRegistry.register
class CoveredCallStrategy(Strategy):
    """Covered Call strategy implementation."""
    
    name: str = "Covered Call"
    type: StrategyType = StrategyType.BASIC
    description: str = (
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
        options_chain: Dict[str, OptionsChain],
        risk_free_rate: float,
        iv: float,
    ) -> List[StrategyCandidate]:
        """
        Generate Covered Call candidates.
        """
        config = get_config()
        filters = config['strategy_filters']['covered_call']
        delta_range = filters['delta_range']
        dte_range = filters['dte_range']

        candidates = []

        for exp_str, chain in options_chain.items():
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
            dte = (exp_date - datetime.now()).days

            if not (dte_range[0] <= dte <= dte_range[1]):
                continue
                
            for call in chain.calls:
                if call.delta is None or call.open_interest is None:
                    continue

                if not (delta_range[0] <= call.delta <= delta_range[1]):
                    continue

                # This option is a potential candidate
                leg = OptionLeg(
                    option_type=OptionType.CALL,
                    strike=call.strike,
                    expiration_date=exp_str,
                    action=TradeAction.SELL,
                    quantity=1,
                    delta=call.delta,
                    price=call.mid, # Use mid price
                )

                candidate = StrategyCandidate(
                    symbol=symbol,
                    strategy_name=self.name,
                    legs=[leg], # In a real CC, we'd also have a stock leg
                    dte_short=dte,
                    estimated_price=call.mid, # Credit for selling
                    # BP for a true covered call is 0 if you own the stock.
                    # This assumes you are buying the stock.
                    buying_power_effect=(call.strike * 100) - (call.mid * 100),
                    delta=call.delta - 1, # -1 for the 100 shares of stock
                    gamma=call.gamma,
                    theta=call.theta,
                    vega=call.vega,
                    avg_spread_pct=(call.ask - call.bid) / call.mid if call.mid > 0 else 0,
                    avg_open_interest=call.open_interest,
                )

                # Calculate max profit and loss
                # Max profit is strike - spot + premium
                candidate.max_profit = ((call.strike - spot_price) * 100) + (call.mid * 100)
                # Max loss is spot - premium (if stock goes to 0)
                candidate.max_loss = (spot_price * 100) - (call.mid * 100)

                # Simplified PoP
                candidate.probability_of_profit = (1 - call.delta) * 100

                # Simplified Expected Return
                prob_win = candidate.probability_of_profit / 100
                prob_lose = 1 - prob_win
                er = (prob_win * candidate.max_profit) - (prob_lose * candidate.max_loss)
                candidate.expected_return = (er / candidate.buying_power_effect) * 100 if candidate.buying_power_effect > 0 else 0

                candidates.append(candidate)

        return candidates
