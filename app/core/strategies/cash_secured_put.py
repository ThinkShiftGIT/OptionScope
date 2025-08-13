"""
Cash-Secured Put strategy implementation.

This module implements the Cash-Secured Put (CSP) strategy, a basic income strategy
where the trader sells a put option and sets aside enough cash to buy the underlying
at the strike price if assigned.
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
class CashSecuredPutStrategy(Strategy):
    """Cash-Secured Put strategy implementation."""
    
    name: str = "Cash-Secured Put"
    type: StrategyType = StrategyType.BASIC
    description: str = (
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
        options_chain: Dict[str, OptionsChain],
        risk_free_rate: float,
        iv: float,
    ) -> List[StrategyCandidate]:
        """
        Generate Cash-Secured Put candidates.
        """
        config = get_config()
        filters = config['strategy_filters']['cash_secured_put']
        delta_range = filters['delta_range']
        dte_range = filters['dte_range']
        
        candidates = []

        for exp_str, chain in options_chain.items():
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
            dte = (exp_date - datetime.now()).days

            if not (dte_range[0] <= dte <= dte_range[1]):
                continue
                
            for put in chain.puts:
                if put.delta is None or put.open_interest is None:
                    continue

                if not (delta_range[0] <= put.delta <= delta_range[1]):
                    continue

                # This option is a potential candidate
                leg = OptionLeg(
                    option_type=OptionType.PUT,
                    strike=put.strike,
                    expiration_date=exp_str,
                    action=TradeAction.SELL,
                    quantity=1,
                    delta=put.delta,
                    price=put.mid, # Use mid price
                )

                candidate = StrategyCandidate(
                    symbol=symbol,
                    strategy_name=self.name,
                    legs=[leg],
                    dte_short=dte,
                    estimated_price=put.mid, # Credit for selling
                    buying_power_effect=put.strike * 100, # Full strike price in cash
                    delta=put.delta,
                    gamma=put.gamma,
                    theta=put.theta,
                    vega=put.vega,
                    avg_spread_pct=(put.ask - put.bid) / put.mid if put.mid > 0 else 0,
                    avg_open_interest=put.open_interest,
                )

                # Calculate max profit and loss
                candidate.max_profit = put.mid * 100
                candidate.max_loss = (put.strike * 100) - candidate.max_profit

                # Simplified PoP using delta
                candidate.probability_of_profit = (1 - abs(put.delta)) * 100

                # Simplified Expected Return
                prob_win = candidate.probability_of_profit / 100
                prob_lose = 1 - prob_win
                er = (prob_win * candidate.max_profit) - (prob_lose * candidate.max_loss)
                candidate.expected_return = (er / candidate.buying_power_effect) * 100 if candidate.buying_power_effect > 0 else 0

                candidates.append(candidate)

        return candidates
