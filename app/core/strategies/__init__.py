"""
Strategy module initialization.
"""

from app.core.strategies.base import Strategy, StrategyRegistry
from app.core.strategies.cash_secured_put import CashSecuredPutStrategy
from app.core.strategies.covered_call import CoveredCallStrategy
from app.core.strategies.vertical_spread import VerticalSpreadStrategy
