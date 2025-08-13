"""
Unit tests for indicators module.
"""

import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from app.core.indicators import (
    calculate_atr,
    get_atm_iv,
    calculate_iv_term_structure,
    calculate_iv_rank,
    calculate_liquidity_metrics,
    get_vix_level,
    get_realized_vol as calculate_realized_volatility,
    get_iv_skew,
    check_rule_compliance as check_mechanical_rules
)


class TestATRCalculation(unittest.TestCase):
    """Test cases for ATR calculation."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample price data
        dates = pd.date_range('2023-01-01', periods=20)
        self.price_data = pd.DataFrame({
            'open': np.linspace(100, 110, 20),
            'high': np.linspace(102, 112, 20),
            'low': np.linspace(99, 109, 20),
            'close': np.linspace(101, 111, 20)
        }, index=dates)
        
        # Add some volatility
        self.price_data['high'] += np.random.uniform(0, 2, 20)
        self.price_data['low'] -= np.random.uniform(0, 2, 20)
    
    def test_atr_calculation(self):
        """Test ATR calculation with default window."""
        atr = calculate_atr(self.price_data)
        
        # Check that result is a float
        self.assertIsInstance(atr, float)
        
        # Check that ATR is positive
        self.assertGreater(atr, 0)
    
    def test_atr_with_custom_window(self):
        """Test ATR calculation with custom window."""
        # Test with window of 5
        atr_5 = calculate_atr(self.price_data, window=5)
        
        # Test with window of 10
        atr_10 = calculate_atr(self.price_data, window=10)
        
        # Check that both results are floats
        self.assertIsInstance(atr_5, float)
        self.assertIsInstance(atr_10, float)
        
        # Both should be positive
        self.assertGreater(atr_5, 0)
        self.assertGreater(atr_10, 0)
    
    def test_atr_insufficient_data(self):
        """Test ATR calculation with insufficient data."""
        # Create price data with only 2 rows
        short_data = self.price_data.iloc[:2]
        
        # Calculate ATR with window of 14 (default)
        # This should return nan or a reasonable approximation
        atr = calculate_atr(short_data)
        
        # Check that result is a float (even if it's nan)
        self.assertIsInstance(atr, float)








if __name__ == '__main__':
    unittest.main()
