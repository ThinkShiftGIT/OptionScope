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
    calculate_atm_iv,
    calculate_iv_term_structure,
    calculate_iv_rank,
    calculate_liquidity_metrics,
    calculate_vix_level,
    calculate_realized_volatility,
    calculate_iv_skew,
    check_mechanical_rules
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


class TestIVCalculations(unittest.TestCase):
    """Test cases for IV-related calculations."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample options chain with two expirations
        self.options_chain = {
            '2023-02-17': {
                'calls': pd.DataFrame({
                    'strike': [95, 100, 105, 110],
                    'bid': [6.0, 3.0, 1.0, 0.5],
                    'ask': [7.0, 4.0, 2.0, 1.0],
                    'impliedVolatility': [0.25, 0.22, 0.20, 0.19]
                }),
                'puts': pd.DataFrame({
                    'strike': [95, 100, 105, 110],
                    'bid': [0.5, 1.0, 3.0, 6.0],
                    'ask': [1.0, 2.0, 4.0, 7.0],
                    'impliedVolatility': [0.24, 0.23, 0.25, 0.28]
                })
            },
            '2023-03-17': {
                'calls': pd.DataFrame({
                    'strike': [95, 100, 105, 110],
                    'bid': [7.0, 4.0, 2.0, 1.0],
                    'ask': [8.0, 5.0, 3.0, 2.0],
                    'impliedVolatility': [0.28, 0.25, 0.23, 0.22]
                }),
                'puts': pd.DataFrame({
                    'strike': [95, 100, 105, 110],
                    'bid': [1.0, 2.0, 4.0, 7.0],
                    'ask': [2.0, 3.0, 5.0, 8.0],
                    'impliedVolatility': [0.27, 0.26, 0.28, 0.32]
                })
            }
        }
        
        # Create sample IV history
        dates = pd.date_range('2022-01-01', periods=252)
        self.iv_history = pd.Series(
            np.random.uniform(0.15, 0.35, 252),
            index=dates
        )
    
    def test_atm_iv_calculation(self):
        """Test ATM IV calculation."""
        # Test with spot price of 100 (ATM)
        atm_iv = calculate_atm_iv(self.options_chain, 100)
        
        # Check that result is a float
        self.assertIsInstance(atm_iv, float)
        
        # Check that IV is positive and reasonable
        self.assertGreater(atm_iv, 0)
        self.assertLess(atm_iv, 100)  # Should be a percentage
    
    def test_iv_term_structure(self):
        """Test IV term structure calculation."""
        # Calculate IV term structure
        term_structure = calculate_iv_term_structure(self.options_chain, 100)
        
        # Check that result is a Series
        self.assertIsInstance(term_structure, pd.Series)
        
        # Check that we have two expiration dates
        self.assertEqual(len(term_structure), 2)
        
        # Check that all values are positive
        self.assertTrue((term_structure > 0).all())
    
    def test_iv_rank(self):
        """Test IV rank calculation."""
        # Calculate IV rank
        iv_rank = calculate_iv_rank(self.iv_history, current_iv=0.25)
        
        # Check that result is a float
        self.assertIsInstance(iv_rank, float)
        
        # Check that IV rank is between 0 and 100
        self.assertGreaterEqual(iv_rank, 0)
        self.assertLessEqual(iv_rank, 100)
    
    def test_iv_skew(self):
        """Test IV skew calculation."""
        # Calculate IV skew
        iv_skew = calculate_iv_skew(self.options_chain)
        
        # Check that result is a float
        self.assertIsInstance(iv_skew, float)
    
    def test_empty_options_chain(self):
        """Test handling of empty options chain."""
        # Create an empty options chain
        empty_chain = {}
        
        # Calculate ATM IV
        atm_iv = calculate_atm_iv(empty_chain, 100)
        
        # Should return None or a default value
        self.assertIsNone(atm_iv)
        
        # Calculate IV term structure
        term_structure = calculate_iv_term_structure(empty_chain, 100)
        
        # Should return an empty Series
        self.assertIsInstance(term_structure, pd.Series)
        self.assertTrue(term_structure.empty)


class TestLiquidityMetrics(unittest.TestCase):
    """Test cases for liquidity metrics."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample options chain
        self.options_chain = {
            '2023-02-17': {
                'calls': pd.DataFrame({
                    'strike': [95, 100, 105, 110],
                    'bid': [6.0, 3.0, 1.0, 0.5],
                    'ask': [7.0, 4.0, 2.0, 1.0],
                    'volume': [500, 1000, 800, 200],
                    'openInterest': [2000, 5000, 3000, 1000]
                }),
                'puts': pd.DataFrame({
                    'strike': [95, 100, 105, 110],
                    'bid': [0.5, 1.0, 3.0, 6.0],
                    'ask': [1.0, 2.0, 4.0, 7.0],
                    'volume': [300, 800, 1200, 400],
                    'openInterest': [1500, 4000, 6000, 2000]
                })
            }
        }
    
    def test_liquidity_metrics(self):
        """Test calculation of liquidity metrics."""
        # Calculate liquidity metrics
        metrics = calculate_liquidity_metrics(self.options_chain)
        
        # Check that we have the expected keys
        self.assertIn('avg_spread_pct', metrics)
        self.assertIn('avg_open_interest', metrics)
        self.assertIn('avg_volume', metrics)
        
        # Check that all metrics are non-negative
        self.assertGreaterEqual(metrics['avg_spread_pct'], 0)
        self.assertGreaterEqual(metrics['avg_open_interest'], 0)
        self.assertGreaterEqual(metrics['avg_volume'], 0)
    
    def test_empty_options_chain(self):
        """Test handling of empty options chain."""
        # Create an empty options chain
        empty_chain = {}
        
        # Calculate liquidity metrics
        metrics = calculate_liquidity_metrics(empty_chain)
        
        # Check that we have the expected keys with default values
        self.assertIn('avg_spread_pct', metrics)
        self.assertIn('avg_open_interest', metrics)
        self.assertIn('avg_volume', metrics)
        
        # Default values should be reasonable
        self.assertEqual(metrics['avg_spread_pct'], float('inf'))  # Worst case
        self.assertEqual(metrics['avg_open_interest'], 0)
        self.assertEqual(metrics['avg_volume'], 0)


class TestMechanicalRules(unittest.TestCase):
    """Test cases for mechanical rules checking."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample market data
        self.market_data = {
            'iv_rank': 40.0,
            'term_spread': 3.0,
            'vix_level': 20.0,
            'days_to_event': 15,
            'atr': 2.0
        }
        
        # Create sample rules
        self.rules = {
            'iv_rank': {'min': 30, 'max': 60},
            'term_structure': {'min_gap': 2.0},
            'vix_range': {'min': 15, 'max': 30},
            'event_window': {'min_days': 10},
            'atr_width': {'max_ratio': 0.5}
        }
    
    def test_all_rules_pass(self):
        """Test when all rules should pass."""
        # Check rules
        result = check_mechanical_rules(self.market_data, self.rules)
        
        # All rules should pass
        for rule_name, passed in result.items():
            self.assertTrue(passed, f"Rule {rule_name} should have passed")
    
    def test_iv_rank_fail(self):
        """Test when IV rank rule fails."""
        # Modify market data to make IV rank rule fail
        self.market_data['iv_rank'] = 20.0  # Below min
        
        # Check rules
        result = check_mechanical_rules(self.market_data, self.rules)
        
        # IV rank rule should fail
        self.assertFalse(result['iv_rank'])
        
        # Other rules should still pass
        self.assertTrue(result['term_structure'])
        self.assertTrue(result['vix_range'])
        self.assertTrue(result['event_window'])
    
    def test_term_structure_fail(self):
        """Test when term structure rule fails."""
        # Modify market data to make term structure rule fail
        self.market_data['term_spread'] = 1.0  # Below min_gap
        
        # Check rules
        result = check_mechanical_rules(self.market_data, self.rules)
        
        # Term structure rule should fail
        self.assertFalse(result['term_structure'])
    
    def test_event_window_fail(self):
        """Test when event window rule fails."""
        # Modify market data to make event window rule fail
        self.market_data['days_to_event'] = 5  # Below min_days
        
        # Check rules
        result = check_mechanical_rules(self.market_data, self.rules)
        
        # Event window rule should fail
        self.assertFalse(result['event_window'])
    
    def test_missing_market_data(self):
        """Test handling of missing market data."""
        # Remove a key from market data
        incomplete_data = self.market_data.copy()
        del incomplete_data['iv_rank']
        
        # Check rules
        result = check_mechanical_rules(incomplete_data, self.rules)
        
        # IV rank rule should fail or return false
        self.assertFalse(result['iv_rank'])
        
        # Other rules should still be checked
        self.assertIn('term_structure', result)
        self.assertIn('vix_range', result)
        self.assertIn('event_window', result)


if __name__ == '__main__':
    unittest.main()
