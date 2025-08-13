"""
Unit tests for risk module.
"""

import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from app.core.risk import (
    calculate_max_loss,
    calculate_max_loss_percentage,
    generate_price_scenario_table,
    calculate_capital_efficiency,
    calculate_liquidity_penalty,
    calculate_event_penalty
)
from app.core.strategies.base import StrategyCandidate, OptionLeg, TradeAction, OptionType


class TestMaxLossCalculations(unittest.TestCase):
    """Test cases for maximum loss calculations."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample strategy candidate
        self.candidate = StrategyCandidate(
            strategy_name="Test Strategy",
            symbol="AAPL",
            spot_price=150.0,
            dte_short=30,
            max_loss=1000.0,
            max_profit=500.0,
            buying_power_effect=2000.0,
            probability_of_profit=0.6,
            expected_return=5.0,
            legs=[
                OptionLeg(
                    action=TradeAction.BUY,
                    option_type=OptionType.CALL,
                    strike=155.0,
                    expiration_date="2023-06-15",
                    quantity=1,
                    price=3.0,
                    delta=0.4,
                    gamma=0.02,
                    theta=-0.1,
                    vega=0.2
                ),
                OptionLeg(
                    action=TradeAction.SELL,
                    option_type=OptionType.CALL,
                    strike=160.0,
                    expiration_date="2023-06-15",
                    quantity=1,
                    price=1.5,
                    delta=0.3,
                    gamma=0.015,
                    theta=-0.08,
                    vega=0.15
                )
            ],
            delta=0.1,
            gamma=0.005,
            theta=-0.02,
            vega=0.05,
            avg_spread_pct=0.03,
            avg_open_interest=500,
            composite_score=75.0,
            estimated_price=1.5,
            notes="Test strategy candidate"
        )
    
    def test_calculate_max_loss_with_predefined_value(self):
        """Test max loss calculation when the candidate has a predefined max_loss."""
        max_loss = calculate_max_loss(self.candidate)
        
        # Should return the predefined value
        self.assertEqual(max_loss, 1000.0)
    
    def test_calculate_max_loss_without_predefined_value(self):
        """Test max loss calculation when the candidate has no predefined max_loss."""
        # Set max_loss to 0 to trigger calculation
        self.candidate.max_loss = 0
        
        max_loss = calculate_max_loss(self.candidate)
        
        # Should use buying_power_effect as a fallback
        self.assertEqual(max_loss, 2000.0)
    
    def test_calculate_max_loss_percentage(self):
        """Test max loss percentage calculation."""
        # Test with account equity of 10000
        account_equity = 10000.0
        
        max_loss_pct = calculate_max_loss_percentage(self.candidate, account_equity)
        
        # Expected: (1000 / 10000) * 100 = 10%
        self.assertEqual(max_loss_pct, 10.0)
    
    def test_calculate_max_loss_percentage_zero_equity(self):
        """Test max loss percentage calculation with zero account equity."""
        # This should handle division by zero gracefully
        account_equity = 0.0
        
        with self.assertRaises(ZeroDivisionError):
            calculate_max_loss_percentage(self.candidate, account_equity)


class TestScenarioAnalysis(unittest.TestCase):
    """Test cases for scenario analysis."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample strategy candidate
        self.candidate = StrategyCandidate(
            strategy_name="Test Strategy",
            symbol="AAPL",
            spot_price=150.0,
            dte_short=30,
            max_loss=1000.0,
            max_profit=500.0,
            buying_power_effect=2000.0,
            probability_of_profit=0.6,
            expected_return=5.0,
            legs=[],
            delta=0.5,
            gamma=0.02,
            theta=-0.1,
            vega=0.2
        )
    
    def test_generate_price_scenario_table(self):
        """Test generation of price scenario table."""
        # Generate price scenario table
        scenario_table = generate_price_scenario_table(
            candidate=self.candidate,
            spot_price=150.0,
            iv=0.3,
            days_to_expiration=30
        )
        
        # Check that result is a DataFrame
        self.assertIsInstance(scenario_table, pd.DataFrame)
        
        # Check that it has the expected columns
        expected_columns = [
            'Scenario', 'Move %', 'P/L $', 'P/L %', 'IV Change'
        ]
        for col in expected_columns:
            self.assertIn(col, scenario_table.columns)
        
        # Check that it has at least 7 rows (price scenarios)
        self.assertGreaterEqual(len(scenario_table), 7)
    
    def test_generate_price_scenario_table_with_custom_price_points(self):
        """Test generation of price scenario table with custom price points."""
        # Define custom price points
        price_points = [120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0]
        
        # Generate price scenario table
        scenario_table = generate_price_scenario_table(
            candidate=self.candidate,
            spot_price=150.0,
            iv=0.3,
            days_to_expiration=30,
            price_points=price_points
        )
        
        # Check that result is a DataFrame
        self.assertIsInstance(scenario_table, pd.DataFrame)
        
        # Check that it has the correct number of rows for price scenarios
        # (plus additional rows for IV scenarios)
        self.assertGreaterEqual(len(scenario_table), len(price_points))
    
    def test_generate_price_scenario_table_without_iv_scenarios(self):
        """Test generation of price scenario table without IV scenarios."""
        # Generate price scenario table without IV scenarios
        scenario_table = generate_price_scenario_table(
            candidate=self.candidate,
            spot_price=150.0,
            iv=0.3,
            days_to_expiration=30,
            include_iv_scenarios=False
        )
        
        # Check that result is a DataFrame
        self.assertIsInstance(scenario_table, pd.DataFrame)
        
        # Check that all IV Change values are 0
        self.assertTrue((scenario_table['IV Change'] == 0).all())


class TestRiskMetrics(unittest.TestCase):
    """Test cases for risk metrics."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample strategy candidates with different characteristics
        
        # Candidate with good capital efficiency
        self.good_capital_candidate = StrategyCandidate(
            strategy_name="Good Capital Efficiency",
            symbol="AAPL",
            spot_price=150.0,
            dte_short=30,
            max_loss=1000.0,
            max_profit=500.0,
            buying_power_effect=2000.0,
            probability_of_profit=0.6,
            expected_return=5.0,
            legs=[],
            theta=-0.2,  # Good theta decay
            avg_spread_pct=0.01,  # Tight spreads
            avg_open_interest=2000  # Good open interest
        )
        
        # Candidate with poor liquidity
        self.poor_liquidity_candidate = StrategyCandidate(
            strategy_name="Poor Liquidity",
            symbol="AAPL",
            spot_price=150.0,
            dte_short=30,
            max_loss=1000.0,
            max_profit=500.0,
            buying_power_effect=2000.0,
            probability_of_profit=0.6,
            expected_return=5.0,
            legs=[],
            theta=-0.1,
            avg_spread_pct=0.08,  # Wide spreads
            avg_open_interest=50  # Low open interest
        )
    
    def test_calculate_capital_efficiency(self):
        """Test calculation of capital efficiency."""
        # Calculate capital efficiency for good candidate
        good_efficiency = calculate_capital_efficiency(self.good_capital_candidate)
        
        # Calculate capital efficiency for regular candidate
        poor_efficiency = calculate_capital_efficiency(self.poor_liquidity_candidate)
        
        # Good candidate should have better capital efficiency
        self.assertGreater(good_efficiency, poor_efficiency)
        
        # Both should be positive
        self.assertGreaterEqual(good_efficiency, 0)
        self.assertGreaterEqual(poor_efficiency, 0)
    
    def test_calculate_liquidity_penalty(self):
        """Test calculation of liquidity penalty."""
        # Calculate liquidity penalty for good liquidity
        good_penalty = calculate_liquidity_penalty(self.good_capital_candidate)
        
        # Calculate liquidity penalty for poor liquidity
        poor_penalty = calculate_liquidity_penalty(self.poor_liquidity_candidate)
        
        # Poor liquidity should have a higher penalty
        self.assertGreater(poor_penalty, good_penalty)
        
        # Both should be between 0 and 100
        self.assertGreaterEqual(good_penalty, 0)
        self.assertLessEqual(good_penalty, 100)
        self.assertGreaterEqual(poor_penalty, 0)
        self.assertLessEqual(poor_penalty, 100)
    
    def test_calculate_event_penalty(self):
        """Test calculation of event penalty."""
        # Create a candidate with expiration close to event
        event_risk_candidate = StrategyCandidate(
            strategy_name="Event Risk",
            symbol="AAPL",
            spot_price=150.0,
            dte_short=10,  # Expires in 10 days
            max_loss=1000.0,
            max_profit=500.0,
            buying_power_effect=2000.0,
            probability_of_profit=0.6,
            expected_return=5.0,
            legs=[]
        )
        
        # Calculate event penalty for various days to event
        no_event_penalty = calculate_event_penalty(event_risk_candidate, days_to_event=None)
        far_event_penalty = calculate_event_penalty(event_risk_candidate, days_to_event=20)
        close_event_penalty = calculate_event_penalty(event_risk_candidate, days_to_event=11)
        very_close_penalty = calculate_event_penalty(event_risk_candidate, days_to_event=9)
        
        # No event should have no penalty
        self.assertEqual(no_event_penalty, 0)
        
        # Far event should have low or no penalty
        self.assertLess(far_event_penalty, 50)
        
        # Close event should have significant penalty
        self.assertGreater(close_event_penalty, far_event_penalty)
        
        # Very close event should have very high penalty
        self.assertGreater(very_close_penalty, close_event_penalty)
        self.assertGreaterEqual(very_close_penalty, 75)


if __name__ == '__main__':
    unittest.main()
