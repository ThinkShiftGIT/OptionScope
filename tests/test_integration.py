"""
Integration tests for OptionScope.

This module contains integration tests that verify different components
of the application work together correctly.
"""

import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

from app.core.data_providers import create_data_provider
from app.core.indicators import check_mechanical_rules
from app.core.strategies.base import StrategyRegistry
from app.core.risk import calculate_max_loss
from app.core.scoring import rank_candidates
from app.utils.config import get_config


class TestEndToEndFlow(unittest.TestCase):
    """Integration test for the entire application flow."""
    
    @patch('app.core.data_providers.get_config')
    @patch('app.core.data_providers.yf')
    def test_data_to_candidates_flow(self, mock_yf, mock_get_config):
        """Test the flow from data providers to strategy candidates."""
        # Set up mocks for configuration
        mock_get_config.return_value = {
            'data_providers': {
                'price': {'provider': 'yfinance'},
                'options': {'provider': 'yfinance'},
                'volatility': {'provider': 'yfinance'},
                'events': {'provider': 'user_input'},
                'rates': {'provider': 'config', 'risk_free_rate': 0.05}
            },
            'mechanical_rules': {
                'iv_rank': {'min': 30, 'max': 60},
                'term_structure': {'min_gap': 2.0},
                'vix_range': {'min': 15, 'max': 30},
                'event_window': {'min_days': 10},
                'atr_width': {'max_ratio': 0.5},
                'max_risk_pct': 5
            },
            'strategy_parameters': {
                'cash_secured_put': {
                    'delta_targets': [0.3, 0.4, 0.5],
                    'dte_ranges': [[30, 45], [45, 60]]
                },
                'covered_call': {
                    'delta_targets': [0.2, 0.3, 0.4],
                    'dte_ranges': [[15, 30], [30, 45]]
                }
            }
        }
        
        # Set up mock for yfinance ticker
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        
        # Mock price data
        dates = pd.date_range('2023-01-01', periods=30)
        price_data = pd.DataFrame({
            'Open': np.linspace(100, 110, 30),
            'High': np.linspace(102, 112, 30),
            'Low': np.linspace(99, 109, 30),
            'Close': np.linspace(101, 111, 30),
            'Volume': np.random.randint(1000, 5000, 30)
        }, index=dates)
        
        mock_yf.download.return_value = price_data
        
        # Mock options chain
        expiry1 = '2023-06-16'
        expiry2 = '2023-07-21'
        
        mock_ticker.options = [expiry1, expiry2]
        
        # Create calls and puts for first expiration
        calls1 = pd.DataFrame({
            'strike': [100, 105, 110, 115, 120],
            'lastPrice': [12.0, 8.0, 5.0, 3.0, 1.5],
            'bid': [11.5, 7.5, 4.5, 2.5, 1.0],
            'ask': [12.5, 8.5, 5.5, 3.5, 2.0],
            'volume': [500, 1000, 1500, 800, 300],
            'openInterest': [2000, 4000, 6000, 3000, 1000],
            'impliedVolatility': [0.25, 0.23, 0.22, 0.21, 0.20]
        })
        
        puts1 = pd.DataFrame({
            'strike': [100, 105, 110, 115, 120],
            'lastPrice': [1.5, 3.0, 5.0, 8.0, 12.0],
            'bid': [1.0, 2.5, 4.5, 7.5, 11.5],
            'ask': [2.0, 3.5, 5.5, 8.5, 12.5],
            'volume': [300, 800, 1500, 1000, 500],
            'openInterest': [1000, 3000, 6000, 4000, 2000],
            'impliedVolatility': [0.24, 0.22, 0.23, 0.25, 0.27]
        })
        
        # Create calls and puts for second expiration
        calls2 = pd.DataFrame({
            'strike': [100, 105, 110, 115, 120],
            'lastPrice': [14.0, 10.0, 7.0, 4.0, 2.0],
            'bid': [13.5, 9.5, 6.5, 3.5, 1.5],
            'ask': [14.5, 10.5, 7.5, 4.5, 2.5],
            'volume': [400, 900, 1400, 700, 200],
            'openInterest': [1800, 3800, 5800, 2800, 800],
            'impliedVolatility': [0.28, 0.26, 0.25, 0.24, 0.23]
        })
        
        puts2 = pd.DataFrame({
            'strike': [100, 105, 110, 115, 120],
            'lastPrice': [2.0, 4.0, 7.0, 10.0, 14.0],
            'bid': [1.5, 3.5, 6.5, 9.5, 13.5],
            'ask': [2.5, 4.5, 7.5, 10.5, 14.5],
            'volume': [200, 700, 1400, 900, 400],
            'openInterest': [800, 2800, 5800, 3800, 1800],
            'impliedVolatility': [0.27, 0.25, 0.26, 0.28, 0.30]
        })
        
        # Set up mock for option_chain method
        mock_ticker.option_chain.side_effect = [
            MagicMock(calls=calls1, puts=puts1),
            MagicMock(calls=calls2, puts=puts2)
        ]
        
        # Mock volatility data
        dates = pd.date_range('2022-01-01', periods=252)
        iv_history = pd.Series(
            np.random.uniform(0.15, 0.35, 252),
            index=dates
        )
        
        mock_vix = MagicMock()
        mock_vix_hist = pd.DataFrame({
            'Close': np.random.uniform(15, 30, 252)
        }, index=dates)
        
        mock_yf.Ticker.return_value.history.side_effect = [
            iv_history.to_frame('Close'),
            mock_vix_hist
        ]
        
        try:
            # Create data providers
            price_provider = create_data_provider('price')
            options_provider = create_data_provider('options')
            volatility_provider = create_data_provider('volatility')
            events_provider = create_data_provider('events')
            rates_provider = create_data_provider('rates')
            
            # Test fetching data
            symbol = 'AAPL'
            
            # Get price data
            price_data = price_provider.get_price_history(symbol, days=30)
            self.assertIsInstance(price_data, pd.DataFrame)
            
            # Get options chain
            options_chain = options_provider.get_options_chain(symbol)
            self.assertIsInstance(options_chain, dict)
            self.assertEqual(len(options_chain), 2)
            
            # Get IV history
            iv_history = volatility_provider.get_iv_history(symbol, days=252)
            self.assertIsInstance(iv_history, pd.Series)
            
            # Get VIX level
            vix = volatility_provider.get_vix_level()
            self.assertIsInstance(vix, float)
            
            # Get risk-free rate
            risk_free_rate = rates_provider.get_risk_free_rate()
            self.assertEqual(risk_free_rate, 0.05)
            
            # Create market data dictionary
            spot_price = price_data['close'].iloc[-1]
            
            # Get strategy registry
            registry = StrategyRegistry()
            strategies = registry.get_all_strategies()
            
            # Ensure strategies are registered
            self.assertGreater(len(strategies), 0)
            
            # Try generating candidates for first strategy
            strategy_class = strategies[0]
            strategy = strategy_class(mock_get_config.return_value['strategy_parameters'])
            
            candidates = strategy.generate_candidates(
                symbol=symbol,
                spot_price=spot_price,
                options_chain=options_chain,
                risk_free_rate=risk_free_rate,
                iv=0.25
            )
            
            # Verify candidates were generated
            self.assertGreater(len(candidates), 0)
            
            # Test ranking candidates
            ranked = rank_candidates(candidates, account_equity=100000)
            
            # Verify ranking worked
            self.assertEqual(len(ranked), len(candidates))
            
            # Check that first candidate has a composite score
            self.assertIsNotNone(ranked[0].composite_score)
            
            # Success if we reached here without exceptions
            self.assertTrue(True)
            
        except Exception as e:
            self.fail(f"Integration test failed with exception: {e}")


if __name__ == '__main__':
    unittest.main()
