"""
Unit tests for data providers module.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import yaml

from app.core.data_providers import (
    PriceDataProvider,
    OptionsDataProvider,
    VolatilityDataProvider,
    EventsDataProvider,
    RatesDataProvider,
    YFinancePriceProvider,
    YFinanceOptionsProvider,
    YFinanceVolatilityProvider,
    UserInputEventProvider,
    ConfigRatesProvider
)


class TestYFinancePriceProvider(unittest.TestCase):
    """Test cases for the YFinancePriceProvider."""
    
    @patch('app.core.data_providers.yf.download')
    def test_get_price_history(self, mock_download):
        """Test getting price history."""
        # Create mock data
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 2000, 3000]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        # Set up the mock
        mock_download.return_value = mock_data
        
        # Create provider and call the method
        provider = YFinancePriceProvider()
        result = provider.get_price_history('AAPL', days=3)
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertTrue('open' in result.columns)
        self.assertTrue('close' in result.columns)
        
        # Verify the mock was called correctly
        mock_download.assert_called_once()


class TestYFinanceOptionsProvider(unittest.TestCase):
    """Test cases for the YFinanceOptionsProvider."""
    
    @patch('app.core.data_providers.yf.Ticker')
    def test_get_options_chain(self, mock_ticker):
        """Test getting options chain."""
        # Create mock ticker instance
        mock_ticker_instance = MagicMock()
        mock_ticker.return_value = mock_ticker_instance
        
        # Create mock options data
        expiry1 = '2023-01-20'
        expiry2 = '2023-02-17'
        
        # Mock options.expirations
        mock_ticker_instance.options = [expiry1, expiry2]
        
        # Mock option_chain for each expiration
        mock_calls = []
        
        # First expiration - calls
        calls1 = pd.DataFrame({
            'strike': [100, 105, 110],
            'lastPrice': [5.0, 3.0, 1.0],
            'bid': [4.8, 2.9, 0.9],
            'ask': [5.2, 3.1, 1.1],
            'volume': [100, 200, 300],
            'openInterest': [1000, 2000, 3000],
            'impliedVolatility': [0.2, 0.18, 0.16]
        })
        
        # First expiration - puts
        puts1 = pd.DataFrame({
            'strike': [100, 105, 110],
            'lastPrice': [2.0, 4.0, 6.0],
            'bid': [1.9, 3.9, 5.9],
            'ask': [2.1, 4.1, 6.1],
            'volume': [150, 250, 350],
            'openInterest': [1500, 2500, 3500],
            'impliedVolatility': [0.22, 0.20, 0.18]
        })
        
        # Second expiration - calls
        calls2 = pd.DataFrame({
            'strike': [100, 105, 110],
            'lastPrice': [6.0, 4.0, 2.0],
            'bid': [5.8, 3.9, 1.9],
            'ask': [6.2, 4.1, 2.1],
            'volume': [120, 220, 320],
            'openInterest': [1200, 2200, 3200],
            'impliedVolatility': [0.22, 0.20, 0.18]
        })
        
        # Second expiration - puts
        puts2 = pd.DataFrame({
            'strike': [100, 105, 110],
            'lastPrice': [3.0, 5.0, 7.0],
            'bid': [2.9, 4.9, 6.9],
            'ask': [3.1, 5.1, 7.1],
            'volume': [170, 270, 370],
            'openInterest': [1700, 2700, 3700],
            'impliedVolatility': [0.24, 0.22, 0.20]
        })
        
        # Set up the mocks for option_chain method
        mock_ticker_instance.option_chain.side_effect = [
            MagicMock(calls=calls1, puts=puts1),
            MagicMock(calls=calls2, puts=puts2)
        ]
        
        # Create provider and call the method
        provider = YFinanceOptionsProvider()
        result = provider.get_options_chain('AAPL')
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertTrue(expiry1 in result)
        self.assertTrue(expiry2 in result)
        
        # Check structure of first expiration
        self.assertTrue('calls' in result[expiry1])
        self.assertTrue('puts' in result[expiry1])
        self.assertEqual(len(result[expiry1]['calls']), 3)
        self.assertEqual(len(result[expiry1]['puts']), 3)
        
        # Verify the mock was called correctly
        mock_ticker.assert_called_once_with('AAPL')
        self.assertEqual(mock_ticker_instance.option_chain.call_count, 2)


class TestConfigRatesProvider(unittest.TestCase):
    """Test cases for the ConfigRatesProvider."""
    
    @patch('app.core.data_providers.get_config')
    def test_get_risk_free_rate(self, mock_get_config):
        """Test getting the risk-free rate."""
        # Set up the mock
        mock_get_config.return_value = {
            'data_providers': {
                'rates': {
                    'provider': 'config',
                    'risk_free_rate': 0.05
                }
            }
        }
        
        # Create provider and call the method
        provider = ConfigRatesProvider()
        result = provider.get_risk_free_rate()
        
        # Assertions
        self.assertEqual(result, 0.05)
        
        # Verify the mock was called correctly
        mock_get_config.assert_called_once()


if __name__ == '__main__':
    unittest.main()
