import unittest
from unittest.mock import patch
from app.core.data_providers import (
    YFinancePriceProvider,
    YFinanceOptionsProvider,
    YFinanceVolatilityProvider,
    UserInputEventProvider,
    ConfigRatesProvider
)

class TestDataProviderFactory(unittest.TestCase):
    """Test cases for the data provider factory."""

    @patch('app.core.data_providers.get_config')
    def test_create_price_provider(self, mock_get_config):
        """Test creating a price data provider."""
        mock_get_config.return_value = {
            'data_providers': {'price': {'provider': 'yfinance'}}
        }
        from app.core.data_providers import create_data_provider
        provider = create_data_provider('price')
        self.assertIsInstance(provider, YFinancePriceProvider)

    @patch('app.core.data_providers.get_config')
    def test_create_options_provider(self, mock_get_config):
        """Test creating an options data provider."""
        mock_get_config.return_value = {
            'data_providers': {'options': {'provider': 'yfinance'}}
        }
        from app.core.data_providers import create_data_provider
        provider = create_data_provider('options')
        self.assertIsInstance(provider, YFinanceOptionsProvider)

    @patch('app.core.data_providers.get_config')
    def test_create_volatility_provider(self, mock_get_config):
        """Test creating a volatility data provider."""
        mock_get_config.return_value = {
            'data_providers': {'volatility': {'provider': 'yfinance'}}
        }
        from app.core.data_providers import create_data_provider
        provider = create_data_provider('volatility')
        self.assertIsInstance(provider, YFinanceVolatilityProvider)

    @patch('app.core.data_providers.get_config')
    def test_create_events_provider(self, mock_get_config):
        """Test creating an events data provider."""
        mock_get_config.return_value = {
            'data_providers': {'events': {'provider': 'user_input'}}
        }
        from app.core.data_providers import create_data_provider
        provider = create_data_provider('events')
        self.assertIsInstance(provider, UserInputEventProvider)

    @patch('app.core.data_providers.get_config')
    def test_create_rates_provider(self, mock_get_config):
        """Test creating a rates data provider."""
        mock_get_config.return_value = {
            'data_providers': {'rates': {'provider': 'config'}}
        }
        from app.core.data_providers import create_data_provider
        provider = create_data_provider('rates')
        self.assertIsInstance(provider, ConfigRatesProvider)

    @patch('app.core.data_providers.get_config')
    def test_invalid_provider_type(self, mock_get_config):
        """Test creating an invalid provider type."""
        mock_get_config.return_value = {'data_providers': {}}
        from app.core.data_providers import create_data_provider
        with self.assertRaises(ValueError):
            create_data_provider('invalid_type')

    @patch('app.core.data_providers.get_config')
    def test_invalid_provider_implementation(self, mock_get_config):
        """Test creating a provider with an invalid implementation."""
        mock_get_config.return_value = {
            'data_providers': {'price': {'provider': 'invalid'}}
        }
        from app.core.data_providers import create_data_provider
        with self.assertRaises(ValueError):
            create_data_provider('price')
