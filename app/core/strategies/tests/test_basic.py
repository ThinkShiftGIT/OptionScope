
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from app.core.data_providers import create_data_provider, DataProviderType, ConfigRatesProvider

def test_create_data_provider():
    # Mock get_config
    with patch('app.core.data_providers.get_config') as mock_config:
        mock_config.return_value = {
            'data_providers': {
                'rates': {
                    'provider': 'config',
                    'risk_free_rate': 0.05
                }
            }
        }
        provider = create_data_provider('rates')
        assert isinstance(provider, ConfigRatesProvider)
        assert provider.get_risk_free_rate() == 0.05
