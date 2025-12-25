"""
Data provider interfaces and implementations for OptionScope.

This module contains the abstract base classes and concrete implementations 
for various data providers used in the application, including:
- Price data providers
- Options chain data providers
- Volatility data providers
- Event calendar data providers
- Interest rate data providers
"""

import abc
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache, wraps
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field

from app.utils.config import get_config


def timed_cache(seconds: int = 120) -> Callable:
    """
    Create a timed cache decorator with a specified expiration time.
    
    Args:
        seconds: Seconds before cached values expire
        
    Returns:
        Callable: Decorator for caching function results
    """
    def decorator(func: Callable) -> Callable:
        # Store cache and timestamps
        cache: Dict[Any, Any] = {}
        timestamps: Dict[Any, float] = {}
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create a key for the cache based on function arguments
            key = str(args) + str(kwargs)
            current_time = time.time()
            
            # Check if we have a cached value and if it's still valid
            if key in cache and current_time - timestamps[key] < seconds:
                return cache[key]
            
            # Call the function and update cache
            result = func(*args, **kwargs)
            cache[key] = result
            timestamps[key] = current_time
            return result
        
        return wrapper
    
    return decorator


class OptionContract(BaseModel):
    """Model representing an option contract."""
    symbol: str
    expiration: datetime
    strike: float
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

    @property
    def mid(self) -> float:
        """Calculate the mid price between bid and ask."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread_pct(self) -> float:
        """Calculate the bid-ask spread as a percentage of the mid price."""
        if self.mid == 0:
            return float('inf')
        return (self.ask - self.bid) / self.mid


class OptionsChain(BaseModel):
    """Model representing an options chain for a specific expiration date."""
    symbol: str
    expiration: datetime
    calls: List[OptionContract] = Field(default_factory=list)
    puts: List[OptionContract] = Field(default_factory=list)

    def get_atm_iv(self, spot_price: float) -> Tuple[float, float]:
        """
        Get the at-the-money implied volatility for both calls and puts.
        
        Args:
            spot_price: Current spot price of the underlying
            
        Returns:
            Tuple[float, float]: (call_iv, put_iv) for the most ATM options
        """
        # Find the closest strike for calls
        if not self.calls:
            call_iv = float('nan')
        else:
            closest_call = min(self.calls, key=lambda x: abs(x.strike - spot_price))
            call_iv = closest_call.implied_volatility
        
        # Find the closest strike for puts
        if not self.puts:
            put_iv = float('nan')
        else:
            closest_put = min(self.puts, key=lambda x: abs(x.strike - spot_price))
            put_iv = closest_put.implied_volatility
        
        return call_iv, put_iv


class DataProviderType(Enum):
    """Enumeration of data provider types."""
    PRICE = "price"
    OPTIONS = "options"
    VOLATILITY = "volatility"
    EVENTS = "events"
    RATES = "rates"


class DataProvider(abc.ABC):
    """Abstract base class for all data providers."""
    
    @property
    @abc.abstractmethod
    def provider_type(self) -> DataProviderType:
        """Return the type of data provider."""
        pass
    
    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Return the name of the data provider implementation."""
        pass


class PriceDataProvider(DataProvider):
    """Interface for price data providers."""
    
    @property
    def provider_type(self) -> DataProviderType:
        return DataProviderType.PRICE
    
    @abc.abstractmethod
    def get_price(self, symbol: str) -> float:
        """Get the current price of a symbol."""
        pass
    
    @abc.abstractmethod
    def get_ohlc(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Get OHLC price data for a symbol.
        
        Args:
            symbol: The ticker symbol
            period: Data period (e.g., '1d', '1y', '5y')
            interval: Data interval (e.g., '1m', '1h', '1d')
            
        Returns:
            DataFrame with columns: [Open, High, Low, Close, Volume]
        """
        pass


class OptionsDataProvider(DataProvider):
    """Interface for options data providers."""
    
    @property
    def provider_type(self) -> DataProviderType:
        return DataProviderType.OPTIONS
    
    @abc.abstractmethod
    def get_expirations(self, symbol: str) -> List[datetime]:
        """Get list of available option expiration dates for a symbol."""
        pass
    
    @abc.abstractmethod
    def get_options_chain(self, symbol: str, expiration: datetime) -> OptionsChain:
        """
        Get the options chain for a specific symbol and expiration date.
        
        Args:
            symbol: The ticker symbol
            expiration: Expiration date
            
        Returns:
            OptionsChain object with calls and puts
        """
        pass


class VolatilityDataProvider(DataProvider):
    """Interface for volatility data providers."""
    
    @property
    def provider_type(self) -> DataProviderType:
        return DataProviderType.VOLATILITY
    
    @abc.abstractmethod
    def get_index_value(self, symbol: str) -> float:
        """
        Get the current value of a volatility index.
        
        Args:
            symbol: The volatility index symbol (e.g., '^VIX', '^VXN')
            
        Returns:
            Current value of the volatility index
        """
        pass
    
    @abc.abstractmethod
    def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Get historical data for a volatility index.
        
        Args:
            symbol: The volatility index symbol
            period: Data period (e.g., '1d', '1y', '5y')
            
        Returns:
            DataFrame with historical volatility data
        """
        pass

# Renaming to match what is expected in tests/imports if needed,
# but mostly just fixing the missing EventsDataProvider which was named EventDataProvider
class EventsDataProvider(DataProvider):
    """Interface for economic and earnings event data providers."""
    
    @property
    def provider_type(self) -> DataProviderType:
        return DataProviderType.EVENTS
    
    @abc.abstractmethod
    def get_earnings_date(self, symbol: str) -> Optional[datetime]:
        """
        Get the next earnings date for a symbol.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Next earnings announcement date, if available
        """
        pass
    
    @abc.abstractmethod
    def get_upcoming_events(self, symbol: str) -> Dict[str, Any]:
        """
        Get upcoming events for a symbol.
        
        Args:
             symbol: The ticker symbol

        Returns:
             Dict with event details
        """
        pass


class RatesDataProvider(DataProvider):
    """Interface for interest rate data providers."""
    
    @property
    def provider_type(self) -> DataProviderType:
        return DataProviderType.RATES
    
    @abc.abstractmethod
    def get_risk_free_rate(self, term_days: int = 30) -> float:
        """
        Get the risk-free interest rate for a specific term.
        
        Args:
            term_days: Term in days (e.g., 30, 90, 365)
            
        Returns:
            Annual risk-free rate as a decimal (e.g., 0.05 for 5%)
        """
        pass


class YFinancePriceProvider(PriceDataProvider):
    """Yahoo Finance implementation of PriceDataProvider."""
    
    @property
    def provider_name(self) -> str:
        return "yfinance"
    
    def get_price_history(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get price history."""
        # Alias for get_ohlc but with days argument
        # Convert days to period string?
        # Rough approximation
        return self.get_ohlc(symbol, period=f"{days}d")

    @timed_cache(seconds=60)
    def get_price(self, symbol: str) -> float:
        """Get current price from Yahoo Finance."""
        ticker = yf.Ticker(symbol)
        # get fast price
        try:
             # Fast way
             info = ticker.fast_info
             return info.last_price
        except:
             data = ticker.history(period="1d")
             if data.empty:
                 raise ValueError(f"No data found for {symbol}")
             return float(data['Close'].iloc[-1])
    
    @timed_cache(seconds=300)
    def get_ohlc(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Get OHLC data from Yahoo Finance."""
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No OHLC data found for {symbol}")
        # Rename columns to lowercase for consistency if needed, but existing code expects Title Case or lower?
        # App uses ['close'] lower case in load_market_data: price_data['close'].iloc[-1]
        # But get_ohlc defined earlier returned Title Case.
        # Let's standardize on lowercase
        data.columns = [c.lower() for c in data.columns]
        return data[['open', 'high', 'low', 'close', 'volume']]


class YFinanceOptionsProvider(OptionsDataProvider):
    """Yahoo Finance implementation of OptionsDataProvider."""
    
    @property
    def provider_name(self) -> str:
        return "yfinance"
    
    @timed_cache(seconds=120)
    def get_expirations(self, symbol: str) -> List[datetime]:
        """Get available option expirations from Yahoo Finance."""
        ticker = yf.Ticker(symbol)
        expiration_dates = ticker.options
        
        # Convert string dates to datetime objects
        return [datetime.strptime(exp_date, '%Y-%m-%d') for exp_date in expiration_dates]
    
    @timed_cache(seconds=120)
    def get_options_chain(self, symbol: str, expiration: datetime = None) -> OptionsChain:
        """Get options chain from Yahoo Finance. If expiration is None, gets nearest."""
        ticker = yf.Ticker(symbol)

        if expiration is None:
             if not ticker.options:
                 raise ValueError(f"No options found for {symbol}")
             exp_str = ticker.options[0]
             expiration = datetime.strptime(exp_str, '%Y-%m-%d')
        else:
             exp_str = expiration.strftime('%Y-%m-%d')
        
        # Get the options data
        try:
            opt_data = ticker.option_chain(exp_str)
            calls_df = opt_data.calls
            puts_df = opt_data.puts
        except Exception as e:
            raise ValueError(f"Error fetching options data for {symbol} ({exp_str}): {str(e)}")
        
        # Create the OptionsChain object
        chain = OptionsChain(
            symbol=symbol,
            expiration=expiration,
            calls=[],
            puts=[]
        )
        
        # Process calls
        for _, row in calls_df.iterrows():
            contract = OptionContract(
                symbol=f"{symbol}_{exp_str}_C{row['strike']}",
                expiration=expiration,
                strike=float(row['strike']),
                option_type="call",
                bid=float(row['bid']),
                ask=float(row['ask']),
                last=float(row['lastPrice']) if 'lastPrice' in row else None,
                volume=int(row['volume']) if 'volume' in row and not pd.isna(row['volume']) else 0,
                open_interest=int(row['openInterest']) if 'openInterest' in row and not pd.isna(row['openInterest']) else 0,
                implied_volatility=float(row['impliedVolatility']),
                delta=float(row['delta']) if 'delta' in row and not pd.isna(row['delta']) else None,
                gamma=float(row['gamma']) if 'gamma' in row and not pd.isna(row['gamma']) else None,
                theta=float(row['theta']) if 'theta' in row and not pd.isna(row['theta']) else None,
                vega=float(row['vega']) if 'vega' in row and not pd.isna(row['vega']) else None,
                rho=float(row['rho']) if 'rho' in row and not pd.isna(row['rho']) else None
            )
            chain.calls.append(contract)
        
        # Process puts
        for _, row in puts_df.iterrows():
            contract = OptionContract(
                symbol=f"{symbol}_{exp_str}_P{row['strike']}",
                expiration=expiration,
                strike=float(row['strike']),
                option_type="put",
                bid=float(row['bid']),
                ask=float(row['ask']),
                last=float(row['lastPrice']) if 'lastPrice' in row else None,
                volume=int(row['volume']) if 'volume' in row and not pd.isna(row['volume']) else 0,
                open_interest=int(row['openInterest']) if 'openInterest' in row and not pd.isna(row['openInterest']) else 0,
                implied_volatility=float(row['impliedVolatility']),
                delta=float(row['delta']) if 'delta' in row and not pd.isna(row['delta']) else None,
                gamma=float(row['gamma']) if 'gamma' in row and not pd.isna(row['gamma']) else None,
                theta=float(row['theta']) if 'theta' in row and not pd.isna(row['theta']) else None,
                vega=float(row['vega']) if 'vega' in row and not pd.isna(row['vega']) else None,
                rho=float(row['rho']) if 'rho' in row and not pd.isna(row['rho']) else None
            )
            chain.puts.append(contract)
        
        return chain


class YFinanceVolatilityProvider(VolatilityDataProvider):
    """Yahoo Finance implementation of VolatilityDataProvider."""
    
    @property
    def provider_name(self) -> str:
        return "yfinance"
    
    def get_iv_history(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """Get IV history."""
        # Map symbol to volatility index
        vol_indices = {
            'SPY': '^VIX',
            'QQQ': '^VXN'
        }
        vol_index = vol_indices.get(symbol, '^VIX') # Default to VIX
        return self.get_historical_data(vol_index, period=f"{days}d")

    @timed_cache(seconds=120)
    def get_index_value(self, symbol: str) -> float:
        """Get current volatility index value from Yahoo Finance."""
        # If symbol passed is not an index but a ticker, try to map it
        if not symbol.startswith('^'):
             vol_indices = {
                'SPY': '^VIX',
                'QQQ': '^VXN'
            }
             symbol = vol_indices.get(symbol, '^VIX')

        ticker = yf.Ticker(symbol)
        try:
             info = ticker.fast_info
             return info.last_price
        except:
             data = ticker.history(period="1d")
             if data.empty:
                 raise ValueError(f"No data found for volatility index {symbol}")
             return float(data['Close'].iloc[-1])
    
    @timed_cache(seconds=600)
    def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical volatility index data from Yahoo Finance."""
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            raise ValueError(f"No historical data found for volatility index {symbol}")
        # Lowercase columns
        data.columns = [c.lower() for c in data.columns]
        return data[['close']]


class UserInputEventsProvider(EventsDataProvider):
    """User input implementation of EventDataProvider."""
    
    @property
    def provider_name(self) -> str:
        return "user_input"
    
    def get_earnings_date(self, symbol: str) -> Optional[datetime]:
        """
        Get the next earnings date from user input.
        This is a placeholder that would be connected to UI inputs.
        """
        # In a real implementation, this would retrieve from Streamlit session state
        # For now, we'll just return None
        return None

    def get_upcoming_events(self, symbol: str) -> Dict[str, Any]:
         """Placeholder."""
         return {'days_to_event': None}
    
    def get_economic_events(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get economic events from user input.
        This is a placeholder that would be connected to UI inputs.
        """
        # In a real implementation, this would retrieve from Streamlit session state
        # For now, we'll just return an empty DataFrame with the right structure
        return pd.DataFrame(columns=['Date', 'Event', 'Importance', 'Expected', 'Actual'])

# Alias for compatibility with tests if they use UserInputEventProvider
UserInputEventProvider = UserInputEventsProvider

class ConfigRatesProvider(RatesDataProvider):
    """Configuration-based implementation of RatesDataProvider."""
    
    @property
    def provider_name(self) -> str:
        return "config"
    
    def get_risk_free_rate(self, term_days: int = 30) -> float:
        """Get risk-free rate from configuration."""
        config = get_config()
        return float(config['data_providers']['rates']['risk_free_rate'])


def create_data_provider(provider_type_str: str) -> DataProvider:
    """
    Factory function to get the appropriate data provider based on configuration.
    
    Args:
        provider_type_str: Type of data provider to create (string)
        
    Returns:
        DataProvider: Implementation of the requested provider type
    """
    
    try:
        provider_type = DataProviderType(provider_type_str)
    except ValueError:
        # Map simple strings if they don't match exactly
        mapping = {
            'price': DataProviderType.PRICE,
            'options': DataProviderType.OPTIONS,
            'volatility': DataProviderType.VOLATILITY,
            'events': DataProviderType.EVENTS,
            'rates': DataProviderType.RATES
        }
        provider_type = mapping.get(provider_type_str)
        if not provider_type:
            raise ValueError(f"Invalid provider type: {provider_type_str}")

    config = get_config()
    # Check if provider config exists, if not use default
    if provider_type.value in config['data_providers']:
        provider_name = config['data_providers'][provider_type.value]['provider']
    else:
        # Fallback to defaults
        if provider_type == DataProviderType.PRICE: provider_name = 'yfinance'
        elif provider_type == DataProviderType.OPTIONS: provider_name = 'yfinance'
        elif provider_type == DataProviderType.VOLATILITY: provider_name = 'yfinance'
        elif provider_type == DataProviderType.EVENTS: provider_name = 'user_input'
        elif provider_type == DataProviderType.RATES: provider_name = 'config'
        else: provider_name = 'yfinance'

    # Lazy import to avoid circular dependency issues if any
    from app.core.providers.public_provider import PublicPriceProvider

    # Handle provider instantiation safely
    providers = {
        (DataProviderType.PRICE, "yfinance"): YFinancePriceProvider, # Class not instance
        (DataProviderType.PRICE, "public"): PublicPriceProvider,
        (DataProviderType.OPTIONS, "yfinance"): YFinanceOptionsProvider,
        (DataProviderType.VOLATILITY, "yfinance"): YFinanceVolatilityProvider,
        (DataProviderType.EVENTS, "user_input"): UserInputEventsProvider,
        (DataProviderType.RATES, "config"): ConfigRatesProvider,
    }
    
    provider_cls = providers.get((provider_type, provider_name))
    
    if provider_cls:
        try:
            return provider_cls()
        except ValueError:
            # Fallback if init fails (e.g. missing credentials for Public)
            if provider_name == "public":
                print("Public.com provider failed to init, falling back to yfinance")
                return YFinancePriceProvider()
            raise

    # Fallback logic if config asks for something we don't have
    if provider_name == "public" and provider_type == DataProviderType.PRICE:
         # Try instantiating anyway? No, already checked map.
         # Just fallback to YFinance
         return YFinancePriceProvider()

    raise ValueError(f"No {provider_type.value} provider found with name '{provider_name}'")

# Alias for backward compatibility if needed
get_provider = create_data_provider
