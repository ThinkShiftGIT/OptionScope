"""
Caching utilities for OptionScope.

This module provides caching decorators and utilities to help prevent
rate limiting when fetching data from external sources.
"""

import functools
import time
from typing import Any, Callable, Dict, TypeVar, cast

import streamlit as st

# Type variable for the function return type
T = TypeVar('T')


def streamlit_cache(ttl_seconds: int = 300) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to cache function results using Streamlit's cache mechanism.
    
    Args:
        ttl_seconds: Time to live in seconds for cached values
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Create a unique key for this function
        cache_key = f"cache_{func.__module__}_{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create a key for this specific call
            call_key = str(args) + str(kwargs)
            
            # Initialize cache in session state if needed
            if cache_key not in st.session_state:
                st.session_state[cache_key] = {}
            
            cache = st.session_state[cache_key]
            current_time = time.time()
            
            # Check if we have a cached value and if it's still valid
            if call_key in cache:
                value, timestamp = cache[call_key]
                if current_time - timestamp < ttl_seconds:
                    return value
            
            # Call the function and cache the result
            result = func(*args, **kwargs)
            cache[call_key] = (result, current_time)
            return result
        
        return wrapper
    
    return decorator


def memory_cache(ttl_seconds: int = 300) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to cache function results in memory.
    Useful when Streamlit's cache is not appropriate.
    
    Args:
        ttl_seconds: Time to live in seconds for cached values
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Create cache and timestamps dictionaries
        cache: Dict[str, Any] = {}
        timestamps: Dict[str, float] = {}
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create a key for this specific call
            call_key = str(args) + str(kwargs)
            current_time = time.time()
            
            # Check if we have a cached value and if it's still valid
            if call_key in cache and current_time - timestamps[call_key] < ttl_seconds:
                return cache[call_key]
            
            # Call the function and cache the result
            result = func(*args, **kwargs)
            cache[call_key] = result
            timestamps[call_key] = current_time
            return result
        
        return wrapper
    
    return decorator


def clear_cache() -> None:
    """
    Clear all cached values from the Streamlit session state.
    """
    # Find and clear all cache keys
    cache_keys = [k for k in st.session_state.keys() if k.startswith("cache_")]
    for key in cache_keys:
        del st.session_state[key]
