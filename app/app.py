"""
OptionScope - Options Strategy Scanner and Optimizer

Main Streamlit application file that integrates all components and provides
the user interface for the options strategy scanner and optimizer.
"""

import os
import pandas as pd
import streamlit as st
import yaml

from app.core.data_providers import (
    create_data_provider,
    PriceDataProvider,
    OptionsDataProvider,
    VolatilityDataProvider,
    EventsDataProvider,
    RatesDataProvider
)
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
from app.core.strategies.base import StrategyRegistry
from app.core.risk import calculate_max_loss, generate_price_scenario_table
from app.core.scoring import rank_candidates, get_top_candidates, generate_pareto_chart_data
from app.ui.components import (
    status_dashboard,
    strategy_candidate_table,
    strategy_detail_panel,
    pareto_chart,
    iv_term_structure_heatmap,
    trade_log_table,
    trade_log_form
)
from app.utils.config import get_config
from app.utils.caching import streamlit_cache, clear_cache


# Set page configuration
st.set_page_config(
    page_title="OptionScope",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@streamlit_cache(ttl_seconds=300)
def load_market_data(symbol):
    """
    Load market data for a symbol using data providers.
    
    Args:
        symbol: Ticker symbol to load data for
        
    Returns:
        Dict of market data
    """
    config = get_config()
    
    # Initialize data providers
    price_provider = create_data_provider('price')
    options_provider = create_data_provider('options')
    volatility_provider = create_data_provider('volatility')
    events_provider = create_data_provider('events')
    rates_provider = create_data_provider('rates')
    
    # Fetch price data
    price_data = price_provider.get_price_history(symbol, days=30)
    spot_price = price_data['close'].iloc[-1]
    
    # Calculate ATR
    atr = calculate_atr(price_data, window=5)
    
    # Fetch options chain
    options_chain = options_provider.get_options_chain(symbol)
    
    # Calculate ATM IV
    atm_iv = calculate_atm_iv(options_chain, spot_price)
    
    # Calculate IV Term Structure
    iv_term_structure = calculate_iv_term_structure(options_chain, spot_price)
    term_spread = iv_term_structure.max() - iv_term_structure.min() if not iv_term_structure.empty else 0
    
    # Calculate IV Rank
    iv_history = volatility_provider.get_iv_history(symbol, days=252)
    iv_rank = calculate_iv_rank(iv_history)
    
    # Calculate VIX Level
    vix_level = calculate_vix_level(volatility_provider)
    
    # Get upcoming events
    events = events_provider.get_upcoming_events(symbol)
    days_to_event = events['days_to_event'] if events else None
    
    # Get risk-free rate
    risk_free_rate = rates_provider.get_risk_free_rate()
    
    # Calculate realized volatility
    realized_vol = calculate_realized_volatility(price_data, window=30)
    
    # Calculate IV Skew
    iv_skew = calculate_iv_skew(options_chain)
    
    # Calculate liquidity metrics
    liquidity = calculate_liquidity_metrics(options_chain)
    
    # Compile market data
    market_data = {
        'symbol': symbol,
        'spot_price': spot_price,
        'atr': atr,
        'atm_iv': atm_iv,
        'iv_term_structure': iv_term_structure,
        'term_spread': term_spread,
        'iv_rank': iv_rank,
        'vix_level': vix_level,
        'days_to_event': days_to_event,
        'risk_free_rate': risk_free_rate,
        'realized_volatility': realized_vol,
        'iv_skew': iv_skew,
        'bid_ask_spreads': liquidity['avg_spread_pct'],
        'open_interest': liquidity['avg_open_interest'],
        'volume': liquidity['avg_volume'],
        'options_chain': options_chain
    }
    
    return market_data


@streamlit_cache(ttl_seconds=300)
def generate_strategy_candidates(market_data):
    """
    Generate strategy candidates using registered strategies.
    
    Args:
        market_data: Market data dictionary
        
    Returns:
        List of StrategyCandidate objects
    """
    config = get_config()
    
    symbol = market_data['symbol']
    spot_price = market_data['spot_price']
    options_chain = market_data['options_chain']
    risk_free_rate = market_data['risk_free_rate']
    iv = market_data['atm_iv']
    
    # Get all registered strategies
    registry = StrategyRegistry()
    strategies = registry.get_all_strategies()
    
    all_candidates = []
    
    # Generate candidates for each strategy
    for strategy_class in strategies:
        strategy = strategy_class(config['strategy_parameters'])
        candidates = strategy.generate_candidates(
            symbol=symbol,
            spot_price=spot_price,
            options_chain=options_chain,
            risk_free_rate=risk_free_rate,
            iv=iv
        )
        all_candidates.extend(candidates)
    
    return all_candidates


def load_trade_log():
    """
    Load trade log from file.
    
    Returns:
        DataFrame containing the trade log
    """
    log_path = os.path.join(os.path.dirname(__file__), "..", "trade_log.csv")
    
    if os.path.exists(log_path):
        return pd.read_csv(log_path)
    else:
        return pd.DataFrame(columns=[
            "Date", "Symbol", "Strategy", "Expiration", 
            "Strikes", "Price", "Quantity", "Notes"
        ])


def save_trade_log(log_df):
    """
    Save trade log to file.
    
    Args:
        log_df: DataFrame containing trade log entries
    """
    log_path = os.path.join(os.path.dirname(__file__), "..", "trade_log.csv")
    log_df.to_csv(log_path, index=False)


def main():
    """Main application function."""
    # Display title
    st.title("OptionScope 📊")
    st.write("Options Strategy Scanner and Optimizer")
    
    # Load configuration
    config = get_config()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Symbol selection
        symbols = config['symbols']
        symbol = st.selectbox("Select Symbol", symbols)
        
        # Account equity
        account_equity = st.number_input(
            "Account Equity ($)",
            min_value=1000,
            value=100000,
            step=1000
        )
        
        # Risk per trade
        max_risk_pct = st.slider(
            "Max Risk per Trade (%)",
            min_value=1,
            max_value=10,
            value=config['mechanical_rules']['max_risk_pct']
        )
        
        # Strategy filter
        registry = StrategyRegistry()
        strategy_names = [s.__name__ for s in registry.get_all_strategies()]
        selected_strategies = st.multiselect(
            "Strategy Filter",
            options=strategy_names,
            default=strategy_names
        )
        
        # Cache control
        if st.button("Clear Cache"):
            clear_cache()
            st.success("Cache cleared!")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Strategy Candidates", "Trade Log"])
    
    # Get market data
    with st.spinner("Loading market data..."):
        market_data = load_market_data(symbol)
    
    # Check mechanical rules
    rule_compliance = check_mechanical_rules(market_data, config['mechanical_rules'])
    
    # Dashboard tab
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Status dashboard
            all_rules_pass = status_dashboard(symbol, rule_compliance, market_data)
            
            # Market overview
            st.subheader("Market Overview")
            
            # Price and volatility metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Price", f"${market_data['spot_price']:.2f}")
            with metric_col2:
                st.metric("IV", f"{market_data['atm_iv']:.1f}%")
            with metric_col3:
                st.metric("IV Rank", f"{market_data['iv_rank']:.1f}%")
        
        with col2:
            # Current positions (placeholder)
            st.subheader("Current Positions")
            st.info("Position tracking will be added in a future update.")
        
        # IV term structure visualization
        st.subheader("IV Term Structure")
        iv_data = {
            pd.Timestamp.now().strftime('%Y-%m-%d'): 
            pd.DataFrame(market_data['iv_term_structure'])
        }
        iv_term_structure_heatmap(symbol, iv_data)
    
    # Strategy candidates tab
    with tab2:
        # Generate candidates
        if all_rules_pass:
            with st.spinner("Generating strategy candidates..."):
                candidates = generate_strategy_candidates(market_data)
                
                # Filter by selected strategies
                if selected_strategies and len(selected_strategies) < len(strategy_names):
                    candidates = [c for c in candidates if c.strategy_name in selected_strategies]
                
                # Rank candidates
                top_candidates = get_top_candidates(
                    candidates,
                    account_equity,
                    market_data.get('days_to_event'),
                    top_n=10,
                    include_categories=True
                )
                
                # Display Pareto chart
                st.subheader("Strategy Trade-offs")
                pareto_chart(candidates)
                
                # Display candidate table
                st.subheader("Top Strategy Candidates")
                selected_idx = strategy_candidate_table(top_candidates)
                
                # Display selected candidate details
                if selected_idx >= 0:
                    st.subheader("Strategy Details")
                    selected_candidate = top_candidates[selected_idx]
                    strategy_detail_panel(
                        selected_candidate,
                        symbol,
                        market_data['spot_price']
                    )
                    
                    # Add to trade log button
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button("Add to Trade Log"):
                            st.session_state['add_to_log'] = selected_candidate
                            st.rerun()
        else:
            st.warning("Entry rules not met. Strategy generation disabled.")
    
    # Trade log tab
    with tab3:
        st.subheader("Trade Log")
        
        # Load trade log
        trade_log = load_trade_log()
        
        # Display trade log table
        trade_log_table(trade_log)
        
        # Add to trade log form
        st.subheader("Add to Trade Log")
        
        # Check if we have a candidate to add from the strategy tab
        candidate_to_add = st.session_state.get('add_to_log', None)
        if candidate_to_add:
            # Remove from session state to avoid duplicate adds
            del st.session_state['add_to_log']
            
            # Display form with pre-filled values
            form_data = trade_log_form(symbol, candidate_to_add)
        else:
            # Display empty form
            form_data = trade_log_form(symbol)
        
        # Add to log if form submitted
        if form_data:
            trade_log = trade_log.append(form_data, ignore_index=True)
            save_trade_log(trade_log)
            st.success("Trade added to log!")
            st.rerun()


if __name__ == "__main__":
    main()
