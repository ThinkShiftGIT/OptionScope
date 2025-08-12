"""
Reusable UI components for the Streamlit interface.

This module contains various UI components and helpers for building
the OptionScope Streamlit interface, including status lights,
strategy tables, and visualization components.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.core.strategies.base import StrategyCandidate
from app.utils.config import get_config


def status_light(
    label: str, 
    status: bool, 
    value: Optional[Union[float, str]] = None,
    threshold: Optional[Union[float, str]] = None
) -> None:
    """
    Display a status light with label and optional value/threshold.
    
    Args:
        label: Label for the status light
        status: Whether the status is good (True) or bad (False)
        value: Optional current value to display
        threshold: Optional threshold value to display
    """
    # Create columns for the status light
    col1, col2 = st.columns([1, 4])
    
    # Display the status indicator
    with col1:
        if status:
            st.markdown(
                '<div style="background-color:#28a745; height:20px; width:20px; '
                'border-radius:50%; margin:auto;"></div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background-color:#dc3545; height:20px; width:20px; '
                'border-radius:50%; margin:auto;"></div>', 
                unsafe_allow_html=True
            )
    
    # Display the label and value
    with col2:
        if value is not None and threshold is not None:
            st.markdown(f"**{label}**: {value} (Threshold: {threshold})")
        elif value is not None:
            st.markdown(f"**{label}**: {value}")
        else:
            st.markdown(f"**{label}**")


def status_dashboard(
    symbol: str,
    rule_compliance: Dict[str, bool],
    market_data: Dict[str, Any]
) -> bool:
    """
    Display a dashboard with status lights for various rules and metrics.
    
    Args:
        symbol: Ticker symbol
        rule_compliance: Dictionary of rule names and their compliance status
        market_data: Dictionary of market data values
        
    Returns:
        bool: Whether all rules pass
    """
    st.subheader("Market Conditions Dashboard")
    
    # Get config for thresholds
    config = get_config()
    
    # Volatility Structure
    term_structure_status = rule_compliance.get('term_structure', False)
    term_structure_value = market_data.get('term_spread', None)
    term_structure_threshold = config['mechanical_rules']['term_structure']['min_gap']
    
    status_light(
        "Volatility Structure",
        term_structure_status,
        f"{term_structure_value:.2f} pts" if term_structure_value is not None else None,
        f"{term_structure_threshold:.2f} pts"
    )
    
    # Market Volatility (IV Rank)
    iv_rank_status = rule_compliance.get('iv_rank', False)
    iv_rank_value = market_data.get('iv_rank', None)
    iv_rank_min = config['mechanical_rules']['iv_rank']['min']
    iv_rank_max = config['mechanical_rules']['iv_rank']['max']
    
    status_light(
        "IV Rank",
        iv_rank_status,
        f"{iv_rank_value:.1f}%" if iv_rank_value is not None else None,
        f"{iv_rank_min}% - {iv_rank_max}%"
    )
    
    # VIX/VXN Level
    vix_status = rule_compliance.get('vix_range', False)
    vix_value = market_data.get('vix_level', None)
    vix_min = config['mechanical_rules']['vix_range']['min']
    vix_max = config['mechanical_rules']['vix_range']['max']
    
    status_light(
        "VIX/VXN Level",
        vix_status,
        f"{vix_value:.1f}" if vix_value is not None else None,
        f"{vix_min} - {vix_max}"
    )
    
    # Event Timing
    event_status = rule_compliance.get('event_window', True)  # Default to True if not set
    days_to_event = market_data.get('days_to_event', None)
    event_min_days = config['mechanical_rules']['event_window']['min_days']
    
    if days_to_event is not None:
        event_value = f"{days_to_event} days"
        event_threshold = f">= {event_min_days} days"
    else:
        event_value = "No events"
        event_threshold = None
    
    status_light(
        "Event Timing",
        event_status,
        event_value,
        event_threshold
    )
    
    # ATR Guardrail
    atr_status = rule_compliance.get('atr_width', True)  # Default to True if not set
    atr_value = market_data.get('atr', None)
    
    status_light(
        "ATR (5-Day)",
        atr_status,
        f"${atr_value:.2f}" if atr_value is not None else None,
        "< Strike Width"
    )
    
    # Overall verdict
    all_pass = all(rule_compliance.values())
    
    st.markdown("---")
    
    if all_pass:
        st.success("✅ All entry rules PASS - Entry allowed")
    else:
        st.error("❌ Some entry rules FAIL - Entry not recommended")
    
    return all_pass


def strategy_candidate_table(candidates: List[StrategyCandidate]) -> int:
    """
    Display a table of strategy candidates with key metrics.
    
    Args:
        candidates: List of strategy candidates to display
        
    Returns:
        int: Index of selected candidate (-1 if none selected)
    """
    if not candidates:
        st.warning("No strategy candidates available.")
        return -1
    
    # Create DataFrame for display
    data = []
    for i, candidate in enumerate(candidates):
        # Handle different strategy types appropriately
        if candidate.dte_long:
            dte = f"{candidate.dte_short}-{candidate.dte_long}"
        else:
            dte = f"{candidate.dte_short}"
        
        # Create a summary of the strikes
        strikes = [f"{leg.strike}" for leg in candidate.legs 
                  if hasattr(leg, 'strike') and leg.strike > 0]
        strike_summary = ", ".join(strikes)
        
        # Add row to data
        data.append({
            "ID": i,
            "Strategy": candidate.strategy_name,
            "DTE": dte,
            "Strikes": strike_summary,
            "Pop %": f"{candidate.probability_of_profit:.1f}%",
            "Exp. Return %": f"{candidate.expected_return:.1f}%",
            "Max Loss %": f"{candidate.max_loss/candidate.buying_power_effect*100:.1f}%" if candidate.buying_power_effect else "N/A",
            "BP Effect": f"${candidate.buying_power_effect:.0f}",
            "Score": f"{candidate.composite_score:.1f}"
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Display table with selection
    st.dataframe(
        df,
        column_config={
            "ID": st.column_config.NumberColumn(
                "ID",
                help="Candidate ID",
                width="small",
            ),
            "Strategy": st.column_config.TextColumn(
                "Strategy",
                help="Strategy name",
                width="medium",
            ),
            "DTE": st.column_config.TextColumn(
                "DTE",
                help="Days to expiration",
                width="small",
            ),
            "Strikes": st.column_config.TextColumn(
                "Strikes",
                help="Option strikes",
                width="medium",
            ),
            "Pop %": st.column_config.TextColumn(
                "PoP %",
                help="Probability of profit",
                width="small",
            ),
            "Exp. Return %": st.column_config.TextColumn(
                "Exp. Return %",
                help="Expected return percentage",
                width="small",
            ),
            "Max Loss %": st.column_config.TextColumn(
                "Max Loss %",
                help="Maximum loss as percentage of buying power",
                width="small",
            ),
            "BP Effect": st.column_config.TextColumn(
                "BP Effect",
                help="Buying power effect",
                width="small",
            ),
            "Score": st.column_config.TextColumn(
                "Score",
                help="Composite score",
                width="small",
            ),
        },
        hide_index=True,
        use_container_width=True,
    )
    
    # Allow selection of a candidate
    selected = st.selectbox(
        "Select a candidate to view details:",
        range(len(candidates)),
        format_func=lambda i: f"{candidates[i].strategy_name} (DTE: {candidates[i].dte_short})"
    )
    
    return selected


def strategy_detail_panel(candidate: StrategyCandidate, symbol: str, spot_price: float) -> None:
    """
    Display detailed information about a strategy candidate.
    
    Args:
        candidate: Strategy candidate to display
        symbol: Ticker symbol
        spot_price: Current spot price
    """
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Risk Profile", "Greeks", "Notes"])
    
    with tab1:
        st.subheader("Risk Profile")
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Profit", f"${candidate.max_profit:.2f}")
        with col2:
            st.metric("Max Loss", f"${candidate.max_loss:.2f}")
        with col3:
            st.metric("BP Effect", f"${candidate.buying_power_effect:.2f}")
        
        # Display scenario analysis if available
        if candidate.scenario_table is not None:
            st.subheader("Scenario Analysis")
            
            # Create heatmap for scenario table
            fig = go.Figure()
            
            # Add table
            scenario_table = candidate.scenario_table.copy()
            
            # Format columns for display
            scenario_table["P/L $"] = scenario_table["P/L $"].map("${:.2f}".format)
            scenario_table["ROI %"] = scenario_table["ROI %"].map("{:.2f}%".format)
            
            fig.add_trace(go.Table(
                header=dict(
                    values=list(scenario_table.columns),
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[scenario_table[col] for col in scenario_table.columns],
                    fill_color=[
                        ['lightgreen' if x > 0 else 'lightcoral' for x in candidate.scenario_table["P/L $"]]
                    ],
                    align='left'
                )
            ))
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Option Greeks")
        
        # Display Greeks
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Delta", f"{candidate.delta:.3f}")
        with col2:
            st.metric("Gamma", f"{candidate.gamma:.5f}")
        with col3:
            st.metric("Theta", f"{candidate.theta:.3f}")
        with col4:
            st.metric("Vega", f"{candidate.vega:.3f}")
        
        # Display strategy structure
        st.subheader("Strategy Structure")
        
        # Create table of legs
        leg_data = []
        for i, leg in enumerate(candidate.legs):
            if hasattr(leg, 'option_type') and hasattr(leg, 'strike'):
                leg_type = leg.option_type.value.upper() if hasattr(leg.option_type, 'value') else "STOCK"
                action = leg.action.value.upper() if hasattr(leg.action, 'value') else "BUY"
                strike = leg.strike if leg.strike > 0 else spot_price
                
                leg_data.append({
                    "Leg": i + 1,
                    "Action": action,
                    "Type": leg_type,
                    "Strike": f"${strike:.2f}",
                    "Expiry": leg.expiration_date,
                    "Qty": leg.quantity,
                    "Delta": f"{leg.delta:.3f}" if leg.delta is not None else "N/A",
                    "Price": f"${leg.price:.2f}" if leg.price is not None else "N/A"
                })
        
        st.dataframe(pd.DataFrame(leg_data), hide_index=True)
    
    with tab3:
        st.subheader("Notes & Details")
        
        # Display strategy notes
        st.markdown(f"**Strategy Notes:**")
        st.write(candidate.notes)
        
        # Display liquidity metrics if available
        st.markdown("**Liquidity Metrics:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Average Spread %", 
                f"{candidate.avg_spread_pct:.2f}%" if hasattr(candidate, 'avg_spread_pct') else "N/A"
            )
        with col2:
            st.metric(
                "Average Open Interest",
                f"{candidate.avg_open_interest:.0f}" if hasattr(candidate, 'avg_open_interest') else "N/A"
            )


def pareto_chart(candidates: List[StrategyCandidate]) -> None:
    """
    Display a Pareto chart showing the trade-offs between profit, probability, and risk.
    
    Args:
        candidates: List of strategy candidates
    """
    if not candidates:
        st.warning("No candidates available for visualization.")
        return
    
    # Extract data for the chart
    data = []
    for candidate in candidates:
        max_loss_pct = candidate.max_loss / candidate.buying_power_effect * 100 if candidate.buying_power_effect else 0
        
        data.append({
            'Strategy': f"{candidate.strategy_name} (DTE: {candidate.dte_short})",
            'Expected Return (%)': candidate.expected_return,
            'Probability of Profit (%)': candidate.probability_of_profit,
            'Max Loss (%)': max_loss_pct,
            'Score': candidate.composite_score
        })
    
    df = pd.DataFrame(data)
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='Expected Return (%)',
        y='Probability of Profit (%)',
        size='Max Loss (%)',
        color='Score',
        hover_name='Strategy',
        size_max=50,
        color_continuous_scale='viridis',
    )
    
    # Update layout
    fig.update_layout(
        title='Strategy Trade-offs: Return vs. Probability vs. Risk',
        xaxis_title='Expected Return (%)',
        yaxis_title='Probability of Profit (%)',
        coloraxis_colorbar_title='Score',
        height=500
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)


def iv_term_structure_heatmap(
    symbol: str,
    iv_data: Dict[str, pd.DataFrame],
    days: int = 30
) -> None:
    """
    Display a heatmap of IV term structure over time.
    
    Args:
        symbol: Ticker symbol
        iv_data: Dictionary of IV data by date
        days: Number of days to show
    """
    # Check if we have enough data
    if len(iv_data) < 2:
        st.warning("Not enough historical IV data available for visualization.")
        return
    
    # Create DataFrame for heatmap
    dates = sorted(iv_data.keys())[-days:]
    
    # Get DTE values from the first date (assuming all dates have the same DTEs)
    dte_values = iv_data[dates[0]].index.tolist()
    
    # Create data for heatmap
    heatmap_data = []
    for date in dates:
        row = {'Date': date}
        for dte in dte_values:
            row[f'DTE {dte}'] = iv_data[date].loc[dte, 0]  # ATM IV (delta = 0)
        heatmap_data.append(row)
    
    df = pd.DataFrame(heatmap_data)
    df = df.set_index('Date')
    
    # Create heatmap
    fig = px.imshow(
        df,
        labels=dict(x="Days to Expiration", y="Date", color="IV (%)"),
        x=df.columns,
        y=df.index,
        color_continuous_scale="RdYlGn_r",  # Higher IV is more red
        title=f"{symbol} IV Term Structure Over Time"
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        xaxis_title="Days to Expiration",
        yaxis_title="Date"
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)


def trade_log_table(log_df: pd.DataFrame) -> None:
    """
    Display a table of past trades from the trade log.
    
    Args:
        log_df: DataFrame containing trade log entries
    """
    if log_df.empty:
        st.warning("No trades in the log.")
        return
    
    # Format the DataFrame for display
    display_df = log_df.copy()
    
    # Sort by date (most recent first)
    display_df = display_df.sort_values('Date', ascending=False)
    
    # Format date column
    if 'Date' in display_df.columns:
        display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d')
    
    # Display the table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


def trade_log_form(symbol: str, candidate: Optional[StrategyCandidate] = None) -> Dict[str, Any]:
    """
    Display a form for adding a trade to the log.
    
    Args:
        symbol: Ticker symbol
        candidate: Optional strategy candidate to pre-fill form
        
    Returns:
        Dict with form values if submitted, empty dict otherwise
    """
    with st.form("trade_log_form"):
        st.write("Add Trade to Log")
        
        # Pre-fill values if candidate provided
        strategy = st.text_input(
            "Strategy",
            value=candidate.strategy_name if candidate else ""
        )
        
        expiration = st.text_input(
            "Expiration",
            value=candidate.legs[0].expiration_date if candidate and candidate.legs else ""
        )
        
        strikes = st.text_input(
            "Strikes",
            value=", ".join([f"{leg.strike}" for leg in candidate.legs]) if candidate else ""
        )
        
        price = st.number_input(
            "Price",
            value=float(candidate.estimated_price) if candidate else 0.0,
            step=0.01
        )
        
        quantity = st.number_input(
            "Quantity",
            value=1,
            min_value=1
        )
        
        notes = st.text_area(
            "Notes",
            value=candidate.notes if candidate else ""
        )
        
        submitted = st.form_submit_button("Add to Log")
        
        if submitted:
            return {
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Symbol": symbol,
                "Strategy": strategy,
                "Expiration": expiration,
                "Strikes": strikes,
                "Price": price,
                "Quantity": quantity,
                "Notes": notes
            }
    
    return {}
