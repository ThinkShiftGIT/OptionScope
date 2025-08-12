# OptionScope: Options Strategy Scanner & Optimizer

OptionScope is a production-grade options scanner and optimizer that helps traders find optimal options strategies based on current market conditions, volatility metrics, and mechanical rules.

## Features

- **Live Data Ingestion**: Real-time market data, volatility metrics, and event calendars
- **Mechanical Rule Evaluation**: Pre-defined entry criteria with configurable thresholds
- **Comprehensive Strategy Library**: From basic cash-secured puts to advanced calendar+vertical hybrids
- **Strike & DTE Optimization**: No arbitrary caps on days-to-expiration
- **Multi-Objective Scoring**: Balances expected return, probability of profit, capital efficiency, and risk
- **Intuitive Streamlit Interface**: Dashboard, candidate comparison, and trade logging

## Installation

OptionScope uses Poetry for dependency management. Follow these steps to install and run the application:

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/) for dependency management

### Setup

```bash
# Install Poetry (if not already installed)
pipx install poetry    # or: pip install --user poetry

# Clone the repository
git clone https://github.com/yourusername/optionscope.git
cd optionscope

# Create and activate virtual environment with dependencies
poetry env use 3.11
poetry install

# Run the application
poetry run streamlit run app/app_trade_entry.py
```

### Development Commands

```bash
# Run tests
poetry run pytest -q

# Lint code
poetry run ruff check .

# Type check
poetry run mypy app
```

## Configuration

Strategy parameters, scoring weights, and mechanical rules are configurable in `config.yml`.

## Usage

1. Select a symbol and adjust parameters in the sidebar
2. The dashboard will display current market conditions and rule compliance
3. Review candidate strategies ranked by the multi-objective score
4. Click on a strategy to view detailed risk/reward metrics and scenarios
5. Save promising trades to the trade log for future reference

## Strategy Library

OptionScope includes strategies across all complexity levels:

- **Basic**: Cash-Secured Put, Covered Call, Vertical Spreads
- **Intermediate**: Calendars, Diagonals, Iron Condors/Butterflies, Collars, Jade Lizards
- **Advanced**: Ratio Spreads, Backspreads, Unbalanced Condors, Straddles/Strangles

## Assumptions

- Options data quality depends on the yfinance provider
- Default rules are set for moderate volatility environments
- Risk calculations use standard models and may not capture all market conditions
- Liquidity metrics are simplified and should be verified before trade execution

## License

[MIT License](LICENSE)
