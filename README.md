# Portfolio Risk Analysis & Volatility Forecasting Dashboard

A comprehensive Streamlit application for portfolio optimization, risk analysis, and volatility forecasting using GARCH and EWMA models.

## Features

- **Portfolio Overview**: Normalized price evolution, correlation matrices, and returns distribution
- **Volatility Analysis**: GARCH vs EWMA model comparison with forecasting
- **Portfolio Optimization**: Monte Carlo simulation and efficient frontier calculation
- **Risk Metrics**: VaR, Expected Shortfall, Sharpe Ratio, and drawdown analysis
- **Performance Comparison**: Historical performance tracking and monthly returns heatmaps
- **GARCH Model Analysis**: Comprehensive econometric framework with diagnostics

## Project Structure

```
portfolio_dashboard/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── config/                     # Configuration files
│   ├── __init__.py
│   ├── style.py               # CSS styling
│   └── constants.py           # Application constants
├── data/                       # Data management
│   ├── __init__.py
│   ├── loader.py              # Data loading functions
│   └── assets.yaml            # Asset definitions
├── models/                     # Financial models
│   ├── __init__.py
│   ├── volatility.py          # GARCH and volatility models
│   └── portfolio.py           # Portfolio analytics
├── visualization/              # Plotting functions
│   ├── __init__.py
│   └── charts.py              # Plotly visualizations
└── utils/                      # Utility functions
    ├── __init__.py
    └── calculations.py        # Helper calculations
```

## Installation

1. Clone the repository or extract the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser.

## Configuration

### Asset Selection
- Update `data/assets.yaml` to add or modify available assets
- The YAML file contains organized categories of stocks, indices, ETFs, and other securities

### Portfolio Parameters
Configure via the sidebar:
- **Date Range**: Select historical period for analysis
- **Asset Selection**: Choose from available assets in categories
- **Portfolio Weights**: Equal-weighted or custom allocation
- **Risk Parameters**: VaR confidence level, portfolio value, risk-free rate
- **Monte Carlo Settings**: Number of simulation portfolios

## Key Models

### GARCH Volatility Modeling
- Automatic model order selection (AIC, BIC, HQIC)
- ARCH effects testing (Engle's LM test)
- Comprehensive diagnostics (Ljung-Box, Jarque-Bera)
- 30-day ahead volatility forecasts

### Portfolio Optimization
- Monte Carlo simulation (10,000+ portfolios)
- Efficient frontier calculation
- Maximum Sharpe ratio optimization
- Minimum volatility portfolios

### Risk Metrics
- Value at Risk (VaR) - Historical simulation
- Expected Shortfall (CVaR)
- Maximum drawdown
- Rolling volatility

## Dependencies

- `streamlit`: Web application framework
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `yfinance`: Market data fetching
- `plotly`: Interactive visualizations
- `scipy`: Statistical functions
- `statsmodels`: Time series analysis
- `arch`: GARCH model estimation

## Notes

- Data is fetched from Yahoo Finance via yfinance
- GARCH models require sufficient historical data (minimum 100 observations)
- Cache is implemented for efficient data loading (TTL: 30 minutes for prices, 1 hour for asset definitions)

## License

This project is provided as-is for educational and research purposes.
