"""
Constants and configuration settings for the Portfolio Dashboard
"""

# Page configuration
PAGE_CONFIG = {
    "page_title": "Portfolio Risk Analysis Dashboard",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Default parameters
DEFAULT_YAML_PATH = "LIS_QUANT_PROJECT\\assets.yaml"
DEFAULT_DATE_RANGE_DAYS = 3 * 365  # 3 years
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_PORTFOLIO_VALUE = 1000000
DEFAULT_RISK_FREE_RATE = 0.02
DEFAULT_NUM_PORTFOLIOS = 10000
DEFAULT_BATCH_SIZE = 5

# GARCH parameters
DEFAULT_LAMBDA_PARAM = 0.94  # EWMA decay factor
DEFAULT_FORECAST_HORIZON = 20
DEFAULT_MAX_P = 3
DEFAULT_MAX_Q = 3
DEFAULT_GARCH_CRITERION = 'bic'

# Volatility parameters
ANNUALIZATION_FACTOR = 252  # Trading days per year
DEFAULT_ROLLING_WINDOW = 30
