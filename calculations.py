"""
Utility calculation functions for Portfolio Dashboard
"""

import pandas as pd
import numpy as np


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log returns from price data with memory efficiency.
    
    Args:
        prices: DataFrame with price data
        
    Returns:
        DataFrame with log returns
    """
    returns = np.log(prices / prices.shift(1))
    returns = returns.dropna()
    return returns
