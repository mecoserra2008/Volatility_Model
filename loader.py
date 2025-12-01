"""
Data loading and management functions for Portfolio Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import yaml
from typing import List, Dict


@st.cache_data(ttl=3600)
def load_assets_from_yaml(yaml_path: str) -> Dict:
    """
    Load and parse assets from YAML file with efficient memory management.
    
    Args:
        yaml_path: Path to the YAML file containing asset definitions
        
    Returns:
        Dictionary containing parsed asset information
    """
    try:
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except Exception as e:
        st.error(f"Error loading YAML file: {str(e)}")
        return {}


def extract_all_symbols(assets_data: Dict) -> pd.DataFrame:
    """
    Extract all symbols from the loaded assets data into a structured DataFrame.
    
    Args:
        assets_data: Dictionary containing asset categories and symbols
        
    Returns:
        DataFrame with columns: symbol, name, category, description, main_category
    """
    all_assets = []
    
    if 'asset_categories' in assets_data:
        for main_cat, cat_data in assets_data['asset_categories'].items():
            if 'symbols' in cat_data:
                for asset in cat_data['symbols']:
                    all_assets.append({
                        'symbol': asset['symbol'],
                        'name': asset['name'],
                        'category': asset['category'],
                        'description': asset['description'],
                        'main_category': main_cat
                    })
    
    df = pd.DataFrame(all_assets)
    # Clean up memory
    del all_assets
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data_in_batches(
    symbols: List[str],
    start_date: str,
    end_date: str,
    batch_size: int = 5
) -> pd.DataFrame:
    """
    Fetch price data for multiple symbols in batches to avoid yfinance limits.
    Memory-efficient implementation using generators and chunking.

    Args:
        symbols: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        batch_size: Number of symbols to fetch per batch

    Returns:
        DataFrame with adjusted close prices for all symbols
    """
    all_data = {}
    failed_symbols = []
    total_batches = (len(symbols) + batch_size - 1) // batch_size

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        batch_num = i // batch_size + 1

        status_text.text(f"Fetching batch {batch_num}/{total_batches}: {', '.join(batch)}")

        try:
            # Fetch data for current batch
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                progress=False,
                threads=True
            )

            # Extract adjusted close prices
            if len(batch) == 1:
                # Single symbol case
                if not data.empty and 'Close' in data.columns:
                    all_data[batch[0]] = data['Close']
                else:
                    failed_symbols.append(batch[0])
            else:
                # Multiple symbols case
                for symbol in batch:
                    if 'Close' in data.columns:
                        if isinstance(data['Close'], pd.DataFrame):
                            if symbol in data['Close'].columns:
                                # Check if the data is valid (not all NaN)
                                if not data['Close'][symbol].isna().all():
                                    all_data[symbol] = data['Close'][symbol]
                                else:
                                    failed_symbols.append(symbol)
                            else:
                                failed_symbols.append(symbol)
                        else:
                            # Single column returned
                            if not data['Close'].isna().all():
                                all_data[symbol] = data['Close']
                            else:
                                failed_symbols.append(symbol)

            # Clean up batch data from memory
            del data

        except Exception as e:
            st.warning(f"Error fetching batch {batch}: {str(e)}")
            failed_symbols.extend(batch)

        # Update progress
        progress_bar.progress((batch_num) / total_batches)

    progress_bar.empty()
    status_text.empty()

    # Show warning about failed symbols
    if failed_symbols:
        st.warning(f"Failed to download data for: {', '.join(failed_symbols)}. These symbols will be excluded from the analysis.")

    # Combine all data efficiently
    df = pd.DataFrame(all_data)

    # Forward fill then backward fill missing values
    if not df.empty:
        df = df.ffill().bfill()

    # Clean up
    del all_data

    return df
