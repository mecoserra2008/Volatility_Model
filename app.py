"""
Portfolio Risk Analysis and Volatility Forecasting Dashboard
===========================================================
A comprehensive Streamlit application for portfolio optimization,
risk analysis, and volatility forecasting using GARCH and EWMA models.

Main application entry point.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List
import warnings
warnings.filterwarnings('ignore')

# Import local modules
from config import CUSTOM_CSS, PAGE_CONFIG, DEFAULT_YAML_PATH, ANNUALIZATION_FACTOR
from data import load_assets_from_yaml, extract_all_symbols, fetch_data_in_batches
from models import VolatilityModels, PortfolioAnalytics
from visualization import Visualizations
from utils import calculate_returns

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Custom CSS for better styling
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    """Main application function."""
    
    st.title("📊 Portfolio Risk Analysis & Volatility Forecasting Dashboard")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙ Configuration")
    
    # Load assets
    yaml_path = st.sidebar.text_input(
        "YAML File Path",
        value=DEFAULT_YAML_PATH
    )
    
    if not yaml_path:
        st.warning("Please provide the path to the assets YAML file.")
        return
    
    with st.spinner("Loading assets from YAML..."):
        assets_data = load_assets_from_yaml(yaml_path)
    
    if not assets_data:
        st.error("Failed to load assets. Please check the YAML file path.")
        return
    
    assets_df = extract_all_symbols(assets_data)
    
    # Date range selection
    st.sidebar.subheader("📅 Date Range")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)  # 3 years default
    
    start_date_input = st.sidebar.date_input(
        "Start Date",
        value=start_date,
        max_value=end_date
    )
    
    end_date_input = st.sidebar.date_input(
        "End Date",
        value=end_date,
        min_value=start_date_input
    )
    
    # Asset selection
    st.sidebar.subheader("📈 Asset Selection")
    
    # Filter by category
    categories = ['All'] + sorted(assets_df['main_category'].unique().tolist())
    selected_category = st.sidebar.selectbox("Filter by Category", categories)
    
    if selected_category != 'All':
        filtered_assets = assets_df[assets_df['main_category'] == selected_category]
    else:
        filtered_assets = assets_df
    
    # Asset multiselect
    asset_options = filtered_assets.apply(
        lambda x: f"{x['symbol']} - {x['name']}", axis=1
    ).tolist()
    
    selected_assets = st.sidebar.multiselect(
        "Select Assets (No Limit)",
        asset_options,
        default=asset_options[:3] if len(asset_options) >= 3 else asset_options
    )
    
    if not selected_assets:
        st.warning("Please select at least one asset.")
        return
    
    # Extract symbols
    selected_symbols = [asset.split(' - ')[0] for asset in selected_assets]
    
    # Portfolio weights configuration
    st.sidebar.subheader("⚖ Portfolio Weights")
    weights_input_method = st.sidebar.radio(
        "Weight Input Method",
        ["Equal Weights", "Custom Weights"]
    )
    
    if weights_input_method == "Equal Weights":
        user_weights = np.array([1.0 / len(selected_symbols)] * len(selected_symbols))
    else:
        st.sidebar.write("Enter weights (must sum to 1.0):")
        weight_inputs = []
        for symbol in selected_symbols:
            weight = st.sidebar.number_input(
                f"{symbol}",
                min_value=0.0,
                max_value=1.0,
                value=1.0 / len(selected_symbols),
                step=0.01,
                key=f"weight_{symbol}"
            )
            weight_inputs.append(weight)
        
        user_weights = np.array(weight_inputs)
        
        # Normalize weights
        if user_weights.sum() > 0:
            user_weights = user_weights / user_weights.sum()
            st.sidebar.success(f"Weights normalized. Sum: {user_weights.sum():.4f}")
        else:
            st.sidebar.error("Weights must sum to a positive value!")
            return
    
    # Risk parameters
    st.sidebar.subheader("🎯 Risk Parameters")
    confidence_level = st.sidebar.slider(
        "VaR/ES Confidence Level",
        min_value=0.90,
        max_value=0.99,
        value=0.95,
        step=0.01
    )
    
    portfolio_value = st.sidebar.number_input(
        "Portfolio Value ($)",
        min_value=1000,
        max_value=100000000,
        value=1000000,
        step=10000
    )
    
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (Annual)",
        min_value=0.0,
        max_value=0.10,
        value=0.02,
        step=0.001,
        format="%.3f"
    )
    
    # Optimization parameters
    st.sidebar.subheader("🎲 Monte Carlo Settings")
    num_portfolios = st.sidebar.number_input(
        "Number of Simulations",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000
    )
    
    # Fetch data button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        
        # Store in session state
        st.session_state.run_analysis = True
        st.session_state.selected_symbols = selected_symbols
        st.session_state.user_weights = user_weights
        st.session_state.start_date = start_date_input
        st.session_state.end_date = end_date_input
        st.session_state.confidence_level = confidence_level
        st.session_state.portfolio_value = portfolio_value
        st.session_state.risk_free_rate = risk_free_rate
        st.session_state.num_portfolios = num_portfolios
    
    # Run analysis if button clicked
    if hasattr(st.session_state, 'run_analysis') and st.session_state.run_analysis:
        
        # Fetch data
        with st.spinner("Fetching market data..."):
            prices = fetch_data_in_batches(
                st.session_state.selected_symbols,
                st.session_state.start_date.strftime('%Y-%m-%d'),
                st.session_state.end_date.strftime('%Y-%m-%d')
            )
        
        if prices.empty:
            st.error("No data available for selected assets and date range.")
            return

        # Calculate returns
        returns = calculate_returns(prices)

        # Update selected symbols and weights based on available data
        available_symbols = [sym for sym in st.session_state.selected_symbols if sym in returns.columns]

        if not available_symbols:
            st.error("No data available for any of the selected assets.")
            return

        # Adjust weights for available symbols only
        if len(available_symbols) < len(st.session_state.selected_symbols):
            # Some symbols failed - adjust weights
            failed_symbols = [sym for sym in st.session_state.selected_symbols if sym not in available_symbols]
            st.warning(f"Some symbols failed to download: {', '.join(failed_symbols)}. Adjusting portfolio weights for available assets only.")

            # Create new weight array for available symbols
            old_weights = st.session_state.user_weights
            adjusted_weights = []

            for i, sym in enumerate(st.session_state.selected_symbols):
                if sym in available_symbols:
                    adjusted_weights.append(old_weights[i])

            # Normalize weights
            adjusted_weights = np.array(adjusted_weights)
            if adjusted_weights.sum() > 0:
                adjusted_weights = adjusted_weights / adjusted_weights.sum()
            else:
                # Equal weights if all original weights were zero
                adjusted_weights = np.ones(len(available_symbols)) / len(available_symbols)

            # Update session state
            st.session_state.user_weights = adjusted_weights
            st.session_state.selected_symbols = available_symbols

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Overview",
            "Volatility Analysis",
            "Portfolio Optimization",
            "Risk Metrics",
            "Performance Comparison",
            "GARCH Model Analysis & Forecasting"
        ])
        
        # ===== TAB 1: OVERVIEW =====
        with tab1:
            st.header("Portfolio Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Selected Assets", len(st.session_state.selected_symbols))
            
            with col2:
                st.metric("Data Points", len(prices))
            
            with col3:
                avg_return = (returns.mean() * 252 * 100).mean()
                st.metric("Avg Annual Return", f"{avg_return:.2f}%")
            
            with col4:
                avg_vol = (returns.std() * np.sqrt(252) * 100).mean()
                st.metric("Avg Volatility", f"{avg_vol:.2f}%")
            
            st.markdown("---")
            
            # Price evolution
            st.subheader("Asset Price Evolution (Normalized)")
            fig_prices = Visualizations.plot_price_evolution(prices)
            st.plotly_chart(fig_prices, use_container_width=True)
            
            # Correlation matrix
            st.subheader("Asset Correlation Matrix")
            fig_corr = Visualizations.plot_correlation_matrix(returns)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Returns distribution
            st.subheader("Returns Distribution")
            fig_dist = go.Figure()
            for col in returns.columns:
                fig_dist.add_trace(go.Histogram(
                    x=returns[col] * 100,
                    name=col,
                    opacity=0.7,
                    nbinsx=50
                ))
            fig_dist.update_layout(
                title="Daily Returns Distribution",
                xaxis_title="Return (%)",
                yaxis_title="Frequency",
                barmode='overlay',
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # ===== TAB 2: VOLATILITY ANALYSIS =====
        with tab2:
            st.header("Volatility Forecasting & Model Comparison")

            # Get available symbols (only those that have data)
            available_symbols = [sym for sym in st.session_state.selected_symbols if sym in returns.columns]

            if not available_symbols:
                st.error("No data available for volatility analysis. All selected symbols failed to download.")
            else:
                # Select asset for detailed volatility analysis
                vol_symbol = st.selectbox(
                    "Select Asset for Volatility Analysis",
                    available_symbols
                )

                # Validate that the symbol exists in returns
                if vol_symbol not in returns.columns:
                    st.error(f"Data not available for {vol_symbol}")
                else:
                    vol_returns = returns[vol_symbol]

                    # Validate that vol_returns is not empty
                    if len(vol_returns) == 0:
                        st.error(f"No return data available for {vol_symbol}")
                    else:
                        # Volatility comparison plot
                        st.subheader(f"Volatility Models - {vol_symbol}")
                        fig_vol = Visualizations.plot_volatility_comparison(vol_returns, vol_symbol)
                        st.plotly_chart(fig_vol, use_container_width=True)

                        # Model performance metrics
                        st.subheader("Model Performance Metrics")
                        with st.spinner("Comparing models..."):
                            metrics = VolatilityModels.compare_models(vol_returns)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**EWMA Model**")
                            if metrics['EWMA_RMSE'] is not None:
                                st.metric("RMSE", f"{metrics['EWMA_RMSE']:.6f}")
                                st.metric("MAE", f"{metrics['EWMA_MAE']:.6f}")
                            else:
                                st.warning("EWMA metrics unavailable")

                        with col2:
                            st.write("**GARCH(1,1) Model**")
                            if metrics['GARCH_RMSE'] is not None:
                                st.metric("RMSE", f"{metrics['GARCH_RMSE']:.6f}")
                                st.metric("MAE", f"{metrics['GARCH_MAE']:.6f}")
                            else:
                                st.warning("GARCH model unavailable")

                        # Forecast table
                        st.subheader("GARCH Volatility Forecast (Next 20 Days)")
                        try:
                            _, garch_forecast = VolatilityModels.garch_volatility(vol_returns, 20)
                            if garch_forecast is not None and not garch_forecast.empty:
                                forecast_df = pd.DataFrame({
                                    'Date': garch_forecast.index,
                                    'Forecasted Volatility (%)': garch_forecast.values * 100
                                })
                                st.dataframe(forecast_df, use_container_width=True)
                            else:
                                st.info("GARCH forecast not available")
                        except Exception as e:
                            st.warning(f"Unable to generate GARCH forecast: {str(e)}")
        
        # ===== TAB 3: PORTFOLIO OPTIMIZATION =====
        with tab3:
            st.header("Portfolio Optimization via Monte Carlo Simulation")
            
            with st.spinner(f"Running {st.session_state.num_portfolios:,} simulations..."):
                mc_results = PortfolioAnalytics.monte_carlo_optimization(
                    returns,
                    st.session_state.num_portfolios,
                    st.session_state.risk_free_rate
                )
            
            # Calculate user portfolio metrics
            user_return = (returns * st.session_state.user_weights).sum(axis=1).mean() * 252
            user_vol = PortfolioAnalytics.portfolio_volatility(
                returns,
                st.session_state.user_weights
            )
            user_sharpe = PortfolioAnalytics.sharpe_ratio(
                returns,
                st.session_state.user_weights,
                st.session_state.risk_free_rate
            )
            
            user_portfolio = {
                'return': user_return,
                'volatility': user_vol,
                'sharpe': user_sharpe
            }
            
            # Display metrics
            st.subheader("Portfolio Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Your Portfolio**")
                st.metric("Expected Return", f"{user_return * 100:.2f}%")
                st.metric("Volatility", f"{user_vol * 100:.2f}%")
                st.metric("Sharpe Ratio", f"{user_sharpe:.3f}")
            
            with col2:
                st.write("**Maximum Sharpe Portfolio**")
                max_sharpe = mc_results['max_sharpe_portfolio']
                st.metric("Expected Return", f"{max_sharpe['return'] * 100:.2f}%")
                st.metric("Volatility", f"{max_sharpe['volatility'] * 100:.2f}%")
                st.metric("Sharpe Ratio", f"{max_sharpe['sharpe']:.3f}")
            
            # Efficient frontier
            st.subheader("Efficient Frontier")
            with st.spinner("Calculating efficient frontier..."):
                try:
                    ef_df = PortfolioAnalytics.efficient_frontier(
                        returns,
                        num_points=50,
                        risk_free_rate=st.session_state.risk_free_rate
                    )
                except:
                    ef_df = pd.DataFrame()
                    st.warning("Could not calculate efficient frontier")
            
            fig_ef = Visualizations.plot_efficient_frontier(
                mc_results,
                ef_df,
                user_portfolio,
                st.session_state.risk_free_rate
            )
            st.plotly_chart(fig_ef, use_container_width=True)
            
            # Weight allocation
            st.subheader("Portfolio Weight Allocations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_user_weights = Visualizations.plot_portfolio_weights(
                    st.session_state.user_weights,
                    st.session_state.selected_symbols,
                    "Your Portfolio Weights"
                )
                st.plotly_chart(fig_user_weights, use_container_width=True)
            
            with col2:
                fig_optimal_weights = Visualizations.plot_portfolio_weights(
                    max_sharpe['weights'],
                    st.session_state.selected_symbols,
                    "Max Sharpe Portfolio Weights"
                )
                st.plotly_chart(fig_optimal_weights, use_container_width=True)
            
            # Detailed weights table
            st.subheader("Detailed Weight Comparison")
            weights_comparison = pd.DataFrame({
                'Asset': st.session_state.selected_symbols,
                'Your Weights': st.session_state.user_weights * 100,
                'Optimal Weights': max_sharpe['weights'] * 100
            })
            weights_comparison['Difference'] = (
                weights_comparison['Optimal Weights'] - weights_comparison['Your Weights']
            )
            st.dataframe(
                weights_comparison.style.format({
                    'Your Weights': '{:.2f}%',
                    'Optimal Weights': '{:.2f}%',
                    'Difference': '{:+.2f}%'
                }),
                use_container_width=True
            )
        
        # ===== TAB 4: RISK METRICS =====
        with tab4:
            st.header("Risk Metrics & Analysis")
            
            # Calculate portfolio returns
            portfolio_returns = PortfolioAnalytics.portfolio_returns(
                returns,
                st.session_state.user_weights
            )
            
            # Calculate risk metrics
            var = PortfolioAnalytics.value_at_risk(
                portfolio_returns,
                st.session_state.confidence_level,
                st.session_state.portfolio_value
            )
            
            es = PortfolioAnalytics.expected_shortfall(
                portfolio_returns,
                st.session_state.confidence_level,
                st.session_state.portfolio_value
            )
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    f"Value at Risk ({int(st.session_state.confidence_level*100)}%)",
                    f"${var:,.0f}"
                )
            
            with col2:
                st.metric(
                    f"Expected Shortfall ({int(st.session_state.confidence_level*100)}%)",
                    f"${es:,.0f}"
                )
            
            with col3:
                sharpe = PortfolioAnalytics.sharpe_ratio(
                    returns,
                    st.session_state.user_weights,
                    st.session_state.risk_free_rate
                )
                st.metric("Sharpe Ratio", f"{sharpe:.3f}")
            
            with col4:
                max_drawdown = ((portfolio_returns + 1).cumprod() / 
                               (portfolio_returns + 1).cumprod().cummax() - 1).min()
                st.metric("Max Drawdown", f"{max_drawdown * 100:.2f}%")
            
            st.markdown("---")
            
            # Portfolio returns distribution
            st.subheader("Portfolio Returns Distribution")
            
            fig_dist = go.Figure()
            
            # Histogram
            fig_dist.add_trace(go.Histogram(
                x=portfolio_returns * 100,
                nbinsx=50,
                name='Returns',
                marker_color='lightblue',
                opacity=0.7
            ))
            
            # VaR line
            var_percentile = np.percentile(
                portfolio_returns * 100,
                (1 - st.session_state.confidence_level) * 100
            )
            fig_dist.add_vline(
                x=var_percentile,
                line_dash="dash",
                line_color="red",
                annotation_text=f"VaR ({int(st.session_state.confidence_level*100)}%)"
            )
            
            fig_dist.update_layout(
                title="Portfolio Daily Returns Distribution",
                xaxis_title="Return (%)",
                yaxis_title="Frequency",
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Rolling volatility
            st.subheader("Rolling Volatility (30-Day Window)")
            rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(252) * 100
            
            fig_rolling = go.Figure()
            fig_rolling.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                mode='lines',
                name='Rolling Volatility',
                fill='tozeroy',
                line=dict(color='purple')
            ))
            
            fig_rolling.update_layout(
                title="Portfolio Rolling Volatility (Annualized)",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_rolling, use_container_width=True)
            
            # Drawdown chart
            st.subheader("Portfolio Drawdown")
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max - 1) * 100
            
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red')
            ))
            
            fig_dd.update_layout(
                title="Portfolio Drawdown Over Time",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_dd, use_container_width=True)
        
        # ===== TAB 5: PERFORMANCE COMPARISON =====
        with tab5:
            st.header("Performance Comparison")
            
            # Get optimal weights
            max_sharpe = mc_results['max_sharpe_portfolio']
            
            # Plot performance comparison
            fig_perf = Visualizations.plot_portfolio_performance(
                prices,
                st.session_state.user_weights,
                max_sharpe['weights'],
                st.session_state.selected_symbols
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # Performance metrics table
            st.subheader("Performance Metrics Comparison")
            
            portfolio_returns = PortfolioAnalytics.portfolio_returns(
                returns, st.session_state.user_weights
            )
            optimal_returns = PortfolioAnalytics.portfolio_returns(
                returns, max_sharpe['weights']
            )
            
            metrics_data = []
            
            # User portfolio
            user_total_return = ((1 + portfolio_returns).prod() - 1) * 100
            user_annual_return = (returns * st.session_state.user_weights).sum(axis=1).mean() * 252 * 100
            user_annual_vol = portfolio_returns.std() * np.sqrt(252) * 100
            user_sharpe = PortfolioAnalytics.sharpe_ratio(
                returns, st.session_state.user_weights, st.session_state.risk_free_rate
            )
            
            metrics_data.append({
                'Portfolio': 'Your Portfolio',
                'Total Return (%)': user_total_return,
                'Annual Return (%)': user_annual_return,
                'Annual Volatility (%)': user_annual_vol,
                'Sharpe Ratio': user_sharpe
            })
            
            # Optimal portfolio
            opt_total_return = ((1 + optimal_returns).prod() - 1) * 100
            opt_annual_return = (returns * max_sharpe['weights']).sum(axis=1).mean() * 252 * 100
            opt_annual_vol = optimal_returns.std() * np.sqrt(252) * 100
            opt_sharpe = PortfolioAnalytics.sharpe_ratio(
                returns, max_sharpe['weights'], st.session_state.risk_free_rate
            )
            
            metrics_data.append({
                'Portfolio': 'Max Sharpe Portfolio',
                'Total Return (%)': opt_total_return,
                'Annual Return (%)': opt_annual_return,
                'Annual Volatility (%)': opt_annual_vol,
                'Sharpe Ratio': opt_sharpe
            })
            
            # Individual assets
            for i, symbol in enumerate(st.session_state.selected_symbols):
                asset_returns = returns.iloc[:, i]
                total_ret = ((1 + asset_returns).prod() - 1) * 100
                annual_ret = asset_returns.mean() * 252 * 100
                annual_vol = asset_returns.std() * np.sqrt(252) * 100
                sharpe = (annual_ret / 100 - st.session_state.risk_free_rate) / (annual_vol / 100)
                
                metrics_data.append({
                    'Portfolio': symbol,
                    'Total Return (%)': total_ret,
                    'Annual Return (%)': annual_ret,
                    'Annual Volatility (%)': annual_vol,
                    'Sharpe Ratio': sharpe
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            st.dataframe(
                metrics_df.style.format({
                    'Total Return (%)': '{:.2f}',
                    'Annual Return (%)': '{:.2f}',
                    'Annual Volatility (%)': '{:.2f}',
                    'Sharpe Ratio': '{:.3f}'
                }),
                use_container_width=True
            )
            
            # Monthly returns heatmap
            st.subheader("Monthly Returns Heatmap (Your Portfolio)")
            
            monthly_returns = portfolio_returns.resample('M').apply(
                lambda x: (1 + x).prod() - 1
            ) * 100
            
            # Pivot for heatmap
            monthly_pivot = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values
            })
            monthly_pivot = monthly_pivot.pivot(
                index='Month',
                columns='Year',
                values='Return'
            )
            
            month_names = [
                'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
            ]
            monthly_pivot.index = [month_names[i-1] for i in monthly_pivot.index]
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=monthly_pivot.values,
                x=monthly_pivot.columns,
                y=monthly_pivot.index,
                colorscale='RdYlGn',
                zmid=0,
                text=monthly_pivot.values,
                texttemplate='%{text:.1f}%',
                textfont={"size": 10},
                colorbar=dict(title="Return (%)")
            ))
            
            fig_heatmap.update_layout(
                title="Monthly Returns Heatmap",
                xaxis_title="Year",
                yaxis_title="Month",
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)

        # ===== TAB 6: GARCH MODEL ANALYSIS & FORECASTING =====
        with tab6:
            st.header("GARCH Model Analysis & Forecasting Framework")
            st.markdown("""
            This section implements the complete econometric framework for GARCH volatility modeling:

            **Framework Steps:**
            1. **ARCH Effects Testing** - Engle's LM test to justify GARCH modeling
            2. **Model Order Selection** - Information criteria comparison (AIC, BIC, HQIC)
            3. **Parameter Estimation** - Maximum likelihood with significance tests
            4. **Diagnostic Checking** - Ljung-Box tests and residual analysis
            5. **Volatility Forecasting** - 30-day ahead conditional volatility
            """)

            # Calculate Monte Carlo results if not already done
            if 'mc_results' not in locals():
                with st.spinner("Running portfolio optimization..."):
                    mc_results = PortfolioAnalytics.monte_carlo_optimization(
                        returns,
                        st.session_state.num_portfolios,
                        st.session_state.risk_free_rate
                    )

            # Get max Sharpe portfolio
            max_sharpe = mc_results['max_sharpe_portfolio']

            # Calculate portfolio returns
            user_portfolio_returns = PortfolioAnalytics.portfolio_returns(
                returns,
                st.session_state.user_weights
            )

            optimal_portfolio_returns = PortfolioAnalytics.portfolio_returns(
                returns,
                max_sharpe['weights']
            )

            # Selector for entity to analyze
            st.markdown("---")
            st.subheader("Select Entity for GARCH Analysis")

            entity_type = st.radio(
                "Analysis Target",
                ["Individual Asset", "Your Portfolio", "Max Sharpe Portfolio", "All Entities Comparison"],
                horizontal=True
            )

            st.markdown("---")

            # Function to display complete GARCH framework for one entity
            def display_garch_framework(returns_series, entity_name):
                """Display complete GARCH econometric framework for one entity."""

                st.header(f"GARCH Analysis: {entity_name}")

                # STEP 1: ARCH EFFECTS TESTING
                st.subheader("Step 1: Testing for ARCH Effects")
                st.markdown("""
                **Engle's ARCH LM Test** - Tests null hypothesis of no ARCH effects.
                If p-value < 0.05, ARCH effects are present and GARCH modeling is justified.
                """)

                arch_test = VolatilityModels.arch_lm_test(returns_series, lags=10)

                if arch_test['success']:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("LM Statistic", f"{arch_test['lm_statistic']:.4f}")
                    with col2:
                        st.metric("p-value", f"{arch_test['lm_pvalue']:.4f}")
                    with col3:
                        st.metric("F-Statistic", f"{arch_test['f_statistic']:.4f}")
                    with col4:
                        conclusion = "ARCH Present" if arch_test['lm_pvalue'] < 0.05 else "No ARCH"
                        st.metric("Conclusion", conclusion)

                    if arch_test['lm_pvalue'] < 0.05:
                        st.success(f"ARCH effects detected (p={arch_test['lm_pvalue']:.4f}). GARCH modeling is appropriate.")
                    else:
                        st.warning(f"No significant ARCH effects (p={arch_test['lm_pvalue']:.4f}). GARCH may not be necessary.")
                else:
                    st.error("ARCH test failed. Proceeding with caution.")

                st.markdown("---")

                # STEP 2: MODEL ORDER SELECTION
                st.subheader("Step 2: GARCH Model Order Selection")
                st.markdown("""
                **Information Criteria Comparison** - Testing GARCH(p,q) for p,q ∈ {1,2,3}.
                Lower values indicate better fit penalized for complexity.
                """)

                with st.spinner("Testing multiple GARCH specifications..."):
                    model_selection = VolatilityModels.garch_auto_select(
                        returns_series,
                        max_p=3,
                        max_q=3,
                        criterion='bic'
                    )

                if model_selection['success']:
                    # Display comparison table
                    comparison_df = model_selection['all_models'].copy()
                    comparison_df = comparison_df.sort_values('bic')
                    comparison_df['rank'] = range(1, len(comparison_df) + 1)
                    comparison_df = comparison_df[['rank', 'p', 'q', 'aic', 'bic', 'hqic', 'loglik']]

                    st.dataframe(
                        comparison_df.style.format({
                            'aic': '{:.2f}',
                            'bic': '{:.2f}',
                            'hqic': '{:.2f}',
                            'loglik': '{:.2f}'
                        }).highlight_min(subset=['aic', 'bic', 'hqic'], color='lightgreen'),
                        use_container_width=True
                    )

                    selected_p = model_selection['best_p']
                    selected_q = model_selection['best_q']
                    st.info(f"**Selected Model:** GARCH({selected_p},{selected_q}) based on minimum BIC = {model_selection['best_bic']:.2f}")
                else:
                    st.error("Model selection failed. Using GARCH(1,1) as default.")
                    selected_p, selected_q = 1, 1

                st.markdown("---")

                # STEP 3: PARAMETER ESTIMATION
                st.subheader("Step 3: Parameter Estimation")
                st.markdown("""
                **Maximum Likelihood Estimation** - Coefficients estimated via MLE with asymptotic standard errors.
                """)

                with st.spinner(f"Fitting GARCH({selected_p},{selected_q}) model..."):
                    garch_results = VolatilityModels.garch_comprehensive(
                        returns_series,
                        forecast_horizon=30,
                        p=selected_p,
                        q=selected_q,
                        auto_select=False  # Already selected
                    )

                if garch_results['success']:
                    # Coefficient table
                    st.markdown("**Estimated Coefficients:**")
                    coef_data = []

                    # Build coefficient list dynamically
                    param_order = ['omega']
                    param_order += [f'alpha[{i}]' for i in range(1, selected_q + 1)]
                    param_order += [f'beta[{i}]' for i in range(1, selected_p + 1)]

                    for param_name in param_order:
                        if param_name in garch_results['coefficients']:
                            coef = garch_results['coefficients'][param_name]
                            stderr = garch_results['std_errors'][param_name]
                            pval = garch_results['pvalues'][param_name]
                            t_stat = coef / stderr if stderr > 0 else 0

                            # Significance
                            if pval < 0.01:
                                sig = "***"
                            elif pval < 0.05:
                                sig = "**"
                            elif pval < 0.10:
                                sig = "*"
                            else:
                                sig = ""

                            coef_data.append({
                                'Parameter': param_name,
                                'Estimate': f"{coef:.6f}",
                                'Std. Error': f"{stderr:.6f}",
                                't-statistic': f"{t_stat:.4f}",
                                'p-value': f"{pval:.4f}",
                                'Sig.': sig
                            })

                    coef_df = pd.DataFrame(coef_data)
                    st.dataframe(coef_df, use_container_width=True)
                    st.caption("Significance: *** p<0.01, ** p<0.05, * p<0.10")

                    # Model properties
                    st.markdown("**Model Properties:**")
                    persistence = garch_results['coefficients']['persistence']

                    prop_col1, prop_col2, prop_col3 = st.columns(3)
                    with prop_col1:
                        st.metric("Persistence (Σα + Σβ)", f"{persistence:.6f}")
                    with prop_col2:
                        status = "Stationary" if persistence < 1 else "Non-stationary"
                        st.metric("Stationarity", status)
                    with prop_col3:
                        half_life = -np.log(2) / np.log(persistence) if 0 < persistence < 1 else np.inf
                        st.metric("Half-life (days)", f"{half_life:.1f}" if half_life < 1000 else "∞")

                    if persistence < 1:
                        st.success(f"Model is covariance stationary (persistence = {persistence:.6f} < 1)")
                    else:
                        st.warning(f"Model shows high persistence (persistence = {persistence:.6f} ≥ 1). Consider IGARCH.")

                    # Information criteria
                    st.markdown("**Model Fit Statistics:**")
                    fit_col1, fit_col2, fit_col3, fit_col4 = st.columns(4)
                    with fit_col1:
                        st.metric("Log-Likelihood", f"{garch_results['diagnostics']['log_likelihood']:.2f}")
                    with fit_col2:
                        st.metric("AIC", f"{garch_results['diagnostics']['aic']:.2f}")
                    with fit_col3:
                        st.metric("BIC", f"{garch_results['diagnostics']['bic']:.2f}")
                    with fit_col4:
                        st.metric("Observations", f"{garch_results['diagnostics']['num_obs']}")

                    st.markdown("---")

                    # STEP 4: DIAGNOSTIC CHECKING
                    st.subheader("Step 4: Diagnostic Checking")
                    st.markdown("""
                    **Residual Tests** - Check if model adequately captures volatility dynamics.
                    All tests should have p-values > 0.05 for adequate model specification.
                    """)

                    diag_col1, diag_col2, diag_col3 = st.columns(3)

                    with diag_col1:
                        st.markdown("**Ljung-Box (Residuals)**")
                        st.markdown("Tests for autocorrelation")
                        lb_pval = garch_results['diagnostics']['ljungbox_pvalue']
                        st.metric("p-value", f"{lb_pval:.4f}")
                        if lb_pval > 0.05:
                            st.success("Pass: No autocorrelation")
                        else:
                            st.warning("Fail: Autocorrelation detected")

                    with diag_col2:
                        st.markdown("**Ljung-Box (Squared)**")
                        st.markdown("Tests for remaining ARCH")
                        lb_sq_pval = garch_results['diagnostics']['ljungbox_sq_pvalue']
                        st.metric("p-value", f"{lb_sq_pval:.4f}")
                        if lb_sq_pval > 0.05:
                            st.success("Pass: No ARCH effects")
                        else:
                            st.warning("Fail: ARCH effects remain")

                    with diag_col3:
                        st.markdown("**Jarque-Bera Test**")
                        st.markdown("Tests for normality")
                        jb_pval = garch_results['diagnostics']['jarque_bera_pvalue']
                        st.metric("p-value", f"{jb_pval:.4f}")
                        if jb_pval > 0.05:
                            st.success("Pass: Normally distributed")
                        else:
                            st.info("Non-normal: Consider t-dist")

                    # Diagnostic plots
                    st.markdown("**Diagnostic Plots:**")
                    fig_diag = Visualizations.plot_garch_diagnostics(garch_results, f"Diagnostics: {entity_name}")
                    st.plotly_chart(fig_diag, use_container_width=True)

                    st.markdown("---")

                    # STEP 5: VOLATILITY FORECASTING
                    st.subheader("Step 5: Volatility Forecasting")
                    st.markdown("""
                    **30-Day Ahead Forecasts** - Conditional volatility forecasts based on fitted model.
                    """)

                    if garch_results['forecasts'] is not None and not garch_results['forecasts'].empty:
                        # Forecast plot
                        fig_forecast = go.Figure()

                        # Historical volatility
                        hist_vol = garch_results['conditional_volatility']
                        fig_forecast.add_trace(go.Scatter(
                            x=hist_vol.index,
                            y=hist_vol * 100,
                            mode='lines',
                            name='Historical Volatility',
                            line=dict(color='blue', width=2)
                        ))

                        # Forecast
                        fig_forecast.add_trace(go.Scatter(
                            x=garch_results['forecasts'].index,
                            y=garch_results['forecasts'] * 100,
                            mode='lines',
                            name='30-Day Forecast',
                            line=dict(color='red', width=2, dash='dash')
                        ))

                        fig_forecast.update_layout(
                            title=f"GARCH Volatility Forecast: {entity_name}",
                            xaxis_title="Date",
                            yaxis_title="Volatility (%)",
                            height=500,
                            template='plotly_white'
                        )

                        st.plotly_chart(fig_forecast, use_container_width=True)

                        # Forecast table
                        st.markdown("**Forecast Values:**")
                        forecast_df = pd.DataFrame({
                            'Date': garch_results['forecasts'].index.strftime('%Y-%m-%d'),
                            'Day': range(1, 31),
                            'Volatility (%)': garch_results['forecasts'].values * 100,
                            'Variance': (garch_results['forecasts'].values ** 2) * 10000
                        })

                        # Display in two columns
                        col_left, col_right = st.columns(2)
                        with col_left:
                            st.dataframe(forecast_df.iloc[:15].style.format({
                                'Volatility (%)': '{:.4f}',
                                'Variance': '{:.6f}'
                            }), use_container_width=True)
                        with col_right:
                            st.dataframe(forecast_df.iloc[15:].style.format({
                                'Volatility (%)': '{:.4f}',
                                'Variance': '{:.6f}'
                            }), use_container_width=True)

                        # Forecast statistics
                        st.markdown("**Forecast Statistics:**")
                        fcst_col1, fcst_col2, fcst_col3, fcst_col4 = st.columns(4)
                        with fcst_col1:
                            st.metric("Mean Volatility", f"{garch_results['forecasts'].mean() * 100:.4f}%")
                        with fcst_col2:
                            st.metric("Min Volatility", f"{garch_results['forecasts'].min() * 100:.4f}%")
                        with fcst_col3:
                            st.metric("Max Volatility", f"{garch_results['forecasts'].max() * 100:.4f}%")
                        with fcst_col4:
                            st.metric("Final (Day 30)", f"{garch_results['forecasts'].iloc[-1] * 100:.4f}%")

                    else:
                        st.warning("Forecast generation failed.")

                else:
                    st.error(f"GARCH model estimation failed: {garch_results.get('error', 'Unknown error')}")

            # Execute based on selection
            if entity_type == "Individual Asset":
                available_symbols = [sym for sym in st.session_state.selected_symbols if sym in returns.columns]
                selected_asset = st.selectbox("Select Asset:", available_symbols)

                if selected_asset:
                    asset_returns = returns[selected_asset]
                    display_garch_framework(asset_returns, selected_asset)

            elif entity_type == "Your Portfolio":
                display_garch_framework(user_portfolio_returns, "Your Portfolio")

            elif entity_type == "Max Sharpe Portfolio":
                display_garch_framework(optimal_portfolio_returns, "Maximum Sharpe Ratio Portfolio")

            elif entity_type == "All Entities Comparison":
                st.header("Comprehensive GARCH Comparison")
                st.markdown("Comparing GARCH results across your portfolio, optimal portfolio, and individual assets.")

                # Collect all entities
                entities = {
                    'Your Portfolio': user_portfolio_returns,
                    'Max Sharpe Portfolio': optimal_portfolio_returns
                }

                # Add individual assets (limit to selected ones for display)
                available_symbols = [sym for sym in st.session_state.selected_symbols if sym in returns.columns]
                for symbol in available_symbols[:3]:  # Limit to 3 assets for comparison
                    entities[symbol] = returns[symbol]

                # Run GARCH for all entities
                all_results = {}

                progress_bar = st.progress(0)
                for idx, (name, ret_series) in enumerate(entities.items()):
                    with st.spinner(f"Analyzing {name}..."):
                        # Model selection
                        model_sel = VolatilityModels.garch_auto_select(ret_series)
                        if model_sel['success']:
                            p, q = model_sel['best_p'], model_sel['best_q']
                        else:
                            p, q = 1, 1

                        # Fit model
                        garch_res = VolatilityModels.garch_comprehensive(
                            ret_series,
                            forecast_horizon=30,
                            p=p,
                            q=q,
                            auto_select=False
                        )

                        all_results[name] = {
                            'model_spec': (p, q),
                            'results': garch_res
                        }

                    progress_bar.progress((idx + 1) / len(entities))

                progress_bar.empty()

                # Comparison table
                st.subheader("Model Comparison Summary")

                comparison_data = []
                for name, data in all_results.items():
                    if data['results']['success']:
                        res = data['results']
                        p, q = data['model_spec']

                        comparison_data.append({
                            'Entity': name,
                            'Model': f"GARCH({p},{q})",
                            'Persistence': res['coefficients']['persistence'],
                            'AIC': res['diagnostics']['aic'],
                            'BIC': res['diagnostics']['bic'],
                            'LB p-value': res['diagnostics']['ljungbox_pvalue'],
                            'ARCH p-value': res['diagnostics']['ljungbox_sq_pvalue'],
                            'Mean Forecast (%)': res['forecasts'].mean() * 100 if res['forecasts'] is not None else np.nan
                        })

                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    st.dataframe(comp_df.style.format({
                        'Persistence': '{:.6f}',
                        'AIC': '{:.2f}',
                        'BIC': '{:.2f}',
                        'LB p-value': '{:.4f}',
                        'ARCH p-value': '{:.4f}',
                        'Mean Forecast (%)': '{:.4f}'
                    }), use_container_width=True)

                    # Key insights
                    st.markdown("---")
                    st.subheader("Key Insights")

                    # Find entity with lowest/highest persistence
                    min_pers_idx = comp_df['Persistence'].idxmin()
                    max_pers_idx = comp_df['Persistence'].idxmax()

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Lowest Persistence (Fastest Mean Reversion):**")
                        st.info(f"{comp_df.loc[min_pers_idx, 'Entity']}: {comp_df.loc[min_pers_idx, 'Persistence']:.6f}")
                    with col2:
                        st.markdown("**Highest Persistence (Slowest Mean Reversion):**")
                        st.info(f"{comp_df.loc[max_pers_idx, 'Entity']}: {comp_df.loc[max_pers_idx, 'Persistence']:.6f}")

                    # Forecast comparison plot
                    st.markdown("---")
                    st.subheader("Forecast Comparison")

                    fig_comp = go.Figure()
                    for name, data in all_results.items():
                        if data['results']['success'] and data['results']['forecasts'] is not None:
                            fig_comp.add_trace(go.Scatter(
                                x=data['results']['forecasts'].index,
                                y=data['results']['forecasts'] * 100,
                                mode='lines',
                                name=name
                            ))

                    fig_comp.update_layout(
                        title="30-Day Volatility Forecasts Comparison",
                        xaxis_title="Date",
                        yaxis_title="Volatility (%)",
                        height=500,
                        template='plotly_white'
                    )

                    st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.error("No successful GARCH estimates for comparison.")


if __name__ == "__main__":
if __name__ == '__main__':
    main()
