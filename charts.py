"""
Visualization functions for Portfolio Dashboard
All plotting functions using Plotly
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List

from models.volatility import VolatilityModels
from utils.calculations import calculate_returns


class Visualizations:
    """
    Class containing all visualization functions using Plotly.
    """
    
    @staticmethod
    def plot_price_evolution(prices: pd.DataFrame, title: str = "Asset Price Evolution"):
        """Plot normalized price evolution of all assets."""
        fig = go.Figure()
        
        # Normalize prices to 100
        normalized_prices = (prices / prices.iloc[0]) * 100
        
        for col in normalized_prices.columns:
            fig.add_trace(go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[col],
                mode='lines',
                name=col,
                hovertemplate='%{x}<br>%{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Normalized Price (Base = 100)",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_correlation_matrix(returns: pd.DataFrame):
        """Plot correlation matrix heatmap."""
        corr_matrix = returns.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Asset Correlation Matrix",
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_volatility_comparison(
        returns: pd.Series,
        symbol: str,
        forecast_horizon: int = 20
    ):
        """Plot GARCH vs EWMA volatility with forecasts."""
        # Validate input
        if returns is None or len(returns) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for volatility analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title=f"Volatility Models Comparison - {symbol}",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                height=500,
                template='plotly_white'
            )
            return fig

        # Calculate volatilities
        ewma_vol = VolatilityModels.ewma_volatility(returns)
        garch_vol, garch_forecast = VolatilityModels.garch_volatility(
            returns, forecast_horizon
        )
        realized_vol = VolatilityModels.realized_volatility(returns, window=20)

        fig = go.Figure()

        # Realized volatility
        if not realized_vol.empty:
            fig.add_trace(go.Scatter(
                x=realized_vol.index,
                y=realized_vol * 100,
                mode='lines',
                name='Realized Volatility (20-day)',
                line=dict(color='gray', width=1)
            ))

        # EWMA volatility
        if not ewma_vol.empty:
            fig.add_trace(go.Scatter(
                x=ewma_vol.index,
                y=ewma_vol * 100,
                mode='lines',
                name='EWMA Volatility',
                line=dict(color='blue', width=2)
            ))

        # GARCH volatility
        if not garch_vol.empty:
            fig.add_trace(go.Scatter(
                x=garch_vol.index,
                y=garch_vol * 100,
                mode='lines',
                name='GARCH(1,1) Volatility',
                line=dict(color='red', width=2)
            ))

        # GARCH forecast
        if garch_forecast is not None and not garch_forecast.empty:
            fig.add_trace(go.Scatter(
                x=garch_forecast.index,
                y=garch_forecast * 100,
                mode='lines',
                name='GARCH Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))

        fig.update_layout(
            title=f"Volatility Models Comparison - {symbol}",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )

        return fig
    
    @staticmethod
    def plot_efficient_frontier(
        mc_results: Dict,
        efficient_frontier: pd.DataFrame,
        user_portfolio: Dict,
        risk_free_rate: float = 0.02
    ):
        """Plot efficient frontier with Monte Carlo portfolios."""
        fig = go.Figure()
        
        # Monte Carlo portfolios
        fig.add_trace(go.Scatter(
            x=np.array(mc_results['volatility']) * 100,
            y=np.array(mc_results['returns']) * 100,
            mode='markers',
            name='Random Portfolios',
            marker=dict(
                size=4,
                color=mc_results['sharpe'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio"),
                opacity=0.6
            ),
            hovertemplate='Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        # Efficient frontier
        if not efficient_frontier.empty:
            fig.add_trace(go.Scatter(
                x=efficient_frontier['volatility'] * 100,
                y=efficient_frontier['return'] * 100,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='red', width=3)
            ))
        
        # Max Sharpe portfolio
        max_sharpe = mc_results['max_sharpe_portfolio']
        fig.add_trace(go.Scatter(
            x=[max_sharpe['volatility'] * 100],
            y=[max_sharpe['return'] * 100],
            mode='markers',
            name='Max Sharpe Portfolio',
            marker=dict(size=15, color='red', symbol='star')
        ))
        
        # User portfolio
        fig.add_trace(go.Scatter(
            x=[user_portfolio['volatility'] * 100],
            y=[user_portfolio['return'] * 100],
            mode='markers',
            name='Your Portfolio',
            marker=dict(size=15, color='blue', symbol='diamond')
        ))
        
        # Capital Market Line
        max_sharpe_slope = (max_sharpe['return'] - risk_free_rate) / max_sharpe['volatility']
        x_cml = np.linspace(0, max(mc_results['volatility']) * 1.2, 100)
        y_cml = risk_free_rate + max_sharpe_slope * x_cml
        
        fig.add_trace(go.Scatter(
            x=x_cml * 100,
            y=y_cml * 100,
            mode='lines',
            name='Capital Market Line',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Efficient Frontier and Portfolio Optimization",
            xaxis_title="Volatility (Annualized %)",
            yaxis_title="Expected Return (Annualized %)",
            hovermode='closest',
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_portfolio_performance(
        prices: pd.DataFrame,
        user_weights: np.ndarray,
        optimal_weights: np.ndarray,
        asset_names: List[str]
    ):
        """Compare portfolio performance over time."""
        returns = calculate_returns(prices)
        
        # Calculate cumulative returns
        user_portfolio_returns = (returns * user_weights).sum(axis=1)
        optimal_portfolio_returns = (returns * optimal_weights).sum(axis=1)
        
        user_cumulative = (1 + user_portfolio_returns).cumprod()
        optimal_cumulative = (1 + optimal_portfolio_returns).cumprod()
        
        # Individual assets
        individual_cumulative = {}
        for i, asset in enumerate(asset_names):
            individual_cumulative[asset] = (1 + returns.iloc[:, i]).cumprod()
        
        fig = go.Figure()
        
        # User portfolio
        fig.add_trace(go.Scatter(
            x=user_cumulative.index,
            y=(user_cumulative - 1) * 100,
            mode='lines',
            name='Your Portfolio',
            line=dict(color='blue', width=3)
        ))
        
        # Optimal portfolio
        fig.add_trace(go.Scatter(
            x=optimal_cumulative.index,
            y=(optimal_cumulative - 1) * 100,
            mode='lines',
            name='Max Sharpe Portfolio',
            line=dict(color='red', width=3)
        ))
        
        # Individual assets
        for asset, cum_ret in individual_cumulative.items():
            fig.add_trace(go.Scatter(
                x=cum_ret.index,
                y=(cum_ret - 1) * 100,
                mode='lines',
                name=asset,
                line=dict(width=1),
                opacity=0.6
            ))
        
        fig.update_layout(
            title="Portfolio Performance Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            hovermode='x unified',
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_portfolio_weights(weights: np.ndarray, labels: List[str], title: str):
        """Plot portfolio weights as pie chart."""
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=weights,
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])

        fig.update_layout(
            title=title,
            height=400,
            template='plotly_white'
        )

        return fig

    @staticmethod
    def plot_garch_diagnostics(garch_results: Dict, title: str = "GARCH Model Diagnostics"):
        """
        Plot comprehensive GARCH model diagnostics including:
        - Standardized residuals
        - ACF of standardized residuals
        - ACF of squared standardized residuals
        - QQ plot
        """
        if not garch_results['success']:
            fig = go.Figure()
            fig.add_annotation(
                text="GARCH model diagnostics not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        from scipy import stats
        from statsmodels.tsa.stattools import acf

        std_resid = garch_results['standardized_residuals'].dropna()

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Standardized Residuals',
                'ACF of Standardized Residuals',
                'ACF of Squared Standardized Residuals',
                'Q-Q Plot'
            )
        )

        # 1. Standardized residuals time series
        fig.add_trace(
            go.Scatter(
                x=std_resid.index,
                y=std_resid.values,
                mode='lines',
                name='Std. Residuals',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )

        # 2. ACF of standardized residuals
        acf_vals = acf(std_resid, nlags=20)
        fig.add_trace(
            go.Bar(
                x=list(range(len(acf_vals))),
                y=acf_vals,
                name='ACF',
                marker_color='steelblue'
            ),
            row=1, col=2
        )
        # Add confidence bands
        conf_level = 1.96 / np.sqrt(len(std_resid))
        fig.add_hline(y=conf_level, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=-conf_level, line_dash="dash", line_color="red", row=1, col=2)

        # 3. ACF of squared standardized residuals
        acf_sq_vals = acf(std_resid**2, nlags=20)
        fig.add_trace(
            go.Bar(
                x=list(range(len(acf_sq_vals))),
                y=acf_sq_vals,
                name='ACF Squared',
                marker_color='coral'
            ),
            row=2, col=1
        )
        fig.add_hline(y=conf_level, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-conf_level, line_dash="dash", line_color="red", row=2, col=1)

        # 4. Q-Q plot
        sorted_resid = np.sort(std_resid.values)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_resid)))

        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_resid,
                mode='markers',
                name='Q-Q',
                marker=dict(color='purple', size=4)
            ),
            row=2, col=2
        )
        # Add 45-degree line
        min_val = min(theoretical_quantiles.min(), sorted_resid.min())
        max_val = max(theoretical_quantiles.max(), sorted_resid.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='45° Line',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=2
        )

        fig.update_layout(
            title=title,
            height=800,
            showlegend=False,
            template='plotly_white'
        )

        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Lag", row=1, col=2)
        fig.update_xaxes(title_text="Lag", row=2, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)

        fig.update_yaxes(title_text="Std. Residuals", row=1, col=1)
        fig.update_yaxes(title_text="ACF", row=1, col=2)
        fig.update_yaxes(title_text="ACF", row=2, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)

        return fig
