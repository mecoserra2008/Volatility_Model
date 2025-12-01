"""
Portfolio Analytics for Portfolio Dashboard
Implements portfolio metrics, optimization, and risk calculations
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict


class PortfolioAnalytics:
    """
    Class for portfolio performance, risk metrics, and optimization.
    """
    
    @staticmethod
    def portfolio_returns(
        returns: pd.DataFrame,
        weights: np.ndarray
    ) -> pd.Series:
        """Calculate portfolio returns given asset returns and weights."""
        return (returns * weights).sum(axis=1)
    
    @staticmethod
    def portfolio_volatility(
        returns: pd.DataFrame,
        weights: np.ndarray
    ) -> float:
        """Calculate portfolio volatility (annualized)."""
        cov_matrix = returns.cov() * 252  # Annualize
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return np.sqrt(portfolio_variance)
    
    @staticmethod
    def sharpe_ratio(
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio."""
        portfolio_return = (returns * weights).sum(axis=1).mean() * 252
        portfolio_vol = PortfolioAnalytics.portfolio_volatility(returns, weights)
        
        if portfolio_vol == 0:
            return 0
        
        return (portfolio_return - risk_free_rate) / portfolio_vol
    
    @staticmethod
    def value_at_risk(
        returns: pd.Series,
        confidence_level: float = 0.95,
        portfolio_value: float = 1000000
    ) -> float:
        """
        Calculate Value at Risk (VaR) using historical simulation.
        
        Args:
            returns: Portfolio returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            portfolio_value: Current portfolio value
            
        Returns:
            VaR in monetary terms
        """
        var_percentile = np.percentile(returns, (1 - confidence_level) * 100)
        return abs(var_percentile * portfolio_value)
    
    @staticmethod
    def expected_shortfall(
        returns: pd.Series,
        confidence_level: float = 0.95,
        portfolio_value: float = 1000000
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Portfolio returns
            confidence_level: Confidence level
            portfolio_value: Current portfolio value
            
        Returns:
            Expected Shortfall in monetary terms
        """
        var_percentile = np.percentile(returns, (1 - confidence_level) * 100)
        # Average of returns below VaR
        tail_losses = returns[returns <= var_percentile]
        es = abs(tail_losses.mean() * portfolio_value)
        return es
    
    @staticmethod
    def monte_carlo_optimization(
        returns: pd.DataFrame,
        num_portfolios: int = 10000,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """
        Perform Monte Carlo simulation for portfolio optimization.
        
        Args:
            returns: DataFrame of asset returns
            num_portfolios: Number of random portfolios to generate
            risk_free_rate: Risk-free rate for Sharpe calculation
            
        Returns:
            Dictionary with optimization results
        """
        num_assets = len(returns.columns)
        results = {
            'returns': [],
            'volatility': [],
            'sharpe': [],
            'weights': []
        }
        
        # Progress indicator
        progress_bar = st.progress(0)
        
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            
            # Calculate metrics
            portfolio_return = (returns * weights).sum(axis=1).mean() * 252
            portfolio_vol = PortfolioAnalytics.portfolio_volatility(returns, weights)
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
            
            results['returns'].append(portfolio_return)
            results['volatility'].append(portfolio_vol)
            results['sharpe'].append(sharpe)
            results['weights'].append(weights)
            
            # Update progress every 1000 iterations
            if i % 1000 == 0:
                progress_bar.progress((i + 1) / num_portfolios)
        
        progress_bar.empty()
        
        # Find optimal portfolios
        max_sharpe_idx = np.argmax(results['sharpe'])
        min_vol_idx = np.argmin(results['volatility'])
        
        results['max_sharpe_portfolio'] = {
            'return': results['returns'][max_sharpe_idx],
            'volatility': results['volatility'][max_sharpe_idx],
            'sharpe': results['sharpe'][max_sharpe_idx],
            'weights': results['weights'][max_sharpe_idx]
        }
        
        results['min_volatility_portfolio'] = {
            'return': results['returns'][min_vol_idx],
            'volatility': results['volatility'][min_vol_idx],
            'sharpe': results['sharpe'][min_vol_idx],
            'weights': results['weights'][min_vol_idx]
        }
        
        return results
    
    @staticmethod
    def efficient_frontier(
        returns: pd.DataFrame,
        num_points: int = 100,
        risk_free_rate: float = 0.02
    ) -> pd.DataFrame:
        """
        Calculate the efficient frontier.
        
        Args:
            returns: DataFrame of asset returns
            num_points: Number of points on the frontier
            risk_free_rate: Risk-free rate
            
        Returns:
            DataFrame with efficient frontier points
        """
        from scipy.optimize import minimize
        
        num_assets = len(returns.columns)
        cov_matrix = returns.cov() * 252
        mean_returns = returns.mean() * 252
        
        def portfolio_stats(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_return, portfolio_vol
        
        def neg_sharpe(weights):
            p_return, p_vol = portfolio_stats(weights)
            return -(p_return - risk_free_rate) / p_vol
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Find min and max return portfolios
        def min_return(weights):
            return -np.dot(weights, mean_returns)
        
        result = minimize(
            min_return,
            num_assets * [1. / num_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        min_ret = -result.fun
        
        def max_return(weights):
            return np.dot(weights, mean_returns)
        
        result = minimize(
            lambda x: -max_return(x),
            num_assets * [1. / num_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        max_ret = result.fun
        
        # Generate efficient frontier
        target_returns = np.linspace(min_ret, max_ret, num_points)
        efficient_portfolios = []
        
        for target in target_returns:
            constraints_with_return = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target}
            ]
            
            result = minimize(
                lambda weights: np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))),
                num_assets * [1. / num_assets],
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_with_return,
                options={'ftol': 1e-9, 'maxiter': 1000}
            )
            
            if result.success:
                efficient_portfolios.append({
                    'return': target,
                    'volatility': result.fun,
                    'weights': result.x
                })
        
        return pd.DataFrame(efficient_portfolios)
