"""
Volatility Models for Portfolio Dashboard
Implements GARCH, EWMA, and realized volatility calculations
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from datetime import timedelta


class VolatilityModels:
    """
    Class implementing various volatility estimation and forecasting models.
    Includes GARCH, EWMA, and realized volatility calculations.
    """

    @staticmethod
    def arch_lm_test(returns: pd.Series, lags: int = 10) -> Dict:
        """
        Engle's ARCH Lagrange Multiplier test for ARCH effects.

        Tests the null hypothesis that there are no ARCH effects up to lag order.

        Args:
            returns: Series of returns
            lags: Number of lags to test

        Returns:
            Dictionary with test statistic, p-value, and conclusion
        """
        from statsmodels.stats.diagnostic import het_arch

        if returns is None or len(returns) == 0:
            return {'success': False, 'error': 'Empty returns'}

        try:
            # Engle's ARCH test
            lm_stat, lm_pval, f_stat, f_pval = het_arch(returns, nlags=lags)

            return {
                'success': True,
                'lm_statistic': lm_stat,
                'lm_pvalue': lm_pval,
                'f_statistic': f_stat,
                'f_pvalue': f_pval,
                'lags': lags,
                'conclusion': 'ARCH effects detected' if lm_pval < 0.05 else 'No ARCH effects'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    @staticmethod
    def garch_auto_select(
        returns: pd.Series,
        max_p: int = 3,
        max_q: int = 3,
        criterion: str = 'bic'
    ) -> Dict:
        """
        Automatically select the best GARCH(p,q) model based on information criteria.

        Implements proper econometric model selection using AIC, BIC, and HQIC.

        Args:
            returns: Series of returns
            max_p: Maximum GARCH lag order to test
            max_q: Maximum ARCH lag order to test
            criterion: 'aic', 'bic', or 'hqic' for model selection

        Returns:
            Dictionary with best model specification and comparison results
        """
        from arch import arch_model

        # Validate input
        if returns is None or len(returns) == 0:
            return {'success': False, 'error': 'Empty returns'}

        if len(returns) < 100:
            return {'success': False, 'error': 'Insufficient data for model selection'}

        try:
            # Demean returns
            returns_demeaned = returns - returns.mean()
            returns_scaled = returns_demeaned * 100

            model_results = []

            # Test different GARCH specifications
            for p in range(1, max_p + 1):
                for q in range(1, max_q + 1):
                    try:
                        model = arch_model(
                            returns_scaled,
                            vol='Garch',
                            p=p,
                            q=q,
                            mean='Zero',
                            dist='normal'
                        )
                        fitted = model.fit(disp='off', show_warning=False)

                        # Calculate HQIC (Hannan-Quinn Information Criterion)
                        n = len(returns)
                        k = p + q + 1  # number of parameters
                        hqic = -2 * fitted.loglikelihood + 2 * k * np.log(np.log(n))

                        model_results.append({
                            'p': p,
                            'q': q,
                            'aic': fitted.aic,
                            'bic': fitted.bic,
                            'hqic': hqic,
                            'loglik': fitted.loglikelihood,
                            'converged': True
                        })
                    except:
                        continue

            if not model_results:
                return {'success': False, 'error': 'No models converged'}

            # Select best model based on criterion
            model_df = pd.DataFrame(model_results)

            # Normalize criterion name
            criterion_lower = criterion.lower()
            if criterion_lower not in ['aic', 'bic', 'hqic']:
                criterion_lower = 'bic'  # default to BIC

            best_idx = model_df[criterion_lower].idxmin()
            best_spec = model_df.loc[best_idx]

            return {
                'success': True,
                'best_p': int(best_spec['p']),
                'best_q': int(best_spec['q']),
                'all_models': model_df,
                'criterion': criterion_lower.upper(),
                'best_aic': best_spec['aic'],
                'best_bic': best_spec['bic'],
                'best_hqic': best_spec['hqic'],
                'best_loglik': best_spec['loglik']
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    @staticmethod
    def garch_comprehensive(
        returns: pd.Series,
        forecast_horizon: int = 30,
        p: int = 1,
        q: int = 1,
        auto_select: bool = True
    ) -> Dict:
        """
        Fit GARCH model with comprehensive diagnostics and statistical tests.

        Args:
            returns: Series of returns
            forecast_horizon: Number of periods to forecast (default 30 days)
            p: GARCH lag order (used if auto_select=False)
            q: ARCH lag order (used if auto_select=False)
            auto_select: If True, automatically select best (p,q) based on BIC

        Returns:
            Dictionary containing model results, diagnostics, and forecasts
        """
        results = {
            'success': False,
            'model': None,
            'fitted_model': None,
            'conditional_volatility': None,
            'forecasts': None,
            'coefficients': {},
            'pvalues': {},
            'std_errors': {},
            'diagnostics': {},
            'residuals': None,
            'standardized_residuals': None,
            'model_spec': {'p': p, 'q': q}
        }

        # Validate input
        if returns is None or len(returns) == 0:
            return results

        # Automatic model selection
        if auto_select:
            auto_result = VolatilityModels.garch_auto_select(returns)
            if auto_result['success']:
                p = auto_result['best_p']
                q = auto_result['best_q']
                results['model_selection'] = auto_result
                results['model_spec'] = {'p': p, 'q': q}

        if len(returns) < max(50, 10 * (p + q)):  # Need sufficient data
            return results

        try:
            from arch import arch_model
            from scipy import stats

            # Demean returns for GARCH (standard practice)
            returns_demeaned = returns - returns.mean()

            # Scale to percentage for numerical stability
            returns_scaled = returns_demeaned * 100

            # Fit GARCH model
            model = arch_model(
                returns_scaled,
                vol='Garch',
                p=p,
                q=q,
                mean='Zero',  # Already demeaned
                dist='normal'
            )

            fitted_model = model.fit(disp='off', show_warning=False)

            # Extract results
            results['success'] = True
            results['model'] = model
            results['fitted_model'] = fitted_model

            # Conditional volatility (unscale)
            results['conditional_volatility'] = fitted_model.conditional_volatility / 100

            # Coefficients with p-values and standard errors
            params = fitted_model.params
            pvalues = fitted_model.pvalues
            std_errors = fitted_model.std_err

            # Extract all parameters dynamically
            for param_name in params.index:
                results['coefficients'][param_name] = params[param_name]
                results['pvalues'][param_name] = pvalues[param_name]
                results['std_errors'][param_name] = std_errors[param_name]

            # Calculate persistence (sum of all alpha and beta coefficients)
            alpha_sum = sum([v for k, v in results['coefficients'].items() if 'alpha' in k])
            beta_sum = sum([v for k, v in results['coefficients'].items() if 'beta' in k])
            results['coefficients']['persistence'] = alpha_sum + beta_sum

            # Model fit statistics
            results['diagnostics']['aic'] = fitted_model.aic
            results['diagnostics']['bic'] = fitted_model.bic
            results['diagnostics']['log_likelihood'] = fitted_model.loglikelihood
            results['diagnostics']['num_obs'] = fitted_model.nobs

            # Residuals and standardized residuals
            results['residuals'] = fitted_model.resid / 100  # Unscale
            results['standardized_residuals'] = fitted_model.std_resid

            # Ljung-Box test on standardized residuals
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(results['standardized_residuals'], lags=[10], return_df=True)
            results['diagnostics']['ljungbox_stat'] = lb_result['lb_stat'].iloc[0]
            results['diagnostics']['ljungbox_pvalue'] = lb_result['lb_pvalue'].iloc[0]

            # Ljung-Box test on squared standardized residuals (ARCH effects)
            lb_sq_result = acorr_ljungbox(results['standardized_residuals']**2, lags=[10], return_df=True)
            results['diagnostics']['ljungbox_sq_stat'] = lb_sq_result['lb_stat'].iloc[0]
            results['diagnostics']['ljungbox_sq_pvalue'] = lb_sq_result['lb_pvalue'].iloc[0]

            # Jarque-Bera test for normality
            jb_stat, jb_pvalue = stats.jarque_bera(results['standardized_residuals'].dropna())
            results['diagnostics']['jarque_bera_stat'] = jb_stat
            results['diagnostics']['jarque_bera_pvalue'] = jb_pvalue

            # Generate forecasts
            forecasts = fitted_model.forecast(horizon=forecast_horizon)
            forecast_vol = np.sqrt(forecasts.variance.values[-1, :]) / 100

            last_date = returns.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_horizon,
                freq='D'
            )
            results['forecasts'] = pd.Series(forecast_vol, index=forecast_dates)

            return results

        except Exception as e:
            results['error'] = str(e)
            return results

    @staticmethod
    def ewma_volatility(
        returns: pd.Series,
        lambda_param: float = 0.94
    ) -> pd.Series:
        """
        Calculate Exponentially Weighted Moving Average (EWMA) volatility.

        Args:
            returns: Series of returns
            lambda_param: Decay factor (typically 0.94 for daily data)

        Returns:
            Series of EWMA volatility estimates
        """
        # Validate input
        if returns is None or len(returns) == 0:
            return pd.Series(dtype=float)

        if len(returns) < 2:
            return pd.Series([returns.std()], index=returns.index)

        # Square returns
        squared_returns = returns ** 2

        # Initialize variance with sample variance
        variance = pd.Series(index=returns.index, dtype=float)
        variance.iloc[0] = squared_returns.iloc[0]

        # Calculate EWMA variance iteratively
        for i in range(1, len(returns)):
            variance.iloc[i] = (
                lambda_param * variance.iloc[i-1] +
                (1 - lambda_param) * squared_returns.iloc[i]
            )

        # Convert to volatility (standard deviation)
        volatility = np.sqrt(variance)

        return volatility
    
    @staticmethod
    def garch_volatility(
        returns: pd.Series,
        forecast_horizon: int = 20
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Fit GARCH(1,1) model and generate volatility forecasts.

        Args:
            returns: Series of returns
            forecast_horizon: Number of periods to forecast

        Returns:
            Tuple of (fitted volatility, forecasted volatility)
        """
        # Validate input
        if returns is None or len(returns) == 0:
            return pd.Series(dtype=float), None

        if len(returns) < 30:  # GARCH needs sufficient data
            st.warning("Insufficient data for GARCH model. Using EWMA fallback.")
            ewma_vol = VolatilityModels.ewma_volatility(returns)
            return ewma_vol, None

        try:
            from arch import arch_model

            # Scale returns to percentage for numerical stability
            returns_scaled = returns * 100

            # Fit GARCH(1,1) model
            model = arch_model(
                returns_scaled,
                vol='Garch',
                p=1,
                q=1,
                dist='normal'
            )

            # Fit with reduced output
            fitted_model = model.fit(disp='off', show_warning=False)

            # Extract conditional volatility
            conditional_vol = fitted_model.conditional_volatility / 100

            # Generate forecasts
            forecasts = fitted_model.forecast(horizon=forecast_horizon)
            forecast_vol = np.sqrt(forecasts.variance.values[-1, :]) / 100

            # Create forecast series
            last_date = returns.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_horizon,
                freq='D'
            )
            forecast_series = pd.Series(forecast_vol, index=forecast_dates)

            return conditional_vol, forecast_series

        except Exception as e:
            st.warning(f"GARCH model failed: {str(e)}. Using EWMA fallback.")
            # Fallback to EWMA
            ewma_vol = VolatilityModels.ewma_volatility(returns)
            return ewma_vol, None
    
    @staticmethod
    def realized_volatility(
        returns: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate realized volatility using rolling window.

        Args:
            returns: Series of returns
            window: Rolling window size

        Returns:
            Series of realized volatility
        """
        # Validate input
        if returns is None or len(returns) == 0:
            return pd.Series(dtype=float)

        if len(returns) < window:
            return pd.Series(dtype=float)

        return returns.rolling(window=window).std()
    
    @staticmethod
    def compare_models(
        returns: pd.Series,
        test_size: int = 60
    ) -> Dict[str, float]:
        """
        Compare GARCH and EWMA models against realized volatility.

        Args:
            returns: Series of returns
            test_size: Size of test set for comparison

        Returns:
            Dictionary with performance metrics (RMSE, MAE)
        """
        # Validate input
        if returns is None or len(returns) == 0:
            return {
                'EWMA_RMSE': None,
                'EWMA_MAE': None,
                'GARCH_RMSE': None,
                'GARCH_MAE': None
            }

        if len(returns) < test_size + 30:  # Need enough data for train and test
            return {
                'EWMA_RMSE': None,
                'EWMA_MAE': None,
                'GARCH_RMSE': None,
                'GARCH_MAE': None
            }

        # Split data
        train_returns = returns[:-test_size]
        test_returns = returns[-test_size:]

        # Calculate realized volatility on test set
        realized_vol = VolatilityModels.realized_volatility(test_returns, window=1)
        realized_vol = realized_vol.dropna()

        if realized_vol.empty:
            return {
                'EWMA_RMSE': None,
                'EWMA_MAE': None,
                'GARCH_RMSE': None,
                'GARCH_MAE': None
            }

        # EWMA predictions
        ewma_vol = VolatilityModels.ewma_volatility(returns)
        if ewma_vol.empty:
            return {
                'EWMA_RMSE': None,
                'EWMA_MAE': None,
                'GARCH_RMSE': None,
                'GARCH_MAE': None
            }

        ewma_test = ewma_vol[-test_size:].reindex(realized_vol.index)

        # Calculate metrics
        metrics = {}

        # EWMA metrics
        try:
            ewma_rmse = np.sqrt(np.mean((ewma_test - realized_vol) ** 2))
            ewma_mae = np.mean(np.abs(ewma_test - realized_vol))
            metrics['EWMA_RMSE'] = ewma_rmse
            metrics['EWMA_MAE'] = ewma_mae
        except:
            metrics['EWMA_RMSE'] = None
            metrics['EWMA_MAE'] = None

        # Try GARCH
        try:
            garch_vol, _ = VolatilityModels.garch_volatility(train_returns)
            if not garch_vol.empty:
                garch_test = garch_vol[-test_size:].reindex(realized_vol.index)
                garch_rmse = np.sqrt(np.mean((garch_test - realized_vol) ** 2))
                garch_mae = np.mean(np.abs(garch_test - realized_vol))
                metrics['GARCH_RMSE'] = garch_rmse
                metrics['GARCH_MAE'] = garch_mae
            else:
                metrics['GARCH_RMSE'] = None
                metrics['GARCH_MAE'] = None
        except:
            metrics['GARCH_RMSE'] = None
            metrics['GARCH_MAE'] = None

        return metrics
