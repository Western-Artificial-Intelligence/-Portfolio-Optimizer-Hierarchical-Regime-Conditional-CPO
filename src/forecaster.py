"""
Uncertainty / Forecasting module for the AI Supervisor.

Provides forecast uncertainty features as inputs to the meta-labeling
classifier. Uses rolling volatility-based proxies that capture the same
signal as DeepAR variance (widening uncertainty -> higher risk).

This module is designed as a drop-in replaceable component: you can swap
the rolling estimators for full DeepAR probabilistic forecasts later.
"""

import pandas as pd
import numpy as np


def compute_uncertainty_features(returns, benchmark_col="SPY US Equity",
                                  windows=(5, 21, 63), verbose=True):
    """
    Compute uncertainty/risk features from return series.

    These proxy the DeepAR variance signal: high values = high uncertainty.

    Parameters
    ----------
    returns : pd.DataFrame
        DatetimeIndex × tickers (including benchmark).
    windows : tuple of int
        Short, medium, long rolling windows.

    Returns
    -------
    uncertainty : pd.DataFrame
        DatetimeIndex × feature columns.
    """
    features = {}

    # --- Portfolio-level uncertainty ---
    canadian = returns.drop(columns=[benchmark_col], errors="ignore")
    mean_ret = canadian.mean(axis=1)

    for w in windows:
        # 1. Rolling volatility (annualized)
        features[f"vol_{w}d"] = mean_ret.rolling(w).std() * np.sqrt(252)

        # 2. Volatility of volatility (vol-of-vol) — key uncertainty signal
        daily_vol = mean_ret.rolling(w).std()
        features[f"vol_of_vol_{w}d"] = daily_vol.rolling(w).std()

        # 3. Cross-sectional return dispersion (how much stocks disagree)
        features[f"dispersion_{w}d"] = canadian.rolling(w).std().mean(axis=1)

    # --- EWMA variance (exponentially weighted — reacts faster) ---
    features["ewma_var_12"] = mean_ret.ewm(span=12).var() * 252
    features["ewma_var_26"] = mean_ret.ewm(span=26).var() * 252

    # --- SPY-specific uncertainty ---
    if benchmark_col in returns.columns:
        spy = returns[benchmark_col]
        features["spy_vol_21d"] = spy.rolling(21).std() * np.sqrt(252)
        features["spy_vol_63d"] = spy.rolling(63).std() * np.sqrt(252)

        # Relative vol: Canadian vs SPY (>1 means Canadians more volatile)
        # Add epsilon to avoid inf when SPY vol is near zero (e.g. synthetic paths)
        spy_vol = spy.rolling(21).std().clip(lower=1e-10)
        features["relative_vol"] = mean_ret.rolling(21).std() / spy_vol

    # --- Downside risk ---
    neg_returns = mean_ret.clip(upper=0)
    features["downside_vol_21d"] = neg_returns.rolling(21).std() * np.sqrt(252)

    # --- Max drawdown over rolling window ---
    cum = (1 + mean_ret).cumprod()
    for w in [21, 63]:
        rolling_max = cum.rolling(w).max().clip(lower=1e-10)
        rolling_dd = (cum - rolling_max) / rolling_max
        features[f"rolling_dd_{w}d"] = rolling_dd

    result = pd.DataFrame(features, index=returns.index)
    if verbose:
        print(f"[forecaster] Computed {result.shape[1]} uncertainty features")

    return result
