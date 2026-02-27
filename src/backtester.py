"""
Backtesting and performance evaluation for the Portfolio Optimizer.

Computes standard portfolio performance metrics and generates
comparison tables against benchmarks.
"""

import pandas as pd
import numpy as np


def compute_metrics(returns, name="Portfolio", risk_free_rate=0.02):
    """
    Compute standard portfolio performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    name : str
        Name for display.
    risk_free_rate : float
        Annual risk-free rate for Sharpe/Sortino.

    Returns
    -------
    metrics : dict
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return {"Name": name, "Sharpe": 0, "Max Drawdown": np.nan, "Ann Return (%)": 0,
                "Ann Vol (%)": 0, "Sortino": 0, "Calmar": 0, "Skewness": 0, "Kurtosis": 0}

    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    rf_daily = risk_free_rate / 252
    excess = returns - rf_daily

    sharpe = excess.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    # Sortino (downside deviation)
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-8
    sortino = (ann_return - risk_free_rate) / downside_std

    # Max drawdown
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Tracking error will be computed separately when comparing

    try:
        skew_val = returns.skew()
        kurt_val = returns.kurtosis()
    except (ValueError, ZeroDivisionError):
        skew_val = kurt_val = 0

    metrics = {
        "Name": name,
        "Ann Return (%)": round(ann_return * 100, 2),
        "Ann Vol (%)": round(ann_vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Max DD (%)": round(max_dd * 100, 2),
        "Calmar": round(calmar, 3),
        "Skewness": round(skew_val, 3),
        "Kurtosis": round(kurt_val, 3),
    }

    return metrics


def compute_tracking_error(portfolio_returns, benchmark_returns):
    """
    Compute annualized tracking error between portfolio and benchmark.
    """
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    diff = portfolio_returns.loc[common] - benchmark_returns.loc[common]
    te = diff.std() * np.sqrt(252)
    return te


def compute_turnover(weights_history):
    """
    Compute average portfolio turnover per rebalance.

    Returns
    -------
    avg_turnover : float
        Mean absolute weight change per rebalance.
    """
    if len(weights_history) < 2:
        return 0.0

    turnovers = []
    for i in range(1, len(weights_history)):
        prev = weights_history.iloc[i - 1]
        curr = weights_history.iloc[i]
        turnover = (curr - prev).abs().sum() / 2  # one-way
        turnovers.append(turnover)

    return np.mean(turnovers)


def compare_benchmarks(results_dict, spy_returns):
    """
    Compare multiple strategies against SPY.

    Parameters
    ----------
    results_dict : dict[str, pd.Series]
        Strategy name -> daily returns.
    spy_returns : pd.Series
        SPY daily returns.

    Returns
    -------
    comparison : pd.DataFrame
        One row per strategy with all metrics.
    """
    rows = []

    for name, returns in results_dict.items():
        m = compute_metrics(returns, name=name)
        te = compute_tracking_error(returns, spy_returns)
        m["Tracking Error (%)"] = round(te * 100, 2)

        # Correlation with SPY
        common = returns.index.intersection(spy_returns.index)
        corr = returns.loc[common].corr(spy_returns.loc[common])
        m["Corr w/ SPY"] = round(corr, 3)

        rows.append(m)

    df = pd.DataFrame(rows).set_index("Name")

    print("\n" + "=" * 90)
    print("STRATEGY COMPARISON")
    print("=" * 90)
    print(df.to_string())
    print("=" * 90)

    return df
