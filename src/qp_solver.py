"""
Quadratic Programming solver for S&P 500 Canadian Clone.

Minimizes tracking error between a basket of Canadian stocks and SPY
using cvxpy, subject to fully-invested and max-weight constraints.
"""

import pandas as pd
import numpy as np
import cvxpy as cp

from src.config import MAX_WEIGHT, MIN_WEIGHT, BENCHMARK


def optimize_tracking(returns_canadian, returns_spy, max_weight=MAX_WEIGHT):
    """
    Solve the QP to minimize tracking error vs SPY.

    minimize  (1/T) || R_can @ w - r_spy ||²
    s.t.      sum(w) = 1
              0 <= w_i <= max_weight

    Parameters
    ----------
    returns_canadian : pd.DataFrame
        DatetimeIndex × Canadian tickers, daily returns.
    returns_spy : pd.Series
        Daily SPY returns, same index as returns_canadian.

    Returns
    -------
    weights : pd.Series
        Optimal weights indexed by ticker name.
    tracking_error : float
        Annualized tracking error (std of daily return difference × sqrt(252)).
    status : str
        Solver status.
    """
    # Align indices
    common_idx = returns_canadian.index.intersection(returns_spy.index)
    R = returns_canadian.loc[common_idx].dropna(axis=1, how="all")
    r_spy = returns_spy.loc[common_idx]

    # Drop rows where any value is NaN
    valid = R.notna().all(axis=1) & r_spy.notna()
    R = R.loc[valid].values
    r_spy_vals = r_spy.loc[valid].values
    tickers = returns_canadian.loc[common_idx].dropna(axis=1, how="all").columns.tolist()

    n_assets = R.shape[1]
    T = R.shape[0]

    # Define the QP
    w = cp.Variable(n_assets)

    # Tracking error = (1/T) * sum_t (R_t @ w - r_spy_t)^2
    tracking_diff = R @ w - r_spy_vals
    objective = cp.Minimize((1 / T) * cp.sum_squares(tracking_diff))

    constraints = [
        cp.sum(w) == 1,
        w >= MIN_WEIGHT,
        w <= max_weight,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        print(f"[qp_solver] ⚠️ Solver status: {problem.status}")
        return pd.Series(np.zeros(n_assets), index=tickers), np.nan, problem.status

    weights = pd.Series(w.value, index=tickers)
    weights = weights.clip(lower=0)  # Clean tiny negatives from numerical noise
    weights = weights / weights.sum()  # Re-normalize

    # Compute annualized tracking error
    port_returns = R @ weights.values
    te_daily = np.std(port_returns - r_spy_vals)
    te_annual = te_daily * np.sqrt(252)

    return weights, te_annual, problem.status


def rolling_optimization(prices, benchmark_col=BENCHMARK,
                          lookback=252 * 5, rebal_freq="ME",
                          max_weight=MAX_WEIGHT):
    """
    Walk-forward monthly rebalancing using QP optimization.

    Parameters
    ----------
    prices : pd.DataFrame
        DatetimeIndex × all tickers (including benchmark).
    benchmark_col : str
        Column for benchmark returns.
    lookback : int
        Number of trading days to look back for each optimization.
    rebal_freq : str
        Rebalancing frequency ('ME' for month-end, 'QE' for quarter-end).

    Returns
    -------
    weights_history : pd.DataFrame
        DatetimeIndex (rebalance dates) × tickers, weights at each rebalance.
    portfolio_returns : pd.Series
        Daily portfolio returns.
    metrics : dict
        Summary metrics.
    """
    # Compute returns
    returns = prices.pct_change().iloc[1:]

    spy_returns = returns[benchmark_col]
    canadian_returns = returns.drop(columns=[benchmark_col], errors="ignore")

    # Get rebalance dates (end of each month)
    rebal_dates = returns.resample(rebal_freq).last().index
    # Filter dates that have enough history
    min_date = returns.index[lookback] if lookback < len(returns) else returns.index[0]
    rebal_dates = rebal_dates[rebal_dates >= min_date]

    weights_history = {}
    current_weights = None

    print(f"[qp_solver] Running rolling optimization:")
    print(f"  Lookback: {lookback} days, Rebalance: {rebal_freq}")
    print(f"  Rebalance dates: {len(rebal_dates)} (from {rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    for i, date in enumerate(rebal_dates):
        # Get lookback window
        mask = (returns.index <= date) & (returns.index > date - pd.Timedelta(days=lookback * 2))
        window_returns = canadian_returns.loc[mask].tail(lookback)
        window_spy = spy_returns.loc[mask].tail(lookback)

        # Only optimize with tickers that have data in this window
        valid_cols = window_returns.columns[window_returns.notna().mean() > 0.8]
        window_clean = window_returns[valid_cols].fillna(0)

        weights, te, status = optimize_tracking(window_clean, window_spy, max_weight)

        if status in ("optimal", "optimal_inaccurate"):
            # Expand to full universe (0 for missing tickers)
            full_weights = pd.Series(0.0, index=canadian_returns.columns)
            full_weights[weights.index] = weights.values
            current_weights = full_weights
            weights_history[date] = full_weights

            if (i + 1) % 12 == 0 or i == 0:
                print(f"  [{date.date()}] TE={te:.4f}, "
                      f"Top 3: {weights.nlargest(3).to_dict()}")

    weights_df = pd.DataFrame(weights_history).T
    weights_df.index.name = "rebalance_date"

    # Compute daily portfolio returns using period weights
    portfolio_returns = _apply_weights(weights_df, canadian_returns)

    print(f"\n[qp_solver] Optimization complete: {len(weights_df)} rebalance periods")

    return weights_df, portfolio_returns


def _apply_weights(weights_df, returns):
    """
    Apply a weight schedule (rebalanced periodically) to daily returns.

    Between rebalance dates, weights drift with returns.
    """
    daily_returns = pd.Series(0.0, index=returns.index, dtype=float)

    rebal_dates = weights_df.index.tolist()

    for i, rebal_date in enumerate(rebal_dates):
        # Determine the period this weight applies to
        start = rebal_date
        end = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else returns.index[-1]

        period_mask = (returns.index > start) & (returns.index <= end)
        period_returns = returns.loc[period_mask]

        weights = weights_df.loc[rebal_date]
        # Align weights with available columns
        common = weights.index.intersection(period_returns.columns)
        w = weights[common].values

        daily_returns.loc[period_mask] = period_returns[common].values @ w

    return daily_returns
