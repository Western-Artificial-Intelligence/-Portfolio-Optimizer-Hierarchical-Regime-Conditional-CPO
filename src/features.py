"""
Feature engineering for the Portfolio Optimizer.

Computes returns, rolling statistics, and merges macro indicators
with price data on a common date index.
"""

import pandas as pd
import numpy as np


def compute_returns(prices, method="simple"):
    """
    Compute daily returns from a prices DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        DatetimeIndex × tickers, containing close prices.
    method : str
        "simple" for arithmetic returns, "log" for log returns.

    Returns
    -------
    returns : pd.DataFrame
        Same shape as prices (minus 1 row), containing daily returns.
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()

    returns = returns.iloc[1:]  # drop first NaN row
    print(f"[features] Computed {method} returns: {returns.shape}")
    return returns


def merge_macro(returns, econ, yield_curve):
    """
    Merge macro indicators with returns on the date index.
    Aligns dates using forward-fill for macro data gaps on trading days.

    Parameters
    ----------
    returns : pd.DataFrame
        DatetimeIndex × tickers.
    econ : pd.DataFrame
        Economic indicators (T10Y2Y, DXY, MOVE, etc.).
    yield_curve : pd.DataFrame
        Yield curve spread data.

    Returns
    -------
    merged : pd.DataFrame
        Returns augmented with macro columns.
    """
    # Reindex macro data to match trading days, forward-fill
    econ_aligned = econ.reindex(returns.index, method="ffill")
    yc_aligned = yield_curve.reindex(returns.index, method="ffill")

    # Combine
    merged = returns.copy()

    for col in econ_aligned.columns:
        merged[f"macro_{col}"] = econ_aligned[col]

    for col in yc_aligned.columns:
        merged[f"yc_{col}"] = yc_aligned[col]

    print(f"[features] Merged dataset: {merged.shape[0]} rows × {merged.shape[1]} columns")
    return merged


def compute_rolling_stats(returns, benchmark_col="SPY US Equity",
                           windows=(21, 63, 126)):
    """
    Compute rolling volatility and rolling correlation to the benchmark.

    Parameters
    ----------
    returns : pd.DataFrame
        DatetimeIndex × tickers.
    benchmark_col : str
        Column name of the benchmark (SPY).
    windows : tuple of int
        Rolling window sizes in trading days (21=1mo, 63=3mo, 126=6mo).

    Returns
    -------
    rolling_vol : dict[int, pd.DataFrame]
        Mapping from window → rolling volatility DataFrame.
    rolling_corr : dict[int, pd.Series]
        Mapping from window → rolling correlation to benchmark.
    """
    rolling_vol = {}
    rolling_corr = {}

    benchmark = returns.get(benchmark_col)
    if benchmark is None:
        print(f"[features] ⚠️ Benchmark '{benchmark_col}' not found, skipping correlations")
        benchmark = None

    canadian = returns.drop(columns=[benchmark_col], errors="ignore")

    for w in windows:
        # Rolling annualized volatility
        vol = canadian.rolling(window=w).std() * np.sqrt(252)
        rolling_vol[w] = vol

        # Rolling correlation to SPY
        if benchmark is not None:
            corr = canadian.rolling(window=w).apply(
                lambda x: x.corr(benchmark.loc[x.index]) if len(x.dropna()) > 5 else np.nan,
                raw=False
            )
            rolling_corr[w] = corr

        print(f"[features] Computed rolling stats (window={w})")

    return rolling_vol, rolling_corr


def compute_rolling_correlation_fast(returns, benchmark_col="SPY US Equity",
                                      windows=(21, 63)):
    """
    Faster rolling correlation using pandas rolling.corr().

    Returns
    -------
    corr_dict : dict[int, pd.DataFrame]
        window → DataFrame of rolling correlations to benchmark.
    """
    benchmark = returns[benchmark_col]
    canadian = returns.drop(columns=[benchmark_col], errors="ignore")

    corr_dict = {}
    for w in windows:
        corr_df = canadian.apply(lambda col: col.rolling(w).corr(benchmark))
        corr_dict[w] = corr_df
        print(f"[features] Fast rolling correlation (window={w}): {corr_df.shape}")

    return corr_dict


def filter_sparse_tickers(prices, min_coverage=0.5):
    """
    Remove tickers with less than `min_coverage` fraction of non-null prices.

    Returns
    -------
    filtered : pd.DataFrame
        Only tickers with sufficient data.
    removed : list[str]
        Names of removed tickers.
    """
    coverage = prices.notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    removed = coverage[coverage < min_coverage].index.tolist()

    filtered = prices[keep]

    if removed:
        print(f"[features] Removed {len(removed)} sparse tickers: {removed}")
    print(f"[features] Kept {len(keep)} tickers with ≥{min_coverage*100:.0f}% coverage")

    return filtered, removed
