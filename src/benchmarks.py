"""
Benchmark strategies for comparison against the Canadian Clone.

Implements:
1. VIX Rule-Based: Dynamic allocation based on volatility levels
2. Static 60/40: 60% clone, 40% cash
3. Equal-Weight TSX: Equal allocation across all Canadian stocks
"""

import pandas as pd
import numpy as np

from src.config import BENCHMARK, TEST_START


def compute_vix_proxy(spy_returns, window=21):
    """
    Compute VIX proxy using SPY's rolling 21-day annualized volatility.
    
    VIX ≈ 21-day rolling std × sqrt(252) × 100
    
    Parameters
    ----------
    spy_returns : pd.Series
        Daily SPY returns.
    window : int
        Rolling window in days.
    
    Returns
    -------
    vix_proxy : pd.Series
        Annualized volatility as percentage (VIX-like scale).
    """
    rolling_vol = spy_returns.rolling(window).std() * np.sqrt(252) * 100
    return rolling_vol


def run_vix_rule_benchmark(clone_returns, spy_returns, test_start=TEST_START):
    """
    VIX Rule-Based Benchmark.
    
    Dynamic allocation between clone and cash based on VIX proxy levels:
    - VIX < 25: 100% clone allocation
    - VIX 25-35: 50% clone, 50% cash
    - VIX > 35: 25% clone, 75% cash
    
    Parameters
    ----------
    clone_returns : pd.Series
        Daily returns of the Canadian Clone portfolio.
    spy_returns : pd.Series
        Daily SPY returns (used to compute VIX proxy).
    test_start : str
        Start date for test period evaluation.
    
    Returns
    -------
    vix_rule_returns : pd.Series
        Daily returns of the VIX rule-based strategy.
    """
    # Compute VIX proxy
    vix_proxy = compute_vix_proxy(spy_returns)
    
    # Align all series
    common_idx = clone_returns.index.intersection(vix_proxy.index)
    clone = clone_returns.loc[common_idx]
    vix = vix_proxy.loc[common_idx]
    
    # Filter to test period
    test_mask = clone.index >= pd.Timestamp(test_start)
    clone = clone.loc[test_mask]
    vix = vix.loc[test_mask]
    
    # Determine allocation based on VIX levels
    allocation = pd.Series(index=clone.index, dtype=float)
    allocation[vix > 35] = 0.25   # High fear: 25% clone
    allocation[(vix > 25) & (vix <= 35)] = 0.50   # Elevated: 50% clone
    allocation[vix <= 25] = 1.00  # Normal: 100% clone
    allocation = allocation.fillna(1.0)  # Default to full allocation
    
    # Apply allocation (rest goes to cash = 0% return)
    vix_rule_returns = clone * allocation
    
    return vix_rule_returns


def run_60_40_benchmark(clone_returns, risk_free_rate=0.02, test_start=TEST_START):
    """
    Static 60/40 Benchmark.
    
    Allocation:
    - 60% Canadian Clone
    - 40% Cash (earning risk-free rate)
    
    Parameters
    ----------
    clone_returns : pd.Series
        Daily returns of the Canadian Clone portfolio.
    risk_free_rate : float
        Annual risk-free rate for cash portion (default 2%).
    test_start : str
        Start date for test period evaluation.
    
    Returns
    -------
    returns_60_40 : pd.Series
        Daily returns of the 60/40 strategy.
    """
    # Filter to test period
    clone = clone_returns.loc[clone_returns.index >= pd.Timestamp(test_start)]
    
    # Daily risk-free return
    rf_daily = risk_free_rate / 252
    
    # 60% clone + 40% cash
    returns_60_40 = 0.60 * clone + 0.40 * rf_daily
    
    return returns_60_40


def run_equal_weight_benchmark(canadian_prices, test_start=TEST_START):
    """
    Equal-Weight TSX Benchmark.
    
    Equal allocation across all Canadian stocks, rebalanced daily.
    
    Parameters
    ----------
    canadian_prices : pd.DataFrame
        DatetimeIndex × Canadian tickers, closing prices.
    test_start : str
        Start date for test period evaluation.
    
    Returns
    -------
    ew_returns : pd.Series
        Daily returns of equal-weight portfolio.
    """
    # Compute returns
    returns = canadian_prices.pct_change(fill_method=None).iloc[1:]
    
    # Filter to test period
    returns = returns.loc[returns.index >= pd.Timestamp(test_start)]
    
    # Equal weight = simple average of all stock returns
    ew_returns = returns.mean(axis=1)
    
    return ew_returns


def run_all_benchmarks(clone_returns, spy_returns, canadian_prices, test_start=TEST_START):
    """
    Run all benchmark strategies.
    
    Parameters
    ----------
    clone_returns : pd.Series
        Daily returns of the Canadian Clone portfolio.
    spy_returns : pd.Series
        Daily SPY returns.
    canadian_prices : pd.DataFrame
        Canadian stock prices.
    test_start : str
        Start date for test period.
    
    Returns
    -------
    benchmarks : dict[str, pd.Series]
        Strategy name -> daily returns.
    """
    print("\n" + "=" * 60)
    print("RUNNING BENCHMARK STRATEGIES")
    print("=" * 60)
    
    benchmarks = {}
    
    # 1. VIX Rule-Based
    print("\n[benchmarks] Running VIX Rule-Based...")
    vix_proxy = compute_vix_proxy(spy_returns)
    test_vix = vix_proxy.loc[vix_proxy.index >= pd.Timestamp(test_start)]
    high_vix_days = (test_vix > 25).sum()
    print(f"  VIX proxy > 25 on {high_vix_days} days ({100*high_vix_days/len(test_vix):.1f}%)")
    
    benchmarks["VIX Rule-Based"] = run_vix_rule_benchmark(
        clone_returns, spy_returns, test_start
    )
    
    # 2. Static 60/40
    print("[benchmarks] Running Static 60/40 (2% risk-free)...")
    benchmarks["Static 60/40"] = run_60_40_benchmark(
        clone_returns, risk_free_rate=0.02, test_start=test_start
    )
    
    # 3. Equal-Weight TSX
    print("[benchmarks] Running Equal-Weight TSX...")
    benchmarks["Equal-Weight TSX"] = run_equal_weight_benchmark(
        canadian_prices, test_start
    )
    
    print(f"\n[benchmarks] Generated {len(benchmarks)} benchmark strategies")
    
    return benchmarks
