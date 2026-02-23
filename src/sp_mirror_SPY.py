"""
SP_mirror_SPY: derivative-overlay extension targeting SPY (S&P 500 price index)
instead of SPY_Total_Returns (total return). Reuses all infrastructure from
sp_mirror_SPY_Total_Returns.

How to run:
  From project root: python run_sp_mirror.py
"""

from __future__ import annotations

from src.sp_mirror_SPY_Total_Returns import (
    SPMirrorConfig,
    load_derivatives_csv_folder,
    align_price_and_fx_series,
    compute_returns_from_prices,
    compute_alpha_series,
    get_rebalance_dates,
    compute_sp_mirror_returns,
    apply_supervisor,
    compute_sp_mirror_metrics,
    metrics_table,
    run_sp_mirror_single,
    OPTIONAL_SERIES,
)

REQUIRED_SERIES = {
    "ES1__PX_SETTLE",
    "USDCAD__PX_LAST",
    "SPX__PX_LAST",      # SPY / S&P 500 price index as primary benchmark
}

BENCH_COL = "SPX__PX_LAST"
BENCH_NAME = "SPY"


def run_sp_mirror_single_spy(
    clone_return,
    derivatives_df,
    config: SPMirrorConfig,
):
    """Wrapper that calls the shared pipeline with SPX as the benchmark."""
    return run_sp_mirror_single(
        clone_return, derivatives_df, config, bench_col=BENCH_COL,
    )
