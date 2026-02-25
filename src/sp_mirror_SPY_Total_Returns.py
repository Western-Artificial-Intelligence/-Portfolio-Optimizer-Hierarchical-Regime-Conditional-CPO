"""
SP_mirror: derivative-overlay extension to the Canadian SP500 tracking system.

How to run:
  From project root: python run_sp_mirror.py
  Or: python -c "from src.sp_mirror_SPY_Total_Returns import *; ..."

Compatibility: rebal_freq "M" | "W" | "D" (no pandas offset parsing); works with pandas>=2.0,<2.2.

Assumptions and limitations (documented in code):
- No futures margin/funding rates; ES1 daily returns used as overlay stream.
- No explicit roll yield; Bloomberg generic continuous ES1 is used as-is.
- No FX forward rates; FX hedge is spot-only with no carry.
- Transaction cost model: cost_bps * |alpha_t - alpha_{t-1}| (simplified proxy).
- All series aligned on intersection of trading dates; forward-fill used
  carefully (macro/yields: ffill; ES/SPY_Total_Returns: ffill only for short gaps).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Required / optional tickers (Bloomberg BDH export: one CSV per ticker_field)
# Filenames use double underscore, e.g. ES1__PX_SETTLE.csv, SPXT__PX_LAST.csv (SPY Total Returns)
# -----------------------------------------------------------------------------
REQUIRED_SERIES = {
    "ES1__PX_SETTLE",   # Equity futures proxy for S&P exposure
    "USDCAD__PX_LAST",  # Spot FX USD/CAD
    "SPXT__PX_LAST",    # SPY Total Returns (S&P 500 total return index)
}
OPTIONAL_SERIES = {
    "SPX__PX_LAST",
    "VIX__PX_LAST", "VIX3M__PX_LAST", "MOVE__PX_LAST", "DXY__PX_LAST",
    "USGG2YR__PX_LAST", "USGG10YR__PX_LAST",
    "Yield_Spread__PX_LAST_USGG10YR-USGG2YR",
    "S5INFT__PX_LAST", "S5HLTH__PX_LAST", "S5FINL__PX_LAST", "S5COND__PX_LAST",
    "S5CONS__PX_LAST", "S5INDU__PX_LAST", "S5MATR__PX_LAST",
    "S5ENER__PX_LAST", "S5UTL__PX_LAST", "S5TELS__PX_LAST",
}


@dataclass
class SPMirrorConfig:
    """Configuration for SP_mirror overlay and evaluation."""

    # Overlay sizing
    overlay_mode: str = "beta"           # "beta" | "vol" | "hybrid" | "regime"
    overlay_lookback: int = 252         # Rolling window L for alpha estimation
    hybrid_lambda: float = 0.5          # hybrid: alpha = lambda*(1/beta) + (1-lambda)*(sigma_bench/sigma_clone)
    alpha_min: float = 0.8
    alpha_max: float = 1.2

    # Regime-conditional alpha: fixed alpha per vol regime.
    # Thresholds are annualized benchmark realized vol boundaries.
    # Only used when overlay_mode = "regime".
    regime_vol_lookback: int = 20       # trailing window to measure realized vol
    regime_thresholds: tuple = (0.15, 0.25)   # (normal/elevated boundary, elevated/crisis boundary)
    regime_alphas: tuple = (1.20, 1.10, 1.00) # (normal alpha, elevated alpha, crisis alpha)

    # Shrinkage: alpha_final = shrinkage_w * alpha_raw + (1 - shrinkage_w) * 1.0
    # Pulls alpha toward 1.0 (no overlay) to reduce noise sensitivity.
    # Not applied in regime mode (regime alphas are already hand-specified).
    shrinkage_w: float = 0.5

    # Vol-regime dampening: when trailing realized vol exceeds its long-run
    # percentile threshold, shrink alpha further toward 1.0.
    vol_dampening: bool = True
    vol_dampening_lookback: int = 20           # trailing window for realized vol
    vol_dampening_percentile: float = 0.90     # threshold (vs full-sample vol distribution)
    vol_dampening_floor: float = 0.3           # dampening multiplier at max vol (0 = full kill)

    # Drawdown circuit-breaker: force alpha = 1.0 when strategy drawdown
    # exceeds threshold; re-enable when drawdown recovers past reset level.
    dd_circuit_breaker: bool = True
    dd_trigger: float = -0.15                  # trigger when rolling DD < -15%
    dd_reset: float = -0.10                    # re-enable when DD > -10%
    dd_lookback: int = 60                      # rolling window for DD measurement

    # Expanding window: use all history from inception instead of fixed L
    use_expanding_window: bool = False

    # Rebalance
    rebal_freq: str = "M"               # "M" | "W" | "D" (monthly default)
    rebal_anchor: str = "end"           # "end" = month-end on trading calendar

    # FX hedge: spot only, no carry
    fx_hedge: bool = True               # Hedge overlay USD exposure via spot USDCAD

    # Cost model (simplified proxy): cost_t = cost_bps * |alpha_t - alpha_{t-1}|
    cost_bps: float = 0.0               # Decimal, e.g. 0.0001 = 1 bp; set 0 to switch off
    cost_per_trade_note: str = "cost_bps * |alpha_t - alpha_{t-1}| (proxy for overlay notional change)"

    # Supervisor
    supervisor_mode: str = "none"        # "none" | "dial" | "cash"
    prob_series: Optional[pd.Series] = None  # P for dial/cash; index = date

    def __post_init__(self):
        if self.overlay_mode not in ("beta", "vol", "hybrid", "regime"):
            raise ValueError("overlay_mode must be 'beta', 'vol', 'hybrid', or 'regime'")
        if self.supervisor_mode not in ("none", "dial", "cash"):
            raise ValueError("supervisor_mode must be 'none', 'dial', or 'cash'")


# -----------------------------------------------------------------------------
# CSV loading: folder of individual Bloomberg BDH CSVs (one ticker/field per file)
# -----------------------------------------------------------------------------

def _infer_date_col(df: pd.DataFrame) -> str:
    """Return name of column to use as date index."""
    for c in df.columns:
        cstr = str(c).strip().lower()
        if cstr in ("date", "dates", "time", "datetime"):
            return c
    return df.columns[0]


def _infer_value_col(df: pd.DataFrame, key: str) -> str:
    """Return name of column to use as value (numeric)."""
    for c in df.columns:
        if c == key or str(c).strip() == key:
            return c
    # Last numeric column
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric:
        return numeric[-1]
    return df.columns[-1]


def load_derivatives_csv_folder(
    folder: str | Path,
    required: Optional[set] = None,
    optional: Optional[set] = None,
) -> pd.DataFrame:
    """
    Load a folder of Bloomberg BDH-style CSVs (one series per file).

    Expected: each file named like "ES1_PX_SETTLE.csv" or "SPXT_PX_LAST.csv" (SPY Total Returns)
    with at least a date column and a value column. Date column is inferred
    (e.g. "date", "Date") or first column; value is inferred from filename or
    last numeric column.

    Parameters
    ----------
    folder : path
        Directory containing CSV files.
    required : set, optional
        Set of required keys (filename without .csv). If any missing, raises.
    optional : set, optional
        Additional optional keys to load if present.

    Returns
    -------
    df : pd.DataFrame
        DatetimeIndex, columns = series keys (e.g. ES1_PX_SETTLE).
        Timezone-naive, sorted index. No duplicate index.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Derivatives folder not found: {folder}")

    required = required or set()
    optional = optional or set()
    all_keys = required | optional

    series_dict = {}
    for path in folder.glob("*.csv"):
        key = path.stem
        if key not in all_keys and not required:
            all_keys = all_keys | {key}
        try:
            raw = pd.read_csv(path, low_memory=False)
        except Exception as e:
            raise RuntimeError(f"Failed to read {path}: {e}") from e

        date_col = _infer_date_col(raw)
        value_col = _infer_value_col(raw, key)

        raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
        raw = raw.dropna(subset=[date_col])
        raw = raw.rename(columns={date_col: "date", value_col: key})
        s = raw[["date", key]].copy()
        s[key] = pd.to_numeric(s[key], errors="coerce")
        s = s.dropna(subset=["date"])
        s = s.set_index("date").sort_index()
        s = s[~s.index.duplicated(keep="first")]
        series_dict[key] = s[key]

    missing = required - set(series_dict)
    if missing:
        raise FileNotFoundError(
            f"Required series missing in {folder}: {sorted(missing)}. "
            f"Loaded: {sorted(series_dict.keys())}"
        )

    df = pd.DataFrame(series_dict)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    print(f"[sp_mirror] Loaded {len(df.columns)} series from {folder}: {list(df.columns)}")
    print(f"[sp_mirror] Date range: {df.index.min().date()} to {df.index.max().date()}")
    return df


# -----------------------------------------------------------------------------
# Align series: intersection of dates, careful forward-fill
# -----------------------------------------------------------------------------

def align_price_and_fx_series(
    derivatives: pd.DataFrame,
    clone_return: pd.Series,
    price_keys: Optional[list] = None,
    macro_keys: Optional[list] = None,
    ffill_limit_price: int = 5,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Align derivatives and clone_return on intersection of trading dates.

    - Price/FX series (ES1, SPY Total Returns, USDCAD): forward-fill with limit for short gaps only.
    - Macro keys: forward-fill (no limit) for weekends/holidays.
    - Returns aligned (clone_return, r_ES, r_bench, r_USDCAD, etc.) and common index.

    Parameters
    ----------
    derivatives : pd.DataFrame
        From load_derivatives_csv_folder.
    clone_return : pd.Series
        Daily clone returns; index = trading dates.
    price_keys : list, optional
        Columns to treat as price/FX (limited ffill). Default: ES1_PX_SETTLE, SPXT_PX_LAST (SPY Total Returns), USDCAD_PX_LAST.
    macro_keys : list, optional
        Columns to treat as macro (full ffill). Rest filled with limit.
    ffill_limit_price : int
        Max forward-fill days for price series to avoid artificial returns across long gaps.

    Returns
    -------
    aligned_prices : pd.DataFrame
        Aligned levels on common index.
    clone_aligned : pd.Series
        Clone returns on common index.
    common_index : pd.DatetimeIndex
        Sorted, timezone-naive.
    """
    price_keys = price_keys or ["ES1__PX_SETTLE", "SPXT__PX_LAST", "USDCAD__PX_LAST"]
    macro_keys = macro_keys or []

    clone_return = clone_return.dropna()
    if clone_return.index.tz is not None:
        clone_return = clone_return.tz_localize(None)
    clone_return = clone_return.sort_index()

    idx = clone_return.index.intersection(derivatives.index)
    idx = idx.sort_values()
    if len(idx) == 0:
        raise ValueError("No overlapping dates between clone_return and derivatives.")

    # Reindex derivatives: for macro use ffill(); for price use ffill(limit=ffill_limit_price)
    out = pd.DataFrame(index=idx)
    for col in derivatives.columns:
        s = derivatives[col].reindex(idx)
        if col in macro_keys:
            s = s.ffill()
        else:
            s = s.ffill(limit=ffill_limit_price)
        out[col] = s

    clone_aligned = clone_return.reindex(idx)
    # Drop rows where required price series are still NaN (long gaps)
    required_cols = [c for c in price_keys if c in out.columns]
    valid = out[required_cols].notna().all(axis=1)
    out = out.loc[valid]
    clone_aligned = clone_aligned.reindex(out.index)
    common = out.index.intersection(clone_aligned.index)
    common = common[clone_aligned.loc[common].notna()].sort_values()
    out = out.loc[common]
    clone_aligned = clone_aligned.loc[common]

    return out, clone_aligned, common


def compute_returns_from_prices(aligned_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns via pct_change(). Drops first row (NaN)."""
    ret = aligned_prices.pct_change()
    return ret.iloc[1:]


# -----------------------------------------------------------------------------
# Alpha estimation (rolling)
# -----------------------------------------------------------------------------

def rolling_beta(clone_ret: pd.Series, bench_ret: pd.Series, window: int) -> pd.Series:
    """Rolling beta = cov(clone, bench) / var(bench) over window."""
    common = clone_ret.align(bench_ret, join="inner")[0]
    cov = common.rolling(window).cov(bench_ret.reindex(common.index).ffill().bfill())
    var_b = bench_ret.reindex(common.index).ffill().bfill().rolling(window).var()
    beta = cov / var_b
    return beta


def compute_alpha_series(
    clone_return: pd.Series,
    bench_return: pd.Series,
    r_ES: pd.Series,
    config: SPMirrorConfig,
    rebal_dates: pd.DatetimeIndex,
) -> pd.Series:
    """
    Compute overlay alpha on rebalance dates and forward-fill until next rebalance.

    Modes:
      beta/vol/hybrid — rolling estimation with shrinkage:
        1. Estimate raw alpha from beta/vol/hybrid.
        2. Apply shrinkage: alpha = w * raw + (1 - w) * 1.0
        3. Clip to [alpha_min, alpha_max].
        4. Forward-fill between rebalance dates.

      regime — fixed alpha per vol regime (no rolling estimation):
        1. Compute trailing realized vol of benchmark.
        2. Classify into normal / elevated / crisis.
        3. Look up alpha from regime_alphas table.
        4. Update on rebalance dates, forward-fill.

    After construction:
      5. Vol-regime dampening (if enabled, only for non-regime modes).
    """
    common = clone_return.index.intersection(bench_return.index).intersection(r_ES.index)
    common = common.sort_values()
    c = clone_return.reindex(common)
    b = bench_return.reindex(common)
    es = r_ES.reindex(common)

    alpha_daily = pd.Series(index=common, dtype=float)
    alpha_daily.iloc[:] = np.nan

    if config.overlay_mode == "regime":
        # Regime-conditional: use trailing benchmark vol to select alpha
        rvl = config.regime_vol_lookback
        trailing_vol = b.rolling(rvl).std() * np.sqrt(252)
        lo, hi = config.regime_thresholds
        a_normal, a_elevated, a_crisis = config.regime_alphas

        for d in rebal_dates:
            if d not in common:
                continue
            vol_d = trailing_vol.loc[d] if d in trailing_vol.index and not np.isnan(trailing_vol.loc[d]) else None
            if vol_d is None:
                continue
            if vol_d >= hi:
                alpha_daily.loc[d] = a_crisis
            elif vol_d >= lo:
                alpha_daily.loc[d] = a_elevated
            else:
                alpha_daily.loc[d] = a_normal

        alpha_daily = alpha_daily.ffill()
        alpha_daily = alpha_daily.fillna(a_normal)

    else:
        # Dynamic estimation modes: beta / vol / hybrid
        L = config.overlay_lookback
        min_obs = L // 2
        w = config.shrinkage_w

        for d in rebal_dates:
            if d not in common:
                continue
            loc = common.get_loc(d)
            if config.use_expanding_window:
                start = 0
            else:
                start = max(0, loc - L)
            end = loc + 1
            c_win = c.iloc[start:end].dropna()
            b_win = b.iloc[start:end].dropna()
            if len(c_win) < min_obs or len(b_win) < min_obs:
                continue
            idx_win = c_win.index.intersection(b_win.index)
            if len(idx_win) < min_obs:
                continue
            if not config.use_expanding_window:
                c_win = c_win.loc[idx_win].tail(L)
                b_win = b_win.loc[idx_win].tail(L)
            else:
                c_win = c_win.loc[idx_win]
                b_win = b_win.loc[idx_win]
            cov_cb = c_win.cov(b_win)
            var_b = b_win.var()
            if var_b <= 0:
                continue
            beta = cov_cb / var_b
            sigma_clone = c_win.std()
            sigma_bench = b_win.std()
            if sigma_clone <= 0:
                continue

            if config.overlay_mode == "beta":
                a_raw = 1.0 / beta if beta > 0 else 1.0
            elif config.overlay_mode == "vol":
                a_raw = sigma_bench / sigma_clone if sigma_bench > 0 else 1.0
            else:
                a_beta = 1.0 / beta if beta > 0 else 1.0
                a_vol = sigma_bench / sigma_clone if sigma_bench > 0 else 1.0
                a_raw = config.hybrid_lambda * a_beta + (1 - config.hybrid_lambda) * a_vol

            a = w * a_raw + (1.0 - w) * 1.0
            a = np.clip(a, config.alpha_min, config.alpha_max)
            alpha_daily.loc[d] = a

        alpha_daily = alpha_daily.ffill()
        alpha_daily = alpha_daily.fillna(1.0)

    # Vol-regime dampening: reduce overlay when trailing vol is elevated
    # (skipped for regime mode — it already uses vol to select alpha)
    if config.vol_dampening and config.overlay_mode != "regime":
        vl = config.vol_dampening_lookback
        trailing_vol = b.rolling(vl).std() * np.sqrt(252)
        trailing_vol = trailing_vol.reindex(common)
        vol_threshold = trailing_vol.quantile(config.vol_dampening_percentile)

        if vol_threshold > 0:
            vol_ratio = (trailing_vol / vol_threshold).clip(lower=0.0)
            dampening = np.where(
                vol_ratio > 1.0,
                np.clip(
                    1.0 - (1.0 - config.vol_dampening_floor) * (vol_ratio - 1.0),
                    config.vol_dampening_floor,
                    1.0,
                ),
                1.0,
            )
            dampening = pd.Series(dampening, index=common)
            alpha_daily = 1.0 + (alpha_daily - 1.0) * dampening

    return alpha_daily


def get_rebalance_dates(
    index: pd.DatetimeIndex,
    freq: str = "M",
    anchor: str = "end",
) -> pd.DatetimeIndex:
    """Rebalance dates: last available trading day per period (month/week)."""
    if freq == "D":
        return index
    s = pd.Series(range(len(index)), index=index)
    if freq == "W":
        grouped = s.groupby([s.index.isocalendar().year, s.index.isocalendar().week])
    else:
        grouped = s.groupby([s.index.year, s.index.month])
    last_dates = grouped.apply(lambda g: g.index[-1])
    rebal = pd.DatetimeIndex(last_dates.values).sort_values()
    return rebal


# -----------------------------------------------------------------------------
# SP_mirror return computation
# -----------------------------------------------------------------------------

def compute_sp_mirror_returns(
    clone_return: pd.Series,
    r_ES: pd.Series,
    r_USDCAD: pd.Series,
    bench_return: pd.Series,
    config: SPMirrorConfig,
    rebal_dates: Optional[pd.DatetimeIndex] = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute SP_mirror daily returns: clone + overlay + fx_hedge - costs.

    - overlay_return = (alpha - 1) * r_ES
    - fx_hedge_return = -max(0, alpha - 1) * r_USDCAD  (hedge only overlay USD)
    - cost_t = cost_bps * |alpha_t - alpha_{t-1}|
    - sp_mirror_return = clone + overlay + fx_hedge - cost

    Returns
    -------
    sp_mirror_return : pd.Series
    overlay_return : pd.Series
    fx_hedge_return : pd.Series
    alpha_used : pd.Series
    """
    common = clone_return.index.intersection(r_ES.index).intersection(r_USDCAD.index).intersection(bench_return.index)
    common = common.sort_values()
    clone = clone_return.reindex(common).fillna(0)
    r_es = r_ES.reindex(common).fillna(0)
    r_fx = r_USDCAD.reindex(common).fillna(0)
    bench = bench_return.reindex(common).fillna(0)

    if rebal_dates is None:
        rebal_dates = get_rebalance_dates(common, config.rebal_freq, config.rebal_anchor)

    alpha = compute_alpha_series(clone, bench, r_es, config, rebal_dates)
    alpha = alpha.reindex(common).ffill().fillna(1.0)

    # Drawdown circuit-breaker: force alpha=1 when strategy DD exceeds threshold
    if config.dd_circuit_breaker:
        dd_w = config.dd_lookback
        cum = (1 + clone).cumprod()
        rolling_max = cum.rolling(dd_w, min_periods=1).max()
        rolling_dd = cum / rolling_max - 1.0

        breaker_on = False
        for i, d in enumerate(common):
            dd_val = rolling_dd.loc[d]
            if not breaker_on and dd_val < config.dd_trigger:
                breaker_on = True
            elif breaker_on and dd_val > config.dd_reset:
                breaker_on = False
            if breaker_on:
                alpha.loc[d] = 1.0

    overlay = (alpha - 1.0) * r_es
    if config.fx_hedge:
        fx_hedge = -np.maximum(alpha - 1.0, 0.0) * r_fx
    else:
        fx_hedge = pd.Series(0.0, index=common)

    alpha_prev = alpha.shift(1)
    alpha_prev = alpha_prev.fillna(1.0)
    cost = config.cost_bps * (alpha - alpha_prev).abs()

    raw_return = clone + overlay + fx_hedge
    sp_mirror_return = raw_return - cost

    return sp_mirror_return, overlay, fx_hedge, alpha


def apply_supervisor(
    sp_mirror_return: pd.Series,
    overlay_return: pd.Series,
    fx_hedge_return: pd.Series,
    alpha_used: pd.Series,
    r_ES: pd.Series,
    r_USDCAD: pd.Series,
    config: SPMirrorConfig,
) -> pd.Series:
    """
    Apply supervisor mode: none (identity), dial (alpha_final = 1 + P*(alpha-1)),
    or cash (final = P * sp_mirror_return).

    If supervisor_mode is dial/cash and prob_series is None, returns unchanged.
    """
    if config.supervisor_mode == "none":
        return sp_mirror_return
    P = config.prob_series
    if P is None or len(P) == 0:
        return sp_mirror_return
    common = sp_mirror_return.index.intersection(P.index)
    if len(common) == 0:
        return sp_mirror_return
    P = P.reindex(sp_mirror_return.index).ffill().bfill().fillna(1.0)

    if config.supervisor_mode == "cash":
        return P * sp_mirror_return

    # dial: alpha_final = 1 + P*(alpha - 1); recompute overlay and fx_hedge from alpha_final
    alpha_final = 1.0 + P * (alpha_used - 1.0)
    r_es_aligned = r_ES.reindex(alpha_final.index).fillna(0)
    r_fx_aligned = r_USDCAD.reindex(alpha_final.index).fillna(0)
    overlay_new = (alpha_final - 1.0) * r_es_aligned
    fx_new = -np.maximum(alpha_final - 1.0, 0.0) * r_fx_aligned
    alpha_prev = alpha_final.shift(1).fillna(1.0)
    cost_new = config.cost_bps * (alpha_final - alpha_prev).abs()
    # Recover clone component: sp_mirror = clone + overlay + fx_hedge - cost
    # so clone = sp_mirror - overlay - fx_hedge + cost.  Approximate cost from original alpha.
    alpha_prev_orig = alpha_used.shift(1).fillna(1.0)
    cost_orig = config.cost_bps * (alpha_used - alpha_prev_orig).abs()
    clone_component = sp_mirror_return - overlay_return - fx_hedge_return + cost_orig
    idx = clone_component.index.intersection(overlay_new.index)
    clone_component = clone_component.reindex(idx).fillna(0)
    overlay_new = overlay_new.reindex(idx).fillna(0)
    fx_new = fx_new.reindex(idx).fillna(0)
    cost_new = cost_new.reindex(idx).fillna(0)
    final_return = clone_component + overlay_new + fx_new - cost_new
    return final_return


# -----------------------------------------------------------------------------
# Metrics – matches main.py compare_benchmarks() columns, plus overlay extras
# -----------------------------------------------------------------------------

RISK_FREE_RATE = 0.02


def compute_sp_mirror_metrics(
    strategy_return: pd.Series,
    bench_return: pd.Series,
    name: str = "Strategy",
    fx_return: Optional[pd.Series] = None,
    bench_name: str = "SPY_Total_Returns",
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """
    Portfolio metrics matching main.py's compare_benchmarks() output,
    extended with overlay-specific columns.
    """
    common = strategy_return.dropna().index.intersection(bench_return.dropna().index)
    common = common.sort_values()
    if len(common) < 22:
        return {"Name": name, "Error": "Insufficient overlapping data"}
    r = strategy_return.loc[common].fillna(0)
    b = bench_return.loc[common].fillna(0)

    # ── main.py compute_metrics columns ──────────────────────────────────
    ann_return = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    rf_daily = risk_free_rate / 252
    excess = r - rf_daily
    sharpe = excess.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0

    downside = r[r < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-8
    sortino = (ann_return - risk_free_rate) / downside_std

    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # ── main.py compare_benchmarks columns ───────────────────────────────
    active = r - b
    te_ann = active.std() * np.sqrt(252)
    corr = r.corr(b)

    # ── overlay-specific extras ──────────────────────────────────────────
    if b.var() > 0:
        X = np.column_stack([np.ones(len(b)), b.values])
        betas = np.linalg.lstsq(X, r.values, rcond=None)[0]
        beta_full = betas[1]
    else:
        beta_full = np.nan

    mean_active_ann = active.mean() * 252
    ir = mean_active_ann / te_ann if te_ann > 0 else np.nan

    rolling_beta_252 = r.rolling(252).cov(b) / b.rolling(252).var()
    beta_rolling_mean = rolling_beta_252.mean() if rolling_beta_252.notna().any() else np.nan

    cum_strat = cum * 100
    cum_bench = (1 + b).cumprod() * 100
    avg_cum_gap = (cum_strat - cum_bench).mean()

    out = {
        "Name": name,
        # --- same columns as main.py ---
        "Ann Return (%)": round(ann_return * 100, 2),
        "Ann Vol (%)": round(ann_vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Max DD (%)": round(max_dd * 100, 2),
        "Calmar": round(calmar, 3),
        "Skewness": round(r.skew(), 3),
        "Kurtosis": round(r.kurtosis(), 3),
        "Tracking Error (%)": round(te_ann * 100, 2),
        f"Corr w/ {bench_name}": round(corr, 3),
        # --- overlay extras ---
        f"Beta vs {bench_name}": round(beta_full, 3),
        "Beta rolling mean": round(beta_rolling_mean, 3) if not np.isnan(beta_rolling_mean) else None,
        "Mean Active Return (%)": round(mean_active_ann * 100, 2),
        "Information Ratio": round(ir, 3) if not np.isnan(ir) else None,
        "Avg Cumulative Gap (pts)": round(avg_cum_gap, 2),
    }
    if fx_return is not None:
        common_fx = common.intersection(fx_return.dropna().index)
        if len(common_fx) > 10:
            out["FX beta proxy (corr USDCAD)"] = round(r.loc[common_fx].corr(fx_return.loc[common_fx]), 3)
    return out


def metrics_table(
    results_dict: dict[str, pd.Series],
    bench_return: pd.Series,
    fx_return: Optional[pd.Series] = None,
    bench_name: str = "SPY_Total_Returns",
) -> pd.DataFrame:
    """Build a metrics DataFrame for each strategy in results_dict."""
    rows = []
    for name, returns in results_dict.items():
        m = compute_sp_mirror_metrics(returns, bench_return, name=name, fx_return=fx_return, bench_name=bench_name)
        if "Error" in m:
            rows.append(m)
            continue
        rows.append(m)
    return pd.DataFrame(rows).set_index("Name")


# -----------------------------------------------------------------------------
# Full pipeline: load -> align -> compute -> metrics
# -----------------------------------------------------------------------------

def run_sp_mirror_single(
    clone_return: pd.Series,
    derivatives_df: pd.DataFrame,
    config: SPMirrorConfig,
    bench_col: str = "SPXT__PX_LAST",
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Single run: align data, compute SP_mirror returns, return series and benchmark.

    Parameters
    ----------
    bench_col : str
        Column name for the benchmark price series (default SPXT__PX_LAST = SPY Total Returns).

    Returns
    -------
    sp_mirror_return : pd.Series
    overlay_return : pd.Series
    fx_hedge_return : pd.Series
    alpha_used : pd.Series
    bench_return : pd.Series (aligned daily returns of bench_col)
    """
    required = {bench_col, "ES1__PX_SETTLE", "USDCAD__PX_LAST"}
    for key in required:
        if key not in derivatives_df.columns:
            raise FileNotFoundError(f"Required series missing: {key}. Columns: {list(derivatives_df.columns)}")

    overlay_col = "ES1__PX_SETTLE"
    use_spx_fallback = False

    es_raw = derivatives_df["ES1__PX_SETTLE"].pct_change().dropna()
    bench_raw = derivatives_df[bench_col].pct_change().dropna()
    common_check = es_raw.index.intersection(bench_raw.index)
    if len(common_check) > 100:
        es_vol = es_raw.loc[common_check].std() * np.sqrt(252)
        bench_vol = bench_raw.loc[common_check].std() * np.sqrt(252)
        if es_vol < bench_vol * 0.1:
            if "SPX__PX_LAST" in derivatives_df.columns and bench_col != "SPX__PX_LAST":
                print(f"[sp_mirror] WARNING: ES1 ann vol ({es_vol:.4f}) is <10% of {bench_col} "
                      f"({bench_vol:.4f}). Falling back to SPX as overlay proxy.")
                overlay_col = "SPX__PX_LAST"
                use_spx_fallback = True
            elif bench_col == "SPX__PX_LAST":
                print(f"[sp_mirror] WARNING: ES1 ann vol ({es_vol:.4f}) is <10% of SPX "
                      f"({bench_vol:.4f}). Using SPX directly as overlay proxy.")
                overlay_col = "SPX__PX_LAST"
                use_spx_fallback = True
            else:
                print(f"[sp_mirror] WARNING: ES1 ann vol ({es_vol:.4f}) is <10% of {bench_col} "
                      f"({bench_vol:.4f}). No fallback available — overlay returns will be near zero.")

    price_keys = [overlay_col, bench_col, "USDCAD__PX_LAST"]
    aligned_prices, clone_aligned, common_index = align_price_and_fx_series(
        derivatives_df, clone_return,
        price_keys=price_keys,
    )
    returns_df = compute_returns_from_prices(aligned_prices)
    clone_ret = clone_aligned.reindex(returns_df.index)
    clone_ret = clone_ret.dropna()
    common = returns_df.index.intersection(clone_ret.index)
    returns_df = returns_df.loc[common]
    clone_ret = clone_ret.loc[common]

    r_ES = returns_df[overlay_col]
    r_USDCAD = returns_df["USDCAD__PX_LAST"]
    r_bench = returns_df[bench_col]

    if use_spx_fallback:
        print(f"[sp_mirror] Using {overlay_col} as overlay proxy. Date range: "
              f"{common[0].date()} to {common[-1].date()} ({len(common)} days)")

    rebal_dates = get_rebalance_dates(common, config.rebal_freq, config.rebal_anchor)

    sp_mirror_ret, overlay_ret, fx_hedge_ret, alpha_used = compute_sp_mirror_returns(
        clone_ret, r_ES, r_USDCAD, r_bench, config, rebal_dates
    )

    if config.supervisor_mode != "none" and config.prob_series is not None:
        sp_mirror_ret = apply_supervisor(
            sp_mirror_ret, overlay_ret, fx_hedge_ret, alpha_used, r_ES, r_USDCAD, config
        )

    return sp_mirror_ret, overlay_ret, fx_hedge_ret, alpha_used, r_bench
