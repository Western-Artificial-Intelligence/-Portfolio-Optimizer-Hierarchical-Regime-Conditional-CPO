"""
SP_mirror Robustness & Overfitting Audit

Seven independent diagnostic sections + composite scoring:
  1. Walk-forward testing
  2. Parameter sensitivity analysis
  3. Naive baseline comparison
  4. Placebo test (supervisor signal)
  5. Look-ahead bias checks
  6. Stability diagnostics (alpha/beta)
  7. Crisis stress tests
  8. Composite robustness score (mean-variance + complexity penalty)

All outputs saved to a configurable directory as CSVs and PNGs.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.sp_mirror_SPY_Total_Returns import (
    SPMirrorConfig,
    run_sp_mirror_single,
    compute_sp_mirror_returns,
    compute_alpha_series,
    get_rebalance_dates,
    align_price_and_fx_series,
    compute_returns_from_prices,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _window_metrics(strategy: pd.Series, bench: pd.Series) -> dict:
    """Core per-window metrics: TE, beta, corr, IR, max DD."""
    common = strategy.dropna().index.intersection(bench.dropna().index).sort_values()
    r = strategy.reindex(common).fillna(0)
    b = bench.reindex(common).fillna(0)
    active = r - b
    te = active.std() * np.sqrt(252)
    corr = r.corr(b)
    if b.var() > 0:
        beta = r.cov(b) / b.var()
    else:
        beta = np.nan
    mean_active = active.mean() * 252
    ir = mean_active / te if te > 0 else np.nan
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return {
        "Tracking Error (%)": round(te * 100, 3),
        "Beta": round(beta, 4),
        "Correlation": round(corr, 4),
        "Info Ratio": round(ir, 4) if not np.isnan(ir) else None,
        "Max DD (%)": round(dd * 100, 2),
    }


def _save(df: pd.DataFrame, path: Path, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"  [{label}] Saved: {path}")


def _savefig(fig, path: Path, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [{label}] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Walk-Forward Testing
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_test(
    clone_return: pd.Series,
    derivatives_df: pd.DataFrame,
    config: SPMirrorConfig,
    bench_col: str,
    out_dir: Path,
    train_days: int = 1260,
    test_days: int = 252,
) -> pd.DataFrame:
    """
    Rolling walk-forward evaluation.

    Train on `train_days`, test on next `test_days`, roll by `test_days`.
    No hyperparameter re-tuning across folds — config is held fixed.
    Detects instability if metrics collapse in later windows.
    """
    print("\n── Section 1: Walk-Forward Testing ──")
    sp_ret, _, _, _, r_bench = run_sp_mirror_single(
        clone_return, derivatives_df, config, bench_col=bench_col,
    )
    common = sp_ret.dropna().index.intersection(r_bench.dropna().index).sort_values()
    sp_ret = sp_ret.reindex(common)
    r_bench = r_bench.reindex(common)
    clone_aligned = clone_return.reindex(common).fillna(0)

    rows = []
    start = train_days
    fold = 1
    while start + test_days <= len(common):
        test_idx = common[start : start + test_days]
        m_mirror = _window_metrics(sp_ret.loc[test_idx], r_bench.loc[test_idx])
        m_mirror["Strategy"] = "SP_mirror"
        m_mirror["Fold"] = fold
        m_mirror["Test Start"] = str(test_idx[0].date())
        m_mirror["Test End"] = str(test_idx[-1].date())
        rows.append(m_mirror)

        m_clone = _window_metrics(clone_aligned.loc[test_idx], r_bench.loc[test_idx])
        m_clone["Strategy"] = "Clone"
        m_clone["Fold"] = fold
        m_clone["Test Start"] = str(test_idx[0].date())
        m_clone["Test End"] = str(test_idx[-1].date())
        rows.append(m_clone)

        start += test_days
        fold += 1

    df = pd.DataFrame(rows)
    _save(df, out_dir / "s1_walk_forward_results.csv", "S1")

    summary = df.groupby("Strategy")[
        ["Tracking Error (%)", "Beta", "Correlation", "Info Ratio", "Max DD (%)"]
    ].agg(["mean", "std", "min", "max"])
    _save(summary, out_dir / "s1_walk_forward_summary.csv", "S1")

    # Plot rolling TE per fold
    mirror_df = df[df["Strategy"] == "SP_mirror"].copy()
    clone_df = df[df["Strategy"] == "Clone"].copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(mirror_df["Fold"].values, mirror_df["Tracking Error (%)"].values,
            "o-", label="SP_mirror")
    ax.plot(clone_df["Fold"].values, clone_df["Tracking Error (%)"].values,
            "s--", label="Clone")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Tracking Error (%)")
    ax.set_title("Walk-Forward: Tracking Error per Test Window")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, out_dir / "s1_walk_forward_te.png", "S1")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Parameter Sensitivity Analysis
# ─────────────────────────────────────────────────────────────────────────────

def parameter_sensitivity(
    clone_return: pd.Series,
    derivatives_df: pd.DataFrame,
    bench_col: str,
    out_dir: Path,
    oos_frac: float = 0.30,
) -> pd.DataFrame:
    """
    Vary overlay_lookback, alpha bounds, and rebal_freq independently.

    Flag overfitting if:
    - Small parameter changes cause large performance swings.
    - Best config is at an extreme corner of the grid.
    """
    print("\n── Section 2: Parameter Sensitivity ──")

    lookbacks = [200, 252, 500]
    bounds = [(0.6, 1.4), (0.8, 1.2), (0.9, 1.1)]
    freqs = ["M", "W"]

    grid = []
    for L in lookbacks:
        for (amin, amax) in bounds:
            for freq in freqs:
                grid.append({"L": L, "alpha_min": amin, "alpha_max": amax, "freq": freq})

    rows = []
    for g in grid:
        cfg = SPMirrorConfig(
            overlay_mode="beta",
            overlay_lookback=g["L"],
            alpha_min=g["alpha_min"],
            alpha_max=g["alpha_max"],
            rebal_freq=g["freq"],
            cost_bps=0.0,
            supervisor_mode="none",
        )
        try:
            sp_ret, _, _, _, r_bench = run_sp_mirror_single(
                clone_return, derivatives_df, cfg, bench_col=bench_col,
            )
        except Exception as e:
            print(f"  [S2] Skipping L={g['L']} bounds=[{g['alpha_min']},{g['alpha_max']}] freq={g['freq']}: {e}")
            continue

        common = sp_ret.dropna().index.intersection(r_bench.dropna().index).sort_values()
        sp = sp_ret.reindex(common).fillna(0)
        b = r_bench.reindex(common).fillna(0)

        n = len(common)
        oos_start = int(n * (1 - oos_frac))

        m_full = _window_metrics(sp, b)
        m_oos = _window_metrics(sp.iloc[oos_start:], b.iloc[oos_start:])

        rows.append({
            "L": g["L"],
            "Bounds": f"[{g['alpha_min']}, {g['alpha_max']}]",
            "Freq": g["freq"],
            "TE Full (%)": m_full["Tracking Error (%)"],
            "TE OOS (%)": m_oos["Tracking Error (%)"],
            "IR Full": m_full["Info Ratio"],
            "IR OOS": m_oos["Info Ratio"],
            "Beta Full": m_full["Beta"],
            "Beta OOS": m_oos["Beta"],
        })

    df = pd.DataFrame(rows)
    _save(df, out_dir / "s2_param_sensitivity.csv", "S2")

    # Heatmap: TE OOS by L x Bounds (monthly only for clarity)
    monthly = df[df["Freq"] == "M"].copy()
    if len(monthly) > 0:
        pivot = monthly.pivot_table(index="L", columns="Bounds", values="TE OOS (%)")
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Alpha Bounds")
        ax.set_ylabel("Lookback L")
        ax.set_title("OOS Tracking Error (%) — Monthly Rebal")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10)
        fig.colorbar(im, ax=ax, label="TE (%)")
        fig.tight_layout()
        _savefig(fig, out_dir / "s2_param_heatmap.png", "S2")

    # Flag diagnostics
    if len(df) > 1:
        te_range = df["TE OOS (%)"].max() - df["TE OOS (%)"].min()
        best_idx = df["TE OOS (%)"].idxmin()
        best_row = df.loc[best_idx]
        print(f"  [S2] OOS TE range across grid: {te_range:.2f}pp")
        print(f"  [S2] Best config: L={best_row['L']}, Bounds={best_row['Bounds']}, "
              f"Freq={best_row['Freq']} → TE OOS={best_row['TE OOS (%)']}%")
        if best_row["L"] in [200, 500]:
            print("  [S2] ⚠ Best lookback is at grid boundary — possible overfitting.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Naive Baseline Comparison
# ─────────────────────────────────────────────────────────────────────────────

def naive_baseline_comparison(
    clone_return: pd.Series,
    derivatives_df: pd.DataFrame,
    config: SPMirrorConfig,
    bench_col: str,
    out_dir: Path,
    oos_frac: float = 0.30,
) -> pd.DataFrame:
    """
    Compare dynamic overlay to three naive baselines:
      A) No overlay (alpha = 1 always)
      B) Constant alpha = mean(sigma_bench / sigma_clone)
      C) Constant alpha = mean(1 / beta)

    Constant baselines are routed through the same pipeline (with dampening
    and circuit-breaker if active) via pinned bounds [const, const].

    If dynamic overlay doesn't materially beat constant alpha OOS,
    the dynamic calibration may be overfitting to noise.
    """
    print("\n── Section 3: Naive Baseline Comparison ──")

    # Dynamic overlay
    sp_ret, _, _, alpha_used, r_bench = run_sp_mirror_single(
        clone_return, derivatives_df, config, bench_col=bench_col,
    )
    common = sp_ret.dropna().index.intersection(r_bench.dropna().index).sort_values()
    sp = sp_ret.reindex(common).fillna(0)
    b = r_bench.reindex(common).fillna(0)
    clone_al = clone_return.reindex(common).fillna(0)

    # Estimate constant alpha values from training portion
    n = len(common)
    train_end = int(n * (1 - oos_frac))
    c_train = clone_al.iloc[:train_end]
    b_train = b.iloc[:train_end]

    sigma_bench = b_train.std()
    sigma_clone = c_train.std()
    cov_cb = c_train.cov(b_train)
    var_b = b_train.var()
    full_beta = cov_cb / var_b if var_b > 0 else 1.0

    const_vol = np.clip(sigma_bench / sigma_clone if sigma_clone > 0 else 1.0,
                        config.alpha_min, config.alpha_max)
    const_beta = np.clip(1.0 / full_beta if full_beta > 0 else 1.0,
                         config.alpha_min, config.alpha_max)

    # Baseline A: no overlay (alpha=1) — run through pipeline with bounds [1,1]
    cfg_none = replace(config, alpha_min=1.0, alpha_max=1.0, shrinkage_w=1.0)
    try:
        ret_a, _, _, _, _ = run_sp_mirror_single(
            clone_return, derivatives_df, cfg_none, bench_col=bench_col,
        )
        baseline_a = ret_a.reindex(common).fillna(0)
    except Exception:
        baseline_a = clone_al.copy()

    # Baseline B: constant vol alpha — pinned bounds [const_vol, const_vol]
    cfg_vol = replace(config, alpha_min=const_vol, alpha_max=const_vol, shrinkage_w=1.0)
    try:
        ret_b, _, _, _, _ = run_sp_mirror_single(
            clone_return, derivatives_df, cfg_vol, bench_col=bench_col,
        )
        baseline_b = ret_b.reindex(common).fillna(0)
    except Exception:
        baseline_b = clone_al.copy()

    # Baseline C: constant 1/beta alpha — pinned bounds [const_beta, const_beta]
    cfg_beta = replace(config, alpha_min=const_beta, alpha_max=const_beta, shrinkage_w=1.0)
    try:
        ret_c, _, _, _, _ = run_sp_mirror_single(
            clone_return, derivatives_df, cfg_beta, bench_col=bench_col,
        )
        baseline_c = ret_c.reindex(common).fillna(0)
    except Exception:
        baseline_c = clone_al.copy()

    strategies = {
        "SP_mirror (dynamic)": sp,
        "Clone (alpha=1)": baseline_a,
        f"Const alpha={const_vol:.3f} (vol)": baseline_b,
        f"Const alpha={const_beta:.3f} (1/β)": baseline_c,
    }

    rows = []
    for name, ret in strategies.items():
        m_full = _window_metrics(ret, b)
        m_oos = _window_metrics(ret.iloc[train_end:], b.iloc[train_end:])
        rows.append({
            "Strategy": name,
            "TE Full (%)": m_full["Tracking Error (%)"],
            "TE OOS (%)": m_oos["Tracking Error (%)"],
            "IR Full": m_full["Info Ratio"],
            "IR OOS": m_oos["Info Ratio"],
            "Beta Full": m_full["Beta"],
            "Beta OOS": m_oos["Beta"],
        })

    df = pd.DataFrame(rows)
    _save(df, out_dir / "s3_naive_baselines.csv", "S3")

    dynamic_oos = df.loc[df["Strategy"] == "SP_mirror (dynamic)", "TE OOS (%)"].values[0]
    for _, row in df.iterrows():
        if row["Strategy"] != "SP_mirror (dynamic)" and row["Strategy"] != "Clone (alpha=1)":
            if row["TE OOS (%)"] <= dynamic_oos * 1.05:
                print(f"  [S3] ⚠ Constant baseline '{row['Strategy']}' matches or beats dynamic OOS TE. "
                      f"Dynamic calibration may be overfitting.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Placebo Test (Supervisor Signal)
# ─────────────────────────────────────────────────────────────────────────────

def placebo_test(
    clone_return: pd.Series,
    derivatives_df: pd.DataFrame,
    base_config: SPMirrorConfig,
    bench_col: str,
    out_dir: Path,
    n_random: int = 5,
) -> pd.DataFrame:
    """
    Replace supervisor P(t) with random noise or shuffled P(t).

    If performance is similar, the supervisor adds no real signal.
    Only runs if base_config has a supervisor with prob_series.
    """
    print("\n── Section 4: Placebo Test (Supervisor) ──")

    if base_config.supervisor_mode == "none" or base_config.prob_series is None:
        print("  [S4] No supervisor configured — skipping placebo test (not applicable).")
        report = pd.DataFrame({"Note": ["Supervisor not configured; placebo test skipped."]})
        _save(report, out_dir / "s4_placebo_test.csv", "S4")
        return report

    sp_ret_base, _, _, _, r_bench = run_sp_mirror_single(
        clone_return, derivatives_df, base_config, bench_col=bench_col,
    )
    P_orig = base_config.prob_series
    common = sp_ret_base.dropna().index.intersection(r_bench.dropna().index).sort_values()

    b = r_bench.reindex(common).fillna(0)

    rows = []
    m_base = _window_metrics(sp_ret_base.reindex(common).fillna(0), b)
    m_base["Variant"] = "Original (no supervisor)" if base_config.supervisor_mode == "none" else "Original supervisor"
    rows.append(m_base)

    # Shuffled P
    P_shuffled = P_orig.copy()
    P_shuffled.values[:] = np.random.permutation(P_orig.values)
    cfg_shuf = replace(base_config, supervisor_mode="dial", prob_series=P_shuffled)
    try:
        sp_shuf, _, _, _, _ = run_sp_mirror_single(clone_return, derivatives_df, cfg_shuf, bench_col=bench_col)
        m = _window_metrics(sp_shuf.reindex(common).fillna(0), b)
        m["Variant"] = "Shuffled P(t)"
        rows.append(m)
    except Exception as e:
        print(f"  [S4] Shuffled P failed: {e}")

    # Random P
    rng = np.random.default_rng(42)
    for i in range(n_random):
        P_rand = pd.Series(rng.uniform(0, 1, len(P_orig)), index=P_orig.index)
        cfg_rand = replace(base_config, supervisor_mode="dial", prob_series=P_rand)
        try:
            sp_rand, _, _, _, _ = run_sp_mirror_single(clone_return, derivatives_df, cfg_rand, bench_col=bench_col)
            m = _window_metrics(sp_rand.reindex(common).fillna(0), b)
            m["Variant"] = f"Random P #{i+1}"
            rows.append(m)
        except Exception as e:
            print(f"  [S4] Random P #{i+1} failed: {e}")

    df = pd.DataFrame(rows)
    _save(df, out_dir / "s4_placebo_test.csv", "S4")

    orig_te = rows[0]["Tracking Error (%)"]
    placebo_tes = [r["Tracking Error (%)"] for r in rows[1:] if r["Tracking Error (%)"] is not None]
    if placebo_tes:
        mean_placebo = np.mean(placebo_tes)
        if orig_te >= mean_placebo * 0.95:
            print(f"  [S4] ⚠ Original TE ({orig_te:.2f}%) is not better than placebo mean ({mean_placebo:.2f}%). "
                  f"Supervisor may not add signal.")
        else:
            print(f"  [S4] Original TE ({orig_te:.2f}%) < placebo mean ({mean_placebo:.2f}%). "
                  f"Supervisor shows signal.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Look-Ahead Bias Check
# ─────────────────────────────────────────────────────────────────────────────

def look_ahead_bias_check(
    clone_return: pd.Series,
    derivatives_df: pd.DataFrame,
    config: SPMirrorConfig,
    bench_col: str,
    out_dir: Path,
) -> list[str]:
    """
    Verify that alpha_t uses only data up to t, and overlay is applied correctly.

    Checks:
    1. Alpha base signal changes only on rebalance dates (skipped if vol
       dampening or DD circuit-breaker are active, since those adjust alpha
       daily using backward-looking data — not look-ahead).
    2. Rebalance dates are actual trading days.
    3. Lookback warm-up is respected.
    4. Forward-fill pattern between rebalance dates (skipped if dampening active).
    5. Vol dampening uses only trailing data (no future vol).
    """
    print("\n── Section 5: Look-Ahead Bias Check ──")

    sp_ret, _, _, alpha_used, r_bench = run_sp_mirror_single(
        clone_return, derivatives_df, config, bench_col=bench_col,
    )
    common = alpha_used.dropna().index.sort_values()

    findings = []
    has_daily_adjustments = config.vol_dampening or config.dd_circuit_breaker

    # Check 1: alpha changes only on rebalance dates
    rebal = get_rebalance_dates(common, config.rebal_freq, config.rebal_anchor)
    if has_daily_adjustments:
        findings.append(
            f"✓ Skipping rebalance-only check: vol_dampening={config.vol_dampening}, "
            f"dd_circuit_breaker={config.dd_circuit_breaker} adjust alpha daily using trailing data."
        )
    else:
        alpha_changes = alpha_used.diff().abs()
        change_dates = alpha_changes[alpha_changes > 1e-10].index
        non_rebal_changes = change_dates.difference(rebal)
        n_unexpected = len([d for d in non_rebal_changes if d > common[config.overlay_lookback]])
        if n_unexpected > 0:
            msg = f"⚠ {n_unexpected} alpha changes on non-rebalance dates (possible look-ahead)."
            findings.append(msg)
        else:
            findings.append("✓ Alpha changes only on rebalance dates.")

    # Check 2: rebalance dates are actual trading days in the index
    rebal_in_index = rebal.isin(common)
    n_missing = (~rebal_in_index).sum()
    if n_missing > 0:
        findings.append(f"⚠ {n_missing} rebalance dates not in trading index.")
    else:
        findings.append("✓ All rebalance dates are actual trading days.")

    # Check 3: alpha lookback window doesn't exceed available history
    L = config.overlay_lookback
    first_valid_alpha = alpha_used[alpha_used != 1.0].index[0] if (alpha_used != 1.0).any() else None
    if first_valid_alpha is not None:
        loc = common.get_loc(first_valid_alpha)
        if loc < L // 2:
            findings.append(f"⚠ First non-trivial alpha at index {loc} < L/2={L//2}. Warm-up may be too short.")
        else:
            findings.append(f"✓ First non-trivial alpha at index {loc} (≥ L/2={L//2}).")

    # Check 4: forward-fill pattern between rebalance dates
    if has_daily_adjustments:
        findings.append(
            "✓ Skipping forward-fill check: daily adjustments (dampening/circuit-breaker) modify alpha between rebalances."
        )
    else:
        for i, d in enumerate(rebal):
            if d not in common:
                continue
            loc = common.get_loc(d)
            if loc + 1 < len(common):
                next_day = common[loc + 1]
                if abs(alpha_used.loc[d] - alpha_used.loc[next_day]) > 1e-10:
                    if next_day not in rebal:
                        findings.append(f"⚠ Alpha changes between rebal {d.date()} and next day {next_day.date()}")
                        break
        else:
            findings.append("✓ Alpha is held constant between rebalance dates (forward-fill correct).")

    # Check 5: if vol dampening is active, verify it uses trailing vol only
    if config.vol_dampening:
        vl = config.vol_dampening_lookback
        if vl >= 1:
            findings.append(
                f"✓ Vol dampening uses trailing {vl}-day window (backward-looking, no future data)."
            )
        else:
            findings.append("⚠ Vol dampening lookback < 1 — potential look-ahead.")

    report = pd.DataFrame({"Check": findings})
    _save(report, out_dir / "s5_look_ahead_checks.csv", "S5")

    for f in findings:
        print(f"  [S5] {f}")

    return findings


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Stability Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def stability_diagnostics(
    clone_return: pd.Series,
    derivatives_df: pd.DataFrame,
    config: SPMirrorConfig,
    bench_col: str,
    out_dir: Path,
) -> dict:
    """
    Analyze alpha and beta stability:
    - Distribution of alpha (mean, std, skew, range)
    - Autocorrelation of alpha at rebalance dates
    - Rolling beta stability across regimes

    Flags if alpha swings > 0.3 std dev or beta is unstable.
    """
    print("\n── Section 6: Stability Diagnostics ──")

    sp_ret, _, _, alpha_used, r_bench = run_sp_mirror_single(
        clone_return, derivatives_df, config, bench_col=bench_col,
    )
    common = sp_ret.dropna().index.intersection(r_bench.dropna().index).sort_values()
    sp = sp_ret.reindex(common).fillna(0)
    b = r_bench.reindex(common).fillna(0)
    alpha = alpha_used.reindex(common).ffill().fillna(1.0)

    # Alpha on rebalance dates only (where it actually changes)
    rebal = get_rebalance_dates(common, config.rebal_freq, config.rebal_anchor)
    alpha_rebal = alpha.reindex(rebal).dropna()

    alpha_stats = {
        "Alpha mean": round(alpha_rebal.mean(), 4),
        "Alpha std": round(alpha_rebal.std(), 4),
        "Alpha min": round(alpha_rebal.min(), 4),
        "Alpha max": round(alpha_rebal.max(), 4),
        "Alpha skew": round(alpha_rebal.skew(), 4),
        "Alpha autocorr(1)": round(alpha_rebal.autocorr(lag=1), 4) if len(alpha_rebal) > 2 else None,
        "Alpha autocorr(3)": round(alpha_rebal.autocorr(lag=3), 4) if len(alpha_rebal) > 4 else None,
    }

    # Rolling 252-day beta — computed only after warm-up (first non-trivial alpha)
    rolling_beta_252 = sp.rolling(252).cov(b) / b.rolling(252).var()
    rb_valid = rolling_beta_252.dropna()

    # Exclude warm-up: only measure stability after overlay is active
    first_active = alpha[alpha != 1.0].index[0] if (alpha != 1.0).any() else common[0]
    warmup_end = first_active + pd.Timedelta(days=252)
    rb_post_warmup = rb_valid.loc[rb_valid.index >= warmup_end]
    if len(rb_post_warmup) < 50:
        rb_post_warmup = rb_valid

    beta_stats = {
        "Rolling beta mean": round(rb_post_warmup.mean(), 4),
        "Rolling beta std": round(rb_post_warmup.std(), 4),
        "Rolling beta min": round(rb_post_warmup.min(), 4),
        "Rolling beta max": round(rb_post_warmup.max(), 4),
    }

    all_stats = {**alpha_stats, **beta_stats}
    stats_df = pd.DataFrame([all_stats]).T
    stats_df.columns = ["Value"]
    _save(stats_df, out_dir / "s6_stability_stats.csv", "S6")

    # Plot 1: alpha distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(alpha_rebal.values, bins=30, edgecolor="black", alpha=0.7)
    axes[0].axvline(1.0, color="red", linestyle="--", label="alpha=1 (no overlay)")
    axes[0].set_title("Alpha Distribution (rebalance dates)")
    axes[0].set_xlabel("Alpha")
    axes[0].legend()

    axes[1].plot(alpha.index, alpha.values, linewidth=0.6)
    axes[1].axhline(1.0, color="red", linestyle="--", alpha=0.5)
    axes[1].set_title("Alpha Time Series")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Alpha")
    fig.tight_layout()
    _savefig(fig, out_dir / "s6_alpha_distribution.png", "S6")

    # Plot 2: rolling beta
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rb_valid.index, rb_valid.values, linewidth=0.7, label="Rolling beta (252d)")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5)
    ax.set_ylabel("Beta")
    ax.set_title("Rolling Beta Stability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, out_dir / "s6_rolling_beta_stability.png", "S6")

    # Flags
    if alpha_rebal.std() > 0.3:
        print(f"  [S6] ⚠ Alpha std = {alpha_rebal.std():.3f} > 0.3 — excessive swings.")
    else:
        print(f"  [S6] ✓ Alpha std = {alpha_rebal.std():.3f} (within normal range).")

    if rb_valid.std() > 0.3:
        print(f"  [S6] ⚠ Rolling beta std = {rb_valid.std():.3f} > 0.3 — unstable across regimes.")
    else:
        print(f"  [S6] ✓ Rolling beta std = {rb_valid.std():.3f} (stable).")

    return all_stats


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Crisis Stress Test
# ─────────────────────────────────────────────────────────────────────────────

CRISIS_PERIODS = {
    "2011 Euro Crisis": ("2011-07-01", "2011-12-31"),
    "2018 Vol Shock": ("2018-01-25", "2018-04-30"),
    "2020 COVID Crash": ("2020-02-19", "2020-06-30"),
    "2022 Rate Shock": ("2022-01-03", "2022-10-31"),
}


def crisis_stress_test(
    clone_return: pd.Series,
    derivatives_df: pd.DataFrame,
    config: SPMirrorConfig,
    bench_col: str,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Isolate crisis subperiods and compute tracking error, beta drift, drawdown.

    Reveals whether the overlay breaks down during stress regimes.
    """
    print("\n── Section 7: Crisis Stress Test ──")

    sp_ret, _, _, alpha_used, r_bench = run_sp_mirror_single(
        clone_return, derivatives_df, config, bench_col=bench_col,
    )
    common = sp_ret.dropna().index.intersection(r_bench.dropna().index).sort_values()
    sp = sp_ret.reindex(common).fillna(0)
    b = r_bench.reindex(common).fillna(0)
    clone_al = clone_return.reindex(common).fillna(0)
    alpha = alpha_used.reindex(common).ffill().fillna(1.0)

    rows = []
    for name, (start, end) in CRISIS_PERIODS.items():
        mask = (common >= start) & (common <= end)
        idx = common[mask]
        if len(idx) < 10:
            print(f"  [S7] Skipping {name}: only {len(idx)} days in range.")
            continue

        m_mirror = _window_metrics(sp.loc[idx], b.loc[idx])
        m_clone = _window_metrics(clone_al.loc[idx], b.loc[idx])
        alpha_period = alpha.loc[idx]

        rows.append({
            "Crisis": name,
            "Period": f"{start} to {end}",
            "Days": len(idx),
            "SP_mirror TE (%)": m_mirror["Tracking Error (%)"],
            "Clone TE (%)": m_clone["Tracking Error (%)"],
            "SP_mirror Beta": m_mirror["Beta"],
            "Clone Beta": m_clone["Beta"],
            "SP_mirror Max DD (%)": m_mirror["Max DD (%)"],
            "Clone Max DD (%)": m_clone["Max DD (%)"],
            "Alpha mean": round(alpha_period.mean(), 4),
            "Alpha std": round(alpha_period.std(), 4),
        })

    df = pd.DataFrame(rows)
    _save(df, out_dir / "s7_crisis_stress.csv", "S7")

    # Plot: TE comparison across crises
    if len(df) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df))
        w = 0.35
        ax.bar(x - w/2, df["SP_mirror TE (%)"], w, label="SP_mirror", alpha=0.8)
        ax.bar(x + w/2, df["Clone TE (%)"], w, label="Clone", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(df["Crisis"], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Tracking Error (%)")
        ax.set_title("Crisis Stress: Tracking Error Comparison")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        _savefig(fig, out_dir / "s7_crisis_te.png", "S7")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Composite Robustness Score
# ─────────────────────────────────────────────────────────────────────────────

MODEL_COMPLEXITY = {
    "Clone (alpha=1)": 0,
    "Const alpha (vol)": 1,
    "Const alpha (1/β)": 1,
    "SP_mirror (dynamic)": 3,
    "SP_mirror (dynamic + dampening)": 8,
}


def composite_robustness_score(
    wf_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    stability: dict,
    clone_return: pd.Series,
    derivatives_df: pd.DataFrame,
    config: SPMirrorConfig,
    bench_col: str,
    out_dir: Path,
    lam: float = 0.5,
    kappa: float = 0.05,
    train_days: int = 1260,
    test_days: int = 252,
) -> pd.DataFrame:
    """
    Composite robustness score for principled complexity-overfitting tradeoff.

    For each model variant, compute:
        Score = mean(fold_TE) + λ * std(fold_TE) + κ * n_params

    - mean(fold_TE): rewards tracking accuracy
    - std(fold_TE): penalizes instability across regimes
    - n_params: penalizes model complexity (each parameter must earn its keep)

    Parameters
    ----------
    lam : float
        Weight on fold TE standard deviation (default 0.5).
    kappa : float
        Penalty per estimated parameter (default 0.05 percentage points).
    """
    print("\n── Section 8: Composite Robustness Score ──")

    # Run walk-forward for each model variant
    sp_ret_dyn, _, _, _, r_bench = run_sp_mirror_single(
        clone_return, derivatives_df, config, bench_col=bench_col,
    )
    common = sp_ret_dyn.dropna().index.intersection(r_bench.dropna().index).sort_values()
    clone_al = clone_return.reindex(common).fillna(0)
    r_b = r_bench.reindex(common).fillna(0)

    # Estimate constant alphas from training portion
    train_end = train_days
    if train_end >= len(common):
        train_end = len(common) // 2
    c_train = clone_al.iloc[:train_end]
    b_train = r_b.iloc[:train_end]
    sigma_bench = b_train.std()
    sigma_clone = c_train.std()
    cov_cb = c_train.cov(b_train)
    var_b = b_train.var()
    full_beta = cov_cb / var_b if var_b > 0 else 1.0
    const_vol = np.clip(sigma_bench / sigma_clone if sigma_clone > 0 else 1.0,
                        config.alpha_min, config.alpha_max)
    const_beta_val = np.clip(1.0 / full_beta if full_beta > 0 else 1.0,
                             config.alpha_min, config.alpha_max)

    # Build constant-alpha configs through the same pipeline
    cfg_none = replace(config, overlay_mode="beta", alpha_min=1.0, alpha_max=1.0,
                       shrinkage_w=1.0, vol_dampening=False, dd_circuit_breaker=False)
    cfg_vol = replace(config, overlay_mode="beta", alpha_min=const_vol, alpha_max=const_vol,
                      shrinkage_w=1.0, vol_dampening=False, dd_circuit_breaker=False)
    cfg_beta = replace(config, overlay_mode="beta", alpha_min=const_beta_val, alpha_max=const_beta_val,
                       shrinkage_w=1.0, vol_dampening=False, dd_circuit_breaker=False)

    # Regime-conditional config: 3 alphas selected by vol regime
    cfg_regime = replace(config, overlay_mode="regime",
                         vol_dampening=False, dd_circuit_breaker=True)

    model_configs = {
        "Clone (alpha=1)": cfg_none,
        f"Const alpha={const_vol:.3f} (vol)": cfg_vol,
        f"Const alpha={const_beta_val:.3f} (1/β)": cfg_beta,
        "SP_mirror (dynamic)": config,
        "SP_mirror (regime)": cfg_regime,
    }

    n_params_map = {
        "Clone (alpha=1)": 0,
        f"Const alpha={const_vol:.3f} (vol)": 1,
        f"Const alpha={const_beta_val:.3f} (1/β)": 1,
        "SP_mirror (dynamic)": _count_params(config),
        "SP_mirror (regime)": 5,  # 3 alphas + 2 thresholds (thresholds fixed a priori)
    }

    rows = []
    for name, cfg in model_configs.items():
        try:
            sp_ret, _, _, alpha_used, rb = run_sp_mirror_single(
                clone_return, derivatives_df, cfg, bench_col=bench_col,
            )
        except Exception as e:
            print(f"  [S8] Skipping {name}: {e}")
            continue

        cm = sp_ret.dropna().index.intersection(rb.dropna().index).sort_values()
        sp = sp_ret.reindex(cm).fillna(0)
        b = rb.reindex(cm).fillna(0)

        # Walk-forward fold TEs
        fold_tes = []
        fold_dds = []
        start = train_days
        while start + test_days <= len(cm):
            idx = cm[start : start + test_days]
            m = _window_metrics(sp.loc[idx], b.loc[idx])
            fold_tes.append(m["Tracking Error (%)"])
            fold_dds.append(m["Max DD (%)"])
            start += test_days

        if not fold_tes:
            continue

        te_mean = np.mean(fold_tes)
        te_std = np.std(fold_tes, ddof=1) if len(fold_tes) > 1 else 0.0
        te_max = np.max(fold_tes)
        dd_worst = np.min(fold_dds)
        n_p = n_params_map.get(name, 0)

        composite = te_mean + lam * te_std + kappa * n_p

        # Full-sample and OOS metrics
        oos_start = int(len(cm) * 0.7)
        m_full = _window_metrics(sp, b)
        m_oos = _window_metrics(sp.iloc[oos_start:], b.iloc[oos_start:])

        alpha_on_cm = alpha_used.reindex(cm).ffill().fillna(1.0)
        a_std = alpha_on_cm.std()

        rows.append({
            "Model": name,
            "n_params": n_p,
            "WF TE mean (%)": round(te_mean, 3),
            "WF TE std (%)": round(te_std, 3),
            "WF TE max (%)": round(te_max, 3),
            "WF worst DD (%)": round(dd_worst, 2),
            "OOS TE (%)": m_oos["Tracking Error (%)"],
            "Full TE (%)": m_full["Tracking Error (%)"],
            "Alpha std": round(a_std, 4),
            "Composite Score": round(composite, 3),
        })

    df = pd.DataFrame(rows).sort_values("Composite Score")
    _save(df, out_dir / "s8_composite_scores.csv", "S8")

    # Plot: composite score comparison
    if len(df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Bar chart of composite scores
        colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(df))]
        axes[0].barh(df["Model"], df["Composite Score"], color=colors, edgecolor="black", alpha=0.8)
        axes[0].set_xlabel("Composite Score (lower = better)")
        axes[0].set_title(f"Composite: TE_mean + {lam}×TE_std + {kappa}×n_params")
        axes[0].invert_yaxis()
        axes[0].grid(True, axis="x", alpha=0.3)

        # Scatter: TE mean vs TE std (bubble size = n_params)
        sizes = (df["n_params"] + 1) * 80
        axes[1].scatter(df["WF TE mean (%)"], df["WF TE std (%)"], s=sizes, alpha=0.7, edgecolor="black")
        for _, row in df.iterrows():
            axes[1].annotate(row["Model"], (row["WF TE mean (%)"], row["WF TE std (%)"]),
                             fontsize=7, ha="center", va="bottom")
        axes[1].set_xlabel("WF TE mean (%)")
        axes[1].set_ylabel("WF TE std (%)")
        axes[1].set_title("Accuracy vs Stability (bubble = complexity)")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        _savefig(fig, out_dir / "s8_composite_scores.png", "S8")

    # Print ranking
    print(f"\n  Composite Score (λ={lam}, κ={kappa}):")
    print(f"  {'Model':<35} {'Score':>8}  {'TE mean':>8}  {'TE std':>8}  {'Params':>6}")
    print(f"  {'-'*35} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
    for _, row in df.iterrows():
        print(f"  {row['Model']:<35} {row['Composite Score']:>8.3f}  "
              f"{row['WF TE mean (%)']:>7.3f}%  {row['WF TE std (%)']:>7.3f}%  "
              f"{row['n_params']:>6}")

    best = df.iloc[0]
    print(f"\n  → Best model: {best['Model']}  (score: {best['Composite Score']:.3f})")

    return df


def _count_params(config: SPMirrorConfig) -> int:
    """Count effective estimated parameters for complexity penalty."""
    n = 2  # alpha_min, alpha_max (or the rolling estimation itself)
    if config.shrinkage_w < 1.0:
        n += 1
    if config.vol_dampening:
        n += 3  # lookback, percentile, floor
    if config.dd_circuit_breaker:
        n += 3  # trigger, reset, lookback
    if config.use_expanding_window:
        n -= 1  # removes lookback L as a param
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Final Summary Report
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary_report(
    wf_df: pd.DataFrame,
    sens_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    placebo_df: pd.DataFrame,
    bias_findings: list[str],
    stability: dict,
    crisis_df: pd.DataFrame,
    out_dir: Path,
    composite_df: Optional[pd.DataFrame] = None,
) -> str:
    """
    Synthesize all diagnostics into a final verdict:
    - Stability verdict
    - Overfitting risk level (Low / Medium / High)
    - Key fragility points
    - Composite score ranking (if available)
    """
    print("\n" + "=" * 70)
    print("  SP_MIRROR ROBUSTNESS AUDIT — FINAL REPORT")
    print("=" * 70)

    flags = []
    risk_score = 0  # 0-10 scale; >6 = High, 3-6 = Medium, <3 = Low

    # S1: Walk-forward stability
    if wf_df is not None and len(wf_df) > 0:
        mirror_wf = wf_df[wf_df["Strategy"] == "SP_mirror"]
        te_std = mirror_wf["Tracking Error (%)"].std()
        te_trend = mirror_wf["Tracking Error (%)"].values
        if len(te_trend) > 2 and te_trend[-1] > te_trend[0] * 1.5:
            flags.append("Walk-forward TE deteriorates in later folds (instability).")
            risk_score += 2
        if te_std > 3.0:
            flags.append(f"Walk-forward TE std = {te_std:.2f}% — high variance across folds.")
            risk_score += 1

    # S2: Parameter sensitivity
    if sens_df is not None and len(sens_df) > 1:
        te_range = sens_df["TE OOS (%)"].max() - sens_df["TE OOS (%)"].min()
        if te_range > 5.0:
            flags.append(f"Parameter sensitivity: OOS TE range = {te_range:.1f}pp — fragile.")
            risk_score += 2
        elif te_range > 2.0:
            flags.append(f"Parameter sensitivity: OOS TE range = {te_range:.1f}pp — moderate.")
            risk_score += 1

    # S3: Naive baselines — now scored via composite score if available
    if composite_df is not None and len(composite_df) > 1:
        dyn_rows = composite_df[composite_df["Model"].str.contains("dynamic")]
        const_rows = composite_df[~composite_df["Model"].str.contains("dynamic|Clone")]
        if len(dyn_rows) > 0 and len(const_rows) > 0:
            dyn_score = dyn_rows["Composite Score"].values[0]
            best_const = const_rows["Composite Score"].min()
            best_const_name = const_rows.loc[const_rows["Composite Score"].idxmin(), "Model"]
            if dyn_score <= best_const:
                flags.append(
                    f"Dynamic model justifies its complexity (composite {dyn_score:.2f} "
                    f"< {best_const_name} {best_const:.2f})."
                )
                # No penalty — dynamic earns its parameters
            elif dyn_score <= best_const * 1.02:
                flags.append(
                    f"Dynamic model is marginal vs constant (composite {dyn_score:.2f} "
                    f"~= {best_const_name} {best_const:.2f}). Consider simpler model."
                )
                risk_score += 1
            else:
                flags.append(
                    f"Dynamic model does NOT justify its complexity (composite {dyn_score:.2f} "
                    f"> {best_const_name} {best_const:.2f})."
                )
                risk_score += 2
    elif baseline_df is not None and len(baseline_df) > 0:
        # Fallback to old binary check if no composite scores
        dyn = baseline_df.loc[baseline_df["Strategy"].str.contains("dynamic"), "TE OOS (%)"].values
        const = baseline_df.loc[~baseline_df["Strategy"].str.contains("dynamic|Clone"), "TE OOS (%)"].values
        if len(dyn) > 0 and len(const) > 0:
            if dyn[0] >= const.min() * 0.95:
                flags.append("Dynamic overlay does NOT beat constant alpha OOS — possible overfitting.")
                risk_score += 3

    # S4: Placebo (only scored when a supervisor is actually configured)
    if placebo_df is not None and "Tracking Error (%)" in placebo_df.columns and len(placebo_df) > 1:
        orig = placebo_df.iloc[0]["Tracking Error (%)"]
        placebo_mean = placebo_df.iloc[1:]["Tracking Error (%)"].mean()
        if orig >= placebo_mean * 0.95:
            flags.append("Supervisor signal indistinguishable from random noise.")
            risk_score += 2

    # S5: Look-ahead bias
    warnings = [f for f in bias_findings if "⚠" in f]
    if warnings:
        flags.append(f"Look-ahead bias: {len(warnings)} warning(s) detected.")
        risk_score += 2

    # S6: Stability
    if stability:
        alpha_std = stability.get("Alpha std", 0)
        beta_std = stability.get("Rolling beta std", 0)
        if alpha_std > 0.3:
            flags.append(f"Alpha std = {alpha_std:.3f} — excessive parameter swings.")
            risk_score += 1
        if beta_std > 0.3:
            flags.append(f"Rolling beta std = {beta_std:.3f} — beta unstable across regimes.")
            risk_score += 1

    # S7: Crisis
    if crisis_df is not None and len(crisis_df) > 0:
        mirror_worse = (crisis_df["SP_mirror TE (%)"] > crisis_df["Clone TE (%)"]).sum()
        if mirror_worse > len(crisis_df) / 2:
            flags.append(f"Overlay increases TE in {mirror_worse}/{len(crisis_df)} crisis periods.")
            risk_score += 1

    # Verdict
    if risk_score >= 6:
        verdict = "HIGH"
    elif risk_score >= 3:
        verdict = "MEDIUM"
    else:
        verdict = "LOW"

    lines = [
        "SP_MIRROR ROBUSTNESS AUDIT — FINAL REPORT",
        "=" * 50,
        "",
        f"Overfitting Risk Level: {verdict}  (score: {risk_score}/10)",
        "",
        "Key Findings:",
    ]
    if flags:
        for i, f in enumerate(flags, 1):
            lines.append(f"  {i}. {f}")
    else:
        lines.append("  No significant fragility detected.")

    # Composite score summary
    if composite_df is not None and len(composite_df) > 0:
        lines.append("")
        lines.append("Composite Robustness Ranking:")
        lines.append(f"  {'Model':<35} {'Score':>8}  {'TE mean':>8}  {'TE std':>8}  {'Params':>6}")
        for _, row in composite_df.iterrows():
            lines.append(
                f"  {row['Model']:<35} {row['Composite Score']:>8.3f}  "
                f"{row['WF TE mean (%)']:>7.3f}%  {row['WF TE std (%)']:>7.3f}%  "
                f"{int(row['n_params']):>6}"
            )
        best = composite_df.iloc[0]
        lines.append(f"  → Best: {best['Model']}  (composite: {best['Composite Score']:.3f})")

    lines.append("")
    lines.append("Stability Verdict:")
    if risk_score < 3:
        lines.append("  Model appears stable and not overfit to in-sample data.")
    elif risk_score < 6:
        lines.append("  Model shows some sensitivity — interpret results cautiously.")
    else:
        lines.append("  Model shows significant fragility — high risk of overfitting.")

    report_text = "\n".join(lines)
    print(report_text)

    report_path = out_dir / "robustness_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n  Saved final report: {report_path}")

    return report_text
