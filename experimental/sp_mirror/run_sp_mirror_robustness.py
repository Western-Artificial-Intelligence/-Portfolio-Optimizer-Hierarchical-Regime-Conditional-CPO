"""
SP_mirror Robustness Audit — Entrypoint

Runs all 7 diagnostic sections and produces a final summary report.
Outputs saved to results/Derivatives_cloning/robustness_tests/.

How to run:
  python run_sp_mirror_robustness.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sp_mirror_SPY_Total_Returns import (
    SPMirrorConfig,
    load_derivatives_csv_folder,
    OPTIONAL_SERIES,
)
from src.sp_mirror_robustness import (
    walk_forward_test,
    parameter_sensitivity,
    naive_baseline_comparison,
    placebo_test,
    look_ahead_bias_check,
    stability_diagnostics,
    crisis_stress_test,
    composite_robustness_score,
    generate_summary_report,
)
from src.config import DATA_DIR

DATA_DERIVATIVES = DATA_DIR / "derivatives"
CLONE_RETURN_CSV = DATA_DIR / "clone_return.csv"
OUT_DIR = PROJECT_ROOT / "results" / "Derivatives_cloning" / "robustness_tests"

BENCH_COL = "SPXT__PX_LAST"


def load_clone_return():
    path = CLONE_RETURN_CSV
    if path.exists():
        df = pd.read_csv(path)
        date_col = "date" if "date" in df.columns else df.columns[0]
        ret_col = "return" if "return" in df.columns else df.columns[1]
        s = df.set_index(pd.to_datetime(df[date_col]))[ret_col]
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.index = pd.to_datetime(s.index)
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s = s.sort_index()
        print(f"[robustness] Loaded clone_return: {len(s)} days")
        return s

    try:
        from src.data_loader import load_all
        from src.features import filter_sparse_tickers
        from src.qp_solver import rolling_optimization
        from src.config import BENCHMARK
        prices, all_fields, profiles, econ, yield_curve = load_all()
        prices_clean, _ = filter_sparse_tickers(prices, min_coverage=0.5)
        _, clone_returns = rolling_optimization(
            prices_clean, benchmark_col=BENCHMARK, lookback=252 * 5, rebal_freq="M"
        )
        clone_returns = clone_returns.dropna()
        print(f"[robustness] Generated clone_return from pipeline: {len(clone_returns)} days")
        return clone_returns
    except Exception as e:
        print(f"[robustness] Pipeline failed: {e}")

    raise FileNotFoundError(f"Clone return not found at {CLONE_RETURN_CSV}")


def run_audit(label: str, cfg: SPMirrorConfig, clone_return, derivatives_df, out_dir: Path):
    """Run all 8 diagnostic sections for a given config and save to out_dir."""
    print(f"\n{'=' * 70}")
    print(f"  AUDIT: {label}")
    print(f"  Config: bounds=[{cfg.alpha_min}, {cfg.alpha_max}]  shrinkage_w={cfg.shrinkage_w}  "
          f"vol_damp={cfg.vol_dampening}  dd_breaker={cfg.dd_circuit_breaker}")
    print(f"{'=' * 70}")

    out_dir.mkdir(parents=True, exist_ok=True)

    wf_df = walk_forward_test(
        clone_return, derivatives_df, cfg, BENCH_COL, out_dir,
    )
    sens_df = parameter_sensitivity(
        clone_return, derivatives_df, BENCH_COL, out_dir,
    )
    baseline_df = naive_baseline_comparison(
        clone_return, derivatives_df, cfg, BENCH_COL, out_dir,
    )
    placebo_df = placebo_test(
        clone_return, derivatives_df, cfg, BENCH_COL, out_dir,
    )
    bias_findings = look_ahead_bias_check(
        clone_return, derivatives_df, cfg, BENCH_COL, out_dir,
    )
    stability = stability_diagnostics(
        clone_return, derivatives_df, cfg, BENCH_COL, out_dir,
    )
    crisis_df = crisis_stress_test(
        clone_return, derivatives_df, cfg, BENCH_COL, out_dir,
    )
    comp_df = composite_robustness_score(
        wf_df, baseline_df, stability,
        clone_return, derivatives_df, cfg, BENCH_COL, out_dir,
    )
    generate_summary_report(
        wf_df, sens_df, baseline_df, placebo_df,
        bias_findings, stability, crisis_df, out_dir,
        composite_df=comp_df,
    )


def main():
    print("=" * 70)
    print("  SP_mirror — Robustness & Overfitting Audit")
    print("=" * 70)

    clone_return = load_clone_return()

    required = {"ES1__PX_SETTLE", "USDCAD__PX_LAST", "SPXT__PX_LAST"}
    derivatives_df = load_derivatives_csv_folder(
        DATA_DERIVATIVES,
        required=required,
        optional=OPTIONAL_SERIES | {"SPX__PX_LAST"},
    )

    # ── OLD CONFIG (pre-fix baseline for comparison) ──
    old_cfg = SPMirrorConfig(
        overlay_mode="beta",
        overlay_lookback=252,
        alpha_min=0.6,
        alpha_max=1.4,
        shrinkage_w=1.0,           # no shrinkage
        vol_dampening=False,       # no dampening
        dd_circuit_breaker=False,  # no circuit breaker
        rebal_freq="M",
        cost_bps=0.0,
        supervisor_mode="none",
    )

    # ── NEW CONFIG (with all robustness fixes) ──
    new_cfg = SPMirrorConfig(
        overlay_mode="beta",
        overlay_lookback=252,
        # new defaults: bounds [0.8, 1.2], shrinkage 0.5, vol damp on, DD breaker on
        rebal_freq="M",
        cost_bps=0.0,
        supervisor_mode="none",
    )

    old_dir = OUT_DIR / "old_config"
    new_dir = OUT_DIR / "new_config"

    run_audit("OLD CONFIG (pre-fix)", old_cfg, clone_return, derivatives_df, old_dir)
    run_audit("NEW CONFIG (with fixes)", new_cfg, clone_return, derivatives_df, new_dir)

    # ── Side-by-side comparison ──
    print("\n" + "=" * 70)
    print("  SIDE-BY-SIDE COMPARISON")
    print("=" * 70)
    old_report = old_dir / "robustness_report.txt"
    new_report = new_dir / "robustness_report.txt"
    for label, path in [("OLD", old_report), ("NEW", new_report)]:
        if path.exists():
            print(f"\n--- {label} ---")
            print(path.read_text())

    print(f"\n[robustness] All outputs in: {OUT_DIR}")
    print("[robustness] Done.")


if __name__ == "__main__":
    main()
