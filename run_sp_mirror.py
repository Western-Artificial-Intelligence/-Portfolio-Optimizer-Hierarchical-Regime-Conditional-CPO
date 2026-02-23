"""
SP_mirror – Run overlay variants against both SPY_Total_Returns and SPY benchmarks.

How to run:
  From project root:  python run_sp_mirror.py

  Clone returns loaded from data/clone_return.csv or generated via QP pipeline.
  Derivatives CSVs must be in data/derivatives/.
  All outputs saved to results/Derivatives_cloning/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sp_mirror_SPY_Total_Returns import (
    SPMirrorConfig,
    load_derivatives_csv_folder,
    run_sp_mirror_single,
    metrics_table,
    OPTIONAL_SERIES,
)
from src.config import DATA_DIR

DATA_DERIVATIVES = DATA_DIR / "derivatives"
CLONE_RETURN_CSV = DATA_DIR / "clone_return.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "Derivatives_cloning"

EVAL_START = pd.Timestamp("2016-01-04")

BENCHMARKS = {
    "SPY_Total_Returns": {
        "bench_col": "SPXT__PX_LAST",
        "required": {"ES1__PX_SETTLE", "USDCAD__PX_LAST", "SPXT__PX_LAST"},
        "figure": RESULTS_DIR / "sp_mirror_SPY_Total_Returns_comparison.png",
        "metrics_csv": RESULTS_DIR / "sp_mirror_SPY_Total_Returns_metrics.csv",
    },
    "SPY": {
        "bench_col": "SPX__PX_LAST",
        "required": {"ES1__PX_SETTLE", "USDCAD__PX_LAST", "SPX__PX_LAST"},
        "figure": RESULTS_DIR / "sp_mirror_SPY_comparison.png",
        "metrics_csv": RESULTS_DIR / "sp_mirror_SPY_metrics.csv",
    },
}


def load_clone_return(csv_path: Path = None, from_pipeline: bool = True):
    """
    Load clone daily return series.

    Priority: (1) csv_path if provided and exists,
              (2) CLONE_RETURN_CSV if exists,
              (3) if from_pipeline, run Phase 2 QP to get clone_returns.
    """
    path = csv_path or CLONE_RETURN_CSV
    if path and Path(path).exists():
        df = pd.read_csv(path)
        date_col = "date" if "date" in df.columns else df.columns[0]
        ret_col = "return" if "return" in df.columns else df.columns[1]
        s = df.set_index(pd.to_datetime(df[date_col]))[ret_col]
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.index = pd.to_datetime(s.index)
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s = s.sort_index()
        print(f"[run_sp_mirror] Loaded clone_return from {path}: {len(s)} days")
        return s

    if from_pipeline:
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
            print(f"[run_sp_mirror] Generated clone_return from QP pipeline: {len(clone_returns)} days")
            return clone_returns
        except Exception as e:
            print(f"[run_sp_mirror] Could not generate clone from pipeline: {e}")

    raise FileNotFoundError(
        f"Clone return not found. Provide {CLONE_RETURN_CSV} or run Phase 2 to generate it."
    )


def run_variants(
    clone_return: pd.Series,
    derivatives_df: pd.DataFrame,
    bench_col: str,
    bench_name: str,
):
    """Run overlay mode variants (beta, vol, hybrid + cost) against a single benchmark."""
    results = {}
    r_bench = None

    for mode in ["beta", "vol", "hybrid", "regime"]:
        cfg = SPMirrorConfig(
            overlay_mode=mode, rebal_freq="M",
            cost_bps=0.0, supervisor_mode="none",
        )
        try:
            sp_ret, _, _, _, r_bench = run_sp_mirror_single(
                clone_return, derivatives_df, cfg, bench_col=bench_col,
            )
            results[f"SP_mirror ({mode})"] = sp_ret
        except Exception as e:
            print(f"[run_sp_mirror] Skipping {mode} vs {bench_name}: {e}")

    cfg_cost = SPMirrorConfig(
        overlay_mode="beta", rebal_freq="M",
        cost_bps=0.0001, supervisor_mode="none",
    )
    try:
        sp_ret, _, _, _, r_bench = run_sp_mirror_single(
            clone_return, derivatives_df, cfg_cost, bench_col=bench_col,
        )
        results["SP_mirror (beta, 1bp cost)"] = sp_ret
    except Exception as e:
        print(f"[run_sp_mirror] Skipping beta+cost vs {bench_name}: {e}")

    if results and r_bench is not None:
        sample = next(iter(results.values()))
        common = sample.dropna().index
        clone_aligned = clone_return.reindex(common).dropna()
        common = common.intersection(clone_aligned.index)
        if len(common) > 0:
            results["Clone"] = clone_return.reindex(common).fillna(0)
        bench_aligned = r_bench.reindex(common).fillna(0)
        results[bench_name] = bench_aligned
        r_bench = bench_aligned

    return results, r_bench


def trim_to_eval(results: dict, eval_start: pd.Timestamp):
    """Trim all series to dates >= eval_start."""
    return {
        name: ret.loc[ret.index >= eval_start]
        for name, ret in results.items()
        if len(ret.loc[ret.index >= eval_start]) > 0
    }


def plot_comparison(
    results: dict,
    bench_return: pd.Series,
    save_path: Path,
    title: str,
):
    """Plot cumulative returns indexed to 100 at eval start."""
    if not results:
        return
    common = bench_return.dropna().index
    for ret in results.values():
        common = common.intersection(ret.dropna().index)
    if len(common) < 2:
        print("[run_sp_mirror] Not enough common dates for plot.")
        return
    common = common.sort_values()

    fig, ax = plt.subplots(figsize=(14, 8))
    for name, ret in results.items():
        r = ret.reindex(common).fillna(0)
        cum = (1 + r).cumprod()
        cum = cum / cum.iloc[0] * 100
        ax.plot(cum.index, cum.values, label=name, alpha=0.9)
    ax.set_ylabel("Index (start=100)")
    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[run_sp_mirror] Saved plot: {save_path}")


def plot_rolling_betas(
    results: dict,
    bench_return: pd.Series,
    bench_name: str,
    save_path: Path,
    window: int = 252,
):
    """Plot 252-day rolling beta of each strategy vs benchmark, with benchmark at 1.0."""
    if not results:
        return
    common = bench_return.dropna().index
    for ret in results.values():
        common = common.intersection(ret.dropna().index)
    if len(common) < window + 10:
        print("[run_sp_mirror] Not enough data for rolling beta plot.")
        return
    common = common.sort_values()
    b = bench_return.reindex(common).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.axhline(y=1.0, color="black", linewidth=2, label=f"{bench_name} (beta=1.0)")

    for name, ret in results.items():
        if name == bench_name:
            continue
        r = ret.reindex(common).fillna(0)
        rolling_cov = r.rolling(window).cov(b)
        rolling_var = b.rolling(window).var()
        rolling_beta = rolling_cov / rolling_var
        valid = rolling_beta.dropna()
        if len(valid) > 0:
            ax.plot(valid.index, valid.values, label=name, alpha=0.85)

    ax.set_ylabel("Rolling Beta (252-day)")
    ax.set_xlabel("Date")
    ax.set_title(f"Rolling Beta vs {bench_name}  (eval: {EVAL_START.date()} onward)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 2.0)
    ax.set_yticks(np.arange(0.0, 2.001, 0.125))
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[run_sp_mirror] Saved rolling beta plot: {save_path}")


def plot_semiannual_betas(
    results: dict,
    bench_return: pd.Series,
    bench_name: str,
    save_path_png: Path,
    save_path_csv: Path,
    window: int = 252,
):
    """Compute average rolling beta per 6-month interval, plot bar chart, and save CSV."""
    if not results:
        return
    common = bench_return.dropna().index
    for ret in results.values():
        common = common.intersection(ret.dropna().index)
    if len(common) < window + 10:
        print("[run_sp_mirror] Not enough data for semiannual beta plot.")
        return
    common = common.sort_values()
    b = bench_return.reindex(common).fillna(0)

    beta_series = {}
    for name, ret in results.items():
        if name == bench_name:
            continue
        r = ret.reindex(common).fillna(0)
        rolling_cov = r.rolling(window).cov(b)
        rolling_var = b.rolling(window).var()
        rb = (rolling_cov / rolling_var).dropna()
        if len(rb) > 0:
            beta_series[name] = rb

    if not beta_series:
        return

    beta_df = pd.DataFrame(beta_series)
    # Group into 6-month buckets and take the mean
    half_year = beta_df.groupby(pd.Grouper(freq="6ME")).mean()
    half_year = half_year.dropna(how="all")

    # Add SPY as constant 1.0
    half_year[bench_name] = 1.0

    # Period labels like "2016-H1", "2016-H2"
    labels = []
    for d in half_year.index:
        half = "H1" if d.month <= 6 else "H2"
        labels.append(f"{d.year}-{half}")
    half_year.index = labels

    # Save CSV
    save_path_csv.parent.mkdir(parents=True, exist_ok=True)
    half_year.round(4).to_csv(save_path_csv)
    print(f"[run_sp_mirror] Saved semiannual beta CSV: {save_path_csv}")

    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 7))
    cols = [c for c in half_year.columns if c != bench_name]
    x = np.arange(len(half_year))
    n = len(cols)
    width = 0.8 / max(n, 1)

    for i, col in enumerate(cols):
        ax.bar(x + i * width - 0.4 + width / 2, half_year[col], width, label=col, alpha=0.85)

    ax.axhline(y=1.0, color="black", linewidth=2, linestyle="--", label=f"{bench_name} (beta=1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(half_year.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Average Beta (6-month)")
    ax.set_xlabel("Period")
    ax.set_title(f"Semiannual Average Beta vs {bench_name}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0.0, 2.0)
    ax.set_yticks(np.arange(0.0, 2.001, 0.125))
    fig.tight_layout()
    save_path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path_png, dpi=150)
    plt.close(fig)
    print(f"[run_sp_mirror] Saved semiannual beta plot: {save_path_png}")


def run_benchmark(
    bench_name: str,
    bench_cfg: dict,
    clone_return: pd.Series,
    derivatives_df: pd.DataFrame,
    r_usdcad: pd.Series | None,
):
    """Run all variants for one benchmark, save metrics + plot."""
    bench_col = bench_cfg["bench_col"]
    required = bench_cfg["required"]

    missing = required - set(derivatives_df.columns)
    if missing:
        print(f"[run_sp_mirror] Skipping {bench_name}: missing {sorted(missing)}")
        return

    print(f"\n{'─' * 60}")
    print(f"  Benchmark: {bench_name}  ({bench_col})")
    print(f"{'─' * 60}")

    results, r_bench = run_variants(clone_return, derivatives_df, bench_col, bench_name)
    if not results:
        print(f"[run_sp_mirror] No variants produced for {bench_name}.")
        return

    results_eval = trim_to_eval(results, EVAL_START)
    r_bench_eval = r_bench.loc[r_bench.index >= EVAL_START]

    mdf = metrics_table(results_eval, r_bench_eval, fx_return=r_usdcad, bench_name=bench_name)
    print("\n" + "=" * 90)
    print(f"SP_mirror – Metrics (eval: {EVAL_START.date()} onward, vs {bench_name})")
    print("=" * 90)
    print(mdf.to_string())
    print("=" * 90)

    out_metrics = bench_cfg["metrics_csv"]
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    mdf.to_csv(out_metrics)
    print(f"[run_sp_mirror] Saved metrics: {out_metrics}")

    plot_comparison(
        results_eval, r_bench_eval,
        save_path=bench_cfg["figure"],
        title=f"SP_mirror vs {bench_name}  (eval: {EVAL_START.date()} onward)",
    )

    beta_fig = bench_cfg["figure"].parent / f"rolling_beta_vs_{bench_name}.png"
    plot_rolling_betas(results_eval, r_bench_eval, bench_name, save_path=beta_fig)

    semi_fig = bench_cfg["figure"].parent / f"semiannual_beta_vs_{bench_name}.png"
    semi_csv = bench_cfg["figure"].parent / f"semiannual_beta_vs_{bench_name}.csv"
    plot_semiannual_betas(results_eval, r_bench_eval, bench_name, semi_fig, semi_csv)


def main():
    print("=" * 60)
    print("SP_mirror – Derivative overlay (SPY_Total_Returns + SPY)")
    print("=" * 60)

    if not DATA_DERIVATIVES.is_dir():
        print(f"[run_sp_mirror] Derivatives folder not found: {DATA_DERIVATIVES}")
        sys.exit(1)

    clone_return = load_clone_return()

    all_required = set()
    for cfg in BENCHMARKS.values():
        all_required |= cfg["required"]

    derivatives_df = load_derivatives_csv_folder(
        DATA_DERIVATIVES,
        required={"ES1__PX_SETTLE", "USDCAD__PX_LAST"},
        optional=OPTIONAL_SERIES | (all_required - {"ES1__PX_SETTLE", "USDCAD__PX_LAST"}),
    )

    r_usdcad = None
    if "USDCAD__PX_LAST" in derivatives_df.columns:
        r_usdcad = derivatives_df["USDCAD__PX_LAST"].pct_change().dropna()

    print(f"\n[run_sp_mirror] Evaluation start: {EVAL_START.date()}")

    for bench_name, bench_cfg in BENCHMARKS.items():
        run_benchmark(bench_name, bench_cfg, clone_return, derivatives_df, r_usdcad)

    print("\n[run_sp_mirror] Done.")


if __name__ == "__main__":
    main()
