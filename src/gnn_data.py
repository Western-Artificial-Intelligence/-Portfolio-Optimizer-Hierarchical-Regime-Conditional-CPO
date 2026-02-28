"""
GNN Data Pipeline — Dynamic Market Graph Construction.

Builds daily graph snapshots from existing project data files.
No Bloomberg pulls required — all features derived from existing CSVs.

Feature engineering:
  Per-node (per stock, per day):
    - 1-day return
    - Rolling volatility: 5/21/63 days (annualized)
    - Momentum: 5/21/63 days (cumulative return)
    - Volume z-score (21-day trailing)
    - Beta (static, broadcast to all days)
    - GICS sector one-hot (static, broadcast)
  Global (macro, shared across all nodes):
    - T10Y2Y yield curve spread
    - DXY (dollar index)
    - MOVE (bond volatility index)
    - IG credit spread
    - Yield curve spread + inversion flag
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import BENCHMARK, ALL_TICKERS


# ──────────────────────────────────────────────
# 1. Node Feature Engineering
# ──────────────────────────────────────────────

def load_gnn_node_features(returns, all_fields, profiles, econ, yield_curve):
    """
    Build [T, N, F] node feature array from existing project data.

    Parameters
    ----------
    returns     : pd.DataFrame  [T × N]   — daily returns (pct_change of prices)
    all_fields  : dict[str, pd.DataFrame] — from data_loader.load_stock_history()
    profiles    : pd.DataFrame            — from data_loader.load_stock_profiles()
    econ        : pd.DataFrame            — economic_indicators.csv
    yield_curve : pd.DataFrame            — yield_curve_spread.csv

    Returns
    -------
    node_feat_array : np.ndarray [T, N, F]  float32
    dates           : pd.DatetimeIndex      length T
    tickers         : list[str]             length N (same order as axis 1)
    n_features      : int                   F
    """
    # Only keep tickers present in both returns and our universe
    tickers = [t for t in ALL_TICKERS if t in returns.columns]
    N = len(tickers)

    ret = returns.reindex(columns=tickers).copy()
    common_index = ret.index
    T = len(common_index)

    # ── Per-node time-varying features ──────────────────────────────────────

    # Rolling volatility (annualized)
    vol_5  = ret.rolling(5,  min_periods=2).std() * np.sqrt(252)
    vol_21 = ret.rolling(21, min_periods=5).std() * np.sqrt(252)
    vol_63 = ret.rolling(63, min_periods=10).std() * np.sqrt(252)

    # Momentum (cumulative return over window)
    mom_5  = ret.rolling(5,  min_periods=2).sum()
    mom_21 = ret.rolling(21, min_periods=5).sum()
    mom_63 = ret.rolling(63, min_periods=10).sum()

    # Volume z-score (21-day trailing)
    volume_field = all_fields.get("PX_VOLUME", pd.DataFrame())
    if not volume_field.empty and any(t in volume_field.columns for t in tickers):
        vol_data = volume_field.reindex(columns=tickers).reindex(common_index)
        vol_mean = vol_data.rolling(21, min_periods=5).mean()
        vol_std  = vol_data.rolling(21, min_periods=5).std() + 1e-8
        vol_zscore = (vol_data - vol_mean) / vol_std
    else:
        vol_zscore = pd.DataFrame(0.0, index=common_index, columns=tickers)

    per_node_dfs = [ret, vol_5, vol_21, vol_63, mom_5, mom_21, mom_63, vol_zscore]
    n_time_features = len(per_node_dfs)  # 8

    # ── Static node features (broadcast to all days) ─────────────────────────

    # Beta
    beta_lookup = {}
    if "EQY_BETA" in profiles.columns:
        for t in tickers:
            if t in profiles.index and not pd.isna(profiles.loc[t, "EQY_BETA"]):
                beta_lookup[t] = float(profiles.loc[t, "EQY_BETA"])
            else:
                beta_lookup[t] = 1.0
    else:
        beta_lookup = {t: 1.0 for t in tickers}

    # GICS sector one-hot
    if "GICS_SECTOR_NAME" in profiles.columns:
        sectors_raw = profiles["GICS_SECTOR_NAME"].fillna("Unknown")
        unique_sectors = sorted(sectors_raw.unique())
        sector_map = {s: i for i, s in enumerate(unique_sectors)}
        n_sectors = len(unique_sectors)
    else:
        unique_sectors, sector_map, n_sectors = [], {}, 0

    sector_onehots = {}
    for t in tickers:
        onehot = np.zeros(n_sectors, dtype=np.float32)
        if t in profiles.index and "GICS_SECTOR_NAME" in profiles.columns:
            s = profiles.loc[t, "GICS_SECTOR_NAME"]
            if pd.notna(s) and s in sector_map:
                onehot[sector_map[s]] = 1.0
        sector_onehots[t] = onehot

    n_static = 1 + n_sectors  # beta + sector one-hot

    # ── Global (macro) features — same for all nodes each day ────────────────

    macro_cols = [
        "T10Y2Y_PX_LAST", "DXY_PX_LAST", "MOVE_PX_LAST", "IG_SPREAD_PX_LAST"
    ]
    econ_sel = econ.reindex(common_index, method="ffill")[
        [c for c in macro_cols if c in econ.columns]
    ].ffill().fillna(0.0)

    yc_cols = ["YIELD_CURVE_SPREAD", "INVERTED"]
    yc_sel = yield_curve.reindex(common_index, method="ffill")[
        [c for c in yc_cols if c in yield_curve.columns]
    ].ffill().fillna(0.0)

    macro_array = pd.concat([econ_sel, yc_sel], axis=1).values.astype(np.float32)
    n_macro = macro_array.shape[1]

    # ── Assemble [T, N, F] ───────────────────────────────────────────────────

    n_features = n_time_features + n_static + n_macro
    node_feat_array = np.zeros((T, N, n_features), dtype=np.float32)

    for n_idx, ticker in enumerate(tickers):
        col = 0

        # Time-varying features
        for df in per_node_dfs:
            if ticker in df.columns:
                vals = df[ticker].reindex(common_index).values.astype(np.float32)
            else:
                vals = np.zeros(T, dtype=np.float32)
            node_feat_array[:, n_idx, col] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            col += 1

        # Static: beta (broadcast)
        node_feat_array[:, n_idx, col] = beta_lookup[ticker]
        col += 1

        # Static: sector one-hot (broadcast)
        node_feat_array[:, n_idx, col:col + n_sectors] = sector_onehots[ticker]
        col += n_sectors

        # Global macro (same per day for every node)
        macro_vals = np.nan_to_num(macro_array, nan=0.0, posinf=0.0, neginf=0.0)
        node_feat_array[:, n_idx, col:col + n_macro] = macro_vals
        col += n_macro

    print(f"[gnn_data] Feature matrix: {T} days × {N} nodes × {n_features} features")
    print(f"  Per-node time-varying: {n_time_features} | Static: {n_static} "
          f"(1 beta + {n_sectors} sectors) | Macro: {n_macro}")
    print(f"[gnn_data] Tickers: {tickers[:4]}... ({N} total)")

    return node_feat_array, common_index, tickers, n_features


# ──────────────────────────────────────────────
# 2. Sliding Windows
# ──────────────────────────────────────────────

def build_sliding_windows(node_feat_array, window=20):
    """
    Convert [T, N, F] array into sliding windows.

    Window i uses node_feat_array[i : i+window] → shape [window, N, F].
    Prediction date for window i = dates[i + window] (next day after window).

    Returns
    -------
    windows : np.ndarray [T-window, window, N, F]
    """
    T, N, F = node_feat_array.shape
    n_windows = T - window
    windows = np.empty((n_windows, window, N, F), dtype=np.float32)
    for i in range(n_windows):
        windows[i] = node_feat_array[i: i + window]
    return windows  # [T-window, window, N, F]


# ──────────────────────────────────────────────
# 3. Graph Edges
# ──────────────────────────────────────────────

def build_fully_connected_edges(n_nodes):
    """
    Fully connected edge index (no self-loops, no threshold).
    The GAT learns which connections matter — no hard pre-filtering.

    Returns
    -------
    edge_index : torch.LongTensor [2, n_nodes*(n_nodes-1)]
    """
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    return torch.tensor([rows, cols], dtype=torch.long)


# ──────────────────────────────────────────────
# 4. PyTorch Dataset
# ──────────────────────────────────────────────

class GNNDataset(Dataset):
    """
    Dataset of (graph_window, worker_return) pairs.

    windows        : np.ndarray [T, window, N, F]
    worker_returns : np.ndarray [T]   — aligned to window prediction dates
    """

    def __init__(self, windows, worker_returns):
        assert len(windows) == len(worker_returns), (
            f"Shape mismatch: windows={len(windows)}, returns={len(worker_returns)}"
        )
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.returns = torch.tensor(worker_returns, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.returns[idx]


# ──────────────────────────────────────────────
# 5. Full Dataset Builder
# ──────────────────────────────────────────────

def build_graph_dataset(returns, all_fields, profiles, econ, yield_curve,
                        worker_returns=None, window=20):
    """
    Full pipeline: raw data → (windows, worker_returns_aligned, edge_index, dates).

    Parameters
    ----------
    returns        : pd.DataFrame  — daily returns
    all_fields     : dict          — Bloomberg field dict from data_loader
    profiles       : pd.DataFrame  — stock profiles
    econ           : pd.DataFrame  — economic indicators
    yield_curve    : pd.DataFrame
    worker_returns : pd.Series or None — QP solver returns (aligned to dates)
    window         : int           — lookback window in trading days

    Returns
    -------
    windows              : np.ndarray [T-window, window, N, F]
    worker_ret_aligned   : np.ndarray [T-window]
    edge_index           : torch.LongTensor [2, N*(N-1)]
    window_dates         : pd.DatetimeIndex  — prediction date for each window
    n_features           : int
    n_nodes              : int
    """
    node_feats, dates_all, tickers, n_features = load_gnn_node_features(
        returns, all_fields, profiles, econ, yield_curve
    )
    n_nodes = len(tickers)

    windows = build_sliding_windows(node_feats, window=window)

    # Prediction date = day AFTER the window (no lookahead bias)
    # Window i uses days [i, i+window-1]; predicts α for day i+window
    window_dates = dates_all[window:]   # length = T - window  ✓

    # Align worker returns to prediction dates
    if worker_returns is not None:
        worker_ret_aligned = (
            worker_returns.reindex(window_dates).fillna(0.0).values.astype(np.float32)
        )
    else:
        worker_ret_aligned = np.zeros(len(windows), dtype=np.float32)

    edge_index = build_fully_connected_edges(n_nodes)

    print(f"[gnn_data] Windows: {windows.shape[0]} samples "
          f"(date range: {window_dates[0].date()} → {window_dates[-1].date()})")
    print(f"[gnn_data] Edges: {edge_index.shape[1]} directed edges "
          f"(fully connected, {n_nodes} nodes)")

    return windows, worker_ret_aligned, edge_index, window_dates, n_features, n_nodes
