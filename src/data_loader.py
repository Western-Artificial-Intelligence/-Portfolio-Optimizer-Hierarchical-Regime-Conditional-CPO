"""
Data loading and cleaning for all Bloomberg datasets.

The stock_history.csv has a non-standard multi-level Bloomberg format:
    Row 0: ticker names repeated across columns (one per field)
    Row 1: field names (PX_LAST, PX_VOLUME, etc.)
    Row 2: empty "date" row
    Row 3+: actual data

This module handles that parsing and returns clean pandas DataFrames.
"""

import pandas as pd
import numpy as np
from src.config import (
    STOCK_HISTORY_CSV,
    STOCK_PROFILES_CSV,
    ECONOMIC_INDICATORS_CSV,
    YIELD_CURVE_CSV,
    PRICE_FIELD,
    ALL_TICKERS,
)


def load_stock_history(filepath=None):
    """
    Parse the Bloomberg multi-level stock_history.csv.

    Returns
    -------
    prices : pd.DataFrame
        DatetimeIndex × tickers, containing PX_LAST (close prices).
    all_fields : dict[str, pd.DataFrame]
        Mapping from field name -> DataFrame (DatetimeIndex x tickers).
    """
    filepath = filepath or STOCK_HISTORY_CSV

    # Read the raw CSV: first two rows are ticker and field headers
    raw = pd.read_csv(filepath, header=None, low_memory=False)

    # Row 0 = tickers, Row 1 = fields, Row 2 = "date" header, Row 3+ = data
    tickers_row = raw.iloc[0, 1:].values   # skip first col (label "ticker")
    fields_row = raw.iloc[1, 1:].values    # skip first col (label "field")
    dates = pd.to_datetime(raw.iloc[3:, 0].values)

    # Data block (rows 3+, cols 1+)
    data = raw.iloc[3:, 1:].copy()
    data.index = dates
    data.columns = pd.MultiIndex.from_arrays(
        [tickers_row, fields_row], names=["ticker", "field"]
    )

    # Convert everything to numeric
    data = data.apply(pd.to_numeric, errors="coerce")

    # Group by field and create clean DataFrames
    unique_fields = list(set(fields_row))
    all_fields = {}
    for field in unique_fields:
        if not isinstance(field, str) or pd.isna(field):
            continue
        field_data = data.xs(field, level="field", axis=1)
        # Remove duplicate ticker columns (keep first)
        field_data = field_data.loc[:, ~field_data.columns.duplicated()]
        all_fields[field] = field_data

    # Extract closing prices specifically
    prices = all_fields.get(PRICE_FIELD, pd.DataFrame())

    print(f"[data_loader] Loaded stock history: {len(dates)} trading days, "
          f"{prices.shape[1]} tickers, {len(all_fields)} fields")
    print(f"[data_loader] Date range: {dates.min().date()} to {dates.max().date()}")

    return prices, all_fields


def load_stock_profiles(filepath=None):
    """
    Parse the long-format stock_profiles.csv.

    Returns
    -------
    profiles : pd.DataFrame
        Indexed by ticker, columns = field names (GICS_SECTOR_NAME, EQY_BETA, etc.)
    """
    filepath = filepath or STOCK_PROFILES_CSV

    df = pd.read_csv(filepath)
    profiles = df.pivot(index="ticker", columns="field", values="value")

    # Convert numeric columns
    numeric_cols = ["EQY_BETA", "BEST_ANALYST_RATING", "CUR_MKT_CAP", "EQY_SH_OUT"]
    for col in numeric_cols:
        if col in profiles.columns:
            profiles[col] = pd.to_numeric(profiles[col], errors="coerce")

    print(f"[data_loader] Loaded {len(profiles)} stock profiles")
    return profiles


def load_economic_indicators(filepath=None):
    """
    Parse economic_indicators.csv.

    Returns
    -------
    econ : pd.DataFrame
        DatetimeIndex, columns: T10Y2Y, IG_SPREAD, HY_SPREAD, DXY, MOVE
    """
    filepath = filepath or ECONOMIC_INDICATORS_CSV

    econ = pd.read_csv(filepath, parse_dates=["date"], index_col="date")
    econ.sort_index(inplace=True)
    assert isinstance(econ.index, pd.DatetimeIndex), "Economic indicators must have DatetimeIndex for super-state alignment."

    # Forward-fill missing values (weekends/holidays in macro data)
    econ = econ.ffill()

    # Report gaps
    missing = econ.isna().sum()
    if missing.any():
        print(f"[data_loader] Economic indicators remaining NaNs:\n{missing[missing > 0]}")

    print(f"[data_loader] Loaded economic indicators: {len(econ)} rows, "
          f"{econ.shape[1]} columns")
    print(f"[data_loader] Date range: {econ.index.min().date()} to {econ.index.max().date()}")

    return econ


def load_yield_curve(filepath=None):
    """
    Parse yield_curve_spread.csv.

    Returns
    -------
    yc : pd.DataFrame
        DatetimeIndex, columns: US_10Y, US_2Y, YIELD_CURVE_SPREAD, INVERTED
    """
    filepath = filepath or YIELD_CURVE_CSV

    yc = pd.read_csv(filepath, parse_dates=["date"], index_col="date")
    yc.sort_index(inplace=True)
    assert isinstance(yc.index, pd.DatetimeIndex), "Yield curve must have DatetimeIndex for super-state alignment."

    print(f"[data_loader] Loaded yield curve: {len(yc)} rows")
    print(f"[data_loader] Inversions detected: "
          f"{yc['INVERTED'].sum()} days ({yc['INVERTED'].mean()*100:.1f}%)")

    return yc


def load_all():
    """
    Convenience function to load all datasets at once.

    Returns
    -------
    prices : pd.DataFrame
    all_fields : dict
    profiles : pd.DataFrame
    econ : pd.DataFrame
    yield_curve : pd.DataFrame
    """
    print("=" * 60)
    print("Loading all datasets...")
    print("=" * 60)

    prices, all_fields = load_stock_history()
    profiles = load_stock_profiles()
    econ = load_economic_indicators()
    yield_curve = load_yield_curve()

    # Quick data quality summary
    print("\n" + "=" * 60)
    print("DATA QUALITY SUMMARY")
    print("=" * 60)

    # Check which tickers have sufficient data
    coverage = prices.notna().mean()
    sparse = coverage[coverage < 0.5]
    if len(sparse) > 0:
        print(f"\n[WARNING] Sparse tickers (< 50% coverage):")
        for ticker, pct in sparse.items():
            print(f"   {ticker}: {pct*100:.1f}% data available")

    full = coverage[coverage >= 0.5]
    print(f"\n[OK] Tickers with good coverage (>= 50%): {len(full)}")
    print(f"[OK] Total trading days: {len(prices)}")
    print(f"[OK] Economic indicator rows: {len(econ)}")
    print(f"[OK] Yield curve rows: {len(yield_curve)}")

    return prices, all_fields, profiles, econ, yield_curve
