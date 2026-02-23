"""
Tests for the AI Supervisor (Phase 3), including super-state build.

Chunk 6: Regression guard so build_super_state never returns 0 rows
when inputs have overlapping DatetimeIndex and valid data.

Run from project root:
  python -m unittest tests.test_supervisor
  or: pytest tests/test_supervisor.py -v  (if pytest is installed)
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Project root so we can import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.supervisor import build_super_state


class TestBuildSuperState(unittest.TestCase):
    def test_build_super_state_returns_non_empty(self):
        """Unit test: overlapping synthetic data yields len(X) > 0 and expected columns."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2020-01-02", periods=n, freq="B")

        clone_returns = pd.Series(0.001 * np.random.randn(n), index=dates)
        returns_all = pd.DataFrame(
            {
                "SPY US Equity": 0.001 * np.random.randn(n),
                "Ticker A": 0.001 * np.random.randn(n),
            },
            index=dates,
        )
        econ = pd.DataFrame(
            {
                "T10Y2Y_PX_LAST": 270 + np.random.randn(n).cumsum(),
                "IG_SPREAD_PX_LAST": 100 + np.random.randn(n),
                "HY_SPREAD_PX_LAST": 300 + np.random.randn(n),
                "DXY_PX_LAST": 95 + 0.1 * np.random.randn(n),
                "MOVE_PX_LAST": 100 + np.random.randn(n),
            },
            index=dates,
        )
        yield_curve = pd.DataFrame(
            {
                "YIELD_CURVE_SPREAD": 1.5 + 0.01 * np.random.randn(n),
                "INVERTED": (np.random.rand(n) > 0.9).astype(int),
            },
            index=dates,
        )

        X = build_super_state(clone_returns, returns_all, econ, yield_curve)

        self.assertGreater(len(X), 0, "build_super_state must return at least one row when inputs overlap.")
        self.assertIsInstance(X.index, pd.DatetimeIndex)
        critical = ["clone_ret_5d", "vol_5d", "yc_spread"]
        for c in critical:
            self.assertIn(c, X.columns, f"Expected column {c!r} in super-state.")
        self.assertTrue(X.notna().any(axis=1).all(), "No row should be all-NaN.")


if __name__ == "__main__":
    unittest.main()
