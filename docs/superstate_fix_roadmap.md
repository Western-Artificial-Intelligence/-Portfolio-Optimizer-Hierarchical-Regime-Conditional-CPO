# Roadmap: Fixing the Super-State (0 Rows)

The Supervisor reports **"Super-state features: 38 columns, 0 rows"** and then fails with 0 aligned samples. This roadmap chunks the fix into ordered steps.

---

## Root cause (summary)

`build_super_state()` in `src/supervisor.py`:

1. Builds a feature dict from **four sources** with **different indices**:  
   `uncertainty` (from `returns_all`), `econ`, `yield_curve`, and `clone_returns`.
2. Builds `X = pd.DataFrame(features, index=clone_returns.index)` so all columns are aligned to `clone_returns.index`.
3. Calls `X.dropna()`, which drops **any row with a NaN in any column**.

If **any** source has an index that doesn’t align with `clone_returns.index` (e.g. timezone, type, or no overlap), those columns become all-NaN after alignment, and `dropna()` removes every row → **0 rows**.

---

## Chunk 1: Add diagnostics (no behavior change)

**Goal:** See exactly where rows are lost and which indices don’t match.

**Tasks:**

1. **Log index metadata** at the start of `build_super_state()`:
   - `clone_returns.index`: `min`, `max`, `len`, `dtype`, `tz`.
   - `returns_all.index`: same.
   - `econ.index`: same.
   - `yield_curve.index`: same.

2. **Log overlap counts** before building `X`:
   - `clone_returns.index.intersection(returns_all.index).size`
   - Same for `econ` and `yield_curve`.

3. **After** `X = pd.DataFrame(features, index=clone_returns.index)`:
   - `X.isna().sum()` (or per-column NaN count).
   - `len(X)` and `len(X.dropna())`.

4. **Optional:** Log 1–2 sample dates from each index to compare string/datetime representation.

**Deliverable:** Running the pipeline once should print enough info to see which source(s) fail to align.

**Files:** `src/supervisor.py` (inside `build_super_state`).

---

## Chunk 2: Normalize indices to a canonical form

**Goal:** Ensure all inputs use the same index type and timezone so alignment doesn’t produce all-NaN.

**Tasks:**

1. **Define a canonical index** in `build_super_state()`:
   - Use `clone_returns.index` as the target (same as current).
   - Normalize it: e.g. `pd.DatetimeIndex(..., tz=None)` or `tz='UTC'` and ensure it’s `datetime64[ns]`.

2. **Align each source to that canonical index** before building `features`:
   - **Uncertainty:** `uncertainty = compute_uncertainty_features(returns_all)`. Then `uncertainty = uncertainty.reindex(canonical_index)` (optionally with `method="ffill"` for forward-fill).
   - **Econ:** Already `econ.reindex(clone_returns.index, method="ffill")` — use `canonical_index` instead and ensure econ’s index is comparable (e.g. localize/drop tz to match).
   - **Yield curve:** Same as econ: reindex to `canonical_index` with `method="ffill"`.
   - **Clone momentum:** Build from `clone_returns` already on the same index, or reindex clone_returns to `canonical_index` if needed.

3. **Ensure `returns_all` can align:** If `returns_all.index` differs from `clone_returns.index` (e.g. different trimming in Phase 1 vs Phase 2), either:
   - Pass the same underlying prices/returns into both Phase 2 and Phase 3 so indices match, or
   - In Phase 3, restrict `returns_all` to `clone_returns.index` before calling `compute_uncertainty_features`, e.g. `returns_all = returns_all.reindex(clone_returns.index).dropna(how='all')` or align in a way that keeps a common date range.

**Deliverable:** After normalization, `X = pd.DataFrame(features, index=canonical_index)` should have no (or minimal) unintended NaNs from index mismatch.

**Files:** `src/supervisor.py` (`build_super_state`), possibly `main.py` or data flow into `run_supervisor_pipeline`.

---

## Chunk 3: Make `dropna()` less aggressive ✅

**Goal:** Avoid dropping every row when only some columns have NaNs (e.g. leading rolling NaNs or sparse macro data).

**Tasks:**

1. **Option A – Drop only rows that are NaN in “critical” columns:**  
   Define a small set of columns that must be non-NaN (e.g. `clone_ret_5d`, `vol_21d`, one macro column).  
   `X = X.loc[X[critical_cols].notna().all(axis=1)]`.

2. **Option B – Fill instead of drop (where sensible):**  
   For non-critical columns (e.g. some macro changes), fill NaNs with 0 or column median before `dropna()`, then keep a single `dropna()` only for columns that must be valid.

3. **Option C – Drop only all-NaN rows:**  
   Replace `X.dropna()` with `X.dropna(how='all')` so that only rows that are NaN in every column are removed. Then optionally drop rows that are NaN in key columns (as in A).

**Deliverable:** `X` has a non-zero number of rows when at least one source has valid data on `clone_returns.index`, and the pipeline can proceed.

**Files:** `src/supervisor.py` (`build_super_state`).

---

## Chunk 4: Align uncertainty to clone dates explicitly ✅

**Goal:** Remove the hidden dependency that `returns_all.index` equals `clone_returns.index`. Right now, uncertainty is built from `returns_all`; when we put it in `features` and build `X` with `index=clone_returns.index`, pandas aligns by index. If the two indices differ, we get NaNs.

**Tasks:**

1. **After** `uncertainty = compute_uncertainty_features(returns_all)`:
   - Reindex to the canonical index:  
     `uncertainty = uncertainty.reindex(canonical_index, method="ffill")`  
     (or use a join on date part only if timezone/time components differ).
   - Ensure `returns_all` actually contains the dates in `clone_returns.index` (if not, Chunk 2 should fix the data flow so it does).

2. **Add a safeguard:** If `uncertainty.reindex(canonical_index).isna().all(axis=1).all()`, log a warning that uncertainty is all-NaN for the clone index and that index alignment is likely wrong.

**Deliverable:** Uncertainty features are explicitly aligned to clone dates; no silent all-NaN from index mismatch.

**Files:** `src/supervisor.py` (`build_super_state`).

---

## Chunk 5: Data loading and Phase 1/2 consistency ✅

**Goal:** Ensure econ and yield_curve have a date range that overlaps `clone_returns`, and that Phase 1 doesn’t trim or change the index in a way that makes Phase 2’s clone_returns and Phase 3’s returns_all diverge.

**Tasks:**

1. **Inspect loaders:**  
   In `src/data_loader.py`, confirm `econ` and `yield_curve` use `parse_dates=["date"], index_col="date"` and that the resulting index is `datetime64[ns]` (and timezone if you standardize on tz).

2. **Phase 1:**  
   Check whether `prices_clean` is trimmed by date (e.g. only 2020+). If so, either:
   - Extend the trim to include 2010–2019 so that clone_returns and returns_all cover the doc’s train period, or
   - Document that the project now uses a shorter range and adjust `train_end` (or the roadmap in `docs/phase3.md`) accordingly.

3. **Phase 2:**  
   Confirm that `rolling_optimization` uses the same `prices_clean` as Phase 3’s `returns_all` source, so that `clone_returns.index` is a subset of (or equal to) `returns_all.index`.

**Deliverable:** A single, consistent date range and index type from loaders through Phase 2 and into Phase 3, so that super-state inputs all align.

**Files:** `src/data_loader.py`, Phase 1 code that produces `prices_clean`, `main.py` (data flow), and optionally `docs/phase3.md`.

---

## Chunk 6: Tests and regression guard ✅

**Goal:** Lock in the fix and avoid regressions.

**Tasks:**

1. **Unit test for `build_super_state()`:**  
   With small synthetic DataFrames (clone_returns, returns_all, econ, yield_curve) that share a common date range and index type, assert that the returned `X` has `len(X) > 0` and expected columns.

2. **Integration test (optional):**  
   Run Phase 1 → Phase 2 → Phase 3 with the real (or a subset of) data and assert that `X` has rows and that the Supervisor trains and produces a comparison table/plot.

3. **Document expected index contract:**  
   In `build_super_state` docstring or `docs/phase3.md`, state that clone_returns, returns_all, econ, and yield_curve must have a DatetimeIndex and that their date ranges should overlap; recommend passing the same price/return universe from Phase 2 into Phase 3.

**Deliverable:** Tests pass; future index or loader changes are less likely to bring back 0 rows.

**Files:** New test file (e.g. `tests/test_supervisor.py` or under `tests/`), `src/supervisor.py` (docstring), optionally `docs/phase3.md`.

---

## Order of execution

| Order | Chunk | Purpose |
|-------|--------|--------|
| 1 | **Chunk 1 – Diagnostics** | See why we get 0 rows and which index is wrong. |
| 2 | **Chunk 2 – Normalize indices** | Fix index/timezone/type so alignment works. |
| 3 | **Chunk 4 – Align uncertainty** | Explicitly tie uncertainty to clone dates. |
| 4 | **Chunk 3 – Softer dropna** | Avoid dropping all rows when only some columns are NaN. |
| 5 | **Chunk 5 – Data/Phase 1/2** | Ensure data range and pipeline consistency. |
| 6 | **Chunk 6 – Tests** | Guard against future regressions. |

---

## Quick reference: key code locations

- **Super-state build:** `src/supervisor.py` → `build_super_state()`
- **Uncertainty features:** `src/forecaster.py` → `compute_uncertainty_features()`
- **Data flow into Phase 3:** `main.py` → `phase3()` → `run_supervisor_pipeline(clone_returns, returns_all, econ, yield_curve, train_end=...)`
- **Loaders:** `src/data_loader.py` → `load_economic_indicators()`, `load_yield_curve()`
- **Clone returns source:** `main.py` → `phase2()` → `rolling_optimization(prices_clean, ...)` → `clone_returns`
