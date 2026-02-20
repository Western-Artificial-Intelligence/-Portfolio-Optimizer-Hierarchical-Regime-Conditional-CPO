"""
AI Supervisor — Continuous Meta-Labeling layer for the Portfolio Optimizer.

Predicts the probability that the Worker portfolio will experience
a drawdown > threshold over the next horizon. Uses XGBoost trained on
a "super-state" of uncertainty + macro features.

The Supervisor outputs a continuous confidence score P ∈ [0, 1] that
blends between aggressive (Worker weights) and defensive (cash) allocations:
    W_final = P × W_aggressive + (1-P) × W_defensive

References:
  - López de Prado (2018), "Advances in Financial Machine Learning"
  - Chan (2023), "How to Use Machine Learning for Optimization" (CPO)
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

from src.forecaster import compute_uncertainty_features


# ──────────────────────────────────────────────
# 1. Meta-Label Generation
# ──────────────────────────────────────────────

def generate_meta_labels(clone_returns, threshold=0.02, horizon=5, verbose=True):
    """
    Generate binary meta-labels for the AI Supervisor.

    Label = 1 if the Clone does NOT suffer a drawdown > threshold
            over the next `horizon` trading days (success).
    Label = 0 if the Clone DOES suffer a drawdown > threshold (failure).

    Parameters
    ----------
    clone_returns : pd.Series
        Daily returns of the Canadian Clone.
    threshold : float
        Drawdown threshold (e.g., 0.02 = 2%).
    horizon : int
        Forward-looking window in trading days (5 = 1 week).

    Returns
    -------
    labels : pd.Series
        Binary labels (1 = safe, 0 = danger), indexed by date.
    """
    cum = (1 + clone_returns).cumprod()
    labels = pd.Series(np.nan, index=clone_returns.index)

    for i in range(len(cum) - horizon):
        window = cum.iloc[i:i + horizon + 1]
        peak = window.iloc[0]  # Value at decision point
        min_val = window.iloc[1:].min()  # Worst point in next horizon days
        drawdown = (min_val - peak) / peak

        labels.iloc[i] = 1 if drawdown > -threshold else 0  # 1=safe, 0=danger

    labels = labels.dropna()
    if verbose:
        n_danger = (labels == 0).sum()
        n_safe = (labels == 1).sum()
        print(f"[supervisor] Meta-labels: {len(labels)} total, "
              f"{n_safe} safe ({n_safe/len(labels)*100:.1f}%), "
              f"{n_danger} danger ({n_danger/len(labels)*100:.1f}%)")

    return labels


# ──────────────────────────────────────────────
# 2. Super-State Feature Engineering
# ──────────────────────────────────────────────

def build_super_state(clone_returns, returns_all, econ, yield_curve, verbose=True):
    """
    Build the complete super-state feature matrix.

    Features:
    - Uncertainty features (from forecaster.py)
    - Macro indicators (MOVE, DXY, T10Y2Y, IG/HY spreads)
    - Yield curve features (spread, inversion flag)
    - Clone momentum signals

    Returns
    -------
    X : pd.DataFrame
        Feature matrix indexed by date.
    """
    features = {}

    # ── Uncertainty features ──
    uncertainty = compute_uncertainty_features(returns_all, verbose=verbose)
    for col in uncertainty.columns:
        features[col] = uncertainty[col]

    # ── Macro indicators ──
    econ_aligned = econ.reindex(clone_returns.index, method="ffill").bfill()
    for col in econ_aligned.columns:
        # Raw level
        features[f"macro_{col}"] = econ_aligned[col]
        # Rate of change (1-week, 1-month)
        features[f"macro_{col}_chg_5d"] = econ_aligned[col].pct_change(5)
        features[f"macro_{col}_chg_21d"] = econ_aligned[col].pct_change(21)

    # ── Yield curve ──
    yc_aligned = yield_curve.reindex(clone_returns.index, method="ffill").bfill()
    if "YIELD_CURVE_SPREAD" in yc_aligned.columns:
        features["yc_spread"] = yc_aligned["YIELD_CURVE_SPREAD"]
        features["yc_spread_chg_5d"] = yc_aligned["YIELD_CURVE_SPREAD"].pct_change(5)
    if "INVERTED" in yc_aligned.columns:
        features["yc_inverted"] = yc_aligned["INVERTED"]

    # ── Clone-specific momentum signals ──
    features["clone_ret_5d"] = clone_returns.rolling(5).sum()
    features["clone_ret_21d"] = clone_returns.rolling(21).sum()
    features["clone_ret_63d"] = clone_returns.rolling(63).sum()

    X = pd.DataFrame(features, index=clone_returns.index)

    # Replace inf with NaN (e.g. from relative_vol when vol near zero)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are entirely NaN (e.g. vol_of_vol_63d with insufficient warmup)
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
        if verbose:
            print(f"[supervisor] Dropped {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")

    # Drop early rows with NaN from rolling calculations
    X = X.dropna()

    if verbose:
        print(f"[supervisor] Super-state features: {X.shape[1]} columns, "
              f"{X.shape[0]} rows")

    return X


# ──────────────────────────────────────────────
# 3. XGBoost Classifier (TimeSeriesSplit CV)
# ──────────────────────────────────────────────

def train_classifier(X, y, n_splits=5, verbose=True):
    """
    Train an XGBoost classifier with time-series cross-validation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary labels.
    n_splits : int
        Number of TimeSeriesSplit folds.

    Returns
    -------
    model : XGBClassifier
        Fitted on all data.
    cv_results : dict
        Cross-validation metrics.
    """
    # Align X and y
    common = X.index.intersection(y.index)
    X_aligned = X.loc[common]
    y_aligned = y.loc[common]

    if verbose:
        print(f"\n[supervisor] Training XGBoost on {len(X_aligned)} samples, "
              f"{X_aligned.shape[1]} features")

    # Class imbalance handling — upweight the minority class (danger)
    n_pos = int((y_aligned == 1).sum())
    n_neg = int((y_aligned == 0).sum())
    scale_pos_weight = n_pos / n_neg if n_neg > 0 else 1.0
    if verbose:
        print(f"[supervisor] Class balance: {n_pos} safe / {n_neg} danger "
              f"(scale_pos_weight={scale_pos_weight:.2f})")

    # Ensure labels are int (XGBoost requires clean integer classes)
    y_aligned = y_aligned.astype(int)

    # Single-class fallback: XGBoost fails with one class; use DummyClassifier
    if len(np.unique(y_aligned)) < 2:
        majority = int(y_aligned.mode().iloc[0])
        fallback = DummyClassifier(strategy="constant", constant=majority)
        fallback.fit(X_aligned, y_aligned)
        cv_metrics = {"accuracy": [1.0], "precision": [1.0], "recall": [1.0],
                      "f1": [1.0], "auc": [0.5]}
        return fallback, cv_metrics

    # TimeSeriesSplit CV
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_metrics = {"accuracy": [], "precision": [], "recall": [],
                  "f1": [], "auc": []}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_aligned)):
        X_train, X_val = X_aligned.iloc[train_idx], X_aligned.iloc[val_idx]
        y_train, y_val = y_aligned.iloc[train_idx], y_aligned.iloc[val_idx]

        # Skip folds where train or val has only one class
        if len(y_train.unique()) < 2 or len(y_val.unique()) < 2:
            if verbose:
                print(f"  [fold {fold+1}] Skipped — single class in train or val")
            continue

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        cv_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
        cv_metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
        cv_metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
        cv_metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
        try:
            cv_metrics["auc"].append(roc_auc_score(y_val, y_prob))
        except ValueError:
            cv_metrics["auc"].append(0.5)

    if verbose:
        print(f"\n[supervisor] Cross-Validation Results ({n_splits}-fold TimeSeriesSplit):")
        for metric, values in cv_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {metric:>10s}: {mean_val:.4f} ± {std_val:.4f}")

    # Final model: train on all data
    final_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
    )
    final_model.fit(X_aligned, y_aligned, verbose=False)

    return final_model, cv_metrics


# ──────────────────────────────────────────────
# 4. Execution Logic — Continuous Blending
# ──────────────────────────────────────────────

def apply_supervisor(clone_returns, confidence_scores, cash_return=0.0):
    """
    Apply continuous blending based on the Supervisor's confidence score.

    Instead of discrete regime switching (bull/bear thresholds), the
    confidence score P ∈ [0, 1] is used directly as a blending coefficient:

        final_return = P × aggressive_return + (1-P) × defensive_return

    where aggressive = Worker's clone returns, defensive = cash.

    This produces smooth allocation transitions, proportional hedging,
    and reduced turnover compared to step-function approaches.

    Parameters
    ----------
    clone_returns : pd.Series
        Daily unsupervised Worker returns (aggressive allocation).
    confidence_scores : pd.Series
        P ∈ [0, 1] from the XGBoost classifier, indexed by date.
    cash_return : float
        Daily return on defensive allocation (default: 0.0 = cash).

    Returns
    -------
    supervised_returns : pd.Series
        Blended daily returns.
    regime_labels : pd.Series
        "aggressive", "moderate", or "defensive" for reporting.
    allocation : pd.Series
        The blending coefficient P used each day.
    """
    common = clone_returns.index.intersection(confidence_scores.index)
    clone = clone_returns.loc[common]
    P = confidence_scores.loc[common].clip(0.0, 1.0)

    # ── Continuous blending: W_final = P × W_agg + (1-P) × W_def ──
    supervised_returns = P * clone + (1 - P) * cash_return

    # Regime labels (for reporting / visualization only)
    regime = pd.Series("moderate", index=common)
    regime[P > 0.7] = "aggressive"
    regime[P < 0.3] = "defensive"

    agg_pct = (regime == "aggressive").mean() * 100
    mod_pct = (regime == "moderate").mean() * 100
    def_pct = (regime == "defensive").mean() * 100
    avg_P = P.mean()

    print(f"\n[supervisor] Continuous Blending Summary:")
    print(f"  Average confidence:  {avg_P:.3f}")
    print(f"  [Aggressive] (P>0.7): {agg_pct:.1f}%")
    print(f"  [Moderate]  (0.3-0.7): {mod_pct:.1f}%")
    print(f"  [Defensive] (P<0.3):  {def_pct:.1f}%")

    return supervised_returns, regime, P


# ──────────────────────────────────────────────
# 5. End-to-End Pipeline
# ──────────────────────────────────────────────

def run_supervisor_pipeline(clone_returns, returns_all, econ, yield_curve,
                             train_end="2019-12-31"):
    """
    Full Phase 3 pipeline: features → labels → train → predict → apply.

    Uses walk-forward: train on data up to train_end, predict on test period.

    Parameters
    ----------
    clone_returns : pd.Series
        Daily clone returns from Phase 2.
    returns_all : pd.DataFrame
        All stock returns (for computing uncertainty).
    econ : pd.DataFrame
        Economic indicators.
    yield_curve : pd.DataFrame
        Yield curve data.
    train_end : str
        End of training period.

    Returns
    -------
    supervised_returns : pd.Series
    regime : pd.Series
    model : XGBClassifier
    confidence : pd.Series
    """
    print("\n" + "=" * 60)
    print("PHASE 3: AI Supervisor — Meta-Labeling")
    print("=" * 60)

    # Step 1: Generate meta-labels
    labels = generate_meta_labels(clone_returns, threshold=0.02, horizon=5)

    # Step 2: Build super-state features
    X = build_super_state(clone_returns, returns_all, econ, yield_curve)

    # Step 3: Align and split
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    y = labels.loc[common]

    train_mask = X.index <= train_end
    test_mask = X.index > train_end

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    print(f"\n[supervisor] Train: {len(X_train)} samples ({X_train.index.min().date()} → {X_train.index.max().date()})")
    print(f"[supervisor] Test:  {len(X_test)} samples ({X_test.index.min().date()} → {X_test.index.max().date()})")

    print(f"X_train date range: {X_train.index.min()} → {X_train.index.max()}, n={len(X_train)}")
    print(f"y_train date range: {y_train.index.min()} → {y_train.index.max()}, n={len(y_train)}")
    print(f"Intersection: {len(X_train.index.intersection(y_train.index))}")

    # Step 4: Train on training data only
    model, cv_results = train_classifier(X_train, y_train)

    # Step 5: Predict on test set
    confidence_test = pd.Series(
        model.predict_proba(X_test)[:, 1],
        index=X_test.index,
        name="confidence"
    )

    # Also get train confidence for full picture
    confidence_train = pd.Series(
        model.predict_proba(X_train)[:, 1],
        index=X_train.index,
        name="confidence"
    )
    confidence_all = pd.concat([confidence_train, confidence_test]).sort_index()

    # Step 6: Print test set classification report
    y_pred_test = model.predict(X_test)
    print(f"\n[supervisor] Test Set Classification Report:")
    print(classification_report(y_test, y_pred_test,
                                 target_names=["Danger (0)", "Safe (1)"]))

    # Step 7: Apply supervisor to test period (continuous blending)
    test_clone = clone_returns.loc[confidence_test.index]
    supervised_returns, regime, allocation = apply_supervisor(test_clone, confidence_test)

    # Step 8: Feature importance
    importances = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print(f"\n[supervisor] Top 10 Features:")
    for feat, imp in importances.head(10).items():
        print(f"  {feat:>30s}: {imp:.4f}")

    return supervised_returns, regime, model, confidence_all, importances, allocation
