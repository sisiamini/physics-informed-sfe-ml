#!/usr/bin/env python3
"""
Domain_Applicability_Nested_XGBoost.py

Nested applicability-domain analysis for the already-frozen final XGBoost SFE model.

Why this version exists
-----------------------
The previous applicability-domain script searched composition ranges and then
reported reliability on the same out-of-fold predictions used to construct
those ranges. That is useful for exploratory domain construction, but it can
optimistically bias the reported reliability.

This revised script separates domain construction from domain evaluation.

Nested validation
-----------------
1. Use exact-composition-grouped outer cross-validation.
2. For each outer fold:
   a. Use only the outer-training partition.
   b. Generate exact-composition-grouped inner OOF predictions.
   c. Define prediction success as absolute percentage error <= --max_error.
   d. Fit the kNN/PCA dense-composition reference on outer-training compositions.
   e. Search visible composition min-max ranges using outer-training information only.
   f. Freeze that fold-specific applicability rule.
   g. Fit the frozen XGBoost pipeline on the complete outer-training partition.
   h. Evaluate both prediction accuracy and applicability membership on the
      untouched outer-validation partition.
3. Pool all outer-validation predictions to estimate:
   - applicability-domain coverage
   - reliability inside the domain
   - prediction error inside and outside the domain

Exact compositions are grouped without rounding or near-duplicate merging.

Final operational refit
-----------------------
After nested validation, the same frozen model and domain-construction procedure
are refitted using all available records. This produces dashboard-ready visible
composition ranges, a kNN dense-core reference, and a full-data fitted model.

The final full-data domain's internal reliability is descriptive only. The
nested outer-validation estimate is the reliability result to report.

The original reserved-holdout metrics remain the final conventional test result.
This script does not tune or reselect the model.

Main command
------------
python Domain_Applicability_Nested_XGBoost.py ^
  --csv "data/output_with_8_sfe_models.csv" ^
  --model "model_results_8models/final_xgboost_evaluation/final_xgboost_development_pipeline.joblib" ^
  --out_dir "model_results_8models/domain_applicability" ^
  --max_error 0.20 ^
  --min_reliability 0.80 ^
  --knn_k 10 ^
  --knn_quantile 0.90 ^
  --outer_cv 5 ^
  --inner_cv 5 ^
  --absurd_thr 1e6 ^
  --suppress_warnings
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


TARGET_NAMES = [
    "SFE_Experimental (mJ/m²)",
    "SFE_Experimental (mJ/m2)",
    "SFE_Experimental",
]

ID_NAMES = ["No.", "No", "ID", "Id", "id"]

COMPOSITION_COLS = [
    "Fe (%wt)", "Mn (%wt)", "Al (%wt)", "C (%wt)", "Ni (%wt)",
    "Cr (%wt)", "Si (%wt)", "Cu (%wt)", "Mo (%wt)", "Nb (%wt)",
    "V (%wt)", "Ti (%wt)", "Co (%wt)", "N (%wt)", "P (%wt)", "S (%wt)",
]


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def patch_sklearn_compat(model: Any) -> None:
    """Small compatibility patch for old/new sklearn ColumnTransformer objects."""
    def walk(obj: Any) -> None:
        if isinstance(obj, ColumnTransformer):
            if not hasattr(obj, "force_int_remainder_cols"):
                obj.force_int_remainder_cols = False
            elif getattr(obj, "force_int_remainder_cols", None) == "deprecated":
                obj.force_int_remainder_cols = False

        for attr in ["steps", "transformers"]:
            if hasattr(obj, attr):
                for item in getattr(obj, attr):
                    child = item[1] if isinstance(item, tuple) else item
                    if child is not None:
                        walk(child)

        if hasattr(obj, "named_transformers_"):
            for child in obj.named_transformers_.values():
                if child is not None:
                    walk(child)

    walk(model)


def find_target(df: pd.DataFrame) -> str:
    for col in TARGET_NAMES:
        if col in df.columns:
            return col
    raise ValueError("Target column not found. Tried: " + ", ".join(TARGET_NAMES))


def model_features(model: Any, df: pd.DataFrame, target: str) -> list[str]:
    """Use fitted pipeline feature order when available."""
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        pre = model.named_steps["preprocessor"]
        if hasattr(pre, "feature_names_in_"):
            features = list(pre.feature_names_in_)
            missing = [c for c in features if c not in df.columns]
            if missing:
                raise ValueError(
                    "CSV is missing model feature columns:\n  - "
                    + "\n  - ".join(missing)
                )
            return features

    return [c for c in df.columns if c not in ID_NAMES and c != target]


def composition_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in COMPOSITION_COLS if c in df.columns]
    if len(cols) == len(COMPOSITION_COLS):
        return cols

    missing = [c for c in COMPOSITION_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "The exact 16 composition columns required for grouped validation "
            "were not all found:\n  - " + "\n  - ".join(missing)
        )

    return cols


def numeric_frame(
    df: pd.DataFrame,
    fill_missing_with_zero: bool = False,
    absurd_thr: float | None = None,
) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)

    if absurd_thr is not None:
        for col in out.columns:
            bad = out[col].abs() > absurd_thr
            if bad.any():
                print(
                    f"{col}: replacing {int(bad.sum())} absurd values with NaN"
                )
                out.loc[bad, col] = np.nan

    if fill_missing_with_zero:
        out = out.fillna(0.0)

    return out


def make_box(
    lo: pd.Series,
    hi: pd.Series,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return tuple(map(float, lo.to_numpy())), tuple(map(float, hi.to_numpy()))


def inside_box(
    X: np.ndarray,
    box: tuple[tuple[float, ...], tuple[float, ...]],
) -> np.ndarray:
    lo, hi = map(np.asarray, box)
    return ((X >= lo) & (X <= hi)).all(axis=1)


def min_samples_required(n_rows: int) -> int:
    """Require at least 30 rows or 10% of the domain-construction rows."""
    return max(30, int(np.ceil(0.10 * n_rows)))


def exact_composition_groups(X_comp: pd.DataFrame) -> np.ndarray:
    """
    Create exact nominal-composition groups using all 16 composition columns.

    No rounding or near-duplicate merging is performed.
    """
    composition_index = pd.MultiIndex.from_frame(X_comp)
    groups, _ = pd.factorize(composition_index, sort=False)
    return groups


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0.0:
        return np.nan
    return float(r2_score(y_true, y_pred))


def error_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    success_mask: np.ndarray,
) -> dict[str, float | int]:
    if len(y_true) == 0:
        return {
            "n": 0,
            "reliability": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
        }

    return {
        "n": int(len(y_true)),
        "reliability": float(np.mean(success_mask)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": safe_r2(y_true, y_pred),
    }


# -----------------------------------------------------------------------------
# Grouped OOF prediction
# -----------------------------------------------------------------------------

def grouped_oof_predictions(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> np.ndarray:
    """Generate OOF predictions with exact composition groups kept intact."""
    n_groups = len(np.unique(groups))

    if n_groups < n_splits:
        raise ValueError(
            f"Only {n_groups} exact composition groups are available for "
            f"{n_splits}-fold grouped CV."
        )

    cv = GroupKFold(n_splits=n_splits)
    predictions = np.full(len(y), np.nan, dtype=float)

    for train_idx, valid_idx in cv.split(X, y, groups=groups):
        fold_model = clone(model)
        patch_sklearn_compat(fold_model)
        fold_model.fit(X.iloc[train_idx], y[train_idx])
        predictions[valid_idx] = np.asarray(
            fold_model.predict(X.iloc[valid_idx])
        ).ravel()

    if np.isnan(predictions).any():
        raise RuntimeError("Incomplete grouped OOF predictions.")

    return predictions


# -----------------------------------------------------------------------------
# Dense-core calculation
# -----------------------------------------------------------------------------

def fit_dense_core(
    X_comp: pd.DataFrame,
    k: int,
    quantile: float,
    pca_var: float,
    seed: int,
) -> dict[str, Any]:
    if len(X_comp) <= k:
        raise ValueError(
            f"Need more than k={k} rows for kNN dense-core calculation."
        )

    if not 0.0 < quantile < 1.0:
        raise ValueError("--knn_quantile must be between 0 and 1.")

    scaler = StandardScaler()
    Z0 = scaler.fit_transform(X_comp)

    pca = PCA(
        n_components=pca_var,
        whiten=True,
        svd_solver="full",
        random_state=seed,
    )
    Z = pca.fit_transform(Z0)

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(Z)

    # Querying with X=None excludes the training point itself, so the last
    # returned distance is the kth external-neighbour distance.
    kth_distance = nn.kneighbors(
        X=None,
        n_neighbors=k,
        return_distance=True,
    )[0][:, -1]

    threshold = float(np.quantile(kth_distance, quantile))

    return {
        "scaler": scaler,
        "pca": pca,
        "nn": nn,
        "k": int(k),
        "quantile": float(quantile),
        "distance_threshold": threshold,
        "train_kth_external_distance": kth_distance,
        "train_dense_mask": kth_distance <= threshold,
        "explained_variance_sum": float(
            np.sum(pca.explained_variance_ratio_)
        ),
    }


def apply_dense_core(
    dense_reference: dict[str, Any],
    X_comp_new: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a training-fitted dense-core reference to previously unseen rows."""
    scaler = dense_reference["scaler"]
    pca = dense_reference["pca"]
    nn = dense_reference["nn"]
    k = int(dense_reference["k"])
    threshold = float(dense_reference["distance_threshold"])

    Z0 = scaler.transform(X_comp_new)
    Z = pca.transform(Z0)

    kth_distance = nn.kneighbors(
        Z,
        n_neighbors=k,
        return_distance=True,
    )[0][:, -1]

    dense_mask = kth_distance <= threshold
    return dense_mask, kth_distance


# -----------------------------------------------------------------------------
# Domain search
# -----------------------------------------------------------------------------

def evaluate_domain(
    X_comp_values: np.ndarray,
    dense_mask: np.ndarray,
    success_mask: np.ndarray,
    ape: np.ndarray,
    box: tuple[tuple[float, ...], tuple[float, ...]],
) -> dict[str, Any]:
    visible = inside_box(X_comp_values, box)
    final = visible & dense_mask

    n_final = int(final.sum())
    n_success = int(success_mask[final].sum()) if n_final else 0
    reliability = n_success / n_final if n_final else 0.0
    final_ape = ape[final]

    return {
        "n_inside_visible_range": int(visible.sum()),
        "n_inside_final_domain": n_final,
        "n_success_inside_final_domain": n_success,
        "reliability_inside_final_domain": float(reliability),
        "mean_APE_pct_inside_final_domain": (
            float(np.mean(final_ape) * 100.0) if n_final else np.nan
        ),
        "median_APE_pct_inside_final_domain": (
            float(np.median(final_ape) * 100.0) if n_final else np.nan
        ),
        "p80_APE_pct_inside_final_domain": (
            float(np.quantile(final_ape, 0.80) * 100.0)
            if n_final else np.nan
        ),
        "p90_APE_pct_inside_final_domain": (
            float(np.quantile(final_ape, 0.90) * 100.0)
            if n_final else np.nan
        ),
    }


def initial_boxes(
    X_comp: pd.DataFrame,
    q_max: float,
    q_step: float,
) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    boxes = []

    for q in np.arange(0.0, q_max + 0.5 * q_step, q_step):
        lo = X_comp.quantile(q).mask(lambda s: s.abs() < 1e-12, 0.0)
        hi = X_comp.quantile(1.0 - q).mask(
            lambda s: s.abs() < 1e-12,
            0.0,
        )
        boxes.append(make_box(lo, hi))

    boxes.append(make_box(X_comp.min(), X_comp.max()))
    return boxes


def trimmed_boxes(
    X_comp_values: np.ndarray,
    box: tuple[tuple[float, ...], tuple[float, ...]],
    dense_mask: np.ndarray,
    success_mask: np.ndarray,
    max_per_side: int,
) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    """Move one boundary past unsuccessful rows inside the current domain."""
    current = inside_box(X_comp_values, box) & dense_mask
    bad_rows = np.where(current & ~success_mask)[0]

    if len(bad_rows) == 0:
        return []

    lo = np.asarray(box[0], dtype=float)
    hi = np.asarray(box[1], dtype=float)
    new_boxes = []

    for j in range(X_comp_values.shape[1]):
        all_values = np.unique(X_comp_values[:, j])
        bad_values = np.unique(X_comp_values[bad_rows, j])

        lower_candidates = []
        upper_candidates = []

        for value in bad_values:
            greater = all_values[all_values > value]
            smaller = all_values[all_values < value]

            if len(greater):
                new_lo = float(greater[0])
                if lo[j] < new_lo <= hi[j]:
                    lower_candidates.append(new_lo)

            if len(smaller):
                new_hi = float(smaller[-1])
                if lo[j] <= new_hi < hi[j]:
                    upper_candidates.append(new_hi)

        lower_candidates = sorted(
            set(lower_candidates)
        )[:max_per_side]

        upper_candidates = sorted(
            set(upper_candidates),
            reverse=True,
        )[:max_per_side]

        for value in lower_candidates:
            a, b = lo.copy(), hi.copy()
            a[j] = value
            if np.all(a <= b):
                new_boxes.append((tuple(a), tuple(b)))

        for value in upper_candidates:
            a, b = lo.copy(), hi.copy()
            b[j] = value
            if np.all(a <= b):
                new_boxes.append((tuple(a), tuple(b)))

    return new_boxes


def search_domain(
    X_comp: pd.DataFrame,
    dense_mask: np.ndarray,
    success_mask: np.ndarray,
    ape: np.ndarray,
    min_reliability: float,
    min_samples: int,
    q_max: float,
    q_step: float,
    beam_width: int,
    max_steps: int,
    max_candidates_per_side: int,
) -> tuple[
    tuple[tuple[float, ...], tuple[float, ...]] | None,
    pd.DataFrame,
    dict[str, Any] | None,
]:
    X_values = X_comp.to_numpy(dtype=float)
    seen = set()
    records = []
    valid = []

    def score_invalid(item):
        _, stats = item
        return (
            float(stats["reliability_inside_final_domain"]),
            float(np.sqrt(max(0, stats["n_inside_final_domain"]))),
            int(stats["n_success_inside_final_domain"]),
        )

    def add(box, source: str, step: int):
        stats = evaluate_domain(
            X_values,
            dense_mask,
            success_mask,
            ape,
            box,
        )

        is_valid = (
            stats["n_inside_final_domain"] >= min_samples
            and stats["reliability_inside_final_domain"] >= min_reliability
        )

        records.append({
            "source": source,
            "step": step,
            "valid": bool(is_valid),
            **stats,
        })

        if is_valid:
            valid.append((box, stats))

        return stats

    beam = []

    for box in initial_boxes(
        X_comp,
        q_max=q_max,
        q_step=q_step,
    ):
        if box in seen:
            continue

        seen.add(box)
        beam.append((box, add(box, "initial", 0)))

    beam = sorted(
        beam,
        key=score_invalid,
        reverse=True,
    )[:beam_width]

    for step in range(1, max_steps + 1):
        generated = []

        for box, _ in beam:
            candidates = trimmed_boxes(
                X_values,
                box,
                dense_mask,
                success_mask,
                max_candidates_per_side,
            )

            for candidate in candidates:
                if candidate in seen:
                    continue

                seen.add(candidate)
                stats = add(
                    candidate,
                    "adaptive_trim",
                    step,
                )

                if stats["n_inside_final_domain"] >= min_samples:
                    generated.append((candidate, stats))

        if not generated:
            break

        beam = sorted(
            generated,
            key=score_invalid,
            reverse=True,
        )[:beam_width]

    table = pd.DataFrame(records)

    if not valid:
        return None, table, None

    def score_valid(item):
        _, stats = item
        mean_ape = stats["mean_APE_pct_inside_final_domain"]
        mean_ape_score = (
            -float(mean_ape)
            if np.isfinite(mean_ape)
            else -1e9
        )

        return (
            int(stats["n_inside_final_domain"]),
            int(stats["n_inside_visible_range"]),
            float(stats["reliability_inside_final_domain"]),
            mean_ape_score,
        )

    selected = sorted(
        valid,
        key=score_valid,
        reverse=True,
    )[0]

    return selected[0], table, selected[1]


# -----------------------------------------------------------------------------
# Domain construction and external evaluation
# -----------------------------------------------------------------------------

def construct_domain(
    model: Any,
    X_model_train: pd.DataFrame,
    X_comp_train: pd.DataFrame,
    y_train: np.ndarray,
    train_groups: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Construct one applicability rule using training information only.
    """
    inner_pred = grouped_oof_predictions(
        model=model,
        X=X_model_train,
        y=y_train,
        groups=train_groups,
        n_splits=args.inner_cv,
    )

    ape = np.abs(y_train - inner_pred) / np.maximum(
        np.abs(y_train),
        1e-12,
    )

    success_mask = ape <= args.max_error

    dense = fit_dense_core(
        X_comp=X_comp_train,
        k=args.knn_k,
        quantile=args.knn_quantile,
        pca_var=args.pca_var,
        seed=args.seed,
    )

    min_samples = min_samples_required(len(y_train))

    best_box, search_table, best_stats = search_domain(
        X_comp=X_comp_train,
        dense_mask=dense["train_dense_mask"],
        success_mask=success_mask,
        ape=ape,
        min_reliability=args.min_reliability,
        min_samples=min_samples,
        q_max=args.q_max,
        q_step=args.q_step,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        max_candidates_per_side=args.max_candidates_per_side,
    )

    return {
        "box": best_box,
        "dense": dense,
        "search_table": search_table,
        "training_stats": best_stats,
        "training_oof_prediction": inner_pred,
        "training_ape": ape,
        "training_success_mask": success_mask,
        "min_samples": min_samples,
    }


def apply_domain_rule(
    X_comp_new: pd.DataFrame,
    box: tuple[tuple[float, ...], tuple[float, ...]] | None,
    dense_reference: dict[str, Any],
) -> dict[str, np.ndarray]:
    dense_mask, kth_distance = apply_dense_core(
        dense_reference=dense_reference,
        X_comp_new=X_comp_new,
    )

    if box is None:
        visible_mask = np.zeros(
            len(X_comp_new),
            dtype=bool,
        )
        final_mask = np.zeros(
            len(X_comp_new),
            dtype=bool,
        )
    else:
        visible_mask = inside_box(
            X_comp_new.to_numpy(dtype=float),
            box,
        )
        final_mask = visible_mask & dense_mask

    return {
        "inside_visible": visible_mask,
        "inside_dense": dense_mask,
        "inside_final": final_mask,
        "kth_distance": kth_distance,
    }


# -----------------------------------------------------------------------------
# Nested validation
# -----------------------------------------------------------------------------

def run_nested_validation(
    model: Any,
    X_model: pd.DataFrame,
    X_comp: pd.DataFrame,
    y: np.ndarray,
    ids: pd.Series,
    groups: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_groups = len(np.unique(groups))

    if n_groups < args.outer_cv:
        raise ValueError(
            f"Only {n_groups} exact composition groups are available for "
            f"outer_cv={args.outer_cv}."
        )

    outer_cv = GroupKFold(n_splits=args.outer_cv)

    fold_rows = []
    diagnostic_rows = []

    for fold, (train_idx, valid_idx) in enumerate(
        outer_cv.split(
            X_model,
            y,
            groups=groups,
        ),
        start=1,
    ):
        print("\n" + "=" * 100)
        print(f"NESTED DOMAIN VALIDATION — OUTER FOLD {fold}")
        print("=" * 100)

        train_groups = groups[train_idx]
        valid_groups = groups[valid_idx]

        overlap = set(train_groups.tolist()).intersection(
            set(valid_groups.tolist())
        )

        if overlap:
            raise RuntimeError(
                f"Exact-composition leakage in outer fold {fold}: "
                f"{len(overlap)} overlapping groups."
            )

        X_model_train = X_model.iloc[train_idx].reset_index(drop=True)
        X_model_valid = X_model.iloc[valid_idx].reset_index(drop=True)

        X_comp_train = X_comp.iloc[train_idx].reset_index(drop=True)
        X_comp_valid = X_comp.iloc[valid_idx].reset_index(drop=True)

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        groups_train_local = exact_composition_groups(
            X_comp_train
        )

        domain_fit = construct_domain(
            model=model,
            X_model_train=X_model_train,
            X_comp_train=X_comp_train,
            y_train=y_train,
            train_groups=groups_train_local,
            args=args,
        )

        fold_model = clone(model)
        patch_sklearn_compat(fold_model)

        fold_model.fit(
            X_model_train,
            y_train,
        )

        y_pred = np.asarray(
            fold_model.predict(
                X_model_valid
            )
        ).ravel()

        ape = np.abs(
            y_valid - y_pred
        ) / np.maximum(
            np.abs(y_valid),
            1e-12,
        )

        success = ape <= args.max_error

        membership = apply_domain_rule(
            X_comp_new=X_comp_valid,
            box=domain_fit["box"],
            dense_reference=domain_fit["dense"],
        )

        inside = membership["inside_final"]
        outside = ~inside

        inside_stats = error_summary(
            y_valid[inside],
            y_pred[inside],
            success[inside],
        )

        outside_stats = error_summary(
            y_valid[outside],
            y_pred[outside],
            success[outside],
        )

        all_stats = error_summary(
            y_valid,
            y_pred,
            success,
        )

        fold_rows.append({
            "Outer_Fold": fold,
            "Train_n": int(len(train_idx)),
            "Validation_n": int(len(valid_idx)),
            "Train_exact_composition_groups": int(
                len(np.unique(train_groups))
            ),
            "Validation_exact_composition_groups": int(
                len(np.unique(valid_groups))
            ),
            "Group_overlap": 0,
            "Domain_found_in_outer_training": bool(
                domain_fit["box"] is not None
            ),
            "Outer_validation_inside_domain_n": int(
                inside.sum()
            ),
            "Outer_validation_domain_coverage": float(
                np.mean(inside)
            ),
            "Outer_validation_inside_domain_reliability": (
                inside_stats["reliability"]
            ),
            "Outer_validation_inside_domain_RMSE": (
                inside_stats["RMSE"]
            ),
            "Outer_validation_inside_domain_MAE": (
                inside_stats["MAE"]
            ),
            "Outer_validation_inside_domain_R2": (
                inside_stats["R2"]
            ),
            "Outer_validation_outside_domain_n": int(
                outside.sum()
            ),
            "Outer_validation_outside_domain_reliability": (
                outside_stats["reliability"]
            ),
            "Outer_validation_outside_domain_RMSE": (
                outside_stats["RMSE"]
            ),
            "Outer_validation_outside_domain_MAE": (
                outside_stats["MAE"]
            ),
            "Outer_validation_outside_domain_R2": (
                outside_stats["R2"]
            ),
            "Outer_validation_overall_reliability": (
                all_stats["reliability"]
            ),
            "Outer_validation_overall_RMSE": all_stats["RMSE"],
            "Outer_validation_overall_MAE": all_stats["MAE"],
            "Outer_validation_overall_R2": all_stats["R2"],
            "Training_domain_min_samples_required": int(
                domain_fit["min_samples"]
            ),
            "Training_domain_internal_reliability": (
                domain_fit["training_stats"][
                    "reliability_inside_final_domain"
                ]
                if domain_fit["training_stats"] is not None
                else np.nan
            ),
            "Training_domain_internal_n": (
                domain_fit["training_stats"][
                    "n_inside_final_domain"
                ]
                if domain_fit["training_stats"] is not None
                else 0
            ),
            "kNN_distance_threshold": float(
                domain_fit["dense"]["distance_threshold"]
            ),
            "PCA_explained_variance_sum": float(
                domain_fit["dense"]["explained_variance_sum"]
            ),
        })

        for local_pos, dataset_idx in enumerate(valid_idx):
            diagnostic_rows.append({
                "Dataset_Row_Index": int(dataset_idx),
                "ID": ids.iloc[dataset_idx],
                "Outer_Fold": fold,
                "Exact_Composition_Group": int(
                    groups[dataset_idx]
                ),
                "Actual": float(y_valid[local_pos]),
                "Predicted": float(y_pred[local_pos]),
                "Residual": float(
                    y_valid[local_pos] - y_pred[local_pos]
                ),
                "APE": float(ape[local_pos]),
                "APE_percent": float(
                    ape[local_pos] * 100.0
                ),
                f"Error_le_{args.max_error * 100:.1f}pct": bool(
                    success[local_pos]
                ),
                "kNN_kth_external_distance": float(
                    membership["kth_distance"][local_pos]
                ),
                "inside_knn_dense_core": bool(
                    membership["inside_dense"][local_pos]
                ),
                "inside_visible_composition_ranges": bool(
                    membership["inside_visible"][local_pos]
                ),
                "inside_final_applicability_domain": bool(
                    inside[local_pos]
                ),
            })

        print(
            f"Outer validation: n={len(valid_idx)}, "
            f"inside domain={int(inside.sum())} "
            f"({np.mean(inside) * 100.0:.1f}% coverage)"
        )

        if inside.sum() > 0:
            print(
                "Nested reliability inside domain: "
                f"{inside_stats['reliability'] * 100.0:.2f}%"
            )
        else:
            print(
                "Nested reliability inside domain: not estimable "
                "(no validation rows inside fold-specific domain)"
            )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(diagnostic_rows),
    )


def summarize_nested_validation(
    diagnostics: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    actual = diagnostics["Actual"].to_numpy(dtype=float)
    predicted = diagnostics["Predicted"].to_numpy(dtype=float)

    success_col = f"Error_le_{args.max_error * 100:.1f}pct"
    success = diagnostics[
        success_col
    ].astype(bool).to_numpy()

    inside = diagnostics[
        "inside_final_applicability_domain"
    ].astype(bool).to_numpy()

    outside = ~inside

    inside_stats = error_summary(
        actual[inside],
        predicted[inside],
        success[inside],
    )

    outside_stats = error_summary(
        actual[outside],
        predicted[outside],
        success[outside],
    )

    all_stats = error_summary(
        actual,
        predicted,
        success,
    )

    reliability_gain = (
        float(
            inside_stats["reliability"]
            - all_stats["reliability"]
        )
        if inside_stats["n"] > 0
        else np.nan
    )

    return {
        "nested_validation_method": (
            "Exact-composition-grouped outer CV with exact-composition-grouped "
            "inner OOF domain construction"
        ),
        "outer_cv_folds": int(args.outer_cv),
        "inner_cv_folds": int(args.inner_cv),
        "total_nested_outer_validation_rows": int(len(diagnostics)),
        "nested_domain_coverage_fraction": float(
            np.mean(inside)
        ),
        "nested_domain_coverage_percent": float(
            np.mean(inside) * 100.0
        ),
        "nested_inside_domain_n": inside_stats["n"],
        "nested_inside_domain_reliability_fraction": (
            inside_stats["reliability"]
        ),
        "nested_inside_domain_reliability_percent": (
            float(inside_stats["reliability"] * 100.0)
            if inside_stats["n"] > 0
            else np.nan
        ),
        "nested_inside_domain_RMSE": inside_stats["RMSE"],
        "nested_inside_domain_MAE": inside_stats["MAE"],
        "nested_inside_domain_R2": inside_stats["R2"],
        "nested_outside_domain_n": outside_stats["n"],
        "nested_outside_domain_reliability_fraction": (
            outside_stats["reliability"]
        ),
        "nested_outside_domain_reliability_percent": (
            float(outside_stats["reliability"] * 100.0)
            if outside_stats["n"] > 0
            else np.nan
        ),
        "nested_outside_domain_RMSE": outside_stats["RMSE"],
        "nested_outside_domain_MAE": outside_stats["MAE"],
        "nested_outside_domain_R2": outside_stats["R2"],
        "nested_overall_reliability_fraction": (
            all_stats["reliability"]
        ),
        "nested_overall_reliability_percent": float(
            all_stats["reliability"] * 100.0
        ),
        "nested_overall_RMSE": all_stats["RMSE"],
        "nested_overall_MAE": all_stats["MAE"],
        "nested_overall_R2": all_stats["R2"],
        "nested_reliability_gain_inside_vs_overall_fraction": (
            reliability_gain
        ),
        "nested_reliability_gain_inside_vs_overall_percentage_points": (
            float(reliability_gain * 100.0)
            if np.isfinite(reliability_gain)
            else np.nan
        ),
    }


# -----------------------------------------------------------------------------
# Final operational refit
# -----------------------------------------------------------------------------

def fit_final_operational_domain(
    model: Any,
    X_model: pd.DataFrame,
    X_comp: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Refit the frozen model and applicability construction procedure on all rows.

    This final refit is for operational/dashboard use. Its internal domain
    reliability is descriptive; nested outer-validation reliability is the
    performance estimate to report.
    """
    full_domain = construct_domain(
        model=model,
        X_model_train=X_model,
        X_comp_train=X_comp,
        y_train=y,
        train_groups=groups,
        args=args,
    )

    full_model = clone(model)
    patch_sklearn_compat(full_model)
    full_model.fit(X_model, y)

    membership = apply_domain_rule(
        X_comp_new=X_comp,
        box=full_domain["box"],
        dense_reference=full_domain["dense"],
    )

    return {
        **full_domain,
        "full_model": full_model,
        "membership": membership,
    }


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def save_json(
    data: dict[str, Any],
    path: Path,
) -> None:
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(
            f"Object of type {type(obj).__name__} is not JSON serializable"
        )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            default=convert,
        )


def dashboard_header(
    nested_summary: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    inside_n = int(
        nested_summary["nested_inside_domain_n"]
    )

    if inside_n == 0:
        return (
            "Applicability rule was constructed using nested exact-composition-"
            "grouped validation, but no outer-validation observations fell inside "
            "the fold-specific domain. Prospective reliability could therefore "
            "not be estimated."
        )

    return (
        "Nested exact-composition-grouped validation estimated that "
        f"{nested_summary['nested_inside_domain_reliability_percent']:.1f}% "
        "of predictions classified inside the applicability domain had "
        f"absolute percentage error ≤{args.max_error * 100.0:.0f}%, "
        f"with {nested_summary['nested_domain_coverage_percent']:.1f}% "
        "validation coverage. The operational rule requires both the visible "
        "composition ranges and the kNN dense-composition criterion."
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.suppress_warnings:
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
        )

    if args.outer_cv < 2:
        raise ValueError("--outer_cv must be at least 2.")

    if args.inner_cv < 2:
        raise ValueError("--inner_cv must be at least 2.")

    if not 0.0 < args.max_error:
        raise ValueError("--max_error must be > 0.")

    if not 0.0 < args.min_reliability <= 1.0:
        raise ValueError(
            "--min_reliability must be in (0, 1]."
        )

    df = pd.read_csv(args.csv)
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    if df.columns.duplicated().any():
        duplicate_cols = (
            df.columns[
                df.columns.duplicated()
            ]
            .unique()
            .tolist()
        )

        print(
            f"Duplicate columns dropped: {duplicate_cols}"
        )

        df = df.loc[
            :,
            ~df.columns.duplicated(),
        ].copy()

    target = find_target(df)

    model = joblib.load(args.model)
    patch_sklearn_compat(model)

    features = model_features(
        model,
        df,
        target,
    )

    wt_cols = composition_columns(df)

    X_model = numeric_frame(
        df[features],
        absurd_thr=args.absurd_thr,
    )

    X_comp = numeric_frame(
        df[wt_cols],
        fill_missing_with_zero=True,
    )

    y = pd.to_numeric(
        df[target],
        errors="coerce",
    ).to_numpy(dtype=float)

    if any(name in df.columns for name in ID_NAMES):
        id_col = next(
            name
            for name in ID_NAMES
            if name in df.columns
        )
        ids = df[id_col].copy()
    else:
        ids = pd.Series(
            np.arange(1, len(df) + 1),
            name="ID",
        )

    invalid_composition = (
        (X_comp < 0.0)
        | (X_comp > 100.0)
    ).any(axis=1).to_numpy()

    missing_composition = X_comp.isna().any(
        axis=1
    ).to_numpy()

    valid_rows = (
        np.isfinite(y)
        & ~invalid_composition
        & ~missing_composition
    )

    if not valid_rows.all():
        print(
            "Dropping "
            f"{int((~valid_rows).sum())} rows with invalid target/composition."
        )

    df_valid = df.loc[
        valid_rows
    ].reset_index(drop=True)

    X_model = X_model.loc[
        valid_rows
    ].reset_index(drop=True)

    X_comp = X_comp.loc[
        valid_rows
    ].reset_index(drop=True)

    ids = ids.loc[
        valid_rows
    ].reset_index(drop=True)

    y = y[valid_rows]

    if len(y) < max(
        args.outer_cv,
        args.inner_cv,
    ):
        raise ValueError(
            f"Not enough valid rows ({len(y)}) for requested CV."
        )

    groups = exact_composition_groups(
        X_comp
    )

    n_groups = len(
        np.unique(groups)
    )

    print("\nApplicability-domain settings:")
    print(
        f"Valid rows: {len(y)}"
    )
    print(
        f"Exact composition groups: {n_groups}"
    )
    print(
        f"Model predictor count: {len(features)}"
    )
    print(
        f"Outer grouped CV folds: {args.outer_cv}"
    )
    print(
        f"Inner grouped CV folds: {args.inner_cv}"
    )
    print(
        f"Success criterion: absolute percentage error "
        f"≤ {args.max_error * 100.0:.1f}%"
    )
    print(
        f"Training-domain minimum reliability: "
        f"{args.min_reliability * 100.0:.1f}%"
    )
    print(
        "Exact-composition grouping: no rounding or near-duplicate merging"
    )
    print(
        "Model family/hyperparameters: frozen; no tuning or reselection"
    )

    # -------------------------------------------------------------------------
    # 1. Nested external evaluation of the applicability procedure
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("1. NESTED APPLICABILITY-DOMAIN VALIDATION")
    print("=" * 110)

    fold_metrics, nested_diagnostics = run_nested_validation(
        model=model,
        X_model=X_model,
        X_comp=X_comp,
        y=y,
        ids=ids,
        groups=groups,
        args=args,
    )

    nested_summary = summarize_nested_validation(
        diagnostics=nested_diagnostics,
        args=args,
    )

    fold_metrics_path = (
        args.out_dir
        / "nested_domain_fold_metrics.csv"
    )

    diagnostics_path = (
        args.out_dir
        / "nested_domain_oof_diagnostics.csv"
    )

    fold_metrics.to_csv(
        fold_metrics_path,
        index=False,
    )

    nested_diagnostics.to_csv(
        diagnostics_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 2. Final operational full-data refit
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("2. FINAL OPERATIONAL DOMAIN REFIT")
    print("=" * 110)

    final_domain = fit_final_operational_domain(
        model=model,
        X_model=X_model,
        X_comp=X_comp,
        y=y,
        groups=groups,
        args=args,
    )

    final_model_path = (
        args.out_dir
        / "final_xgboost_full_data_pipeline.joblib"
    )

    joblib.dump(
        final_domain["full_model"],
        final_model_path,
    )

    full_search_path = (
        args.out_dir
        / "domain_search_table_full_fit.csv"
    )

    final_domain["search_table"].to_csv(
        full_search_path,
        index=False,
    )

    final_box = final_domain["box"]
    final_stats = final_domain["training_stats"]

    full_ape = final_domain[
        "training_ape"
    ]

    full_success = final_domain[
        "training_success_mask"
    ]

    full_membership = final_domain[
        "membership"
    ]

    full_diagnostics = df_valid.copy()
    full_diagnostics[
        "Grouped_OOF_prediction_for_domain_construction"
    ] = final_domain[
        "training_oof_prediction"
    ]
    full_diagnostics[
        "Grouped_OOF_APE"
    ] = full_ape
    full_diagnostics[
        "Grouped_OOF_APE_pct"
    ] = full_ape * 100.0
    full_diagnostics[
        f"Grouped_OOF_error_le_{args.max_error * 100.0:.1f}pct"
    ] = full_success
    full_diagnostics[
        "kNN_kth_external_distance"
    ] = full_membership[
        "kth_distance"
    ]
    full_diagnostics[
        "inside_knn_dense_core"
    ] = full_membership[
        "inside_dense"
    ]
    full_diagnostics[
        "inside_visible_composition_ranges"
    ] = full_membership[
        "inside_visible"
    ]
    full_diagnostics[
        "inside_final_applicability_domain"
    ] = full_membership[
        "inside_final"
    ]

    full_diagnostics_path = (
        args.out_dir
        / "full_fit_domain_diagnostics.csv"
    )

    full_diagnostics.to_csv(
        full_diagnostics_path,
        index=False,
    )

    header = dashboard_header(
        nested_summary=nested_summary,
        args=args,
    )

    (
        args.out_dir
        / "dashboard_header.txt"
    ).write_text(
        header,
        encoding="utf-8",
    )

    summary_base = {
        "analysis": (
            "Nested applicability-domain evaluation for frozen final XGBoost"
        ),
        "target_column": target,
        "feature_count_used_by_model": int(
            len(features)
        ),
        "composition_columns": wt_cols,
        "total_valid_rows": int(
            len(y)
        ),
        "exact_composition_groups": int(
            n_groups
        ),
        "max_error_threshold_fraction": float(
            args.max_error
        ),
        "max_error_threshold_percent": float(
            args.max_error * 100.0
        ),
        "min_reliability_required_fraction": float(
            args.min_reliability
        ),
        "min_reliability_required_percent": float(
            args.min_reliability * 100.0
        ),
        "knn_k": int(
            args.knn_k
        ),
        "knn_quantile": float(
            args.knn_quantile
        ),
        "pca_variance_target": float(
            args.pca_var
        ),
        "final_trusted_condition": (
            "inside_visible_composition_ranges "
            "AND inside_knn_dense_core"
        ),
        "model_selection_status": (
            "Frozen before applicability-domain analysis"
        ),
        "conventional_holdout_status": (
            "Not reused for model selection; previously reported holdout "
            "metrics remain the final conventional test result"
        ),
        **nested_summary,
    }

    if final_box is None or final_stats is None:
        summary = {
            **summary_base,
            "final_operational_domain_valid": False,
            "message": (
                "Nested validation completed, but no full-data operational "
                "domain satisfying the pre-specified construction criteria "
                "was found."
            ),
            "dashboard_header": header,
        }

        save_json(
            summary,
            args.out_dir / "domain_summary.json",
        )

        print("\n=== Nested applicability-domain result ===")
        print(header)
        print(
            "\nNo full-data operational domain met the "
            "pre-specified construction criteria."
        )
        print(
            "Do not relax thresholds only to force a domain after viewing "
            "this result."
        )

        print("\nSaved:")
        for path in [
            fold_metrics_path,
            diagnostics_path,
            full_search_path,
            full_diagnostics_path,
            final_model_path,
            args.out_dir / "domain_summary.json",
            args.out_dir / "dashboard_header.txt",
        ]:
            print(f"  {path}")

        return

    lo = pd.Series(
        final_box[0],
        index=wt_cols,
    )

    hi = pd.Series(
        final_box[1],
        index=wt_cols,
    )

    ranges = pd.DataFrame({
        "Element": wt_cols,
        "Min_wt_pct": [
            float(lo[column])
            for column in wt_cols
        ],
        "Max_wt_pct": [
            float(hi[column])
            for column in wt_cols
        ],
    })

    ranges_path = (
        args.out_dir
        / "validated_composition_ranges_visible.csv"
    )

    ranges.to_csv(
        ranges_path,
        index=False,
    )

    domain_reference = {
        "wt_cols": wt_cols,
        "lo": lo.to_dict(),
        "hi": hi.to_dict(),
        "knn_ref": final_domain["dense"],
        "max_error": float(
            args.max_error
        ),
        "min_reliability": float(
            args.min_reliability
        ),
        "validation_method": (
            "Nested exact-composition-grouped validation"
        ),
        "nested_validation_summary": nested_summary,
        "final_trusted_condition": (
            "inside_visible_composition_ranges "
            "AND inside_knn_dense_core"
        ),
    }

    reference_path = (
        args.out_dir
        / "composition_domain_reference.joblib"
    )

    joblib.dump(
        domain_reference,
        reference_path,
    )

    summary = {
        **summary_base,
        "final_operational_domain_valid": True,
        "full_fit_internal_domain_n": int(
            final_stats[
                "n_inside_final_domain"
            ]
        ),
        "full_fit_internal_domain_reliability_fraction": float(
            final_stats[
                "reliability_inside_final_domain"
            ]
        ),
        "full_fit_internal_domain_reliability_percent": float(
            final_stats[
                "reliability_inside_final_domain"
            ] * 100.0
        ),
        "full_fit_internal_domain_mean_APE_percent": float(
            final_stats[
                "mean_APE_pct_inside_final_domain"
            ]
        ),
        "full_fit_knn_distance_threshold": float(
            final_domain["dense"][
                "distance_threshold"
            ]
        ),
        "full_fit_pca_explained_variance_sum": float(
            final_domain["dense"][
                "explained_variance_sum"
            ]
        ),
        "dashboard_header": header,
        "interpretation_note": (
            "The nested outer-validation estimates are the reliability and "
            "coverage results to report. Full-fit internal reliability is "
            "descriptive because the final ranges were constructed using "
            "full-data grouped OOF errors."
        ),
        "message": (
            "Nested validation completed and a final operational "
            "dashboard-ready applicability rule was fitted."
        ),
    }

    save_json(
        summary,
        args.out_dir / "domain_summary.json",
    )

    print("\n" + "=" * 110)
    print("NESTED APPLICABILITY-DOMAIN SUMMARY")
    print("=" * 110)

    print(
        f"Nested outer-validation coverage: "
        f"{nested_summary['nested_domain_coverage_percent']:.2f}% "
        f"({nested_summary['nested_inside_domain_n']} / "
        f"{nested_summary['total_nested_outer_validation_rows']})"
    )

    if nested_summary["nested_inside_domain_n"] > 0:
        print(
            f"Nested reliability inside domain: "
            f"{nested_summary['nested_inside_domain_reliability_percent']:.2f}%"
        )
        print(
            f"Nested RMSE inside domain: "
            f"{nested_summary['nested_inside_domain_RMSE']:.4f} mJ/m²"
        )
        print(
            f"Nested MAE inside domain: "
            f"{nested_summary['nested_inside_domain_MAE']:.4f} mJ/m²"
        )

    print(
        f"Nested overall reliability: "
        f"{nested_summary['nested_overall_reliability_percent']:.2f}%"
    )

    print(
        f"Nested overall RMSE: "
        f"{nested_summary['nested_overall_RMSE']:.4f} mJ/m²"
    )

    print(
        f"Nested overall MAE: "
        f"{nested_summary['nested_overall_MAE']:.4f} mJ/m²"
    )

    print("\nDashboard wording:")
    print(header)

    print("\nFinal operational visible composition ranges:")
    for column in wt_cols:
        print(
            f"  {column:<18}: "
            f"{lo[column]:.5g} – {hi[column]:.5g} wt.%"
        )

    print("\nFinal trusted condition:")
    print(
        "  inside visible composition ranges AND inside kNN dense core"
    )

    print("\nImportant:")
    print(
        "  Report nested outer-validation reliability/coverage, not the "
        "full-fit internal reliability, as the evidence for domain usefulness."
    )
    print(
        "  The previously obtained reserved-holdout result remains the "
        "final conventional test result."
    )

    print("\nSaved:")
    for path in [
        fold_metrics_path,
        diagnostics_path,
        ranges_path,
        reference_path,
        full_search_path,
        full_diagnostics_path,
        final_model_path,
        args.out_dir / "domain_summary.json",
        args.out_dir / "dashboard_header.txt",
    ]:
        print(f"  {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Nested exact-composition-grouped applicability-domain validation "
            "for the frozen final SFE regression pipeline."
        )
    )

    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help=(
            "Previously frozen final XGBoost development pipeline. "
            "The fitted state is not used for nested training folds; clones "
            "are refitted using only the appropriate training partition."
        ),
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(
            "domain_applicability_nested"
        ),
    )

    parser.add_argument(
        "--max_error",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--min_reliability",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--knn_k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--knn_quantile",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--pca_var",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--q_max",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--q_step",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--beam_width",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=35,
    )

    parser.add_argument(
        "--max_candidates_per_side",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--outer_cv",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--inner_cv",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--absurd_thr",
        type=float,
        default=1e6,
    )

    parser.add_argument(
        "--suppress_warnings",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
