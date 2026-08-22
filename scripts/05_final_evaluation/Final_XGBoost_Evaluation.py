#!/usr/bin/env python3
# Final_XGBoost_Evaluation.py
"""
Final evaluation of the frozen XGBoost SFE regression model.

This script is intentionally separate from candidate-model screening, Optuna tuning,
and repeated-CV model selection. The XGBoost model family and hyperparameters are
already frozen before this script is run.

Three evaluations are performed:

1. Final untouched holdout evaluation:
   - Reuses the exact development_split_manifest.csv created before model selection.
   - Fits the frozen XGBoost pipeline on the complete 80% development partition.
   - Scores the reserved 20% holdout exactly once.

2. Row-wise five-fold CV on the full dataset:
   - Conventional post-selection validation benchmark.
   - The frozen model and hyperparameters are not changed.

3. Exact-composition-grouped five-fold CV on the full dataset:
   - All records sharing the exact same nominal 16-element composition remain in
     the same fold.
   - This is a stricter post-selection sensitivity analysis for unseen nominal
     compositions.
   - It is not external validation and it is not DOI-grouped validation.

Important:
    - Do not change the model family or hyperparameters after viewing these results.
    - The holdout result is the final conventional test result.
    - Row-wise and grouped CV are post-selection validation analyses.
    - Exact composition groups are created without rounding or near-duplicate merging.

Input:
    CSV dataset containing composition, descriptor, thermodynamic-model output,
    and experimental SFE columns.

Output:
    Folder containing:
        - final_holdout_metrics.csv
        - final_holdout_predictions.csv
        - rowwise_oof_predictions.csv
        - composition_grouped_oof_predictions.csv
        - validation_fold_metrics.csv
        - final_evaluation_summary.csv
        - composition_group_sizes.csv
        - frozen_xgboost_hyperparameters.json
        - final_xgboost_development_pipeline.joblib
        - optional prediction and residual plots

Run:
    python Final_XGBoost_Evaluation.py \
        --csv "data/output_with_8_sfe_models.csv" \
        --output "model_results/final_xgboost_evaluation" \
        --split-manifest "model_results/development_split_manifest.csv"

Optional:
    python Final_XGBoost_Evaluation.py \
        --csv "data/output_with_8_sfe_models.csv" \
        --output "model_results/final_xgboost_evaluation" \
        --split-manifest "model_results/development_split_manifest.csv" \
        --folds 5
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


# =============================================================================
# USER SETTINGS
# =============================================================================

DATASET_PATH = r"output.csv"
TARGET_COLUMN = "SFE_Experimental (mJ/m²)"
ID_COLUMN = "No."

OUTPUT_DIR = "final_xgboost_evaluation"

RANDOM_SEED = 42
N_SPLITS = 5
SAVE_PLOTS = True

COMPOSITION_COLUMNS = [
    "Fe (%wt)", "Mn (%wt)", "Al (%wt)", "C (%wt)", "Ni (%wt)",
    "Cr (%wt)", "Si (%wt)", "Cu (%wt)", "Mo (%wt)", "Nb (%wt)",
    "V (%wt)", "Ti (%wt)", "Co (%wt)", "N (%wt)", "P (%wt)", "S (%wt)",
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_numeric_series(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to numeric.

    Handles examples such as:
    - "26*" -> 26
    - "1,234.5" -> 1234.5
    - non-numeric values -> NaN
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace(r"[^0-9eE+\-\.]", "", regex=True)
    )

    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def get_columns_v_to_ac(df: pd.DataFrame) -> List[str]:
    """
    Return Excel-style columns V through AC by position.

    A = index 0
    V = index 21
    AC = index 28

    Therefore V:AC = df.columns[21:29].
    """
    if df.shape[1] < 29:
        warnings.warn(
            "Dataset has fewer than 29 columns. Cannot identify V–AC columns by position."
        )
        return []

    return list(df.columns[21:29])


def rmse_score(y_true, y_pred) -> float:
    """Version-safe RMSE calculation."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pearson_r(y_true, y_pred) -> float:
    """Calculate Pearson correlation safely."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return np.nan

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculate regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if np.any(np.isclose(y_true, 0.0)):
        mape = np.nan
    else:
        mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)

    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse_score(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE_percent": mape,
        "Pearson_r": pearson_r(y_true, y_pred),
    }


def convert_for_json(obj: Any):
    """Convert numpy objects to JSON-serializable Python objects."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Load CSV dataset."""
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    print(f"Loaded dataset: {path}")
    print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str,
    id_column: Optional[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Prepare predictors, target, identifiers, and composition table.

    Steps:
    1. Validate required composition and target columns.
    2. Clean target.
    3. Drop rows with missing target values.
    4. Exclude identifier column from predictors.
    5. Convert predictors to numeric.
    6. Replace absurd thermodynamic values in columns V–AC with NaN.
    """
    missing_composition = [
        col for col in COMPOSITION_COLUMNS if col not in df.columns
    ]
    if missing_composition:
        raise ValueError(
            f"Missing required composition columns: {missing_composition}"
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    df = df.copy()

    if id_column is not None and id_column in df.columns:
        ids = df[id_column].copy()
    else:
        ids = pd.Series(np.arange(1, len(df) + 1), name="ID")

    y = clean_numeric_series(df[target_column])

    missing_target = int(y.isna().sum())
    if missing_target > 0:
        print(f"Dropping {missing_target} rows with missing/non-numeric target values.")

    valid_mask = y.notna()

    df = df.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)
    ids = ids.loc[valid_mask].reset_index(drop=True)

    drop_cols = [target_column]
    if id_column is not None and id_column in df.columns:
        drop_cols.append(id_column)

    X = df.drop(columns=drop_cols).copy()

    for col in X.columns:
        X[col] = clean_numeric_series(X[col])

    thermo_cols = get_columns_v_to_ac(df)
    thermo_cols = [col for col in thermo_cols if col in X.columns]

    print("Cleaning thermodynamic predictor columns V–AC:")
    for col in thermo_cols:
        absurd_mask = X[col].abs() > 1e6
        n_absurd = int(absurd_mask.sum())

        if n_absurd > 0:
            print(f"  {col}: replacing {n_absurd} absurd values with NaN")
            X.loc[absurd_mask, col] = np.nan

    composition = df[COMPOSITION_COLUMNS].apply(
        pd.to_numeric,
        errors="raise",
    )

    print(f"Predictor matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    return X, y, ids, composition


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build median-imputation preprocessor.

    The imputer is inside the Pipeline, so it is fitted only on training data
    during the holdout fit and separately inside each cross-validation fold.
    """
    numeric_features = list(X.columns)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("median_imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# =============================================================================
# FROZEN XGBOOST MODEL
# =============================================================================

def get_frozen_xgboost_hyperparameters() -> Dict[str, Any]:
    """
    Return the frozen XGBoost hyperparameters selected before final evaluation.

    These are the best Optuna XGBoost hyperparameters carried into the repeated-CV
    comparison. XGBoost was subsequently selected by the pre-specified repeated-CV
    rule: lowest mean RMSE, then lowest mean MAE, with R² as supporting evidence.
    """
    return {
        "objective": "reg:squarederror",
        "n_estimators": 500,
        "learning_rate": 0.011411002001291685,
        "max_depth": 8,
        "min_child_weight": 2.665840052140506,
        "subsample": 0.78310286906201,
        "colsample_bytree": 0.837235999618537,
        "gamma": 0.8987126718179128,
        "reg_alpha": 6.425544478395007e-08,
        "reg_lambda": 0.029156850643502742,
        "random_state": 42,
        "n_jobs": 1,
        "tree_method": "hist",
        "verbosity": 0,
    }


def build_frozen_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build preprocessing + frozen XGBoost pipeline."""
    preprocessor = build_preprocessor(X)

    model = XGBRegressor(
        **get_frozen_xgboost_hyperparameters()
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# =============================================================================
# FIXED DEVELOPMENT / HOLDOUT SPLIT
# =============================================================================

def load_fixed_split_manifest(
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    manifest_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and validate the exact development/holdout split created before model selection.

    The final holdout must be the same observations that were reserved throughout
    candidate screening, Optuna tuning, and repeated-CV model selection.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Development split manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(manifest_path)

    required_columns = {"Row_Index", "Split"}
    missing = required_columns.difference(manifest.columns)
    if missing:
        raise ValueError(
            f"Split manifest missing required columns: {sorted(missing)}"
        )

    if len(manifest) != len(y):
        raise ValueError(
            f"Split manifest contains {len(manifest)} rows but the current dataset "
            f"contains {len(y)} rows."
        )

    manifest = manifest.sort_values("Row_Index").reset_index(drop=True)

    expected_indices = np.arange(len(y))
    if not np.array_equal(
        manifest["Row_Index"].to_numpy(),
        expected_indices,
    ):
        raise ValueError(
            "Split manifest Row_Index must match the current dataset order exactly."
        )

    labels = set(manifest["Split"].astype(str))
    if labels != {"train", "holdout"}:
        raise ValueError(
            "Split manifest must contain exactly the labels 'train' and 'holdout'."
        )

    if "ID" in manifest.columns:
        manifest_ids = manifest["ID"].astype(str).to_numpy()
        current_ids = ids.astype(str).to_numpy()

        if not np.array_equal(manifest_ids, current_ids):
            raise ValueError(
                "Split manifest ID values do not match the current dataset order."
            )

    idx_train = manifest.index[
        manifest["Split"].astype(str) == "train"
    ].to_numpy()
    idx_holdout = manifest.index[
        manifest["Split"].astype(str) == "holdout"
    ].to_numpy()

    if len(idx_train) == 0 or len(idx_holdout) == 0:
        raise ValueError(
            "Split manifest must contain non-empty train and holdout partitions."
        )

    return idx_train, idx_holdout


# =============================================================================
# FINAL UNTOUCHED HOLDOUT EVALUATION
# =============================================================================

def evaluate_final_holdout(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    idx_train: np.ndarray,
    idx_holdout: np.ndarray,
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, float]]:
    """
    Fit on the full development partition and score the reserved holdout once.
    """
    final_pipeline = clone(pipeline)

    X_train = X.iloc[idx_train]
    y_train = y.iloc[idx_train]

    X_holdout = X.iloc[idx_holdout]
    y_holdout = y.iloc[idx_holdout]

    final_pipeline.fit(X_train, y_train)
    holdout_pred = final_pipeline.predict(X_holdout)

    metrics = calculate_metrics(y_holdout, holdout_pred)

    prediction_df = pd.DataFrame(
        {
            "Dataset_Row_Index": idx_holdout,
            "ID": ids.iloc[idx_holdout].values,
            "Actual": y_holdout.values,
            "Predicted": holdout_pred,
            "Residual": y_holdout.values - holdout_pred,
        }
    )

    metrics_row = {
        "Evaluation": "Final untouched 20% holdout",
        "Train_n": len(idx_train),
        "Evaluation_n": len(idx_holdout),
        **metrics,
    }

    return final_pipeline, prediction_df, metrics_row


# =============================================================================
# ROW-WISE AND COMPOSITION-GROUPED POST-SELECTION VALIDATION
# =============================================================================

def make_composition_groups(composition: pd.DataFrame) -> np.ndarray:
    """
    Create exact nominal-composition groups using all 16 composition columns.

    No rounding or near-duplicate merging is performed.
    """
    composition_index = pd.MultiIndex.from_frame(composition)
    groups, _ = pd.factorize(composition_index, sort=False)
    return groups


def run_oof_validation(
    scheme: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    splits: Iterable[Tuple[np.ndarray, np.ndarray]],
    groups: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Run one complete out-of-fold validation scheme with the frozen pipeline.
    """
    oof_predictions = np.full(len(y), np.nan, dtype=float)
    fold_number = np.full(len(y), -1, dtype=int)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = fold_pipeline.predict(X.iloc[valid_idx])

        oof_predictions[valid_idx] = predictions
        fold_number[valid_idx] = fold

        metrics = calculate_metrics(
            y.iloc[valid_idx],
            predictions,
        )

        row = {
            "Scheme": scheme,
            "Fold": fold,
            "Train_n": len(train_idx),
            "Validation_n": len(valid_idx),
            **metrics,
        }

        if groups is not None:
            train_groups = set(groups[train_idx].tolist())
            valid_groups = set(groups[valid_idx].tolist())
            overlap = train_groups.intersection(valid_groups)

            if overlap:
                raise RuntimeError(
                    f"Composition-group leakage detected in fold {fold}: "
                    f"{len(overlap)} overlapping groups."
                )

            row["Train_groups"] = len(train_groups)
            row["Validation_groups"] = len(valid_groups)
            row["Group_overlap"] = 0

        fold_rows.append(row)

    if np.isnan(oof_predictions).any():
        raise RuntimeError(
            f"Incomplete OOF predictions detected for {scheme}."
        )

    if (fold_number < 1).any():
        raise RuntimeError(
            f"Incomplete fold assignments detected for {scheme}."
        )

    aggregate_metrics = calculate_metrics(
        y,
        oof_predictions,
    )

    aggregate_row = {
        "Evaluation": scheme,
        "Train_n": np.nan,
        "Evaluation_n": len(y),
        **aggregate_metrics,
    }

    prediction_df = pd.DataFrame(
        {
            "ID": ids.values,
            "Fold": fold_number,
            "Actual": y.values,
            "Predicted": oof_predictions,
            "Residual": y.values - oof_predictions,
        }
    )

    if groups is not None:
        prediction_df.insert(
            2,
            "Composition_Group",
            groups,
        )

    return prediction_df, pd.DataFrame(fold_rows), aggregate_row


def summarize_cv_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate fold mean and standard deviation for each CV validation scheme."""
    metric_columns = [
        "R2",
        "RMSE",
        "MAE",
        "MAPE_percent",
        "Pearson_r",
    ]

    rows = []

    for scheme, part in fold_df.groupby("Scheme", sort=False):
        row = {"Evaluation": scheme}

        for metric in metric_columns:
            row[f"{metric}_fold_mean"] = float(part[metric].mean())
            row[f"{metric}_fold_std"] = float(part[metric].std(ddof=1))

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def save_frozen_hyperparameters(output_dir: Path) -> Path:
    """Save the exact frozen XGBoost hyperparameters used for final evaluation."""
    path = output_dir / "frozen_xgboost_hyperparameters.json"

    payload = {
        "model": "XGBoost",
        "random_seed": RANDOM_SEED,
        "selection_status": "Frozen before final evaluation",
        "selection_rule": {
            "primary": "lowest mean repeated-CV RMSE",
            "secondary": "lowest mean repeated-CV MAE",
            "supporting": "mean repeated-CV R2 and variability",
        },
        "hyperparameters": get_frozen_xgboost_hyperparameters(),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=4,
            default=convert_for_json,
        )

    return path


def save_prediction_plots(
    prediction_df: pd.DataFrame,
    title_suffix: str,
    filename_prefix: str,
    output_dir: Path,
) -> None:
    """Save predicted-vs-actual and residual plots for one evaluation result."""
    if not SAVE_PLOTS:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn(
            "matplotlib is not installed. Skipping final-evaluation plots."
        )
        return

    actual = prediction_df["Actual"].to_numpy(dtype=float)
    predicted = prediction_df["Predicted"].to_numpy(dtype=float)
    residual = prediction_df["Residual"].to_numpy(dtype=float)

    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, alpha=0.75)

    min_value = min(np.min(actual), np.min(predicted))
    max_value = max(np.max(actual), np.max(predicted))
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
    )

    plt.xlabel("Actual SFE (mJ/m²)")
    plt.ylabel("Predicted SFE (mJ/m²)")
    plt.title(f"Predicted vs Actual: {title_suffix}")
    plt.tight_layout()

    predicted_path = output_dir / f"{filename_prefix}_predicted_vs_actual.png"
    plt.savefig(predicted_path, dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(predicted, residual, alpha=0.75)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted SFE (mJ/m²)")
    plt.ylabel("Residual: Actual - Predicted (mJ/m²)")
    plt.title(f"Residual Plot: {title_suffix}")
    plt.tight_layout()

    residual_path = output_dir / f"{filename_prefix}_residual_plot.png"
    plt.savefig(residual_path, dpi=300)
    plt.close()

    print(f"Saved plots for: {title_suffix}")


# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final holdout, row-wise CV, and exact-composition-grouped CV "
            "evaluation of the frozen XGBoost SFE model."
        )
    )

    parser.add_argument(
        "--csv",
        "--data",
        dest="data",
        type=Path,
        default=Path(DATASET_PATH),
        help="Input modelling CSV dataset.",
    )

    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path(OUTPUT_DIR),
        help="Output directory for final evaluation results.",
    )

    parser.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
        help=(
            "The development_split_manifest.csv created before model selection. "
            "This is required so the exact untouched holdout is reused."
        ),
    )

    parser.add_argument(
        "--target",
        type=str,
        default=TARGET_COLUMN,
        help="Target column name.",
    )

    parser.add_argument(
        "--id-column",
        type=str,
        default=ID_COLUMN,
        help="Identifier column name.",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=N_SPLITS,
        help="Number of folds for row-wise and composition-grouped CV.",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    np.random.seed(RANDOM_SEED)
    warnings.filterwarnings("default")

    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")

    df = load_dataset(args.data)

    X, y, ids, composition = prepare_features_and_target(
        df=df,
        target_column=args.target,
        id_column=args.id_column,
    )

    idx_train, idx_holdout = load_fixed_split_manifest(
        X=X,
        y=y,
        ids=ids,
        manifest_path=args.split_manifest,
    )

    print("\nFinal evaluation settings:")
    print(f"Development samples: {len(idx_train)}")
    print(f"Reserved holdout samples: {len(idx_holdout)}")
    print(f"Row-wise CV folds: {args.folds}")
    print(f"Composition-grouped CV folds: {args.folds}")
    print("Model family: XGBoost")
    print("Hyperparameters: frozen")
    print("No model selection or hyperparameter tuning is performed in this script.")

    pipeline = build_frozen_pipeline(X)

    hyperparameter_path = save_frozen_hyperparameters(
        args.output_dir
    )
    print(f"Saved frozen hyperparameters: {hyperparameter_path}")

    # -------------------------------------------------------------------------
    # 1. Final untouched holdout evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("1. FINAL UNTOUCHED HOLDOUT EVALUATION")
    print("=" * 100)

    final_pipeline, holdout_predictions, holdout_metrics = evaluate_final_holdout(
        pipeline=pipeline,
        X=X,
        y=y,
        ids=ids,
        idx_train=idx_train,
        idx_holdout=idx_holdout,
    )

    holdout_metrics_df = pd.DataFrame([holdout_metrics])

    holdout_metrics_path = args.output_dir / "final_holdout_metrics.csv"
    holdout_predictions_path = args.output_dir / "final_holdout_predictions.csv"
    final_pipeline_path = args.output_dir / "final_xgboost_development_pipeline.joblib"

    holdout_metrics_df.to_csv(
        holdout_metrics_path,
        index=False,
    )
    holdout_predictions.to_csv(
        holdout_predictions_path,
        index=False,
    )
    joblib.dump(
        final_pipeline,
        final_pipeline_path,
    )

    print(
        holdout_metrics_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    # -------------------------------------------------------------------------
    # 2. Row-wise post-selection five-fold CV on all records
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("2. ROW-WISE POST-SELECTION CROSS-VALIDATION")
    print("=" * 100)

    rowwise_cv = KFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    rowwise_predictions, rowwise_folds, rowwise_aggregate = run_oof_validation(
        scheme=f"Row-wise {args.folds}-fold CV",
        pipeline=pipeline,
        X=X,
        y=y,
        ids=ids,
        splits=rowwise_cv.split(X, y),
    )

    # -------------------------------------------------------------------------
    # 3. Exact-composition-grouped post-selection CV on all records
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("3. EXACT-COMPOSITION-GROUPED POST-SELECTION CROSS-VALIDATION")
    print("=" * 100)

    groups = make_composition_groups(composition)
    n_groups = len(np.unique(groups))

    if n_groups < args.folds:
        raise ValueError(
            f"Only {n_groups} exact composition groups are available, "
            f"which is fewer than the requested {args.folds} folds."
        )

    grouped_cv = GroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    grouped_predictions, grouped_folds, grouped_aggregate = run_oof_validation(
        scheme=f"Exact-composition-grouped {args.folds}-fold CV",
        pipeline=pipeline,
        X=X,
        y=y,
        ids=ids,
        splits=grouped_cv.split(
            X,
            y,
            groups=groups,
        ),
        groups=groups,
    )

    # -------------------------------------------------------------------------
    # Save post-selection CV results
    # -------------------------------------------------------------------------
    validation_fold_metrics = pd.concat(
        [rowwise_folds, grouped_folds],
        ignore_index=True,
    )

    cv_fold_summary = summarize_cv_folds(
        validation_fold_metrics
    )

    evaluation_summary = pd.DataFrame(
        [
            holdout_metrics,
            rowwise_aggregate,
            grouped_aggregate,
        ]
    )

    evaluation_summary = evaluation_summary.merge(
        cv_fold_summary,
        on="Evaluation",
        how="left",
    )

    rowwise_predictions_path = args.output_dir / "rowwise_oof_predictions.csv"
    grouped_predictions_path = args.output_dir / "composition_grouped_oof_predictions.csv"
    fold_metrics_path = args.output_dir / "validation_fold_metrics.csv"
    summary_path = args.output_dir / "final_evaluation_summary.csv"
    group_sizes_path = args.output_dir / "composition_group_sizes.csv"

    rowwise_predictions.to_csv(
        rowwise_predictions_path,
        index=False,
    )
    grouped_predictions.to_csv(
        grouped_predictions_path,
        index=False,
    )
    validation_fold_metrics.to_csv(
        fold_metrics_path,
        index=False,
    )
    evaluation_summary.to_csv(
        summary_path,
        index=False,
    )

    group_sizes = pd.Series(groups).value_counts().sort_index()
    pd.DataFrame(
        {
            "Composition_Group": group_sizes.index,
            "Records": group_sizes.values,
        }
    ).to_csv(
        group_sizes_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Optional plots
    # -------------------------------------------------------------------------
    save_prediction_plots(
        prediction_df=holdout_predictions,
        title_suffix="Final Holdout",
        filename_prefix="final_holdout",
        output_dir=args.output_dir,
    )

    save_prediction_plots(
        prediction_df=rowwise_predictions,
        title_suffix=f"Row-wise {args.folds}-fold OOF",
        filename_prefix="rowwise_oof",
        output_dir=args.output_dir,
    )

    save_prediction_plots(
        prediction_df=grouped_predictions,
        title_suffix=f"Exact-composition-grouped {args.folds}-fold OOF",
        filename_prefix="composition_grouped_oof",
        output_dir=args.output_dir,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("FINAL XGBOOST EVALUATION SUMMARY")
    print("=" * 110)

    display_columns = [
        "Evaluation",
        "Evaluation_n",
        "R2",
        "RMSE",
        "MAE",
        "MAPE_percent",
        "Pearson_r",
        "R2_fold_mean",
        "R2_fold_std",
        "RMSE_fold_mean",
        "RMSE_fold_std",
        "MAE_fold_mean",
        "MAE_fold_std",
    ]

    available_columns = [
        col for col in display_columns if col in evaluation_summary.columns
    ]

    print(
        evaluation_summary[available_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print(f"\nExact composition groups: {n_groups}")
    print(
        "Composition-group leakage check: zero overlapping exact-composition "
        "groups in every grouped fold."
    )

    print("\nInterpretation:")
    print("  - Final holdout = final conventional test result.")
    print("  - Row-wise CV = post-selection conventional stability benchmark.")
    print(
        "  - Composition-grouped CV = post-selection sensitivity to unseen "
        "nominal compositions."
    )
    print(
        "  - Do not change the model family or hyperparameters after viewing "
        "these results."
    )

    print(f"\nSaved final holdout metrics: {holdout_metrics_path}")
    print(f"Saved final holdout predictions: {holdout_predictions_path}")
    print(f"Saved row-wise OOF predictions: {rowwise_predictions_path}")
    print(f"Saved grouped OOF predictions: {grouped_predictions_path}")
    print(f"Saved validation fold metrics: {fold_metrics_path}")
    print(f"Saved final evaluation summary: {summary_path}")
    print(f"Saved composition group sizes: {group_sizes_path}")
    print(f"Saved fitted development pipeline: {final_pipeline_path}")
    print(f"All outputs saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
