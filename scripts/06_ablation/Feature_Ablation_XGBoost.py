#!/usr/bin/env python3
# Feature_Ablation_XGBoost.py
"""
Controlled feature-group ablation for the frozen XGBoost SFE regression model.

Purpose:
    Quantify the incremental predictive value of the composition-derived
    elemental-property descriptors and the eight published SFE-model outputs.

Four feature sets are compared:

1. Composition only
2. Composition + descriptors
3. Composition + SFE-model outputs
4. Full feature set: composition + descriptors + SFE-model outputs

Important:
    - The XGBoost model family and hyperparameters are already frozen.
    - The reserved 20% holdout is NOT scored or used in this script.
    - All ablation comparisons are performed only on the fixed 80% development
      partition defined by development_split_manifest.csv.
    - The same repeated-CV splits are reused for every feature set.
    - Primary comparison metric: RMSE.
    - Secondary comparison metric: MAE.
    - R² is reported as supporting evidence.
    - The fixed XGBoost hyperparameters are kept identical across feature sets so
      that the feature groups, rather than a change in model configuration, are
      the controlled experimental variable.

Validation:
    Repeated row-wise CV:
    - 5 folds × 10 repeats by default.
    - Same 50 validation splits for every feature set.

The holdout result obtained previously remains the final conventional test result
for the selected full-feature XGBoost model. This script must not be used to
reselect the final model after viewing the holdout.

Input:
    CSV dataset containing:
        - 16 composition columns
        - 4 descriptor columns
        - 8 SFE-model output columns
        - SFE_Experimental (mJ/m²)

Output:
    Folder containing:
        - ablation_feature_sets.json
        - frozen_xgboost_hyperparameters.json
        - ablation_repeated_cv_fold_metrics.csv
        - ablation_repeated_cv_predictions.csv
        - ablation_repeated_cv_summary.csv
        - ablation_incremental_effects.csv

Run:
    python Feature_Ablation_XGBoost.py \
        --csv "data/output_with_8_sfe_models.csv" \
        --output "model_results/feature_ablation" \
        --split-manifest "model_results/development_split_manifest.csv"

Optional:
    python Feature_Ablation_XGBoost.py \
        --csv "data/output_with_8_sfe_models.csv" \
        --output "model_results/feature_ablation" \
        --split-manifest "model_results/development_split_manifest.csv" \
        --n-splits 5 \
        --n-repeats 10
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


# =============================================================================
# USER SETTINGS
# =============================================================================

DATASET_PATH = r"output.csv"
TARGET_COLUMN = "SFE_Experimental (mJ/m²)"
ID_COLUMN = "No."

OUTPUT_DIR = "feature_ablation"

RANDOM_SEED = 42

N_SPLITS = 5
N_REPEATS = 10


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

COMPOSITION_COLUMNS = [
    "Fe (%wt)", "Mn (%wt)", "Al (%wt)", "C (%wt)", "Ni (%wt)",
    "Cr (%wt)", "Si (%wt)", "Cu (%wt)", "Mo (%wt)", "Nb (%wt)",
    "V (%wt)", "Ti (%wt)", "Co (%wt)", "N (%wt)", "P (%wt)", "S (%wt)",
]

DESCRIPTOR_COLUMNS = [
    "R_mean (pm)",
    "R_std (pm)",
    "Δχ",
    "VEC_mean",
]

SFE_MODEL_COLUMNS = [
    "SFE_Akbari (mJ/m²)",
    "SFE_Allain (mJ/m²)",
    "SFE_Dumay (mJ/m²)",
    "SFE_Elaraby (mJ/m²)",
    "SFE_Wang (mJ/m²)",
    "SFE_Xiong (mJ/m²)",
    "SFE_Yoo (mJ/m²)",
    "SFE_Zambrano (mJ/m²)",
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
        mape = float(
            mean_absolute_percentage_error(y_true, y_pred) * 100.0
        )

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
# DATA LOADING AND VALIDATION
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


def validate_required_columns(
    df: pd.DataFrame,
    target_column: str,
) -> None:
    """Validate all required feature and target columns."""
    required = (
        COMPOSITION_COLUMNS
        + DESCRIPTOR_COLUMNS
        + SFE_MODEL_COLUMNS
        + [target_column]
    )

    missing = [
        col for col in required if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
        )


def prepare_data(
    df: pd.DataFrame,
    target_column: str,
    id_column: Optional[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare all features, target, and identifiers.

    Absurd SFE-model values with abs(value) > 1e6 are replaced with NaN.
    Median imputation is subsequently fitted inside each CV training fold.
    """
    validate_required_columns(
        df=df,
        target_column=target_column,
    )

    df = df.copy()

    if id_column is not None and id_column in df.columns:
        ids = df[id_column].copy()
    else:
        ids = pd.Series(
            np.arange(1, len(df) + 1),
            name="ID",
        )

    y = clean_numeric_series(df[target_column])

    n_missing_target = int(y.isna().sum())
    if n_missing_target > 0:
        print(
            f"Dropping {n_missing_target} rows with missing/non-numeric "
            "target values."
        )

    valid_mask = y.notna()

    df = df.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)
    ids = ids.loc[valid_mask].reset_index(drop=True)

    all_feature_columns = (
        COMPOSITION_COLUMNS
        + DESCRIPTOR_COLUMNS
        + SFE_MODEL_COLUMNS
    )

    X = df[all_feature_columns].copy()

    for col in X.columns:
        X[col] = clean_numeric_series(X[col])

    print("Cleaning SFE-model predictor columns:")
    for col in SFE_MODEL_COLUMNS:
        absurd_mask = X[col].abs() > 1e6
        n_absurd = int(absurd_mask.sum())

        if n_absurd > 0:
            print(
                f"  {col}: replacing {n_absurd} absurd values with NaN"
            )
            X.loc[absurd_mask, col] = np.nan

    print(f"Complete predictor matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    return X, y, ids


# =============================================================================
# FEATURE SETS
# =============================================================================

def get_feature_sets() -> Dict[str, List[str]]:
    """
    Return the four controlled ablation feature sets.

    The four sets form a 2 × 2 augmentation design around the composition baseline:
        descriptors absent/present
        SFE-model outputs absent/present
    """
    return {
        "Composition only": list(COMPOSITION_COLUMNS),
        "Composition + descriptors": (
            list(COMPOSITION_COLUMNS)
            + list(DESCRIPTOR_COLUMNS)
        ),
        "Composition + SFE-model outputs": (
            list(COMPOSITION_COLUMNS)
            + list(SFE_MODEL_COLUMNS)
        ),
        "Full feature set": (
            list(COMPOSITION_COLUMNS)
            + list(DESCRIPTOR_COLUMNS)
            + list(SFE_MODEL_COLUMNS)
        ),
    }


# =============================================================================
# FROZEN XGBOOST MODEL
# =============================================================================

def get_frozen_xgboost_hyperparameters() -> Dict[str, Any]:
    """
    Return the frozen XGBoost hyperparameters selected before ablation.

    These are identical to the final selected XGBoost model.
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


def build_preprocessor(
    feature_columns: List[str],
) -> ColumnTransformer:
    """
    Build median-imputation preprocessor for one feature set.

    The imputer is fitted separately inside every CV training fold.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "median_imputer",
                            SimpleImputer(strategy="median"),
                        ),
                    ]
                ),
                feature_columns,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_frozen_pipeline(
    feature_columns: List[str],
) -> Pipeline:
    """Build preprocessing + frozen XGBoost pipeline for one feature set."""
    preprocessor = build_preprocessor(
        feature_columns=feature_columns,
    )

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
    y: pd.Series,
    ids: pd.Series,
    manifest_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and validate the exact development/holdout split.

    Only the development indices are used for ablation. The holdout indices are
    loaded solely to verify that the original frozen split is being reused.
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
            f"Split manifest contains {len(manifest)} rows but the current "
            f"dataset contains {len(y)} rows."
        )

    manifest = manifest.sort_values(
        "Row_Index"
    ).reset_index(drop=True)

    expected_indices = np.arange(len(y))

    if not np.array_equal(
        manifest["Row_Index"].to_numpy(),
        expected_indices,
    ):
        raise ValueError(
            "Split manifest Row_Index must match the current dataset order exactly."
        )

    labels = set(
        manifest["Split"].astype(str)
    )

    if labels != {"train", "holdout"}:
        raise ValueError(
            "Split manifest must contain exactly the labels 'train' and 'holdout'."
        )

    if "ID" in manifest.columns:
        manifest_ids = manifest["ID"].astype(str).to_numpy()
        current_ids = ids.astype(str).to_numpy()

        if not np.array_equal(
            manifest_ids,
            current_ids,
        ):
            raise ValueError(
                "Split manifest ID values do not match the current dataset order."
            )

    idx_development = manifest.index[
        manifest["Split"].astype(str) == "train"
    ].to_numpy()

    idx_holdout = manifest.index[
        manifest["Split"].astype(str) == "holdout"
    ].to_numpy()

    if len(idx_development) == 0 or len(idx_holdout) == 0:
        raise ValueError(
            "Split manifest must contain non-empty train and holdout partitions."
        )

    return idx_development, idx_holdout


# =============================================================================
# REPEATED ROW-WISE ABLATION
# =============================================================================

def create_repeated_cv_splits(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    n_splits: int,
    n_repeats: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create one repeated-CV split list reused by every feature set.
    """
    repeated_cv = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
    )

    return list(
        repeated_cv.split(
            X_development,
            y_development,
        )
    )


def run_repeated_cv_ablation(
    X_development: pd.DataFrame,
    y_development: pd.Series,
    ids_development: pd.Series,
    feature_sets: Dict[str, List[str]],
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_splits: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate every feature set on the same repeated row-wise CV splits.
    """
    fold_rows = []
    prediction_rows = []

    for feature_set_name, feature_columns in feature_sets.items():
        print("\n" + "=" * 100)
        print(
            f"REPEATED ROW-WISE CV: {feature_set_name} "
            f"({len(feature_columns)} features)"
        )
        print("=" * 100)

        pipeline = build_frozen_pipeline(
            feature_columns=feature_columns,
        )

        for split_number, (
            train_idx,
            valid_idx,
        ) in enumerate(
            cv_splits,
            start=1,
        ):
            repeat_number = (
                (split_number - 1) // n_splits
            ) + 1

            fold_number = (
                (split_number - 1) % n_splits
            ) + 1

            fold_pipeline = clone(pipeline)

            X_fit = X_development.iloc[train_idx]
            y_fit = y_development.iloc[train_idx]

            X_valid = X_development.iloc[valid_idx]
            y_valid = y_development.iloc[valid_idx]

            fold_pipeline.fit(
                X_fit,
                y_fit,
            )

            predictions = fold_pipeline.predict(
                X_valid
            )

            metrics = calculate_metrics(
                y_valid,
                predictions,
            )

            fold_rows.append(
                {
                    "Validation": "Repeated row-wise CV",
                    "Feature_Set": feature_set_name,
                    "Feature_Count": len(feature_columns),
                    "Repeat": repeat_number,
                    "Fold": fold_number,
                    "Split_Number": split_number,
                    "Train_n": len(train_idx),
                    "Validation_n": len(valid_idx),
                    **metrics,
                }
            )

            valid_ids = ids_development.iloc[
                valid_idx
            ].reset_index(drop=True)

            for local_position, development_index in enumerate(
                valid_idx
            ):
                prediction_rows.append(
                    {
                        "Validation": "Repeated row-wise CV",
                        "Feature_Set": feature_set_name,
                        "Repeat": repeat_number,
                        "Fold": fold_number,
                        "Split_Number": split_number,
                        "Development_Row_Index": int(
                            development_index
                        ),
                        "ID": valid_ids.iloc[
                            local_position
                        ],
                        "Actual": float(
                            y_valid.iloc[
                                local_position
                            ]
                        ),
                        "Predicted": float(
                            predictions[
                                local_position
                            ]
                        ),
                        "Residual": float(
                            y_valid.iloc[
                                local_position
                            ]
                            - predictions[
                                local_position
                            ]
                        ),
                    }
                )

            print(
                f"Repeat {repeat_number:02d}, Fold {fold_number}: "
                f"RMSE={metrics['RMSE']:.4f}, "
                f"MAE={metrics['MAE']:.4f}, "
                f"R²={metrics['R2']:.4f}"
            )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
    )


# =============================================================================
# SUMMARY
# =============================================================================

def summarize_fold_metrics(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize repeated-CV fold-level metrics for each feature set.

    RMSE is primary, MAE secondary, and R² is supporting.
    """
    summary = (
        fold_metrics
        .groupby(
            [
                "Validation",
                "Feature_Set",
                "Feature_Count",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            RMSE_median=("RMSE", "median"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            MAE_median=("MAE", "median"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            R2_median=("R2", "median"),
            MAPE_percent_mean=("MAPE_percent", "mean"),
            MAPE_percent_std=("MAPE_percent", "std"),
            Pearson_r_mean=("Pearson_r", "mean"),
            Pearson_r_std=("Pearson_r", "std"),
        )
    )

    summary = summary.sort_values(
        by=[
            "RMSE_mean",
            "MAE_mean",
            "R2_mean",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    summary.insert(
        1,
        "Rank",
        np.arange(
            1,
            len(summary) + 1,
        ),
    )

    return summary


def get_summary_row(
    summary: pd.DataFrame,
    feature_set_name: str,
) -> pd.Series:
    """Return one summary row for one feature set."""
    subset = summary.loc[
        summary["Feature_Set"] == feature_set_name
    ]

    if len(subset) != 1:
        raise RuntimeError(
            f"Expected exactly one summary row for feature set "
            f"'{feature_set_name}'."
        )

    return subset.iloc[0]


def create_incremental_effects(
    repeated_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate controlled incremental feature-group effects.

    Positive Improvement_RMSE and Improvement_MAE values indicate lower error
    after adding the feature group. Positive Improvement_R2 indicates higher R².
    """
    comparisons = [
        (
            "Descriptors added to composition",
            "Composition only",
            "Composition + descriptors",
        ),
        (
            "SFE-model outputs added to composition",
            "Composition only",
            "Composition + SFE-model outputs",
        ),
        (
            "Descriptors added after SFE-model outputs",
            "Composition + SFE-model outputs",
            "Full feature set",
        ),
        (
            "SFE-model outputs added after descriptors",
            "Composition + descriptors",
            "Full feature set",
        ),
        (
            "Full augmentation versus composition baseline",
            "Composition only",
            "Full feature set",
        ),
    ]

    rows = []

    validation_name = "Repeated row-wise CV"

    for (
        comparison_name,
        baseline_name,
        augmented_name,
    ) in comparisons:
        baseline = get_summary_row(
            summary=repeated_summary,
            feature_set_name=baseline_name,
        )

        augmented = get_summary_row(
            summary=repeated_summary,
            feature_set_name=augmented_name,
        )

        rmse_improvement = (
            baseline["RMSE_mean"]
            - augmented["RMSE_mean"]
        )

        mae_improvement = (
            baseline["MAE_mean"]
            - augmented["MAE_mean"]
        )

        r2_improvement = (
            augmented["R2_mean"]
            - baseline["R2_mean"]
        )

        rmse_percent = (
            rmse_improvement
            / baseline["RMSE_mean"]
            * 100.0
        )

        mae_percent = (
            mae_improvement
            / baseline["MAE_mean"]
            * 100.0
        )

        rows.append(
            {
                "Validation": validation_name,
                "Comparison": comparison_name,
                "Baseline_Feature_Set": baseline_name,
                "Augmented_Feature_Set": augmented_name,
                "Baseline_RMSE": baseline["RMSE_mean"],
                "Augmented_RMSE": augmented["RMSE_mean"],
                "Improvement_RMSE": rmse_improvement,
                "Improvement_RMSE_percent": rmse_percent,
                "Baseline_MAE": baseline["MAE_mean"],
                "Augmented_MAE": augmented["MAE_mean"],
                "Improvement_MAE": mae_improvement,
                "Improvement_MAE_percent": mae_percent,
                "Baseline_R2": baseline["R2_mean"],
                "Augmented_R2": augmented["R2_mean"],
                "Improvement_R2": r2_improvement,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def save_feature_sets(
    feature_sets: Dict[str, List[str]],
    output_dir: Path,
) -> Path:
    """Save the exact feature sets used in ablation."""
    path = output_dir / "ablation_feature_sets.json"

    payload = {
        "design": (
            "Controlled 2x2 augmentation around composition baseline"
        ),
        "validation": (
            "Development-only repeated row-wise cross-validation"
        ),
        "holdout_status": (
            "Not used in feature ablation"
        ),
        "feature_sets": feature_sets,
    }

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return path


def save_frozen_hyperparameters(
    output_dir: Path,
) -> Path:
    """Save the frozen XGBoost hyperparameters used for every feature set."""
    path = (
        output_dir
        / "frozen_xgboost_hyperparameters.json"
    )

    payload = {
        "model": "XGBoost",
        "random_seed": RANDOM_SEED,
        "status": (
            "Frozen before feature ablation"
        ),
        "primary_metric": "RMSE",
        "secondary_metric": "MAE",
        "supporting_metric": "R2",
        "hyperparameters": (
            get_frozen_xgboost_hyperparameters()
        ),
    }

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
            default=convert_for_json,
        )

    return path


# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only controlled feature-group ablation for the "
            "frozen XGBoost SFE model using repeated row-wise CV."
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
        help="Output directory for ablation results.",
    )

    parser.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
        help=(
            "The same development_split_manifest.csv created before model "
            "selection. The holdout rows are excluded from all ablation analyses."
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
        "--n-splits",
        type=int,
        default=N_SPLITS,
        help="Number of folds per repeated row-wise CV repeat.",
    )

    parser.add_argument(
        "--n-repeats",
        type=int,
        default=N_REPEATS,
        help="Number of repeated row-wise CV repetitions.",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    np.random.seed(
        RANDOM_SEED
    )

    warnings.filterwarnings(
        "default"
    )

    args = parse_args()

    if args.n_splits < 2:
        raise ValueError(
            "--n-splits must be at least 2."
        )

    if args.n_repeats < 1:
        raise ValueError(
            "--n-repeats must be at least 1."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset(
        args.data
    )

    X, y, ids = prepare_data(
        df=df,
        target_column=args.target,
        id_column=args.id_column,
    )

    idx_development, idx_holdout = load_fixed_split_manifest(
        y=y,
        ids=ids,
        manifest_path=args.split_manifest,
    )

    X_development = X.iloc[
        idx_development
    ].reset_index(drop=True)

    y_development = y.iloc[
        idx_development
    ].reset_index(drop=True)

    ids_development = ids.iloc[
        idx_development
    ].reset_index(drop=True)

    feature_sets = get_feature_sets()

    print("\nFeature-ablation settings:")
    print(
        f"Development samples used: {len(idx_development)}"
    )
    print(
        f"Reserved holdout samples excluded: {len(idx_holdout)}"
    )
    print(
        f"Repeated row-wise CV: {args.n_splits} folds × "
        f"{args.n_repeats} repeats"
    )
    print("Model family: XGBoost")
    print("Hyperparameters: frozen and identical for every feature set")
    print("Primary metric: RMSE")
    print("Secondary metric: MAE")
    print("Supporting metric: R²")
    print("The holdout is NOT scored in this script.")

    print("\nFeature sets:")
    for feature_set_name, columns in feature_sets.items():
        print(
            f"  - {feature_set_name}: {len(columns)} features"
        )

    feature_sets_path = save_feature_sets(
        feature_sets=feature_sets,
        output_dir=args.output_dir,
    )

    hyperparameter_path = save_frozen_hyperparameters(
        output_dir=args.output_dir,
    )

    print(
        f"\nSaved feature definitions: {feature_sets_path}"
    )
    print(
        f"Saved frozen hyperparameters: {hyperparameter_path}"
    )

    # -------------------------------------------------------------------------
    # Repeated row-wise CV on the development partition only
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("DEVELOPMENT-ONLY REPEATED ROW-WISE FEATURE ABLATION")
    print("=" * 110)

    repeated_splits = create_repeated_cv_splits(
        X_development=X_development,
        y_development=y_development,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
    )

    repeated_folds, repeated_predictions = run_repeated_cv_ablation(
        X_development=X_development,
        y_development=y_development,
        ids_development=ids_development,
        feature_sets=feature_sets,
        cv_splits=repeated_splits,
        n_splits=args.n_splits,
    )

    repeated_summary = summarize_fold_metrics(
        repeated_folds
    )

    incremental_effects = create_incremental_effects(
        repeated_summary=repeated_summary,
    )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    repeated_folds_path = (
        args.output_dir
        / "ablation_repeated_cv_fold_metrics.csv"
    )

    repeated_predictions_path = (
        args.output_dir
        / "ablation_repeated_cv_predictions.csv"
    )

    repeated_summary_path = (
        args.output_dir
        / "ablation_repeated_cv_summary.csv"
    )

    effects_path = (
        args.output_dir
        / "ablation_incremental_effects.csv"
    )

    repeated_folds.to_csv(
        repeated_folds_path,
        index=False,
    )

    repeated_predictions.to_csv(
        repeated_predictions_path,
        index=False,
    )

    repeated_summary.to_csv(
        repeated_summary_path,
        index=False,
    )

    incremental_effects.to_csv(
        effects_path,
        index=False,
    )

    print("\n" + "=" * 110)
    print("REPEATED ROW-WISE ABLATION SUMMARY")
    print("=" * 110)

    repeated_display = [
        "Rank",
        "Feature_Set",
        "Feature_Count",
        "RMSE_mean",
        "RMSE_std",
        "MAE_mean",
        "MAE_std",
        "R2_mean",
        "R2_std",
    ]

    print(
        repeated_summary[
            repeated_display
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print("\n" + "=" * 110)
    print("CONTROLLED INCREMENTAL FEATURE EFFECTS")
    print("=" * 110)

    effect_display = [
        "Validation",
        "Comparison",
        "Improvement_RMSE",
        "Improvement_RMSE_percent",
        "Improvement_MAE",
        "Improvement_MAE_percent",
        "Improvement_R2",
    ]

    print(
        incremental_effects[
            effect_display
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print("\nInterpretation rule:")
    print(
        "  Positive Improvement_RMSE / Improvement_MAE = lower prediction error."
    )
    print(
        "  Positive Improvement_R2 = higher explained variance."
    )
    print(
        "  Negative values indicate that adding that feature group worsened the metric."
    )
    print(
        "\nThe reserved holdout was not scored or used for feature-set comparison."
    )

    print(f"\nSaved repeated-CV fold metrics: {repeated_folds_path}")
    print(f"Saved repeated-CV predictions: {repeated_predictions_path}")
    print(f"Saved repeated-CV summary: {repeated_summary_path}")
    print(f"Saved incremental effects: {effects_path}")
    print(f"All outputs saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
