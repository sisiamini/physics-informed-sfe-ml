#!/usr/bin/env python3
# Repeated_CV_Model_Comparison.py
"""
Compare four frozen SFE regression models using repeated cross-validation.

The four model families are:
    - Gradient Boosting
    - XGBoost
    - HistGradientBoosting
    - Extra Trees

Important:
    - The reserved 20% holdout is NOT scored in this script.
    - The same 80% development partition used during Optuna tuning is reused.
    - The same repeated-CV splits are used for every model.
    - Primary comparison metric: mean repeated-CV RMSE.
    - Secondary comparison metric: mean repeated-CV MAE.
    - R² is reported as a supporting metric.
    - Hyperparameters are frozen before this repeated-CV comparison.

Extra Trees and HistGradientBoosting note:
    Their original screening configurations are retained because each achieved
    lower five-fold development-CV RMSE than its best Optuna trial. No Extra Trees
    or HistGradientBoosting hyperparameters are changed in this script.

Input:
    CSV dataset containing predictor columns and the target column:
        SFE_Experimental (mJ/m²)

Output:
    Folder containing:
        - repeated_cv_fold_metrics.csv
        - repeated_cv_predictions.csv
        - repeated_cv_model_summary.csv
        - repeated_cv_pairwise_comparison.csv
        - frozen_model_hyperparameters.json

Run:
    python Repeated_CV_Model_Comparison.py \
        --csv "data/output_with_8_sfe_models.csv" \
        --output "model_results/repeated_cv_comparison" \
        --split-manifest "model_results/development_split_manifest.csv"

Optional:
    python Repeated_CV_Model_Comparison.py \
        --csv "data/output_with_8_sfe_models.csv" \
        --output "model_results/repeated_cv_comparison" \
        --split-manifest "model_results/development_split_manifest.csv" \
        --n-splits 5 \
        --n-repeats 10
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    RepeatedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline


# =============================================================================
# USER SETTINGS
# =============================================================================

DATASET_PATH = r"output.csv"
TARGET_COLUMN = "SFE_Experimental (mJ/m²)"
ID_COLUMN = "No."

OUTPUT_DIR = "repeated_cv_model_comparison"

RANDOM_SEED = 42
TEST_SIZE = 0.20

N_SPLITS = 5
N_REPEATS = 10


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


def create_target_bins(y: pd.Series, n_bins: int = 5):
    """Create approximate stratification bins for regression train/test split."""
    try:
        bins = pd.qcut(y, q=n_bins, duplicates="drop")
        if bins.nunique() < 2:
            return None
        return bins
    except Exception:
        return None


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
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """
    Prepare predictors and target.

    Steps:
    1. Clean target.
    2. Drop rows with missing target.
    3. Exclude identifier column from modeling.
    4. Convert predictors to numeric.
    5. Replace absurd thermodynamic values in columns V–AC with NaN.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    df = df.copy()

    ids = None
    if id_column is not None and id_column in df.columns:
        ids = df[id_column].copy()

    y = clean_numeric_series(df[target_column])

    missing_target = int(y.isna().sum())
    if missing_target > 0:
        print(f"Dropping {missing_target} rows with missing/non-numeric target values.")

    valid_mask = y.notna()

    df = df.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    if ids is not None:
        ids = ids.loc[valid_mask].reset_index(drop=True)

    drop_cols = [target_column]
    if id_column is not None and id_column in df.columns:
        drop_cols.append(id_column)

    X = df.drop(columns=drop_cols)

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

    print(f"Predictor matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    return X, y, ids


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build median-imputation preprocessor.

    The imputer is inside the Pipeline, so it is fitted only on each training fold.
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
# FROZEN MODEL HYPERPARAMETERS
# =============================================================================

def get_frozen_hyperparameters() -> Dict[str, Dict[str, Any]]:
    """
    Return the frozen hyperparameters used for repeated-CV comparison.

    Gradient Boosting and XGBoost use their best Optuna hyperparameters.
    HistGradientBoosting and Extra Trees use their original screening
    configurations because those configurations achieved lower five-fold
    development-CV RMSE than their best Optuna trials.
    """
    return {
        "Gradient Boosting": {
            "n_estimators": 450,
            "learning_rate": 0.0363548167266166,
            "max_depth": 3,
            "min_samples_split": 7,
            "min_samples_leaf": 3,
            "subsample": 0.7406222361513626,
            "max_features": "log2",
            "loss": "huber",
            "criterion": "friedman_mse",
            "random_state": 42,
            "alpha": 0.9472221287070752,
            "n_iter_no_change": 23,
            "validation_fraction": 0.15100329642324195,
            "tol": 6.123021030879258e-05,
        },
        "XGBoost": {
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
        },
        "HistGradientBoosting": {
            "max_iter": 400,
            "learning_rate": 0.03,
            "max_leaf_nodes": 15,
            "l2_regularization": 0.01,
            "random_state": 42,
        },
        "Extra Trees": {
            "n_estimators": 700,
            "criterion": "squared_error",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": False,
            "random_state": 42,
            "n_jobs": -1,
            "max_samples": None,
        },
    }


def create_frozen_models() -> Dict[str, object]:
    """Create the four frozen model instances."""
    params = get_frozen_hyperparameters()

    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required for this comparison but is not installed."
        ) from exc

    return {
        "Gradient Boosting": GradientBoostingRegressor(
            **params["Gradient Boosting"]
        ),
        "XGBoost": XGBRegressor(
            **params["XGBoost"]
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            **params["HistGradientBoosting"]
        ),
        "Extra Trees": ExtraTreesRegressor(
            **params["Extra Trees"]
        ),
    }


def build_pipeline(
    X: pd.DataFrame,
    model,
) -> Pipeline:
    """Build preprocessing + frozen model pipeline."""
    preprocessor = build_preprocessor(X)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clone(model)),
        ]
    )


# =============================================================================
# DEVELOPMENT SPLIT
# =============================================================================

def split_from_manifest(
    X: pd.DataFrame,
    y: pd.Series,
    ids: Optional[pd.Series],
    manifest_path: Path,
):
    """Recreate the fixed train/holdout partition saved by model selection."""
    manifest = pd.read_csv(manifest_path)

    required = {"Row_Index", "Split"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Split manifest missing columns: {sorted(missing)}")

    if len(manifest) != len(y):
        raise ValueError(
            f"Split manifest has {len(manifest)} rows but dataset has {len(y)} rows."
        )

    manifest = manifest.sort_values("Row_Index").reset_index(drop=True)

    expected = np.arange(len(y))
    if not np.array_equal(manifest["Row_Index"].to_numpy(), expected):
        raise ValueError("Split manifest Row_Index must be exactly 0..n-1.")

    valid_labels = {"train", "holdout"}
    labels = set(manifest["Split"].astype(str))

    if not labels.issubset(valid_labels) or labels != valid_labels:
        raise ValueError(
            "Split manifest must contain exactly the labels 'train' and 'holdout'."
        )

    if ids is not None and "ID" in manifest.columns:
        manifest_ids = manifest["ID"].reset_index(drop=True)
        data_ids = ids.reset_index(drop=True)

        if not np.array_equal(
            manifest_ids.astype(str).to_numpy(),
            data_ids.astype(str).to_numpy(),
        ):
            raise ValueError(
                "Split manifest ID values do not match the current dataset order."
            )

    idx_train = manifest.index[manifest["Split"] == "train"].to_numpy()
    idx_holdout = manifest.index[manifest["Split"] == "holdout"].to_numpy()

    X_train = X.iloc[idx_train].reset_index(drop=True)
    y_train = y.iloc[idx_train].reset_index(drop=True)

    ids_train = None
    if ids is not None:
        ids_train = ids.iloc[idx_train].reset_index(drop=True)

    return X_train, y_train, ids_train, idx_train, idx_holdout


def recreate_development_split(
    X: pd.DataFrame,
    y: pd.Series,
    ids: Optional[pd.Series],
):
    """Recreate the deterministic 80/20 development split if no manifest is supplied."""
    stratify_bins = create_target_bins(y)

    idx_all = np.arange(len(y))

    idx_train, idx_holdout = train_test_split(
        idx_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=stratify_bins,
    )

    X_train = X.iloc[idx_train].reset_index(drop=True)
    y_train = y.iloc[idx_train].reset_index(drop=True)

    ids_train = None
    if ids is not None:
        ids_train = ids.iloc[idx_train].reset_index(drop=True)

    return X_train, y_train, ids_train, idx_train, idx_holdout


# =============================================================================
# REPEATED CROSS-VALIDATION
# =============================================================================

def create_repeated_cv_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int,
    n_repeats: int,
):
    """
    Create one fixed repeated-CV split list.

    The resulting split list is reused for every model so all models are
    evaluated on exactly the same training and validation observations.
    """
    repeated_cv = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
    )

    return list(repeated_cv.split(X_train, y_train))


def evaluate_frozen_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    ids_train: Optional[pd.Series],
    models: Dict[str, object],
    cv_splits,
    n_splits: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all frozen models on exactly the same repeated-CV splits.

    Returns:
        fold_metrics_df
        predictions_df
    """
    fold_rows = []
    prediction_rows = []

    for model_name, model in models.items():
        print("\n" + "=" * 90)
        print(f"Repeated-CV evaluation: {model_name}")
        print("=" * 90)

        for split_number, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
            repeat_number = ((split_number - 1) // n_splits) + 1
            fold_number = ((split_number - 1) % n_splits) + 1

            pipeline = build_pipeline(X_train, model)

            X_fit = X_train.iloc[train_idx]
            y_fit = y_train.iloc[train_idx]

            X_valid = X_train.iloc[valid_idx]
            y_valid = y_train.iloc[valid_idx]

            start_time = time.time()
            pipeline.fit(X_fit, y_fit)
            fit_time = time.time() - start_time

            y_pred = pipeline.predict(X_valid)

            rmse = rmse_score(y_valid, y_pred)
            mae = float(mean_absolute_error(y_valid, y_pred))
            r2 = float(r2_score(y_valid, y_pred))

            fold_rows.append(
                {
                    "Model": model_name,
                    "Repeat": repeat_number,
                    "Fold": fold_number,
                    "Split_Number": split_number,
                    "Train_N": len(train_idx),
                    "Validation_N": len(valid_idx),
                    "RMSE": rmse,
                    "MAE": mae,
                    "R2": r2,
                    "Fit_Time_Seconds": fit_time,
                }
            )

            valid_ids = None
            if ids_train is not None:
                valid_ids = ids_train.iloc[valid_idx].reset_index(drop=True)

            for local_pos, development_row_idx in enumerate(valid_idx):
                prediction_row = {
                    "Model": model_name,
                    "Repeat": repeat_number,
                    "Fold": fold_number,
                    "Split_Number": split_number,
                    "Development_Row_Index": int(development_row_idx),
                    "Actual": float(y_valid.iloc[local_pos]),
                    "Predicted": float(y_pred[local_pos]),
                    "Residual": float(y_valid.iloc[local_pos] - y_pred[local_pos]),
                }

                if valid_ids is not None:
                    prediction_row["ID"] = valid_ids.iloc[local_pos]

                prediction_rows.append(prediction_row)

            print(
                f"Repeat {repeat_number:02d}, Fold {fold_number}: "
                f"RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}"
            )

    fold_metrics_df = pd.DataFrame(fold_rows)
    predictions_df = pd.DataFrame(prediction_rows)

    return fold_metrics_df, predictions_df


# =============================================================================
# SUMMARY AND PAIRED COMPARISON
# =============================================================================

def summarize_models(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize repeated-CV performance.

    Ranking rule:
        1. Lowest mean RMSE
        2. Lowest mean MAE
        3. Highest mean R²
    """
    summary = (
        fold_metrics_df
        .groupby("Model", as_index=False)
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
            Fit_Time_mean_seconds=("Fit_Time_Seconds", "mean"),
        )
    )

    summary = summary.sort_values(
        by=["RMSE_mean", "MAE_mean", "R2_mean"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    summary.insert(0, "Rank", np.arange(1, len(summary) + 1))

    best_rmse = summary.loc[0, "RMSE_mean"]
    summary["Delta_RMSE_from_best"] = summary["RMSE_mean"] - best_rmse

    return summary


def create_pairwise_comparison(fold_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create paired fold-by-fold comparisons.

    Because all models use exactly the same repeated-CV splits, fold-level
    metric differences are paired by Repeat and Fold.
    """
    rows = []

    model_names = sorted(fold_metrics_df["Model"].unique())

    indexed = fold_metrics_df.set_index(["Model", "Repeat", "Fold"])

    for model_a, model_b in combinations(model_names, 2):
        a = indexed.loc[model_a].sort_index()
        b = indexed.loc[model_b].sort_index()

        if not a.index.equals(b.index):
            raise RuntimeError(
                f"Repeated-CV folds do not align for {model_a} and {model_b}."
            )

        rmse_diff = a["RMSE"].to_numpy() - b["RMSE"].to_numpy()
        mae_diff = a["MAE"].to_numpy() - b["MAE"].to_numpy()
        r2_diff = a["R2"].to_numpy() - b["R2"].to_numpy()

        rows.append(
            {
                "Model_A": model_a,
                "Model_B": model_b,
                "Mean_RMSE_A_minus_B": float(np.mean(rmse_diff)),
                "Median_RMSE_A_minus_B": float(np.median(rmse_diff)),
                "RMSE_A_better_count": int(np.sum(rmse_diff < 0)),
                "RMSE_B_better_count": int(np.sum(rmse_diff > 0)),
                "RMSE_tie_count": int(np.sum(np.isclose(rmse_diff, 0.0))),
                "Mean_MAE_A_minus_B": float(np.mean(mae_diff)),
                "Median_MAE_A_minus_B": float(np.median(mae_diff)),
                "MAE_A_better_count": int(np.sum(mae_diff < 0)),
                "MAE_B_better_count": int(np.sum(mae_diff > 0)),
                "MAE_tie_count": int(np.sum(np.isclose(mae_diff, 0.0))),
                "Mean_R2_A_minus_B": float(np.mean(r2_diff)),
                "Median_R2_A_minus_B": float(np.median(r2_diff)),
            }
        )

    return pd.DataFrame(rows)


def save_frozen_hyperparameters(
    output_dir: Path,
) -> Path:
    """Save the frozen model hyperparameters used in the comparison."""
    path = output_dir / "frozen_model_hyperparameters.json"

    payload = {
        "random_seed": RANDOM_SEED,
        "selection_rule": {
            "primary": "lowest mean repeated-CV RMSE",
            "secondary": "lowest mean repeated-CV MAE",
            "supporting": "highest mean repeated-CV R2 and lower variability",
        },
        "models": get_frozen_hyperparameters(),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=4,
            default=convert_for_json,
        )

    return path


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeated-CV comparison of four frozen SFE regression models "
            "without scoring the reserved holdout."
        )
    )

    parser.add_argument(
        "--csv",
        "--data",
        dest="data",
        type=str,
        default=DATASET_PATH,
        help="Input modelling CSV dataset.",
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
        "--output",
        "--output-dir",
        dest="output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory.",
    )

    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help=(
            "Optional development_split_manifest.csv produced by SFE_Ensemble.py. "
            "When supplied, the exact same 80/20 split is reused. "
            "The reserved holdout is not evaluated."
        ),
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=N_SPLITS,
        help="Number of folds per repeat.",
    )

    parser.add_argument(
        "--n-repeats",
        type=int,
        default=N_REPEATS,
        help="Number of repeated K-fold repetitions.",
    )

    return parser.parse_args()


def main() -> None:
    np.random.seed(RANDOM_SEED)
    warnings.filterwarnings("default")

    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)

    X, y, ids = prepare_features_and_target(
        df=df,
        target_column=args.target,
        id_column=args.id_column,
    )

    if args.split_manifest is not None:
        X_train, y_train, ids_train, idx_train, idx_holdout = split_from_manifest(
            X=X,
            y=y,
            ids=ids,
            manifest_path=args.split_manifest,
        )
        print(f"Using fixed split manifest: {args.split_manifest}")
    else:
        X_train, y_train, ids_train, idx_train, idx_holdout = recreate_development_split(
            X=X,
            y=y,
            ids=ids,
        )
        print(
            "WARNING: no --split-manifest was supplied. "
            "The deterministic 80/20 split was recreated locally."
        )

    print("\nRepeated-CV comparison settings:")
    print(f"Development samples: {len(y_train)}")
    print(f"Reserved holdout samples: {len(idx_holdout)}")
    print(f"Folds per repeat: {args.n_splits}")
    print(f"Repeats: {args.n_repeats}")
    print(f"Total validation folds per model: {args.n_splits * args.n_repeats}")
    print("Primary selection metric: mean repeated-CV RMSE")
    print("Secondary selection metric: mean repeated-CV MAE")
    print("Supporting metric: mean repeated-CV R²")
    print("Reserved holdout scores will NOT be calculated.")

    models = create_frozen_models()

    print("\nFrozen models:")
    for model_name in models:
        print(f"  - {model_name}")

    hyperparameter_path = save_frozen_hyperparameters(output_dir)
    print(f"\nSaved frozen hyperparameters: {hyperparameter_path}")

    cv_splits = create_repeated_cv_splits(
        X_train=X_train,
        y_train=y_train,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
    )

    fold_metrics_df, predictions_df = evaluate_frozen_models(
        X_train=X_train,
        y_train=y_train,
        ids_train=ids_train,
        models=models,
        cv_splits=cv_splits,
        n_splits=args.n_splits,
    )

    summary_df = summarize_models(fold_metrics_df)
    pairwise_df = create_pairwise_comparison(fold_metrics_df)

    fold_metrics_path = output_dir / "repeated_cv_fold_metrics.csv"
    predictions_path = output_dir / "repeated_cv_predictions.csv"
    summary_path = output_dir / "repeated_cv_model_summary.csv"
    pairwise_path = output_dir / "repeated_cv_pairwise_comparison.csv"

    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)

    print("\n" + "=" * 110)
    print("REPEATED-CV MODEL COMPARISON SUMMARY")
    print("=" * 110)

    display_cols = [
        "Rank",
        "Model",
        "RMSE_mean",
        "RMSE_std",
        "MAE_mean",
        "MAE_std",
        "R2_mean",
        "R2_std",
        "Delta_RMSE_from_best",
    ]

    print(
        summary_df[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print("\nRanking rule:")
    print("  1. Lowest mean repeated-CV RMSE")
    print("  2. Lowest mean repeated-CV MAE")
    print("  3. Highest mean repeated-CV R²")
    print(
        "\nDo not use this script to inspect or score the reserved holdout. "
        "The holdout remains untouched until the final model is selected."
    )

    print(f"\nSaved fold metrics: {fold_metrics_path}")
    print(f"Saved repeated-CV predictions: {predictions_path}")
    print(f"Saved model summary: {summary_path}")
    print(f"Saved paired comparison: {pairwise_path}")
    print(f"All outputs saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
