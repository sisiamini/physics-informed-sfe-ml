#!/usr/bin/env python3
# SFE_Ensemble.py
"""
Compare ensemble regression models for SFE prediction using training-only cross-validation.

A stratified 80/20 development split is created first. Candidate model families are
compared only by five-fold cross-validation within the 80% training partition.
The 20% holdout is reserved and is not evaluated in this script.

Dataset assumption:
    - Column A: identifier, e.g. "No."
    - Columns B–AC: predictors
    - Column AD: target, e.g. "SFE_Experimental (mJ/m²)"
    - Columns V–AC are thermodynamic model outputs and should be used as predictors.
    - Absurd thermodynamic values in V–AC with abs(value) > 1e6 are replaced with NaN.

Input:
    CSV containing composition, descriptor, thermodynamic-model output,
    and experimental SFE columns.

Output:
    Folder containing model evaluation results, prediction CSV files,
    feature importance files, and plots.

Run:
    python SFE_Ensemble.py \
        --csv "data/output_with_8_sfe_models.csv" \
        --output "model_results"

Example:
    python SFE_Ensemble.py \
        --csv "source folder/source file.csv" \
        --output "destination folder/model_results"
"""

from __future__ import annotations

import argparse
import os
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline


# =============================================================================
# USER SETTINGS
# =============================================================================

# -------------------------------------------------------------------------
# 1. Dataset path is supplied from the command line using --csv.
# -------------------------------------------------------------------------
DATASET_PATH = None

# -------------------------------------------------------------------------
# 2. Specify the target column name here
# -------------------------------------------------------------------------
TARGET_COLUMN = "SFE_Experimental (mJ/m²)"

# -------------------------------------------------------------------------
# 3. Specify identifier column, if present
# -------------------------------------------------------------------------
ID_COLUMN = "No."

# Output directory is supplied from the command line using --output.
OUTPUT_DIR = None

# Reproducibility
RANDOM_SEED = 42

# Train/test split settings
TEST_SIZE = 0.20

# Cross-validation settings
RUN_CROSS_VALIDATION = True
N_SPLITS = 5

# Optional plots
SAVE_PLOTS = True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_output_dir(output_dir: str | Path) -> Path:
    """Create output directory if it does not already exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to numeric.

    Handles cases such as:
    - "26*" -> 26
    - " 41 " -> 41
    - "1,234.5" -> 1234.5

    Non-convertible values become NaN.
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


def get_excel_style_columns_v_to_ac(df: pd.DataFrame) -> List[str]:
    """
    Return columns V through AC by position.

    Excel-style positions:
    A  = index 0
    V  = index 21
    AC = index 28

    Therefore, V:AC corresponds to df.columns[21:29].
    """
    if df.shape[1] < 29:
        warnings.warn(
            "Dataset has fewer than 29 columns, so columns V–AC cannot be selected "
            "by position. No thermodynamic columns will be cleaned using the V–AC rule."
        )
        return []

    return list(df.columns[21:29])


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Load CSV dataset."""
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {dataset_path}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str,
    id_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series], List[str]]:
    """
    Prepare X and y.

    - Cleans the target column.
    - Drops rows with missing target values.
    - Excludes the ID column from predictors.
    - Uses all remaining columns as predictors.
    - Converts predictors to numeric where possible.
    - Replaces absurd values in columns V–AC with NaN if abs(value) > 1e6.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found.\n"
            f"Available columns are:\n{list(df.columns)}"
        )

    df = df.copy()

    # Keep IDs separately for saved prediction files.
    ids = None
    if id_column is not None and id_column in df.columns:
        ids = df[id_column].copy()

    # Clean target.
    y = clean_numeric_series(df[target_column])

    n_missing_target = y.isna().sum()
    if n_missing_target > 0:
        print(f"Dropping {n_missing_target} rows with missing/non-numeric target values.")

    valid_target_mask = y.notna()
    df = df.loc[valid_target_mask].reset_index(drop=True)
    y = y.loc[valid_target_mask].reset_index(drop=True)

    if ids is not None:
        ids = ids.loc[valid_target_mask].reset_index(drop=True)

    # Define predictors: all columns except target and ID.
    drop_cols = [target_column]
    if id_column is not None and id_column in df.columns:
        drop_cols.append(id_column)

    X = df.drop(columns=drop_cols)

    # Convert all predictors to numeric.
    for col in X.columns:
        X[col] = clean_numeric_series(X[col])

    # Clean absurd values in columns V–AC.
    thermodynamic_cols = get_excel_style_columns_v_to_ac(df)

    # Remove target and ID from thermodynamic list if they appear there.
    thermodynamic_cols = [
        col for col in thermodynamic_cols
        if col in X.columns
    ]

    if thermodynamic_cols:
        print("Cleaning thermodynamic predictor columns V–AC:")
        for col in thermodynamic_cols:
            absurd_mask = X[col].abs() > 1e6
            n_absurd = int(absurd_mask.sum())
            if n_absurd > 0:
                print(f"  - {col}: replacing {n_absurd} absurd values with NaN")
                X.loc[absurd_mask, col] = np.nan
    else:
        print("No V–AC thermodynamic predictor columns found for absurd-value cleaning.")

    print(f"Predictor matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    return X, y, ids, thermodynamic_cols


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Create preprocessing pipeline.

    Median imputation is fitted only inside the training fold because it is placed
    inside the sklearn Pipeline. This prevents data leakage.
    """
    numeric_features = list(X.columns)

    numeric_transformer = Pipeline(
        steps=[
            ("median_imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def optional_import_xgboost(random_seed: int):
    """Create XGBoost regressor if xgboost is installed."""
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=600,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=random_seed,
            n_jobs=-1,
        )
    except ImportError:
        warnings.warn("XGBoost is not installed. Skipping XGBoostRegressor.")
        return None


def optional_import_lightgbm(random_seed: int):
    """Create LightGBM regressor if lightgbm is installed."""
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression",
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=random_seed,
            n_jobs=-1,
            verbosity=-1,
        )
    except ImportError:
        warnings.warn("LightGBM is not installed. Skipping LGBMRegressor.")
        return None


def optional_import_catboost(random_seed: int):
    """Create CatBoost regressor if catboost is installed."""
    try:
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            loss_function="RMSE",
            iterations=600,
            learning_rate=0.03,
            depth=3,
            l2_leaf_reg=3.0,
            random_seed=random_seed,
            verbose=False,
            allow_writing_files=False,
        )
    except ImportError:
        warnings.warn("CatBoost is not installed. Skipping CatBoostRegressor.")
        return None


def create_models(random_seed: int) -> Dict[str, object]:
    """
    Create candidate regression models.

    Optional models are included only if their packages are installed.
    """
    models: Dict[str, object] = {}

    models["GradientBoosting"] = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.85,
        random_state=random_seed,
    )

    xgb_model = optional_import_xgboost(random_seed)
    if xgb_model is not None:
        models["XGBoost"] = xgb_model

    lgbm_model = optional_import_lightgbm(random_seed)
    if lgbm_model is not None:
        models["LightGBM"] = lgbm_model

    cat_model = optional_import_catboost(random_seed)
    if cat_model is not None:
        models["CatBoost"] = cat_model

    models["HistGradientBoosting"] = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.03,
        max_leaf_nodes=15,
        l2_regularization=0.01,
        random_state=random_seed,
    )

    models["RandomForest"] = RandomForestRegressor(
        n_estimators=700,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=random_seed,
        n_jobs=-1,
    )

    models["ExtraTrees"] = ExtraTreesRegressor(
        n_estimators=700,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=False,
        random_state=random_seed,
        n_jobs=-1,
    )

    models["AdaBoost"] = AdaBoostRegressor(
        n_estimators=300,
        learning_rate=0.03,
        random_state=random_seed,
    )

    # Stacking ensemble.
    # Prefer XGBoost + RandomForest + ExtraTrees when XGBoost is available.
    stacking_estimators = []

    if xgb_model is not None:
        stacking_estimators.append(("xgboost", clone(xgb_model)))

    stacking_estimators.extend(
        [
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=random_seed,
                    n_jobs=-1,
                ),
            ),
            (
                "extra_trees",
                ExtraTreesRegressor(
                    n_estimators=500,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=random_seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    if len(stacking_estimators) >= 2:
        models["Stacking"] = StackingRegressor(
            estimators=stacking_estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            n_jobs=-1,
            passthrough=False,
        )
    else:
        warnings.warn("Not enough base models available for stacking. Skipping StackingRegressor.")

    return models


def make_pipeline(model, preprocessor: ColumnTransformer) -> Pipeline:
    """Create full modeling pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", model),
        ]
    )


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Pearson correlation safely."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size < 2:
        return np.nan

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)    
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson = pearson_correlation(y_true, y_pred)

    # MAPE is inappropriate if actual values are zero or extremely close to zero.
    if np.any(np.isclose(y_true, 0.0)):
        mape = np.nan
    else:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE_percent": mape,
        "Pearson_r": pearson,
    }


def run_cross_validation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    random_seed: int,
) -> Dict[str, float]:
    """
    Run k-fold cross-validation.

    Because preprocessing is inside the pipeline, the median imputer is refit
    separately inside each training fold, preventing data leakage.
    """
    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )

    scoring = {
        "R2": "r2",
        "RMSE": "neg_root_mean_squared_error",
        "MAE": "neg_mean_absolute_error",
    }

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )

    return {
        "CV_R2_mean": float(np.mean(cv_results["test_R2"])),
        "CV_R2_std": float(np.std(cv_results["test_R2"])),
        "CV_RMSE_mean": float(-np.mean(cv_results["test_RMSE"])),
        "CV_RMSE_std": float(np.std(-cv_results["test_RMSE"])),
        "CV_MAE_mean": float(-np.mean(cv_results["test_MAE"])),
        "CV_MAE_std": float(np.std(-cv_results["test_MAE"])),
    }


def create_stratification_bins(y: pd.Series, n_bins: int = 5) -> Optional[pd.Series]:
    """
    Create approximate stratification bins for regression target.

    Returns None if binning is not possible.
    """
    try:
        bins = pd.qcut(y, q=n_bins, duplicates="drop")
        if bins.nunique() < 2:
            return None
        return bins
    except Exception:
        return None


def save_split_manifest(
    n_rows: int,
    idx_train: np.ndarray,
    idx_holdout: np.ndarray,
    ids: Optional[pd.Series],
    output_dir: Path,
) -> None:
    """Save the fixed development split without exposing holdout performance."""
    split = np.full(n_rows, "train", dtype=object)
    split[idx_holdout] = "holdout"

    manifest = pd.DataFrame(
        {
            "Row_Index": np.arange(n_rows),
            "Split": split,
        }
    )

    if ids is not None:
        manifest.insert(1, "ID", ids.reset_index(drop=True).values)

    path = output_dir / "development_split_manifest.csv"
    manifest.to_csv(path, index=False)
    print(f"Saved fixed development split: {path}")


def train_evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    ids: Optional[pd.Series],
    models: Dict[str, object],
    preprocessor: ColumnTransformer,
    output_dir: Path,
    test_size: float,
    random_seed: int,
    run_cv: bool,
    n_splits: int,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """
    Compare candidate model families without using the reserved holdout scores.

    The 80/20 split is created once. Every candidate is compared using the same
    row-wise K-fold cross-validation folds within the training partition only.
    The holdout partition is saved for later final evaluation after model-family
    selection and hyperparameter tuning are complete.
    """
    stratify_bins = create_stratification_bins(y)

    X_train, X_holdout, y_train, y_holdout, idx_train, idx_holdout = train_test_split(
        X,
        y,
        np.arange(len(y)),
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
        stratify=stratify_bins,
    )

    save_split_manifest(
        n_rows=len(y),
        idx_train=idx_train,
        idx_holdout=idx_holdout,
        ids=ids,
        output_dir=output_dir,
    )

    print(f"Training/development samples: {len(y_train)}")
    print(f"Reserved holdout samples: {len(y_holdout)}")
    print("Reserved holdout scores are NOT calculated in this script.")

    trained_pipelines: Dict[str, Pipeline] = {}
    result_rows = []

    for model_name, model in models.items():
        print("\n" + "=" * 80)
        print(f"Evaluating candidate model: {model_name}")
        print("=" * 80)

        pipeline = make_pipeline(model, preprocessor)

        row = {"Model": model_name}

        if run_cv:
            print(
                f"Running {n_splits}-fold training-only cross-validation "
                f"for {model_name}..."
            )
            try:
                cv_metrics = run_cross_validation(
                    pipeline=pipeline,
                    X=X_train,
                    y=y_train,
                    n_splits=n_splits,
                    random_seed=random_seed,
                )
                row.update(cv_metrics)
            except Exception as exc:
                warnings.warn(f"Cross-validation failed for {model_name}: {exc}")
                row.update(
                    {
                        "CV_R2_mean": np.nan,
                        "CV_R2_std": np.nan,
                        "CV_RMSE_mean": np.nan,
                        "CV_RMSE_std": np.nan,
                        "CV_MAE_mean": np.nan,
                        "CV_MAE_std": np.nan,
                    }
                )

        # Fit on the complete development partition only. This fitted pipeline is
        # retained for training diagnostics / feature importance, not holdout scoring.
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_train_pred = pipeline.predict(X_train)
        train_metrics = calculate_metrics(y_train, y_train_pred)

        row.update(
            {
                "Training_time_seconds": training_time,
                "Train_R2": train_metrics["R2"],
                "Train_RMSE": train_metrics["RMSE"],
                "Train_MAE": train_metrics["MAE"],
                "Train_MAPE_percent": train_metrics["MAPE_percent"],
                "Train_Pearson_r": train_metrics["Pearson_r"],
            }
        )

        result_rows.append(row)
        trained_pipelines[model_name] = pipeline

        print(
            f"{model_name} | "
            f"CV R² = {row.get('CV_R2_mean', np.nan):.4f} ± "
            f"{row.get('CV_R2_std', np.nan):.4f}, "
            f"CV RMSE = {row.get('CV_RMSE_mean', np.nan):.4f}, "
            f"CV MAE = {row.get('CV_MAE_mean', np.nan):.4f}"
        )

    results_df = pd.DataFrame(result_rows)

    # Candidate-family selection is based only on training-partition CV.
    results_df = results_df.sort_values(
        by=["CV_R2_mean", "CV_RMSE_mean"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return results_df, trained_pipelines

def sanitize_filename(name: str) -> str:
    """Create safe file name from model name."""
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name)


def get_feature_names_from_pipeline(pipeline: Pipeline) -> List[str]:
    """Extract transformed feature names from fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = []

    return feature_names


def save_feature_importance(
    trained_pipelines: Dict[str, Pipeline],
    output_dir: Path,
    top_n: int = 25,
) -> None:
    """
    Save feature importance CSV files and optional plots for models that support it.
    """
    for model_name, pipeline in trained_pipelines.items():
        model = pipeline.named_steps["model"]
        feature_names = get_feature_names_from_pipeline(pipeline)

        importances = None

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

        if importances is None:
            continue

        if len(feature_names) != len(importances):
            warnings.warn(
                f"Feature-name length mismatch for {model_name}. "
                f"Skipping feature importance."
            )
            continue

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances,
            }
        ).sort_values("Importance", ascending=False)

        importance_path = output_dir / f"feature_importance_{sanitize_filename(model_name)}.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"Saved feature importance: {importance_path}")

        if SAVE_PLOTS:
            try:
                import matplotlib.pyplot as plt

                plot_df = importance_df.head(top_n).iloc[::-1]

                plt.figure(figsize=(8, 7))
                plt.barh(plot_df["Feature"], plot_df["Importance"])
                plt.xlabel("Importance")
                plt.ylabel("Feature")
                plt.title(f"Top {top_n} Feature Importances: {model_name}")
                plt.tight_layout()

                plot_path = output_dir / f"feature_importance_{sanitize_filename(model_name)}.png"
                plt.savefig(plot_path, dpi=300)
                plt.close()
                print(f"Saved feature importance plot: {plot_path}")

            except Exception as exc:
                warnings.warn(f"Could not save feature importance plot for {model_name}: {exc}")


def save_prediction_plots(output_dir: Path) -> None:
    """
    Save predicted-vs-actual and residual plots for each model using saved predictions.
    """
    if not SAVE_PLOTS:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib is not installed. Skipping plots.")
        return

    combined_predictions_path = output_dir / "all_model_predictions.csv"

    if not combined_predictions_path.exists():
        warnings.warn("Combined predictions file not found. Skipping prediction plots.")
        return

    pred_df = pd.read_csv(combined_predictions_path)

    for model_name, model_df in pred_df.groupby("Model"):
        actual = model_df["Actual"].values
        predicted = model_df["Predicted"].values
        residual = model_df["Residual"].values

        # Predicted vs actual plot.
        plt.figure(figsize=(6, 6))
        plt.scatter(actual, predicted, alpha=0.75)

        min_val = min(np.min(actual), np.min(predicted))
        max_val = max(np.max(actual), np.max(predicted))
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

        plt.xlabel("Actual SFE")
        plt.ylabel("Predicted SFE")
        plt.title(f"Predicted vs Actual: {model_name}")
        plt.tight_layout()

        plot_path = output_dir / f"predicted_vs_actual_{sanitize_filename(model_name)}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        # Residual plot.
        plt.figure(figsize=(7, 5))
        plt.scatter(predicted, residual, alpha=0.75)
        plt.axhline(0, linestyle="--")

        plt.xlabel("Predicted SFE")
        plt.ylabel("Residual: Actual - Predicted")
        plt.title(f"Residual Plot: {model_name}")
        plt.tight_layout()

        residual_plot_path = output_dir / f"residual_plot_{sanitize_filename(model_name)}.png"
        plt.savefig(residual_plot_path, dpi=300)
        plt.close()

        print(f"Saved plots for {model_name}")


def save_results(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Save training-only candidate comparison results."""
    results_path = output_dir / "model_evaluation_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 100)
    print("TRAINING-ONLY MODEL COMPARISON TABLE")
    print("=" * 100)

    display_cols = [
        "Model",
        "CV_R2_mean",
        "CV_R2_std",
        "CV_RMSE_mean",
        "CV_RMSE_std",
        "CV_MAE_mean",
        "CV_MAE_std",
        "Train_R2",
        "Train_RMSE",
        "Train_MAE",
        "Training_time_seconds",
    ]

    available_cols = [col for col in display_cols if col in results_df.columns]

    print(results_df[available_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved evaluation results: {results_path}")
    print("The reserved holdout was not scored during candidate-family comparison.")


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def parse_command_line_args() -> argparse.Namespace:
    """Parse command-line input and output paths."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate ensemble SFE regression models."
    )

    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Input CSV dataset file"
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output folder for model results"
    )

    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_command_line_args()

    np.random.seed(RANDOM_SEED)
    warnings.filterwarnings("default")

    output_dir = ensure_output_dir(args.output)

    df = load_dataset(args.csv)

    X, y, ids, thermodynamic_cols = prepare_features_and_target(
        df=df,
        target_column=TARGET_COLUMN,
        id_column=ID_COLUMN,
    )

    preprocessor = create_preprocessor(X)

    models = create_models(RANDOM_SEED)

    if not models:
        raise RuntimeError("No models were created. Please check your environment.")

    print("\nModels to be evaluated:")
    for model_name in models:
        print(f"  - {model_name}")

    results_df, trained_pipelines = train_evaluate_models(
        X=X,
        y=y,
        ids=ids,
        models=models,
        preprocessor=preprocessor,
        output_dir=output_dir,
        test_size=TEST_SIZE,
        random_seed=RANDOM_SEED,
        run_cv=RUN_CROSS_VALIDATION,
        n_splits=N_SPLITS,
    )

    save_results(results_df, output_dir)

    save_feature_importance(
        trained_pipelines=trained_pipelines,
        output_dir=output_dir,
        top_n=25,
    )

    print("\nDone.")
    print(f"All outputs saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()