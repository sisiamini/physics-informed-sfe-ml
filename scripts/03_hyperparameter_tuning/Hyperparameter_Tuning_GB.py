#!/usr/bin/env python3
# Hyperparameter_Tuning_GB.py
"""
Tune Gradient Boosting hyperparameters for SFE prediction using Optuna.

Input:
    CSV dataset containing predictor columns and the target column:
        SFE_Experimental (mJ/m²)

Output:
    Folder containing Optuna tuning results, best hyperparameters,
    development-CV metrics, and the best train-split .joblib pipeline.

Important:
    - The reserved 20% holdout is NOT scored in this script.
    - Optuna minimizes mean five-fold CV RMSE on the 80% development partition.
    - Use the same development_split_manifest.csv for all candidate model families.

Run:
    python Hyperparameter_Tuning_GB.py \
        --csv "data/output.csv" \
        --output "models/GB_optuna_tuning_results" \
        --split-manifest "model_results/development_split_manifest.csv"

Optional:
    python Hyperparameter_Tuning_GB.py \
        --csv "data/output.csv" \
        --output "models/GB_optuna_tuning_results" \
        --split-manifest "model_results/development_split_manifest.csv" \
        --n-trials 100 \
        --cv-folds 5
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import optuna
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    make_scorer,
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

DATASET_PATH = r"output.csv"
TARGET_COLUMN = "SFE_Experimental (mJ/m²)"
ID_COLUMN = "No."

OUTPUT_DIR = "GB_optuna_tuning_results"

RANDOM_SEED = 42
TEST_SIZE = 0.20

CV_FOLDS = 5
N_TRIALS = 100
TIMEOUT_SECONDS = None

N_JOBS_CV = -1

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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) < 2:
        return np.nan

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculate regression metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = rmse_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    if np.any(np.isclose(y_true, 0.0)):
        mape = np.nan
    else:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE_percent": mape,
        "Pearson_r": pearson_r(y_true, y_pred),
    }


def create_target_bins(y: pd.Series, n_bins: int = 5):
    """
    Create approximate stratification bins for regression train/test split.
    """
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

def load_dataset(dataset_path: str) -> pd.DataFrame:
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
# MODEL AND OPTUNA SEARCH SPACE
# =============================================================================

def suggest_gradient_boosting_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Suggest Gradient Boosting hyperparameters.

    Search space follows the previous project tuning script so that the revised
    comparison remains consistent with the earlier Gradient Boosting analysis.
    """
    max_features_choice = trial.suggest_categorical(
        "max_features",
        ["None", "sqrt", "log2", 0.5, 0.7, 0.9, 1.0],
    )

    max_features = None if max_features_choice == "None" else max_features_choice

    loss = trial.suggest_categorical("loss", ["squared_error", "huber"])

    use_early_stopping = trial.suggest_categorical(
        "use_early_stopping",
        [False, True],
    )

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 1200, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "max_features": max_features,
        "loss": loss,
        "criterion": trial.suggest_categorical(
            "criterion",
            ["friedman_mse", "squared_error"],
        ),
        "random_state": RANDOM_SEED,
    }

    if loss == "huber":
        params["alpha"] = trial.suggest_float("alpha", 0.75, 0.95)

    if use_early_stopping:
        params["n_iter_no_change"] = trial.suggest_int("n_iter_no_change", 10, 50)
        params["validation_fraction"] = trial.suggest_float(
            "validation_fraction",
            0.10,
            0.20,
        )
        params["tol"] = trial.suggest_float("tol", 1e-5, 1e-3, log=True)
    else:
        params["n_iter_no_change"] = None

    return params

def build_pipeline(X: pd.DataFrame, params: Dict[str, Any]) -> Pipeline:
    """Build full preprocessing + Gradient Boosting pipeline."""
    preprocessor = build_preprocessor(X)

    model = GradientBoostingRegressor(**params)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline

# =============================================================================
# CROSS-VALIDATION AND OBJECTIVE
# =============================================================================

def get_cv(cv_folds: int) -> KFold:
    """Create reproducible K-fold splitter."""
    return KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_SEED,
    )


def evaluate_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
) -> Dict[str, float]:
    """
    Evaluate pipeline using cross-validation.

    The primary model-selection metric is mean CV RMSE.
    MAE is the secondary error metric and R² is also reported and saved.
    """
    scoring = {
        "R2": "r2",
        "RMSE": make_scorer(rmse_score, greater_is_better=False),
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    }

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=get_cv(cv_folds),
        scoring=scoring,
        n_jobs=N_JOBS_CV,
        return_train_score=False,
        error_score="raise",
    )

    return {
        "CV_R2_mean": float(np.mean(results["test_R2"])),
        "CV_R2_std": float(np.std(results["test_R2"])),
        "CV_RMSE_mean": float(-np.mean(results["test_RMSE"])),
        "CV_RMSE_std": float(np.std(-results["test_RMSE"])),
        "CV_MAE_mean": float(-np.mean(results["test_MAE"])),
        "CV_MAE_std": float(np.std(-results["test_MAE"])),
    }

def create_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
):
    """
    Create Optuna objective.

    Optuna minimizes mean CV RMSE.
    MAE and R² are stored as trial attributes for later analysis.
    """
    def objective(trial: optuna.Trial) -> float:
        params = suggest_gradient_boosting_params(trial)
        pipeline = build_pipeline(X_train, params)

        try:
            cv_metrics = evaluate_cv(
                pipeline=pipeline,
                X=X_train,
                y=y_train,
                cv_folds=cv_folds,
            )
        except Exception as exc:
            trial.set_user_attr("failed_reason", str(exc))
            raise

        trial.set_user_attr("CV_R2_mean", cv_metrics["CV_R2_mean"])
        trial.set_user_attr("CV_R2_std", cv_metrics["CV_R2_std"])
        trial.set_user_attr("CV_RMSE_mean", cv_metrics["CV_RMSE_mean"])
        trial.set_user_attr("CV_RMSE_std", cv_metrics["CV_RMSE_std"])
        trial.set_user_attr("CV_MAE_mean", cv_metrics["CV_MAE_mean"])
        trial.set_user_attr("CV_MAE_std", cv_metrics["CV_MAE_std"])

        return cv_metrics["CV_RMSE_mean"]

    return objective


# =============================================================================
# DEVELOPMENT-MODEL FITTING AND OUTPUTS
# =============================================================================

def fit_best_development_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: Dict[str, Any],
    output_dir: Path,
    cv_folds: int,
) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Fit the best pipeline on the complete development partition.

    The reserved holdout is deliberately not evaluated here.
    """
    pipeline = build_pipeline(X_train, best_params)

    start = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start

    y_train_pred = pipeline.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_train_pred)

    cv_metrics = evaluate_cv(
        pipeline=build_pipeline(X_train, best_params),
        X=X_train,
        y=y_train,
        cv_folds=cv_folds,
    )

    metrics_df = pd.DataFrame(
        [
            {
                "Model": "Gradient Boosting_Optuna",
                "Split": "Development_Train",
                **train_metrics,
                "Training_time_seconds": training_time,
                **cv_metrics,
            }
        ]
    )

    metrics_path = output_dir / "GB_optuna_development_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\nDevelopment evaluation:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nSaved development metrics: {metrics_path}")
    print("Reserved holdout scores were NOT calculated.")

    return pipeline, metrics_df


def save_study_outputs(
    study: optuna.Study,
    best_params: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Save Optuna study history and best hyperparameters."""
    history_df = study.trials_dataframe()
    history_path = output_dir / "GB_optuna_optimization_history.csv"
    history_df.to_csv(history_path, index=False)

    best_params_path = output_dir / "GB_optuna_best_hyperparameters.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(
            best_params,
            f,
            indent=4,
            default=convert_for_json,
        )

    best_summary = {
        "best_trial_number": study.best_trial.number,
        "best_cv_rmse": study.best_value,
        "best_params": best_params,
        "best_trial_user_attrs": study.best_trial.user_attrs,
    }

    best_summary_path = output_dir / "GB_optuna_best_trial_summary.json"
    with open(best_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            best_summary,
            f,
            indent=4,
            default=convert_for_json,
        )

    print(f"\nSaved optimization history: {history_path}")
    print(f"Saved best hyperparameters: {best_params_path}")
    print(f"Saved best trial summary: {best_summary_path}")

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

    idx_train = manifest.index[manifest["Split"] == "train"].to_numpy()
    idx_holdout = manifest.index[manifest["Split"] == "holdout"].to_numpy()

    X_train = X.iloc[idx_train]
    y_train = y.iloc[idx_train]

    return X_train, y_train, idx_train, idx_holdout

# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for Gradient Boosting SFE regression."
    )

    parser.add_argument("--csv", "--data", dest="data", type=str, default=DATASET_PATH)
    parser.add_argument("--target", type=str, default=TARGET_COLUMN)
    parser.add_argument("--id-column", type=str, default=ID_COLUMN)
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help=(
            "Optional development_split_manifest.csv produced by "
            "SFE_Ensemble.py. When supplied, the exact same 80/20 split "
            "is reused. The reserved holdout is not evaluated in this script."
        ),
    )

    return parser.parse_args()


def main():
    np.random.seed(RANDOM_SEED)
    warnings.filterwarnings("default")
    optuna.logging.set_verbosity(optuna.logging.INFO)

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
        X_train, y_train, idx_train, idx_holdout = split_from_manifest(
            X=X,
            y=y,
            ids=ids,
            manifest_path=args.split_manifest,
        )
        print(f"Using fixed split manifest: {args.split_manifest}")
    else:
        stratify_bins = create_target_bins(y)

        X_train, X_holdout, y_train, y_holdout, idx_train, idx_holdout = train_test_split(
            X,
            y,
            np.arange(len(y)),
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True,
            stratify=stratify_bins,
        )

        print(
            "WARNING: no --split-manifest was supplied. "
            "The deterministic 80/20 split was recreated locally."
        )

    print("\nStarting Optuna tuning for Gradient Boosting...")
    print(f"Development samples: {X_train.shape[0]}")
    print(f"Reserved holdout samples: {len(idx_holdout)}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Optuna trials: {args.n_trials}")
    print("Optimization objective: minimize mean CV RMSE")
    print("Reserved holdout scores will NOT be calculated.")

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="GradientBoosting_SFE_Optuna",
    )

    study.optimize(
        create_objective(X_train, y_train, args.cv_folds),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=1,
        show_progress_bar=True,
    )

    best_params = suggest_gradient_boosting_params(study.best_trial)

    print("\nBest Optuna trial:")
    print(f"Trial number: {study.best_trial.number}")
    print(f"Best CV RMSE: {study.best_value:.6f}")
    print(f"Associated CV MAE: {study.best_trial.user_attrs.get('CV_MAE_mean', np.nan):.6f}")
    print(f"Associated CV R²: {study.best_trial.user_attrs.get('CV_R2_mean', np.nan):.6f}")
    print("Best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    save_study_outputs(
        study=study,
        best_params=best_params,
        output_dir=output_dir,
    )

    best_pipeline, metrics_df = fit_best_development_model(
        X_train=X_train,
        y_train=y_train,
        best_params=best_params,
        output_dir=output_dir,
        cv_folds=args.cv_folds,
    )

    model_path = output_dir / "GB_optuna_best_development_pipeline.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"\nSaved best development pipeline: {model_path}")

    print("\nGradient Boosting hyperparameter tuning complete.")
    print("The reserved holdout remains untouched for final model evaluation.")
    print(f"All outputs saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
