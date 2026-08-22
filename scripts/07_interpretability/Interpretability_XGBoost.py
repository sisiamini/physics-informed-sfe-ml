#!/usr/bin/env python3
# Interpretability_XGBoost.py
"""
Interpretability analysis for the frozen final XGBoost SFE regression model.

Purpose:
    Explain how the already-selected full-feature XGBoost model uses the 28
    predictors when making SFE predictions.

This script is post-hoc interpretability only:
    - The XGBoost model family is already frozen.
    - The hyperparameters are already frozen.
    - The original development_split_manifest.csv is reused.
    - The model is fitted on the same 80% development partition.
    - SHAP explanations are calculated on the same reserved 20% holdout.
    - No tuning, model selection, threshold selection, or feature selection is
      performed in this script.
    - Interpretability outputs must not be used to redesign the final model
      after the holdout has already been evaluated.

Interpretability analyses:

1. XGBoost built-in tree importance:
   - gain
   - total gain
   - split count (weight)

2. Holdout SHAP analysis:
   - mean absolute SHAP importance for every feature
   - normalized SHAP importance share
   - aggregate SHAP importance by feature family:
       a. composition
       b. composition-derived descriptors
       c. published SFE-model outputs
   - SHAP beeswarm plot
   - SHAP global bar plot
   - SHAP dependence/scatter plots for the top features
   - local SHAP waterfall plots for representative holdout cases

Important interpretation note:
    SHAP values explain the fitted model, not physical causality. Correlated or
    mathematically related predictors can share or redistribute attribution.
    Therefore, SHAP should be interpreted as model behaviour rather than proof
    that a feature physically causes a change in SFE.

Input:
    CSV dataset containing:
        - 16 composition columns
        - 4 composition-derived descriptor columns
        - 8 published SFE-model output columns
        - SFE_Experimental (mJ/m²)

Output:
    Folder containing:
        - interpretability_feature_importance.csv
        - interpretability_feature_group_shap.csv
        - interpretability_holdout_shap_values.csv
        - interpretability_holdout_predictions.csv
        - interpretability_metadata.json
        - frozen_xgboost_hyperparameters.json
        - shap_beeswarm_holdout.png
        - shap_bar_holdout.png
        - xgboost_gain_importance.png
        - shap_group_importance.png
        - shap_dependence_*.png
        - shap_waterfall_*.png

Run:
    python Interpretability_XGBoost.py \
        --csv "data/output_with_8_sfe_models.csv" \
        --output "model_results/interpretability" \
        --split-manifest "model_results/development_split_manifest.csv"

Optional:
    python Interpretability_XGBoost.py \
        --csv "data/output_with_8_sfe_models.csv" \
        --output "model_results/interpretability" \
        --split-manifest "model_results/development_split_manifest.csv" \
        --top-features 10 \
        --dependence-features 6
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


# =============================================================================
# USER SETTINGS
# =============================================================================

DATASET_PATH = r"output.csv"
TARGET_COLUMN = "SFE_Experimental (mJ/m²)"
ID_COLUMN = "No."

OUTPUT_DIR = "interpretability"

RANDOM_SEED = 42

TOP_FEATURES = 10
DEPENDENCE_FEATURES = 6


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

ALL_FEATURE_COLUMNS = (
    COMPOSITION_COLUMNS
    + DESCRIPTOR_COLUMNS
    + SFE_MODEL_COLUMNS
)


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
        return pd.to_numeric(
            series,
            errors="coerce",
        )

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace(
            r"[^0-9eE+\-\.]",
            "",
            regex=True,
        )
    )

    cleaned = cleaned.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
        }
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    )


def rmse_score(
    y_true,
    y_pred,
) -> float:
    """Version-safe RMSE calculation."""
    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def sanitize_filename(
    name: str,
) -> str:
    """Create a safe filename from a feature name."""
    safe = re.sub(
        r"[^A-Za-z0-9_\-]+",
        "_",
        str(name),
    )
    return safe.strip("_")


def convert_for_json(
    obj: Any,
):
    """Convert numpy objects to JSON-serializable Python objects."""
    if isinstance(
        obj,
        np.integer,
    ):
        return int(obj)

    if isinstance(
        obj,
        np.floating,
    ):
        return float(obj)

    if isinstance(
        obj,
        np.ndarray,
    ):
        return obj.tolist()

    return obj


def get_feature_group(
    feature_name: str,
) -> str:
    """Return the predefined feature family for one predictor."""
    if feature_name in COMPOSITION_COLUMNS:
        return "Composition"

    if feature_name in DESCRIPTOR_COLUMNS:
        return "Composition-derived descriptors"

    if feature_name in SFE_MODEL_COLUMNS:
        return "Published SFE-model outputs"

    raise ValueError(
        f"Unknown feature: {feature_name}"
    )


# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================

def load_dataset(
    dataset_path: str | Path,
) -> pd.DataFrame:
    """Load CSV dataset."""
    path = Path(
        dataset_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(
        path
    )

    print(
        f"Loaded dataset: {path}"
    )
    print(
        f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns"
    )

    return df


def validate_required_columns(
    df: pd.DataFrame,
    target_column: str,
) -> None:
    """Validate the exact predictor and target columns required."""
    required = (
        ALL_FEATURE_COLUMNS
        + [target_column]
    )

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {col}"
                for col in missing
            )
        )


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str,
    id_column: Optional[str],
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Prepare the canonical 28-feature matrix, target, and identifiers.

    Absurd SFE-model values with abs(value) > 1e6 are replaced with NaN.
    Median imputation is fitted later using development data only.
    """
    validate_required_columns(
        df=df,
        target_column=target_column,
    )

    df = df.copy()

    if (
        id_column is not None
        and id_column in df.columns
    ):
        ids = df[
            id_column
        ].copy()
    else:
        ids = pd.Series(
            np.arange(
                1,
                len(df) + 1,
            ),
            name="ID",
        )

    y = clean_numeric_series(
        df[target_column]
    )

    n_missing_target = int(
        y.isna().sum()
    )

    if n_missing_target > 0:
        print(
            f"Dropping {n_missing_target} rows with missing/non-numeric "
            "target values."
        )

    valid_mask = y.notna()

    df = df.loc[
        valid_mask
    ].reset_index(
        drop=True
    )

    y = y.loc[
        valid_mask
    ].reset_index(
        drop=True
    )

    ids = ids.loc[
        valid_mask
    ].reset_index(
        drop=True
    )

    X = df[
        ALL_FEATURE_COLUMNS
    ].copy()

    for col in X.columns:
        X[col] = clean_numeric_series(
            X[col]
        )

    print(
        "Cleaning published SFE-model predictor columns:"
    )

    for col in SFE_MODEL_COLUMNS:
        absurd_mask = (
            X[col].abs() > 1e6
        )

        n_absurd = int(
            absurd_mask.sum()
        )

        if n_absurd > 0:
            print(
                f"  {col}: replacing {n_absurd} absurd values with NaN"
            )

            X.loc[
                absurd_mask,
                col,
            ] = np.nan

    if list(X.columns) != ALL_FEATURE_COLUMNS:
        raise RuntimeError(
            "Feature order does not match the canonical 28-feature order."
        )

    print(
        f"Predictor matrix shape: {X.shape}"
    )
    print(
        f"Target vector shape: {y.shape}"
    )

    return (
        X,
        y,
        ids,
    )


# =============================================================================
# FROZEN XGBOOST MODEL
# =============================================================================

def get_frozen_xgboost_hyperparameters() -> Dict[str, Any]:
    """
    Return the already-selected frozen XGBoost hyperparameters.

    These parameters must not be changed during interpretability analysis.
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


def build_preprocessor() -> ColumnTransformer:
    """
    Build median-imputation preprocessing for the canonical 28 predictors.

    The preprocessor is fitted on development data only.
    """
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "median_imputer",
                            SimpleImputer(
                                strategy="median",
                            ),
                        ),
                    ]
                ),
                ALL_FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_frozen_pipeline() -> Pipeline:
    """Build preprocessing + frozen XGBoost pipeline."""
    preprocessor = build_preprocessor()

    model = XGBRegressor(
        **get_frozen_xgboost_hyperparameters()
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# =============================================================================
# FIXED DEVELOPMENT / HOLDOUT SPLIT
# =============================================================================

def load_fixed_split_manifest(
    y: pd.Series,
    ids: pd.Series,
    manifest_path: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Load and validate the exact development/holdout split used previously.

    The same manifest must be reused so the interpretation corresponds to the
    already-evaluated frozen final model.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Development split manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path
    )

    required_columns = {
        "Row_Index",
        "Split",
    }

    missing = required_columns.difference(
        manifest.columns
    )

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
    ).reset_index(
        drop=True
    )

    expected_indices = np.arange(
        len(y)
    )

    if not np.array_equal(
        manifest["Row_Index"].to_numpy(),
        expected_indices,
    ):
        raise ValueError(
            "Split manifest Row_Index must match the current dataset order exactly."
        )

    labels = set(
        manifest[
            "Split"
        ].astype(
            str
        )
    )

    if labels != {
        "train",
        "holdout",
    }:
        raise ValueError(
            "Split manifest must contain exactly the labels 'train' and 'holdout'."
        )

    if "ID" in manifest.columns:
        manifest_ids = manifest[
            "ID"
        ].astype(
            str
        ).to_numpy()

        current_ids = ids.astype(
            str
        ).to_numpy()

        if not np.array_equal(
            manifest_ids,
            current_ids,
        ):
            raise ValueError(
                "Split manifest ID values do not match the current dataset order."
            )

    idx_development = manifest.index[
        manifest[
            "Split"
        ].astype(
            str
        ) == "train"
    ].to_numpy()

    idx_holdout = manifest.index[
        manifest[
            "Split"
        ].astype(
            str
        ) == "holdout"
    ].to_numpy()

    if (
        len(idx_development) == 0
        or len(idx_holdout) == 0
    ):
        raise ValueError(
            "Split manifest must contain non-empty train and holdout partitions."
        )

    return (
        idx_development,
        idx_holdout,
    )


# =============================================================================
# FIT FROZEN MODEL AND PREPARE EXPLANATION DATA
# =============================================================================

def fit_frozen_model(
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    idx_development: np.ndarray,
    idx_holdout: np.ndarray,
) -> Tuple[
    Pipeline,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    np.ndarray,
]:
    """
    Fit the frozen model on development data and reproduce holdout predictions.

    This is not a new model-selection or evaluation stage. It reproduces the
    already-fixed final model so that its holdout predictions can be explained.
    """
    pipeline = build_frozen_pipeline()

    X_development = X.iloc[
        idx_development
    ].copy()

    y_development = y.iloc[
        idx_development
    ].copy()

    X_holdout = X.iloc[
        idx_holdout
    ].copy()

    y_holdout = y.iloc[
        idx_holdout
    ].copy()

    ids_holdout = ids.iloc[
        idx_holdout
    ].copy()

    pipeline.fit(
        X_development,
        y_development,
    )

    holdout_predictions = pipeline.predict(
        X_holdout
    )

    return (
        pipeline,
        X_development,
        X_holdout,
        y_development,
        y_holdout,
        ids_holdout,
        holdout_predictions,
    )


def get_processed_dataframes(
    pipeline: Pipeline,
    X_development: pd.DataFrame,
    X_holdout: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Apply the fitted development-only preprocessor and restore feature names.
    """
    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    transformed_development = preprocessor.transform(
        X_development
    )

    transformed_holdout = preprocessor.transform(
        X_holdout
    )

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    if feature_names != ALL_FEATURE_COLUMNS:
        raise RuntimeError(
            "Transformed feature names do not match the canonical 28-feature order."
        )

    X_development_processed = pd.DataFrame(
        transformed_development,
        columns=feature_names,
        index=X_development.index,
    )

    X_holdout_processed = pd.DataFrame(
        transformed_holdout,
        columns=feature_names,
        index=X_holdout.index,
    )

    return (
        X_development_processed,
        X_holdout_processed,
    )


# =============================================================================
# XGBOOST BUILT-IN FEATURE IMPORTANCE
# =============================================================================

def map_booster_scores(
    model: XGBRegressor,
    feature_names: List[str],
    importance_type: str,
) -> Dict[str, float]:
    """
    Map XGBoost Booster f0/f1/... importance keys back to feature names.
    """
    raw_scores = model.get_booster().get_score(
        importance_type=importance_type
    )

    mapped = {
        feature_name: 0.0
        for feature_name in feature_names
    }

    for key, value in raw_scores.items():
        if key in mapped:
            mapped[key] = float(
                value
            )
            continue

        match = re.fullmatch(
            r"f(\d+)",
            str(key),
        )

        if match is None:
            warnings.warn(
                f"Could not map XGBoost importance key '{key}'."
            )
            continue

        position = int(
            match.group(1)
        )

        if (
            position < 0
            or position >= len(feature_names)
        ):
            warnings.warn(
                f"XGBoost importance index out of range: {key}"
            )
            continue

        mapped[
            feature_names[
                position
            ]
        ] = float(
            value
        )

    return mapped


def calculate_xgboost_importance(
    pipeline: Pipeline,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Calculate gain, total gain, and split-count importance.
    """
    model = pipeline.named_steps[
        "model"
    ]

    gain = map_booster_scores(
        model=model,
        feature_names=feature_names,
        importance_type="gain",
    )

    total_gain = map_booster_scores(
        model=model,
        feature_names=feature_names,
        importance_type="total_gain",
    )

    weight = map_booster_scores(
        model=model,
        feature_names=feature_names,
        importance_type="weight",
    )

    rows = []

    for feature_name in feature_names:
        rows.append(
            {
                "Feature": feature_name,
                "Feature_Group": get_feature_group(
                    feature_name
                ),
                "XGB_Gain": gain[
                    feature_name
                ],
                "XGB_Total_Gain": total_gain[
                    feature_name
                ],
                "XGB_Split_Count": weight[
                    feature_name
                ],
            }
        )

    importance_df = pd.DataFrame(
        rows
    )

    gain_sum = float(
        importance_df[
            "XGB_Gain"
        ].sum()
    )

    total_gain_sum = float(
        importance_df[
            "XGB_Total_Gain"
        ].sum()
    )

    if gain_sum > 0:
        importance_df[
            "XGB_Gain_Share_percent"
        ] = (
            importance_df[
                "XGB_Gain"
            ]
            / gain_sum
            * 100.0
        )
    else:
        importance_df[
            "XGB_Gain_Share_percent"
        ] = 0.0

    if total_gain_sum > 0:
        importance_df[
            "XGB_Total_Gain_Share_percent"
        ] = (
            importance_df[
                "XGB_Total_Gain"
            ]
            / total_gain_sum
            * 100.0
        )
    else:
        importance_df[
            "XGB_Total_Gain_Share_percent"
        ] = 0.0

    return importance_df


# =============================================================================
# SHAP ANALYSIS
# =============================================================================

def calculate_holdout_shap(
    pipeline: Pipeline,
    X_holdout_processed: pd.DataFrame,
) -> shap.Explanation:
    """
    Calculate Tree SHAP values for the reserved holdout observations.

    tree_path_dependent is used explicitly so no separate background sample is
    required. For XGBoost regression, raw model output corresponds to the
    regression prediction scale.
    """
    model = pipeline.named_steps[
        "model"
    ]

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
        feature_names=list(
            X_holdout_processed.columns
        ),
    )

    explanation = explainer(
        X_holdout_processed
    )

    return explanation


def validate_shap_additivity(
    explanation: shap.Explanation,
    predictions: np.ndarray,
) -> Dict[str, float]:
    """
    Verify that base value + SHAP values reconstruct model predictions.
    """
    values = np.asarray(
        explanation.values,
        dtype=float,
    )

    base_values = np.asarray(
        explanation.base_values,
        dtype=float,
    )

    if base_values.ndim == 0:
        base_values = np.full(
            len(values),
            float(base_values),
        )

    base_values = base_values.reshape(
        -1
    )

    if len(base_values) == 1:
        base_values = np.full(
            len(values),
            float(
                base_values[0]
            ),
        )

    reconstructed = (
        base_values
        + values.sum(
            axis=1
        )
    )

    absolute_error = np.abs(
        reconstructed
        - np.asarray(
            predictions,
            dtype=float,
        )
    )

    return {
        "SHAP_Additivity_Max_Absolute_Error": float(
            np.max(
                absolute_error
            )
        ),
        "SHAP_Additivity_Mean_Absolute_Error": float(
            np.mean(
                absolute_error
            )
        ),
    }


def calculate_shap_importance(
    explanation: shap.Explanation,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Calculate global holdout SHAP importance for every feature.
    """
    shap_values = np.asarray(
        explanation.values,
        dtype=float,
    )

    mean_abs = np.mean(
        np.abs(
            shap_values
        ),
        axis=0,
    )

    median_abs = np.median(
        np.abs(
            shap_values
        ),
        axis=0,
    )

    mean_signed = np.mean(
        shap_values,
        axis=0,
    )

    total_mean_abs = float(
        np.sum(
            mean_abs
        )
    )

    if total_mean_abs > 0:
        share_percent = (
            mean_abs
            / total_mean_abs
            * 100.0
        )
    else:
        share_percent = np.zeros_like(
            mean_abs
        )

    shap_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Feature_Group": [
                get_feature_group(
                    feature_name
                )
                for feature_name in feature_names
            ],
            "SHAP_Mean_Absolute": mean_abs,
            "SHAP_Median_Absolute": median_abs,
            "SHAP_Mean_Signed": mean_signed,
            "SHAP_Importance_Share_percent": share_percent,
        }
    )

    shap_df = shap_df.sort_values(
        "SHAP_Mean_Absolute",
        ascending=False,
    ).reset_index(
        drop=True
    )

    shap_df.insert(
        0,
        "SHAP_Rank",
        np.arange(
            1,
            len(shap_df) + 1,
        ),
    )

    return shap_df


def calculate_group_shap_importance(
    shap_importance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate mean absolute SHAP importance by predefined feature family.

    This aggregation is descriptive and is not a causal decomposition.
    """
    group_df = (
        shap_importance
        .groupby(
            "Feature_Group",
            as_index=False,
            sort=False,
        )
        .agg(
            Feature_Count=(
                "Feature",
                "count",
            ),
            Aggregate_Mean_Absolute_SHAP=(
                "SHAP_Mean_Absolute",
                "sum",
            ),
        )
    )

    total = float(
        group_df[
            "Aggregate_Mean_Absolute_SHAP"
        ].sum()
    )

    if total > 0:
        group_df[
            "SHAP_Group_Share_percent"
        ] = (
            group_df[
                "Aggregate_Mean_Absolute_SHAP"
            ]
            / total
            * 100.0
        )
    else:
        group_df[
            "SHAP_Group_Share_percent"
        ] = 0.0

    group_df = group_df.sort_values(
        "Aggregate_Mean_Absolute_SHAP",
        ascending=False,
    ).reset_index(
        drop=True
    )

    group_df.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(group_df) + 1,
        ),
    )

    return group_df


def merge_importance_tables(
    xgb_importance: pd.DataFrame,
    shap_importance: pd.DataFrame,
) -> pd.DataFrame:
    """Merge XGBoost tree importance and holdout SHAP importance."""
    merged = shap_importance.merge(
        xgb_importance,
        on=[
            "Feature",
            "Feature_Group",
        ],
        how="left",
        validate="one_to_one",
    )

    return merged


def create_holdout_shap_table(
    explanation: shap.Explanation,
    X_holdout_processed: pd.DataFrame,
    idx_holdout: np.ndarray,
    ids_holdout: pd.Series,
    y_holdout: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Save observation-level SHAP values together with final holdout predictions.
    """
    shap_values = np.asarray(
        explanation.values,
        dtype=float,
    )

    prediction_array = np.asarray(
        predictions,
        dtype=float,
    )

    base_values = np.asarray(
        explanation.base_values,
        dtype=float,
    )

    if base_values.ndim == 0:
        base_values = np.full(
            len(shap_values),
            float(base_values),
        )

    base_values = base_values.reshape(
        -1
    )

    if len(base_values) == 1:
        base_values = np.full(
            len(shap_values),
            float(
                base_values[0]
            ),
        )

    table = pd.DataFrame(
        {
            "Dataset_Row_Index": idx_holdout,
            "ID": ids_holdout.to_numpy(),
            "Actual": y_holdout.to_numpy(
                dtype=float
            ),
            "Predicted": prediction_array,
            "Residual": (
                y_holdout.to_numpy(
                    dtype=float
                )
                - prediction_array
            ),
            "Absolute_Residual": np.abs(
                y_holdout.to_numpy(
                    dtype=float
                )
                - prediction_array
            ),
            "SHAP_Base_Value": base_values,
        }
    )

    for position, feature_name in enumerate(
        X_holdout_processed.columns
    ):
        table[
            f"Feature::{feature_name}"
        ] = X_holdout_processed[
            feature_name
        ].to_numpy(
            dtype=float
        )

        table[
            f"SHAP::{feature_name}"
        ] = shap_values[
            :,
            position,
        ]

    return table


# =============================================================================
# PLOTS
# =============================================================================

def save_shap_beeswarm(
    explanation: shap.Explanation,
    output_dir: Path,
    max_display: int,
) -> Path:
    """Save the global SHAP beeswarm plot."""
    path = (
        output_dir
        / "shap_beeswarm_holdout.png"
    )

    plt.figure()

    shap.plots.beeswarm(
        explanation,
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return path


def save_shap_bar(
    explanation: shap.Explanation,
    output_dir: Path,
    max_display: int,
) -> Path:
    """Save the global mean-absolute-SHAP bar plot."""
    path = (
        output_dir
        / "shap_bar_holdout.png"
    )

    plt.figure()

    shap.plots.bar(
        explanation,
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return path


def save_xgboost_gain_plot(
    importance_df: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> Path:
    """Save a simple XGBoost gain-importance plot."""
    path = (
        output_dir
        / "xgboost_gain_importance.png"
    )

    plot_df = (
        importance_df
        .sort_values(
            "XGB_Gain",
            ascending=False,
        )
        .head(
            top_n
        )
        .iloc[
            ::-1
        ]
    )

    plt.figure(
        figsize=(
            8,
            max(
                5,
                0.38 * len(plot_df),
            ),
        )
    )

    plt.barh(
        plot_df["Feature"],
        plot_df["XGB_Gain"],
    )

    plt.xlabel(
        "Average gain"
    )
    plt.ylabel(
        "Feature"
    )
    plt.title(
        "XGBoost built-in feature importance"
    )
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return path


def save_group_shap_plot(
    group_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Save aggregate mean-absolute-SHAP importance by feature family."""
    path = (
        output_dir
        / "shap_group_importance.png"
    )

    plot_df = group_df.sort_values(
        "Aggregate_Mean_Absolute_SHAP",
        ascending=True,
    )

    plt.figure(
        figsize=(
            8,
            4.5,
        )
    )

    plt.barh(
        plot_df[
            "Feature_Group"
        ],
        plot_df[
            "Aggregate_Mean_Absolute_SHAP"
        ],
    )

    plt.xlabel(
        "Aggregate mean |SHAP|"
    )
    plt.ylabel(
        "Feature family"
    )
    plt.title(
        "Holdout SHAP importance by feature family"
    )
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return path


def save_dependence_plots(
    explanation: shap.Explanation,
    shap_importance: pd.DataFrame,
    output_dir: Path,
    n_features: int,
) -> List[Path]:
    """
    Save SHAP scatter/dependence plots for the most important holdout features.
    """
    paths = []

    top_features = (
        shap_importance[
            "Feature"
        ]
        .head(
            n_features
        )
        .tolist()
    )

    for rank, feature_name in enumerate(
        top_features,
        start=1,
    ):
        safe_name = sanitize_filename(
            feature_name
        )

        path = (
            output_dir
            / f"shap_dependence_{rank:02d}_{safe_name}.png"
        )

        plt.figure()

        shap.plots.scatter(
            explanation[
                :,
                feature_name,
            ],
            show=False,
        )

        plt.title(
            f"SHAP dependence: {feature_name}"
        )
        plt.tight_layout()

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        paths.append(
            path
        )

    return paths


def select_representative_holdout_cases(
    y_holdout: pd.Series,
    predictions: np.ndarray,
) -> Dict[str, int]:
    """
    Select three representative holdout observations by absolute residual:
        - lowest-error case
        - median-error case
        - highest-error case
    """
    residuals = np.abs(
        y_holdout.to_numpy(
            dtype=float
        )
        - np.asarray(
            predictions,
            dtype=float,
        )
    )

    order = np.argsort(
        residuals
    )

    best_index = int(
        order[
            0
        ]
    )

    median_index = int(
        order[
            len(order) // 2
        ]
    )

    worst_index = int(
        order[
            -1
        ]
    )

    return {
        "lowest_error": best_index,
        "median_error": median_index,
        "highest_error": worst_index,
    }


def save_waterfall_plots(
    explanation: shap.Explanation,
    y_holdout: pd.Series,
    predictions: np.ndarray,
    ids_holdout: pd.Series,
    output_dir: Path,
    max_display: int,
) -> List[Path]:
    """Save local SHAP waterfall plots for representative holdout observations."""
    case_indices = select_representative_holdout_cases(
        y_holdout=y_holdout,
        predictions=predictions,
    )

    paths = []

    for case_name, position in case_indices.items():
        path = (
            output_dir
            / f"shap_waterfall_{case_name}.png"
        )

        plt.figure()

        shap.plots.waterfall(
            explanation[
                position
            ],
            max_display=max_display,
            show=False,
        )

        actual = float(
            y_holdout.iloc[
                position
            ]
        )

        predicted = float(
            predictions[
                position
            ]
        )

        sample_id = ids_holdout.iloc[
            position
        ]

        plt.title(
            f"{case_name.replace('_', ' ').title()} holdout case "
            f"(ID={sample_id}, actual={actual:.2f}, predicted={predicted:.2f})"
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        paths.append(
            path
        )

    return paths


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def save_frozen_hyperparameters(
    output_dir: Path,
) -> Path:
    """Save the frozen XGBoost hyperparameters used for interpretation."""
    path = (
        output_dir
        / "frozen_xgboost_hyperparameters.json"
    )

    payload = {
        "model": "XGBoost",
        "status": "Frozen before interpretability analysis",
        "random_seed": RANDOM_SEED,
        "hyperparameters": get_frozen_xgboost_hyperparameters(),
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


def save_metadata(
    output_dir: Path,
    idx_development: np.ndarray,
    idx_holdout: np.ndarray,
    y_holdout: pd.Series,
    predictions: np.ndarray,
    shap_additivity: Dict[str, float],
    top_feature: str,
    top_group: str,
) -> Path:
    """Save reproducibility and interpretation metadata."""
    path = (
        output_dir
        / "interpretability_metadata.json"
    )

    payload = {
        "analysis": "Post-hoc interpretability of frozen final XGBoost model",
        "development_n": len(
            idx_development
        ),
        "holdout_n": len(
            idx_holdout
        ),
        "feature_count": len(
            ALL_FEATURE_COLUMNS
        ),
        "shap_evaluation_data": "Reserved 20% holdout",
        "shap_feature_perturbation": "tree_path_dependent",
        "holdout_metrics_reproduced_for_consistency": {
            "R2": float(
                r2_score(
                    y_holdout,
                    predictions,
                )
            ),
            "RMSE": rmse_score(
                y_holdout,
                predictions,
            ),
            "MAE": float(
                mean_absolute_error(
                    y_holdout,
                    predictions,
                )
            ),
        },
        "top_holdout_shap_feature": top_feature,
        "top_holdout_shap_feature_group": top_group,
        **shap_additivity,
        "interpretation_warning": (
            "SHAP and tree importance describe fitted-model behaviour. "
            "They do not establish physical causality, and correlated or "
            "derived predictors may share attribution."
        ),
        "model_change_permitted": False,
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc XGBoost gain and SHAP interpretation for the already-frozen "
            "final SFE prediction model."
        )
    )

    parser.add_argument(
        "--csv",
        "--data",
        dest="data",
        type=Path,
        default=Path(
            DATASET_PATH
        ),
        help="Input modelling CSV dataset.",
    )

    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path(
            OUTPUT_DIR
        ),
        help="Output directory for interpretability results.",
    )

    parser.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
        help=(
            "The same development_split_manifest.csv used throughout model "
            "selection and final evaluation."
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
        "--top-features",
        type=int,
        default=TOP_FEATURES,
        help="Maximum number of features displayed in global importance plots.",
    )

    parser.add_argument(
        "--dependence-features",
        type=int,
        default=DEPENDENCE_FEATURES,
        help="Number of top-SHAP features receiving individual dependence plots.",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the complete interpretability analysis."""
    np.random.seed(
        RANDOM_SEED
    )

    warnings.filterwarnings(
        "default"
    )

    args = parse_args()

    if args.top_features < 1:
        raise ValueError(
            "--top-features must be at least 1."
        )

    if args.dependence_features < 0:
        raise ValueError(
            "--dependence-features cannot be negative."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset(
        args.data
    )

    X, y, ids = prepare_features_and_target(
        df=df,
        target_column=args.target,
        id_column=args.id_column,
    )

    (
        idx_development,
        idx_holdout,
    ) = load_fixed_split_manifest(
        y=y,
        ids=ids,
        manifest_path=args.split_manifest,
    )

    print("\nInterpretability settings:")
    print(
        f"Development samples: {len(idx_development)}"
    )
    print(
        f"Holdout samples explained: {len(idx_holdout)}"
    )
    print(
        f"Predictors: {len(ALL_FEATURE_COLUMNS)}"
    )
    print(
        "Model: frozen final XGBoost"
    )
    print(
        "SHAP evaluation data: reserved 20% holdout"
    )
    print(
        "No tuning, feature selection, or model changes are performed."
    )

    (
        pipeline,
        X_development,
        X_holdout,
        y_development,
        y_holdout,
        ids_holdout,
        holdout_predictions,
    ) = fit_frozen_model(
        X=X,
        y=y,
        ids=ids,
        idx_development=idx_development,
        idx_holdout=idx_holdout,
    )

    print("\nReproduced frozen holdout metrics:")
    print(
        f"R²   = {r2_score(y_holdout, holdout_predictions):.6f}"
    )
    print(
        f"RMSE = {rmse_score(y_holdout, holdout_predictions):.6f} mJ/m²"
    )
    print(
        f"MAE  = {mean_absolute_error(y_holdout, holdout_predictions):.6f} mJ/m²"
    )

    (
        X_development_processed,
        X_holdout_processed,
    ) = get_processed_dataframes(
        pipeline=pipeline,
        X_development=X_development,
        X_holdout=X_holdout,
    )

    # -------------------------------------------------------------------------
    # 1. XGBoost built-in importance
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("1. XGBOOST BUILT-IN TREE IMPORTANCE")
    print("=" * 110)

    xgb_importance = calculate_xgboost_importance(
        pipeline=pipeline,
        feature_names=ALL_FEATURE_COLUMNS,
    )

    # -------------------------------------------------------------------------
    # 2. Holdout SHAP analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("2. HOLDOUT SHAP ANALYSIS")
    print("=" * 110)

    explanation = calculate_holdout_shap(
        pipeline=pipeline,
        X_holdout_processed=X_holdout_processed,
    )

    shap_additivity = validate_shap_additivity(
        explanation=explanation,
        predictions=holdout_predictions,
    )

    print(
        "SHAP additivity max absolute error: "
        f"{shap_additivity['SHAP_Additivity_Max_Absolute_Error']:.8f}"
    )

    shap_importance = calculate_shap_importance(
        explanation=explanation,
        feature_names=ALL_FEATURE_COLUMNS,
    )

    group_importance = calculate_group_shap_importance(
        shap_importance=shap_importance,
    )

    combined_importance = merge_importance_tables(
        xgb_importance=xgb_importance,
        shap_importance=shap_importance,
    )

    # -------------------------------------------------------------------------
    # 3. Observation-level SHAP table
    # -------------------------------------------------------------------------
    shap_table = create_holdout_shap_table(
        explanation=explanation,
        X_holdout_processed=X_holdout_processed,
        idx_holdout=idx_holdout,
        ids_holdout=ids_holdout.reset_index(
            drop=True
        ),
        y_holdout=y_holdout.reset_index(
            drop=True
        ),
        predictions=holdout_predictions,
    )

    holdout_predictions_df = pd.DataFrame(
        {
            "Dataset_Row_Index": idx_holdout,
            "ID": ids_holdout.to_numpy(),
            "Actual": y_holdout.to_numpy(
                dtype=float
            ),
            "Predicted": holdout_predictions,
            "Residual": (
                y_holdout.to_numpy(
                    dtype=float
                )
                - holdout_predictions
            ),
            "Absolute_Residual": np.abs(
                y_holdout.to_numpy(
                    dtype=float
                )
                - holdout_predictions
            ),
        }
    )

    # -------------------------------------------------------------------------
    # 4. Save tables
    # -------------------------------------------------------------------------
    feature_importance_path = (
        args.output_dir
        / "interpretability_feature_importance.csv"
    )

    group_importance_path = (
        args.output_dir
        / "interpretability_feature_group_shap.csv"
    )

    shap_values_path = (
        args.output_dir
        / "interpretability_holdout_shap_values.csv"
    )

    predictions_path = (
        args.output_dir
        / "interpretability_holdout_predictions.csv"
    )

    combined_importance.to_csv(
        feature_importance_path,
        index=False,
    )

    group_importance.to_csv(
        group_importance_path,
        index=False,
    )

    shap_table.to_csv(
        shap_values_path,
        index=False,
    )

    holdout_predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    frozen_params_path = save_frozen_hyperparameters(
        output_dir=args.output_dir,
    )

    # -------------------------------------------------------------------------
    # 5. Save plots
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("3. SAVING INTERPRETABILITY PLOTS")
    print("=" * 110)

    beeswarm_path = save_shap_beeswarm(
        explanation=explanation,
        output_dir=args.output_dir,
        max_display=min(
            args.top_features,
            len(ALL_FEATURE_COLUMNS),
        ),
    )

    shap_bar_path = save_shap_bar(
        explanation=explanation,
        output_dir=args.output_dir,
        max_display=min(
            args.top_features,
            len(ALL_FEATURE_COLUMNS),
        ),
    )

    gain_plot_path = save_xgboost_gain_plot(
        importance_df=combined_importance,
        output_dir=args.output_dir,
        top_n=min(
            args.top_features,
            len(ALL_FEATURE_COLUMNS),
        ),
    )

    group_plot_path = save_group_shap_plot(
        group_df=group_importance,
        output_dir=args.output_dir,
    )

    dependence_paths = save_dependence_plots(
        explanation=explanation,
        shap_importance=shap_importance,
        output_dir=args.output_dir,
        n_features=min(
            args.dependence_features,
            len(ALL_FEATURE_COLUMNS),
        ),
    )

    waterfall_paths = save_waterfall_plots(
        explanation=explanation,
        y_holdout=y_holdout.reset_index(
            drop=True
        ),
        predictions=holdout_predictions,
        ids_holdout=ids_holdout.reset_index(
            drop=True
        ),
        output_dir=args.output_dir,
        max_display=min(
            args.top_features,
            len(ALL_FEATURE_COLUMNS),
        ),
    )

    top_feature = str(
        shap_importance.iloc[
            0
        ][
            "Feature"
        ]
    )

    top_group = str(
        group_importance.iloc[
            0
        ][
            "Feature_Group"
        ]
    )

    metadata_path = save_metadata(
        output_dir=args.output_dir,
        idx_development=idx_development,
        idx_holdout=idx_holdout,
        y_holdout=y_holdout,
        predictions=holdout_predictions,
        shap_additivity=shap_additivity,
        top_feature=top_feature,
        top_group=top_group,
    )

    # -------------------------------------------------------------------------
    # 6. Display concise summaries
    # -------------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("HOLDOUT SHAP FEATURE IMPORTANCE")
    print("=" * 110)

    display_columns = [
        "SHAP_Rank",
        "Feature",
        "Feature_Group",
        "SHAP_Mean_Absolute",
        "SHAP_Importance_Share_percent",
        "XGB_Gain",
    ]

    print(
        combined_importance[
            display_columns
        ]
        .head(
            args.top_features
        )
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n" + "=" * 110)
    print("HOLDOUT SHAP FEATURE-GROUP IMPORTANCE")
    print("=" * 110)

    print(
        group_importance.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nInterpretation reminder:")
    print(
        "  SHAP values describe how the frozen model formed its predictions."
    )
    print(
        "  They do not establish causal or mechanistic effects of alloy variables."
    )
    print(
        "  Correlated and derived predictors can share attribution."
    )
    print(
        "  These outputs must not be used to retune or reselect the final model."
    )

    print(f"\nSaved feature importance: {feature_importance_path}")
    print(f"Saved feature-group SHAP importance: {group_importance_path}")
    print(f"Saved holdout SHAP values: {shap_values_path}")
    print(f"Saved holdout predictions: {predictions_path}")
    print(f"Saved frozen hyperparameters: {frozen_params_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved SHAP beeswarm: {beeswarm_path}")
    print(f"Saved SHAP bar plot: {shap_bar_path}")
    print(f"Saved XGBoost gain plot: {gain_plot_path}")
    print(f"Saved SHAP group plot: {group_plot_path}")

    for path in dependence_paths:
        print(
            f"Saved SHAP dependence plot: {path}"
        )

    for path in waterfall_paths:
        print(
            f"Saved SHAP waterfall plot: {path}"
        )

    print(
        f"\nAll outputs saved in: {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
