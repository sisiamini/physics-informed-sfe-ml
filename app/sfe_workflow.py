"""Prediction workflow for the final SFE Streamlit app.

This module keeps the business logic separate from the UI:
1. parse and validate user-provided wt.% compositions;
2. assess the composition against the final applicability-domain rule;
3. run the corrected descriptor script;
4. run the Curtze-free SFE master script to calculate eight published SFE features;
5. load the frozen full-data XGBoost production pipeline and predict SFE.

Important:
- The production model is the final full-data refit created only after model
  selection/evaluation was complete.
- The applicability-domain rule is the final full-data operational refit.
- Reliability evidence shown by the app comes from nested exact-composition-
  grouped validation, not from the full-fit internal domain reliability.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import math
import subprocess
import sys
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

SCRIPT_DIR = REPO_ROOT / "scripts" / "01_feature_generation"
MODEL_DIR = REPO_ROOT / "models" / "production"

DESCRIPTOR_SCRIPT = "SFE_with_Descriptors.py"
SFE_MASTER_SCRIPT = "SFE_Master.py"

MODEL_PATH = MODEL_DIR / "final_xgboost_full_data_pipeline.joblib"
DOMAIN_REFERENCE_PATH = MODEL_DIR / "composition_domain_reference.joblib"
DOMAIN_SUMMARY_PATH = MODEL_DIR / "domain_summary.json"

ELEMENTS: list[str] = [
    "Mn", "Al", "C", "Ni", "Cr", "Si", "Cu", "Mo",
    "Nb", "V", "Ti", "Co", "N", "P", "S", "Fe",
]

COMPOSITION_COLUMNS: dict[str, str] = {
    "Fe": "Fe (%wt)", "Mn": "Mn (%wt)", "Al": "Al (%wt)",
    "C": "C (%wt)", "Ni": "Ni (%wt)", "Cr": "Cr (%wt)",
    "Si": "Si (%wt)", "Cu": "Cu (%wt)", "Mo": "Mo (%wt)",
    "Nb": "Nb (%wt)", "V": "V (%wt)", "Ti": "Ti (%wt)",
    "Co": "Co (%wt)", "N": "N (%wt)", "P": "P (%wt)",
    "S": "S (%wt)",
}

# Final operational visible ranges from the full-data applicability-domain refit.
# They are UI guidance only. The complete domain also requires the kNN dense-core check.
DOMAIN_RANGES: dict[str, tuple[float, float]] = {
    "Fe": (52.1098, 82.9392),
    "Mn": (0.0, 25.62),
    "Al": (0.0, 3.258000000000001),
    "C": (0.0, 0.6164000000000001),
    "Ni": (0.0, 24.452),
    "Cr": (0.0, 24.682),
    "Si": (0.0, 2.89),
    "Cu": (0.0, 0.056600000000000594),
    "Mo": (0.0, 2.7),
    "Nb": (0.0, 0.0),
    "V": (0.0, 0.0),
    "Ti": (0.0, 0.0),
    "Co": (0.0, 0.0),
    "N": (0.0, 0.48),
    "P": (0.0, 0.03),
    "S": (0.0, 0.011600000000000023),
}

DESCRIPTOR_COLUMNS: list[str] = [
    "R_mean (pm)",
    "R_std (pm)",
    "Δχ",
    "VEC_mean",
]

SFE_FEATURE_COLUMNS: list[str] = [
    "SFE_Akbari (mJ/m²)",
    "SFE_Allain (mJ/m²)",
    "SFE_Dumay (mJ/m²)",
    "SFE_Elaraby (mJ/m²)",
    "SFE_Wang (mJ/m²)",
    "SFE_Xiong (mJ/m²)",
    "SFE_Yoo (mJ/m²)",
    "SFE_Zambrano (mJ/m²)",
]

TOTAL_TARGET = 100.0
TOTAL_TOLERANCE = 1e-6
ABSURD_SFE_THRESHOLD = 1e6


class InputValidationError(ValueError):
    """Raised when the user input violates basic composition rules."""


class PredictionWorkflowError(RuntimeError):
    """Raised when feature generation, domain assessment, or prediction fails."""


@dataclass(frozen=True)
class DomainAssessment:
    inside_visible_ranges: bool
    inside_knn_dense_core: bool
    inside_applicability_domain: bool
    knn_distance: float
    knn_threshold: float
    visible_range_violations: tuple[str, ...]


@dataclass(frozen=True)
class PredictionResult:
    prediction: float
    total: float
    descriptors: dict[str, float]
    sfe_features: dict[str, float]
    model_input: pd.DataFrame
    full_feature_table: pd.DataFrame
    domain: DomainAssessment
    domain_summary: dict
    descriptor_script_output: str
    sfe_master_script_output: str


def parse_composition(raw_values: dict[str, str | float | int | None]) -> dict[str, float]:
    """Convert user input to floats. Blank non-Fe values are 0; blank Fe is balance."""
    composition: dict[str, float] = {}

    for element in ELEMENTS:
        raw = raw_values.get(element, "")
        if raw is None or str(raw).strip() == "":
            if element == "Fe":
                composition[element] = TOTAL_TARGET - sum(composition.values())
            else:
                composition[element] = 0.0
            continue

        try:
            value = float(str(raw).strip())
        except ValueError as exc:
            raise InputValidationError(
                f"Invalid value for {element}: {raw!r}. Enter a numeric wt.% value or leave it blank."
            ) from exc

        if not math.isfinite(value):
            raise InputValidationError(
                f"Invalid value for {element}: {raw!r}. Enter a finite numeric wt.% value."
            )

        composition[element] = value

    return composition


def validate_composition(composition: dict[str, float]) -> float:
    """Validate arithmetic only; applicability-domain limits are not hard rejection limits."""
    missing = [element for element in ELEMENTS if element not in composition]
    if missing:
        raise InputValidationError("Missing composition values for: " + ", ".join(missing))

    for element in ELEMENTS:
        value = composition[element]
        if not math.isfinite(value):
            raise InputValidationError(f"{element} must be finite.")
        if value < 0.0 or value > 100.0:
            raise InputValidationError(
                f"{element} must be between 0 and 100 wt.%; entered {value:g} wt.%."
            )

    total = sum(composition.values())
    if not math.isclose(total, TOTAL_TARGET, rel_tol=0.0, abs_tol=TOTAL_TOLERANCE):
        raise InputValidationError(
            f"Total composition must equal 100 wt.%. Current total is {total:.6f} wt.%."
        )

    return total


def _run_script(script_name: str, input_csv: Path, output_csv: Path) -> str:
    """Run one feature-generation command-line script."""
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        raise PredictionWorkflowError(f"Required script not found: {script_path}")

    completed = subprocess.run(
        [sys.executable, str(script_path), "--csv", str(input_csv), "--output", str(output_csv)],
        cwd=str(SCRIPT_DIR),
        text=True,
        capture_output=True,
        check=False,
    )

    output = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
    if completed.returncode != 0:
        raise PredictionWorkflowError(
            f"{script_name} failed with exit code {completed.returncode}.\n{output}"
        )
    if not output_csv.exists():
        raise PredictionWorkflowError(
            f"{script_name} did not create expected output: {output_csv}"
        )
    return output


@lru_cache(maxsize=1)
def load_model():
    """Load the final full-data production pipeline once per process."""
    if not MODEL_PATH.exists():
        raise PredictionWorkflowError(
            f"Production model not found: {MODEL_PATH}\n"
            "Copy final_xgboost_full_data_pipeline.joblib from the domain-applicability output "
            "into models/production/."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_domain_reference() -> dict:
    """Load the final operational domain reference."""
    if not DOMAIN_REFERENCE_PATH.exists():
        raise PredictionWorkflowError(
            f"Applicability-domain reference not found: {DOMAIN_REFERENCE_PATH}\n"
            "Copy composition_domain_reference.joblib from the domain-applicability output "
            "into models/production/."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = joblib.load(DOMAIN_REFERENCE_PATH)

    required = {"wt_cols", "lo", "hi", "knn_ref", "final_trusted_condition"}
    missing = required.difference(reference)
    if missing:
        raise PredictionWorkflowError(
            "Domain reference is missing required fields: " + ", ".join(sorted(missing))
        )
    return reference


@lru_cache(maxsize=1)
def load_domain_summary() -> dict:
    """Load nested validation evidence for display in the app."""
    if not DOMAIN_SUMMARY_PATH.exists():
        return {}
    try:
        with open(DOMAIN_SUMMARY_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def assess_applicability_domain(composition: dict[str, float]) -> DomainAssessment:
    """Assess one composition against visible ranges plus the fitted kNN dense core."""
    reference = load_domain_reference()
    wt_cols = list(reference["wt_cols"])
    lo = reference["lo"]
    hi = reference["hi"]

    row = pd.DataFrame([
        {COMPOSITION_COLUMNS[element]: composition[element] for element in COMPOSITION_COLUMNS}
    ])
    row = row[wt_cols].apply(pd.to_numeric, errors="coerce")

    violations: list[str] = []
    for column in wt_cols:
        value = float(row.iloc[0][column])
        low = 0.0 if column == "N (%wt)" else float(lo[column])
        high = float(hi[column])
        if value < low or value > high:
            violations.append(f"{column}: {value:g} not in {low:g}–{high:g}")

    inside_visible = len(violations) == 0

    knn_ref = reference["knn_ref"]
    scaler = knn_ref["scaler"]
    pca = knn_ref["pca"]
    nn = knn_ref["nn"]
    k = int(knn_ref["k"])
    threshold = float(knn_ref["distance_threshold"])

    transformed = scaler.transform(row)
    transformed = pca.transform(transformed)
    distances = nn.kneighbors(transformed, n_neighbors=k, return_distance=True)[0]
    kth_distance = float(distances[0, -1])
    inside_dense = kth_distance <= threshold

    return DomainAssessment(
        inside_visible_ranges=inside_visible,
        inside_knn_dense_core=inside_dense,
        inside_applicability_domain=inside_visible and inside_dense,
        knn_distance=kth_distance,
        knn_threshold=threshold,
        visible_range_violations=tuple(violations),
    )


def _make_initial_input_csv(composition: dict[str, float], path: Path) -> None:
    """Create the one-row CSV expected by the descriptor script."""
    row = {"SFE": 0.0}
    for element in ELEMENTS:
        row[COMPOSITION_COLUMNS[element]] = composition[element]
    pd.DataFrame([row]).to_csv(path, index=False, encoding="utf-8-sig")


def _clean_generated_features(row: pd.DataFrame) -> pd.DataFrame:
    """Match the final modelling pipeline's feature cleaning."""
    required_generated_cols = (
        list(COMPOSITION_COLUMNS.values()) + DESCRIPTOR_COLUMNS + SFE_FEATURE_COLUMNS
    )

    missing_generated = [
        column for column in required_generated_cols if column not in row.columns
    ]
    if missing_generated:
        raise PredictionWorkflowError(
            "The generated feature table is missing required columns: "
            + ", ".join(missing_generated)
        )

    cleaned = row.copy()
    for column in required_generated_cols:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    required_finite = list(COMPOSITION_COLUMNS.values()) + DESCRIPTOR_COLUMNS
    invalid_required = [
        column for column in required_finite if cleaned[column].isna().any()
    ]
    if invalid_required:
        raise PredictionWorkflowError(
            "Feature generation returned missing/non-numeric values for required "
            "composition/descriptor features: " + ", ".join(invalid_required)
        )

    # Same cleaning rule used during modelling. NaN is retained for the fitted median imputer.
    for column in SFE_FEATURE_COLUMNS:
        absurd = cleaned[column].abs() > ABSURD_SFE_THRESHOLD
        if absurd.any():
            cleaned.loc[absurd, column] = np.nan

    return cleaned


def run_prediction(composition: dict[str, float]) -> PredictionResult:
    """Execute the complete final descriptor/SFE/domain/model workflow."""
    total = validate_composition(composition)
    domain = assess_applicability_domain(composition)

    with tempfile.TemporaryDirectory(prefix="sfe_prediction_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        raw_csv = temp_dir / "input_composition.csv"
        descriptor_csv = temp_dir / "with_descriptors.csv"
        feature_csv = temp_dir / "with_sfe_features.csv"

        _make_initial_input_csv(composition, raw_csv)
        descriptor_output = _run_script(DESCRIPTOR_SCRIPT, raw_csv, descriptor_csv)
        master_output = _run_script(SFE_MASTER_SCRIPT, descriptor_csv, feature_csv)

        try:
            feature_table = pd.read_csv(feature_csv, encoding="utf-8-sig")
        except Exception as exc:
            raise PredictionWorkflowError(
                f"Could not read generated feature table: {exc}"
            ) from exc

    if feature_table.empty:
        raise PredictionWorkflowError(f"{SFE_MASTER_SCRIPT} produced an empty output table.")

    row = _clean_generated_features(feature_table.iloc[[0]].copy())
    model = load_model()

    model_feature_names = list(getattr(model, "feature_names_in_", []))
    if not model_feature_names:
        model_feature_names = (
            list(COMPOSITION_COLUMNS.values()) + DESCRIPTOR_COLUMNS + SFE_FEATURE_COLUMNS
        )

    missing_for_model = [
        column for column in model_feature_names if column not in row.columns
    ]
    if missing_for_model:
        raise PredictionWorkflowError(
            "The production model expects columns that were not generated: "
            + ", ".join(missing_for_model)
        )

    model_input = row[model_feature_names].apply(pd.to_numeric, errors="coerce")

    try:
        prediction = float(model.predict(model_input)[0])
    except Exception as exc:
        raise PredictionWorkflowError(f"Model prediction failed: {exc}") from exc

    descriptor_values = {
        column: float(model_input.iloc[0][column]) for column in DESCRIPTOR_COLUMNS
    }
    sfe_values = {
        column: (
            float(model_input.iloc[0][column])
            if pd.notna(model_input.iloc[0][column])
            else float("nan")
        )
        for column in SFE_FEATURE_COLUMNS
    }

    return PredictionResult(
        prediction=prediction,
        total=total,
        descriptors=descriptor_values,
        sfe_features=sfe_values,
        model_input=model_input,
        full_feature_table=feature_table,
        domain=domain,
        domain_summary=load_domain_summary(),
        descriptor_script_output=descriptor_output,
        sfe_master_script_output=master_output,
    )
