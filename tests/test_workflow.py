"""Integration and software-integrity tests for the final SFE repository.

Run from the repository root with:

    python -m unittest discover -s .\tests -p "test_*.py" -v

The suite intentionally checks repository path resolution, production-artifact
loading, composition validation, final feature dimensions, applicability-domain
execution, and a complete end-to-end prediction.

It does not rerun model screening, Optuna tuning, repeated cross-validation,
holdout evaluation, SHAP, or nested applicability-domain validation.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
import unittest

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / "app" / "sfe_workflow.py"

EXPECTED_SCRIPT_DIR = REPO_ROOT / "scripts" / "01_feature_generation"
EXPECTED_DESCRIPTOR_SCRIPT = EXPECTED_SCRIPT_DIR / "SFE_with_Descriptors.py"
EXPECTED_SFE_MASTER_SCRIPT = EXPECTED_SCRIPT_DIR / "SFE_Master.py"

EXPECTED_MODEL_PATH = (
    REPO_ROOT
    / "models"
    / "production"
    / "final_xgboost_full_data_pipeline.joblib"
)
EXPECTED_DOMAIN_REFERENCE_PATH = (
    REPO_ROOT
    / "models"
    / "production"
    / "composition_domain_reference.joblib"
)
EXPECTED_DOMAIN_SUMMARY_PATH = (
    REPO_ROOT
    / "models"
    / "production"
    / "domain_summary.json"
)


def load_workflow_module():
    """Load app/sfe_workflow.py directly from the repository."""
    if not WORKFLOW_PATH.is_file():
        raise FileNotFoundError(
            f"Final workflow module not found: {WORKFLOW_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "final_sfe_workflow",
        WORKFLOW_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create import specification for {WORKFLOW_PATH}"
        )

    module = importlib.util.module_from_spec(spec)

    # Register the dynamically loaded module before executing it.
    # Python 3.12 dataclasses inspect sys.modules during class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


workflow = load_workflow_module()


def known_composition_raw() -> dict[str, str]:
    """A valid composition drawn from the final 441-row dataset."""
    return {
        "Mn": "22.0",
        "Al": "0",
        "C": "0.03",
        "Ni": "0",
        "Cr": "0",
        "Si": "0",
        "Cu": "0",
        "Mo": "0",
        "Nb": "0",
        "V": "0",
        "Ti": "0",
        "Co": "0",
        "N": "0",
        "P": "0",
        "S": "0",
        "Fe": "77.97",
    }


class RepositoryPathTests(unittest.TestCase):
    """Catch packaging errors before the Streamlit app is released."""

    def test_workflow_uses_repository_level_script_directory(self):
        self.assertEqual(
            workflow.SCRIPT_DIR.resolve(),
            EXPECTED_SCRIPT_DIR.resolve(),
            "sfe_workflow.py must resolve feature-generation scripts from "
            "scripts/01_feature_generation/ at repository level.",
        )

    def test_descriptor_script_exists(self):
        self.assertTrue(
            EXPECTED_DESCRIPTOR_SCRIPT.is_file(),
            f"Missing descriptor script: {EXPECTED_DESCRIPTOR_SCRIPT}",
        )

    def test_sfe_master_script_exists(self):
        self.assertTrue(
            EXPECTED_SFE_MASTER_SCRIPT.is_file(),
            f"Missing SFE master script: {EXPECTED_SFE_MASTER_SCRIPT}",
        )

    def test_workflow_uses_repository_level_model_paths(self):
        self.assertEqual(
            workflow.MODEL_PATH.resolve(),
            EXPECTED_MODEL_PATH.resolve(),
        )
        self.assertEqual(
            workflow.DOMAIN_REFERENCE_PATH.resolve(),
            EXPECTED_DOMAIN_REFERENCE_PATH.resolve(),
        )
        self.assertEqual(
            workflow.DOMAIN_SUMMARY_PATH.resolve(),
            EXPECTED_DOMAIN_SUMMARY_PATH.resolve(),
        )

    def test_required_production_files_exist(self):
        for path in (
            EXPECTED_MODEL_PATH,
            EXPECTED_DOMAIN_REFERENCE_PATH,
            EXPECTED_DOMAIN_SUMMARY_PATH,
        ):
            with self.subTest(path=path):
                self.assertTrue(
                    path.is_file(),
                    f"Missing required production artefact: {path}",
                )


class CompositionParsingTests(unittest.TestCase):

    def test_blank_fe_is_calculated_by_balance(self):
        raw = {
            element: ""
            for element in workflow.ELEMENTS
        }
        raw["Mn"] = "10"
        raw["Ni"] = "5"

        composition = workflow.parse_composition(raw)

        self.assertAlmostEqual(
            composition["Fe"],
            85.0,
            places=12,
        )
        self.assertAlmostEqual(
            sum(composition.values()),
            100.0,
            places=12,
        )

    def test_blank_non_fe_values_are_zero(self):
        raw = {
            element: ""
            for element in workflow.ELEMENTS
        }

        composition = workflow.parse_composition(raw)

        for element in workflow.ELEMENTS:
            if element == "Fe":
                self.assertAlmostEqual(
                    composition[element],
                    100.0,
                    places=12,
                )
            else:
                self.assertEqual(
                    composition[element],
                    0.0,
                )

    def test_non_numeric_input_is_rejected(self):
        raw = known_composition_raw()
        raw["Mn"] = "not-a-number"

        with self.assertRaises(
            workflow.InputValidationError
        ):
            workflow.parse_composition(raw)


class CompositionValidationTests(unittest.TestCase):

    def test_valid_composition_is_accepted(self):
        composition = workflow.parse_composition(
            known_composition_raw()
        )
        total = workflow.validate_composition(
            composition
        )

        self.assertTrue(
            math.isclose(
                total,
                100.0,
                rel_tol=0.0,
                abs_tol=workflow.TOTAL_TOLERANCE,
            )
        )

    def test_total_not_equal_to_100_is_rejected(self):
        composition = workflow.parse_composition(
            known_composition_raw()
        )
        composition["Fe"] -= 1.0

        with self.assertRaises(
            workflow.InputValidationError
        ):
            workflow.validate_composition(
                composition
            )

    def test_negative_value_is_rejected(self):
        composition = workflow.parse_composition(
            known_composition_raw()
        )
        composition["Mn"] = -0.1
        composition["Fe"] = 100.1

        with self.assertRaises(
            workflow.InputValidationError
        ):
            workflow.validate_composition(
                composition
            )


class FinalSchemaTests(unittest.TestCase):

    def test_composition_variable_count_is_16(self):
        self.assertEqual(
            len(workflow.COMPOSITION_COLUMNS),
            16,
        )

    def test_descriptor_schema_has_four_features(self):
        self.assertEqual(
            workflow.DESCRIPTOR_COLUMNS,
            [
                "R_mean (pm)",
                "R_std (pm)",
                "Δχ",
                "VEC_mean",
            ],
        )

    def test_sfe_output_count_is_eight(self):
        self.assertEqual(
            len(workflow.SFE_FEATURE_COLUMNS),
            8,
        )

    def test_total_predictor_count_is_28(self):
        predictor_count = (
            len(workflow.COMPOSITION_COLUMNS)
            + len(workflow.DESCRIPTOR_COLUMNS)
            + len(workflow.SFE_FEATURE_COLUMNS)
        )

        self.assertEqual(
            predictor_count,
            28,
        )

    def test_operational_nitrogen_range_is_zero_to_point48(self):
        self.assertEqual(
            workflow.DOMAIN_RANGES["N"],
            (0.0, 0.48),
        )

    def test_generated_sfe_cleaning_threshold_is_1e6(self):
        self.assertEqual(
            workflow.ABSURD_SFE_THRESHOLD,
            1e6,
        )


class ProductionArtifactTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure each test run actually loads the current repository artefacts.
        workflow.load_model.cache_clear()
        workflow.load_domain_reference.cache_clear()
        workflow.load_domain_summary.cache_clear()

    def test_production_model_loads(self):
        model = workflow.load_model()
        self.assertTrue(
            hasattr(model, "predict"),
            "Loaded production artefact does not expose predict().",
        )

    def test_domain_reference_loads_and_has_required_structure(self):
        reference = workflow.load_domain_reference()

        required = {
            "wt_cols",
            "lo",
            "hi",
            "knn_ref",
            "final_trusted_condition",
        }

        self.assertTrue(
            required.issubset(reference.keys())
        )
        self.assertEqual(
            len(reference["wt_cols"]),
            16,
        )

    def test_domain_summary_loads_and_matches_final_schema(self):
        summary = workflow.load_domain_summary()

        self.assertIsInstance(
            summary,
            dict,
        )
        self.assertEqual(
            summary.get("feature_count_used_by_model"),
            28,
        )
        self.assertEqual(
            summary.get("total_valid_rows"),
            441,
        )

    def test_domain_summary_contains_final_nested_headline_metrics(self):
        summary = workflow.load_domain_summary()

        self.assertAlmostEqual(
            float(summary["nested_domain_coverage_percent"]),
            47.3922902494331,
            places=8,
        )
        self.assertAlmostEqual(
            float(summary["nested_inside_domain_reliability_percent"]),
            77.51196172248804,
            places=8,
        )
        self.assertAlmostEqual(
            float(summary["nested_inside_domain_RMSE"]),
            6.940985582468638,
            places=8,
        )
        self.assertAlmostEqual(
            float(summary["nested_outside_domain_RMSE"]),
            13.201442468932026,
            places=8,
        )


class EndToEndWorkflowTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        workflow.load_model.cache_clear()
        workflow.load_domain_reference.cache_clear()
        workflow.load_domain_summary.cache_clear()

        cls.composition = workflow.parse_composition(
            known_composition_raw()
        )
        cls.result = workflow.run_prediction(
            cls.composition
        )

    def test_end_to_end_prediction_is_finite(self):
        self.assertTrue(
            math.isfinite(self.result.prediction),
            "End-to-end workflow did not return a finite SFE prediction.",
        )

    def test_end_to_end_total_is_100(self):
        self.assertAlmostEqual(
            self.result.total,
            100.0,
            places=12,
        )

    def test_end_to_end_generates_four_descriptors(self):
        self.assertEqual(
            len(self.result.descriptors),
            4,
        )
        self.assertTrue(
            all(
                math.isfinite(value)
                for value in self.result.descriptors.values()
            )
        )

    def test_end_to_end_generates_eight_sfe_outputs(self):
        self.assertEqual(
            len(self.result.sfe_features),
            8,
        )

    def test_end_to_end_model_input_has_28_predictors(self):
        self.assertIsInstance(
            self.result.model_input,
            pd.DataFrame,
        )
        self.assertEqual(
            self.result.model_input.shape,
            (1, 28),
        )

    def test_end_to_end_domain_assessment_executes(self):
        domain = self.result.domain

        self.assertIsInstance(
            domain.inside_visible_ranges,
            bool,
        )
        self.assertIsInstance(
            domain.inside_knn_dense_core,
            bool,
        )
        self.assertIsInstance(
            domain.inside_applicability_domain,
            bool,
        )
        self.assertTrue(
            math.isfinite(domain.knn_distance),
        )
        self.assertTrue(
            math.isfinite(domain.knn_threshold),
        )

    def test_known_composition_with_zero_n_passes_visible_ranges(self):
        self.assertTrue(
            self.result.domain.inside_visible_ranges,
            "The known final-dataset composition should satisfy the "
            "operational visible ranges, including N = 0 wt.%.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
