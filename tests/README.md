# Tests

This directory contains the final **automated workflow-integrity tests** and **manual application-validation evidence** for the Streamlit research artefact.

The purpose of these checks is to confirm that the deployed application, production artefacts, feature-generation workflow, applicability-domain logic, and final model interface remain consistent with the scientific workflow described in the dissertation.

These tests assess **software and implementation consistency**. They do not replace the statistical evaluation of the predictive model. Predictive performance, cross-validation, ablation, interpretability, and applicability-domain analyses are stored under `results/`.

## Structure

```text
tests/
├── README.md
├── test_workflow.py
└── manual/
    └── app_validation_checklist.md
```

## Automated workflow-integrity tests

Automated tests are implemented in:

```text
tests/test_workflow.py
```

The final release-verification run contains **28 automated tests** covering key parts of the deployed workflow, including:

* expected repository paths and required artefacts;
* loading of the final production model and applicability-domain artefacts;
* input parsing and validation;
* balance-Fe handling;
* feature-generation and final predictor-schema consistency;
* generation of the four composition-derived descriptors;
* generation of the eight retained published SFE-model outputs;
* exclusion of the removed Curtze-derived feature from the final model input;
* applicability-domain execution;
* final 28-predictor model-input construction;
* end-to-end production-model prediction.

Run the automated test suite from the repository root with:

```text
python -m unittest discover -s .\tests -p "test_*.py" -v
```

The canonical final release verification reports:

```text
28 tests passed
```

under the project environment documented in `environment/`.

Passing these tests demonstrates that the submitted software components and production artefacts operate consistently with the intended workflow. It does **not** demonstrate external model generalisation, statistical significance, or subjective application usability.

## Manual application validation

In addition to the automated test suite, the final Streamlit application was manually checked for representative valid, invalid, inside-domain, and outside-domain inputs.

The manual checks confirm the following behaviours:

* a known dataset composition returns the expected production-model prediction;
* blank Fe is calculated correctly as balance Fe;
* a composition satisfying both applicability-domain conditions is labelled `INSIDE`;
* a composition outside the visible composition limits is labelled `OUTSIDE`;
* a composition inside the visible limits but outside the kNN dense-composition region is labelled `OUTSIDE`;
* a total composition different from 100 wt.% is rejected;
* negative composition values are rejected;
* non-numeric composition values are rejected;
* four composition-derived descriptors are displayed;
* eight retained published SFE-model outputs are displayed;
* the final model input contains 28 predictors;
* the operational nitrogen range is displayed as 0–0.48 wt.%.

The completed manual-validation checklist is stored at:

```text
tests/manual/app_validation_checklist.md
```

Screenshot evidence is stored under:

```text
docs/app/screenshots/
```

## Relationship to scientific validation

The automated and manual tests in this directory verify the **implementation and deployment workflow** only.

They do not replace the scientific evaluation of the predictive model.

The final scientific evidence is reported under `results/` and includes:

* candidate-model screening;
* hyperparameter tuning;
* repeated cross-validation for final model selection;
* reserved-holdout evaluation;
* row-wise five-fold cross-validation;
* exact-composition-grouped five-fold cross-validation;
* controlled feature-group ablation;
* SHAP interpretability analysis;
* nested exact-composition-grouped applicability-domain analysis.

This distinction is important: successful software tests establish that the released artefact behaves as designed, whereas the analyses under `results/` provide the evidence for predictive performance, robustness, feature contribution, interpretation, and applicability-domain behaviour.

## Final release check

Before creating or submitting a final repository release:

1. create and activate the canonical Python environment documented under `environment/`;

2. install the locked dependencies from `environment/requirements-lock.txt`;

3. confirm that the final production model and applicability-domain artefacts are present under `models/production/`;

4. run the complete automated test suite:

   ```text
   python -m unittest discover -s .\tests -p "test_*.py" -v
   ```

5. confirm that all **28 automated tests pass**;

6. launch the Streamlit application:

   ```text
   python -m streamlit run .\app\app.py
   ```

7. repeat at least one known-composition prediction;

8. confirm that balance Fe, engineered-feature display, `INSIDE`/`OUTSIDE` domain status, and invalid-input handling behave as documented;

9. confirm that the manual checklist and screenshot evidence correspond to the released application version.

## Scope of this directory

The `tests/` directory provides two complementary forms of release evidence:

* **automated tests** verify reproducible software, artefact, schema, domain, and end-to-end workflow integrity;
* **manual validation** confirms representative behaviour of the user-facing Streamlit application.

Together, these checks demonstrate implementation consistency for the submitted research artefact. The computational and statistical validation of the predictive model remains in the reproducible analysis pipeline under `results/`.