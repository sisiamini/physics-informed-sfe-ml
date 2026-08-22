# Streamlit Application

This directory contains the interactive deployment layer for the final SFE prediction workflow.

## Files

```text
app/
├── app.py
└── sfe_workflow.py
```

`app.py` contains the Streamlit user interface.

`sfe_workflow.py` contains the application logic for:

- parsing user-provided alloy compositions;
- calculating balance Fe when Fe is left blank;
- validating the total composition;
- running the final descriptor-generation script;
- running the final SFE-model feature-generation script;
- loading the production XGBoost pipeline;
- evaluating the operational applicability-domain rule;
- producing the final SFE prediction.

The application deliberately keeps the user-interface layer separate from the scientific prediction workflow.

## Required scientific scripts

The application uses the final feature-generation scripts:

```text
scripts/01_feature_generation/SFE_with_Descriptors.py
scripts/01_feature_generation/SFE_Master.py
```

`sfe_workflow.py` references these canonical feature-generation scripts directly so that the scientific implementation is maintained in a single location rather than duplicated inside the application directory.

## Required production artefacts

The application requires:

```text
models/production/final_xgboost_full_data_pipeline.joblib
models/production/composition_domain_reference.joblib
models/production/domain_summary.json
```

The production XGBoost pipeline is a full-data deployment refit of the model specification that had already been frozen after repeated-CV model selection. The refit uses all 441 observations only after scientific model evaluation is complete; it does not change the selected model family, feature representation, preprocessing, or hyperparameters.

## Model input

The production model expects 28 predictors:

- 16 elemental composition variables;
- four composition-derived descriptors;
- eight retained published SFE-model outputs.

## Applicability-domain behaviour

The application does not reject a prediction solely because the entered composition lies outside the operational applicability domain.

Instead, it:

1. calculates the prediction;
2. checks whether the composition satisfies the visible composition limits;
3. checks whether the composition lies inside the fitted kNN dense-composition region;
4. labels the prediction as `INSIDE` only when both criteria are satisfied;
5. labels the prediction as `OUTSIDE` otherwise.

Predictions outside the applicability domain are displayed with a caution message because they are less well supported by the represented composition space.

The application reports nested cross-validated applicability-domain evidence rather than presenting the full-fit internal reliability as independent validation.

## Nitrogen range

The operational visible nitrogen range used by the application is:

```text
N = 0–0.48 wt.%
```

The original data-derived lower limit was 0.0005 wt.% (5 ppm). The deployment lower bound was rounded to zero so that nominally unspecified or negligible nitrogen can be represented as zero. This is a deployment convention rather than a separately revalidated expansion of the nested applicability-domain analysis.

## Running the app

All commands should be run from the repository root.

Activate the project environment if it is not already active:

```powershell
.\.venv\Scripts\Activate.ps1
```

Confirm that the production artefacts exist:

```powershell
Get-ChildItem .\models\production\
```

The directory should contain:

```text
final_xgboost_full_data_pipeline.joblib
composition_domain_reference.joblib
domain_summary.json
```

Launch the application:

```powershell
python -m streamlit run .\app\app.py
```

Streamlit will print the local application URL in the PowerShell terminal.

The verified environment is documented in `environment/`.

## Functional validation

The final application was manually validated for prediction consistency, balance-Fe handling, applicability-domain classification, invalid-input rejection, and final feature-schema construction.

Detailed validation results are documented in:

- [`../tests/manual/app_validation_checklist.md`](../tests/manual/app_validation_checklist.md)

Screenshot evidence is stored under `docs/app/screenshots/`.