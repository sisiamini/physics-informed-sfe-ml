# Production Models

This directory contains only the final deployment artefacts used by the Streamlit application.

## Structure

```text
models/
└── production/
    ├── final_xgboost_full_data_pipeline.joblib
    ├── composition_domain_reference.joblib
    └── domain_summary.json
```

## final_xgboost_full_data_pipeline.joblib

This is the final full-data XGBoost production pipeline.

It is a deployment refit of the already-selected model family and frozen hyperparameters using all 441 available observations.

The production refit was created only after:

- candidate-model screening;
- Optuna tuning;
- repeated-CV model selection;
- one-time reserved-holdout evaluation;
- row-wise and grouped robustness analysis;
- feature-group ablation;
- SHAP interpretability.

The full-data refit therefore does not replace or invalidate the previously reported holdout result. It exists to maximise the training information available to the deployed application after model selection and evaluation were complete.

## composition_domain_reference.joblib

This file contains the final operational applicability-domain reference used by the Streamlit application.

It stores:

- the composition columns;
- final visible lower and upper composition limits;
- fitted scaling and PCA transformations;
- fitted nearest-neighbour reference;
- kNN distance threshold;
- applicability-rule metadata.

The final trusted condition is:

```text
inside visible composition ranges
AND
inside kNN dense-composition region
```

The application may still produce predictions outside this domain, but they are explicitly flagged as less well supported by the represented composition space.

## domain_summary.json

This file contains the nested applicability-domain evaluation summary used by the application for user-facing context.

The main nested results are:

```text
coverage                         = 47.4%
inside-domain reliability       = 77.5%
outside-domain reliability      = 53.0%
inside-domain RMSE              = 6.94 mJ/m²
outside-domain RMSE             = 13.20 mJ/m²
```

The nested results are the relevant evidence for the usefulness of the applicability-domain procedure.

The full-fit internal reliability is not presented as independent validation because the final operational domain is refitted using all available data.

## Model feature set

The production model uses 28 predictors:

- 16 composition variables;
- four composition-derived descriptors;
- eight retained published SFE-model outputs.

## Software compatibility

The model artefacts were created in the verified project environment documented under `environment/`.

The canonical project environment uses:

```text
Python 3.12.10
scikit-learn 1.7.2
xgboost 3.2.0
joblib 1.5.2
```

For maximum compatibility when loading the serialized pipeline, use `environment/requirements-lock.txt`.

