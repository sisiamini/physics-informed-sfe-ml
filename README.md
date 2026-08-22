# Physics-Informed SFE ML

A reproducible machine-learning workflow for predicting experimental stacking-fault energy (SFE) in multicomponent steels from alloy composition, composition-derived elemental-property descriptors, and retained outputs from published SFE formulations.

The repository was developed as the software artefact for an MSc project and is structured so that the full modelling workflow can be inspected, reproduced, and extended. The final implementation uses 441 experimentally reported SFE observations, 16 composition variables, four composition-derived descriptors, and eight retained published SFE-model outputs, giving 28 predictors in total.

## Research objective

The project investigates whether a reproducible machine-learning workflow can improve prediction of experimental SFE by combining three information sources:

- nominal alloy composition;
- composition-derived elemental-property descriptors;
- published thermodynamic and empirical SFE-model outputs.

The final workflow also evaluates how performance changes when repeated nominal compositions are kept within the same validation fold, quantifies the incremental value of the augmented feature groups, explains the frozen model using SHAP, and evaluates a nested cross-validated applicability-domain procedure.

## Final dataset

The final modelling dataset contains:

- 441 experimental observations;
- 354 unique exact nominal compositions;
- 16 composition variables in wt.%;
- four composition-derived descriptors;
- eight retained published SFE-model outputs;
- one experimental SFE target.

The target is:

`SFE_Experimental (mJ/m²)`

The four descriptors are:

- `R_mean (pm)`
- `R_std (pm)`
- `Δχ`
- `VEC_mean`

The retained published SFE outputs are:

- `SFE_Akbari (mJ/m²)`
- `SFE_Allain (mJ/m²)`
- `SFE_Dumay (mJ/m²)`
- `SFE_Elaraby (mJ/m²)`
- `SFE_Wang (mJ/m²)`
- `SFE_Xiong (mJ/m²)`
- `SFE_Yoo (mJ/m²)`
- `SFE_Zambrano (mJ/m²)`

## Modelling workflow

The final analysis sequence is:

1. feature generation from alloy composition;
2. candidate-model screening on the fixed development partition;
3. Optuna tuning of shortlisted models;
4. repeated 5-fold × 10-repeat cross-validation for final model selection;
5. one-time evaluation on the reserved 20% holdout set;
6. row-wise and exact-composition-grouped robustness evaluation;
7. controlled feature-group ablation;
8. post-hoc SHAP interpretation of the frozen XGBoost model;
9. nested exact-composition-grouped applicability-domain analysis;
10. final full-data production refit for the Streamlit application.

The model family and hyperparameters were frozen before final holdout evaluation and before the later interpretability and applicability-domain analyses.

## Final model and key results

XGBoost was selected after repeated 5-fold × 10-repeat cross-validation on the development partition. At that point, the final model specification — including the 28-feature representation, preprocessing, model family, and hyperparameters — was frozen before evaluation on the reserved holdout.

| Evaluation | R² | RMSE (mJ/m²) | MAE (mJ/m²) |
|---|---:|---:|---:|
| Reserved 20% holdout | 0.706 | 8.17 | 4.81 |
| Row-wise 5-fold CV | 0.618 | 10.27 | 5.83 |
| Exact-composition-grouped 5-fold CV | 0.491 | 11.86 | 6.73 |

Controlled feature-group ablation showed that the full 28-feature representation reduced repeated-CV RMSE from 11.262 mJ/m² for composition alone to 10.980 mJ/m², an improvement of approximately 2.5%.

Post-hoc SHAP analysis attributed 66.91% of mean absolute attribution to composition variables, 20.47% to the eight published SFE-model outputs, and 12.62% to the four composition-derived descriptors.

Nested exact-composition-grouped applicability-domain analysis identified 47.4% of observations as inside domain. Inside-domain RMSE was 6.94 mJ/m² compared with 13.20 mJ/m² outside domain, with nested inside-domain reliability of 77.5% at an absolute-percentage-error threshold of 20%.

Detailed numerical results are documented in [`results/README.md`](results/README.md). The validation design and model-freeze protocol are documented in [`docs/methodology/validation_protocol.md`](docs/methodology/validation_protocol.md).

## Streamlit application

The application accepts alloy composition in wt.%, automatically calculates balance Fe when Fe is left blank, generates all four descriptors and eight retained published SFE outputs, evaluates the applicability-domain rule, and returns the predicted SFE.

The app clearly distinguishes predictions classified inside and outside the operational applicability domain.

Run the application with:

```powershell
python -m streamlit run .\app\app.py
```

The application requires the final production artefacts in `models/production/`.

## Repository structure

```text
physics-informed-sfe-ml/
├── app/
├── data/
├── docs/
├── environment/
├── models/
├── results/
├── scripts/
├── tests/
├── README.md
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
└── .gitignore
```

Major operational directories contain focused README files, while detailed methodology is documented under `docs/methodology/`.

## Quick start

The commands below are written for Windows PowerShell and should be run from the repository root.

Clone the repository:

```powershell
git clone https://github.com/sisiamini/physics-informed-sfe-ml.git
cd physics-informed-sfe-ml
```

Create and activate a Python 3.12.10 virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the verified direct dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\environment\requirements.txt
```

For maximum reproduction fidelity, install the complete locked environment instead:

```powershell
python -m pip install -r .\environment\requirements-lock.txt
```

Launch the Streamlit application:

```powershell
python -m streamlit run .\app\app.py
```

Detailed environment instructions are available in [`environment/README.md`](environment/README.md).

## Reproducing the analysis

The final analysis is organised as eight numbered stages under `scripts/`, from feature generation through nested applicability-domain analysis.

The complete PowerShell reproduction commands are provided in [`scripts/README.md`](scripts/README.md).

The fixed `development_split_manifest.csv` must be reused wherever required; the reserved holdout must not be regenerated when reproducing the reported results.

For the methodological sequence and validation logic, see:

- [`docs/methodology/pipeline.md`](docs/methodology/pipeline.md)
- [`docs/methodology/validation_protocol.md`](docs/methodology/validation_protocol.md)
- [`docs/methodology/applicability_domain.md`](docs/methodology/applicability_domain.md)

## Reproducibility and scope

This repository is designed to make the final computational artefact inspectable and reproducible. The following scope limitations should be retained when interpreting the results:

- the compiled observations come from the available literature dataset rather than an independently designed experimental campaign;
- exact-composition-grouped validation is stricter than ordinary row-wise validation but is still internal validation;
- the applicability-domain procedure is nested cross-validated within the compiled dataset;
- no independent external dataset was available for final external validation;
- SHAP attribution describes model behaviour rather than physical causality.

These limitations are treated as part of the scientific interpretation rather than hidden implementation details.
