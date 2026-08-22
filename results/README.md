# Final Results

This directory contains only the final results used to support the thesis and production artefact.

Superseded experiments, temporary outputs, duplicate figures, and obsolete model results should not be stored here.

## Structure

```text
results/
├── 01_screening/
├── 02_tuning/
├── 03_repeated_cv_selection/
├── 04_final_evaluation/
├── 05_ablation/
├── 06_interpretability/
└── 07_applicability_domain/
```

## 01 — Screening

Contains the final candidate-model screening table produced on the fixed 352-row development partition.

The leading screening models were Extra Trees, HistGradientBoosting, XGBoost, and Gradient Boosting, with mean 5-fold development-CV RMSE values around 10.9 mJ/m².

These four models were shortlisted for the next stage.

## 02 — Hyperparameter tuning

Contains final Optuna outputs for:

- XGBoost;
- Gradient Boosting;
- Extra Trees;
- HistGradientBoosting.

The tuned development-CV RMSE values were approximately:

| Model | Best tuning RMSE |
|---|---:|
| Gradient Boosting | 10.686 |
| XGBoost | 10.719 |
| Extra Trees | 11.092 |
| HistGradientBoosting | 11.164 |

Because tuning did not improve Extra Trees or HistGradientBoosting relative to their screening configurations, the final repeated-CV comparison used the tuned Gradient Boosting and XGBoost configurations alongside the retained screening configurations for Extra Trees and HistGradientBoosting.

## 03 — Repeated-CV selection

The frozen candidate configurations were compared using 50 matched validation folds.

Final mean repeated-CV results:

| Rank | Model | RMSE | MAE | R² |
|---|---|---:|---:|---:|
| 1 | XGBoost | 10.980 | 6.531 | 0.556 |
| 2 | Gradient Boosting | 11.153 | 6.821 | 0.547 |
| 3 | Extra Trees | 11.192 | 7.059 | 0.547 |
| 4 | HistGradientBoosting | 11.428 | 7.144 | 0.518 |

XGBoost was selected according to the pre-specified criterion of lowest mean RMSE, with MAE secondary and R² supporting.

The RMSE advantage over Gradient Boosting was modest, so the result should not be overstated.

## 04 — Final evaluation

This folder contains the first and only conventional reserved-holdout evaluation of the final frozen XGBoost model.

### Holdout

| Metric | Result |
|---|---:|
| R² | 0.706 |
| RMSE | 8.17 mJ/m² |
| MAE | 4.81 mJ/m² |
| MAPE | 22.35% |
| Pearson r | 0.841 |

### Full-dataset row-wise 5-fold CV

| Metric | Result |
|---|---:|
| R² | 0.618 |
| RMSE | 10.27 mJ/m² |
| MAE | 5.83 mJ/m² |
| Pearson r | 0.786 |

### Exact-composition-grouped 5-fold CV

| Metric | Result |
|---|---:|
| R² | 0.491 |
| RMSE | 11.86 mJ/m² |
| MAE | 6.73 mJ/m² |
| Pearson r | 0.714 |

The grouped result is intentionally more conservative and is interpreted as an internal robustness/sensitivity analysis rather than external validation.

## 05 — Feature-group ablation

Final repeated-CV ablation:

| Feature set | Predictors | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| Composition only | 16 | 11.262 | 6.604 | 0.532 |
| Composition + descriptors | 20 | 11.162 | 6.531 | 0.541 |
| Composition + SFE outputs | 24 | 11.004 | 6.575 | 0.554 |
| Full feature set | 28 | 10.980 | 6.531 | 0.556 |

The full feature representation reduced RMSE by approximately 2.5% relative to composition alone.

The eight published SFE outputs provided the larger individual incremental improvement.

## 06 — Interpretability

Final SHAP analysis was performed on the reserved 89-row holdout using the frozen model fitted on the original 352-row development partition.

Feature-family SHAP attribution:

| Feature family | Share |
|---|---:|
| Composition | 66.91% |
| Published SFE-model outputs | 20.47% |
| Composition-derived descriptors | 12.62% |

The top individual SHAP features were:

1. Ni
2. Fe
3. S
4. P
5. `SFE_Elaraby`
6. `VEC_mean`

The SHAP additivity check showed negligible reconstruction error, confirming that the explanation values reproduced the model predictions to numerical precision.

These results must be interpreted as fitted-model attribution, not physical causality.

## 07 — Applicability domain

The nested exact-composition-grouped applicability-domain procedure produced:

| Metric | Result |
|---|---:|
| Validation coverage | 47.4% |
| Inside-domain reliability at APE ≤20% | 77.5% |
| Outside-domain reliability | 53.0% |
| Overall reliability | 64.6% |
| Inside-domain RMSE | 6.94 mJ/m² |
| Outside-domain RMSE | 13.20 mJ/m² |
| Inside-domain MAE | 4.26 mJ/m² |
| Outside-domain MAE | 8.22 mJ/m² |
| Inside-domain R² | 0.704 |

The procedure therefore identified a substantially more reliable subset of the represented composition space, although the nested reliability estimate fell slightly below the 80% construction target.

The final full-data operational domain had higher internal reliability, but that internal figure is descriptive only and is not used as the principal validation claim.
