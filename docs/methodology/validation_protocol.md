# Validation Protocol

This document defines the validation rules used in the final SFE machine-learning workflow and clarifies the point at which the final model specification becomes frozen.

The protocol is designed to separate model development from final evaluation and to reduce optimistic bias from repeated use of the same observations for model selection.

## Core principles

The final workflow follows five rules:

1. the reserved holdout is separated before model screening and tuning;
2. all model-family and hyperparameter decisions are made without using holdout performance;
3. the final model specification is frozen after repeated-CV model selection;
4. the holdout is evaluated once after the model has been frozen;
5. post-selection analyses may describe or stress-test the frozen model but may not alter it.

## Dataset and split

The final dataset contains 441 observations.

A fixed 80/20 development/holdout split is created using random seed 42 and target-stratified binning.

```text
Development partition: 352 observations
Reserved holdout:      89 observations
```

The exact row membership is stored in:

```text
development_split_manifest.csv
```

This manifest is reused rather than regenerated.

## Development-only model screening

Candidate model families are screened using only the 352-row development partition.

The screening protocol is:

```text
5-fold shuffled row-wise KFold
random_state = 42
primary metric = RMSE
secondary metric = MAE
supporting metric = R²
```

The holdout is not scored.

The final shortlist is:

- XGBoost;
- Gradient Boosting;
- Extra Trees;
- HistGradientBoosting.

## Development-only hyperparameter tuning

Shortlisted models are tuned with Optuna.

For each model family:

```text
100 trials
5-fold development cross-validation
objective = mean CV RMSE
```

The holdout remains untouched.

Hyperparameter tuning determines fixed candidate configurations. It does not by itself determine the final model family because several candidate families remain under consideration.

## Repeated-CV model selection

The fixed candidate configurations are compared on identical repeated-CV splits within the development partition.

```text
5 folds
10 repeats
50 matched validation folds
random seed = 42
```

Selection follows the pre-specified hierarchy:

1. lowest mean RMSE;
2. lower MAE as the secondary criterion;
3. R² as supporting evidence.

The final comparison is:

| Rank | Model | RMSE | MAE | R² |
|---|---|---:|---:|---:|
| 1 | XGBoost | 10.980 | 6.531 | 0.556 |
| 2 | Gradient Boosting | 11.153 | 6.821 | 0.547 |
| 3 | Extra Trees | 11.192 | 7.059 | 0.547 |
| 4 | HistGradientBoosting | 11.428 | 7.144 | 0.518 |

XGBoost is selected.

The RMSE difference between XGBoost and Gradient Boosting is modest, so the result is interpreted as a ranking under the pre-specified criterion rather than evidence that XGBoost is universally superior.

## Final model freeze point

The final model specification is frozen immediately after repeated-CV selection of XGBoost.

The frozen specification includes:

- model family;
- complete 28-feature representation;
- preprocessing;
- median-imputation procedure;
- hyperparameters;
- random-state configuration.

The final XGBoost hyperparameters are:

```text
objective          = reg:squarederror
n_estimators       = 500
learning_rate      = 0.011411002001291685
max_depth          = 8
min_child_weight   = 2.665840052140506
subsample          = 0.78310286906201
colsample_bytree   = 0.837235999618537
gamma              = 0.8987126718179128
reg_alpha          = 6.425544478395007e-08
reg_lambda         = 0.029156850643502742
random_state       = 42
n_jobs             = 1
tree_method        = hist
verbosity          = 0
```

From this point onward, no result is used to change the final model specification.

## Reserved-holdout evaluation

The frozen model is fitted on the full 352-row development partition and evaluated once on the 89-row holdout.

Final result:

| Metric | Result |
|---|---:|
| R² | 0.706 |
| RMSE | 8.17 mJ/m² |
| MAE | 4.81 mJ/m² |
| MAPE | 22.35% |
| Pearson r | 0.841 |

The holdout result is reported as obtained. No tuning, feature deletion, threshold adjustment, or model substitution is performed afterwards.

## Post-selection row-wise cross-validation

After final model selection, the frozen specification is evaluated using ordinary row-wise 5-fold cross-validation over all 441 observations.

This provides a conventional internal benchmark:

```text
R²   = 0.618
RMSE = 10.27 mJ/m²
MAE  = 5.83 mJ/m²
r    = 0.786
```

This analysis occurs after model selection and does not feed back into model choice.

## Exact-composition-grouped cross-validation

The dataset contains repeated observations for some exact nominal compositions.

To assess sensitivity to this structure, a second 5-fold analysis keeps identical nominal compositions in the same fold.

A group is defined by the exact numeric tuple of all 16 composition variables.

No rounding, tolerance-based clustering, or near-duplicate matching is applied.

Each grouped fold is checked to confirm zero overlap between training and validation composition groups.

Final grouped result:

```text
R²   = 0.491
RMSE = 11.86 mJ/m²
MAE  = 6.73 mJ/m²
r    = 0.714
```

The reduction relative to row-wise validation is interpreted as evidence that prediction becomes more difficult when exact nominal compositions are not shared between training and validation.

This is an internal composition-level sensitivity analysis. It does not demonstrate transfer to unseen papers, laboratories, measurement protocols, or independent external datasets.

## Feature-group ablation

The ablation experiment uses the same frozen XGBoost configuration.

Four predefined feature sets are compared:

```text
Composition only                 16 predictors
Composition + descriptors        20 predictors
Composition + SFE outputs        24 predictors
Full feature set                 28 predictors
```

The model is not retuned independently for each feature set.

This design asks how performance changes when feature groups are removed while model configuration and validation splits are held constant.

The ablation results do not alter the final model.

## SHAP interpretability

SHAP analysis is explicitly post hoc.

The frozen model is fitted on the development partition and SHAP values are calculated on the reserved holdout.

SHAP is used to describe:

- feature-level model attribution;
- feature-family attribution;
- directional dependence patterns;
- representative local predictions.

SHAP outputs are not used for:

- feature selection;
- model-family selection;
- hyperparameter tuning;
- changing the production model.

## Applicability-domain validation

The applicability-domain procedure is evaluated after the final model has already been selected.

The domain combines:

```text
visible composition ranges
AND
PCA-assisted kNN dense-composition criterion
```

The earlier same-data domain procedure was replaced because selecting a domain using OOF errors and then reporting reliability on those same errors can be optimistic.

The final procedure therefore uses nested exact-composition-grouped validation:

- inner grouped OOF predictions construct the domain using outer-training data only;
- the fold-specific domain rule is frozen;
- the rule and frozen XGBoost model are evaluated on the untouched outer-validation fold.

The pooled nested result is:

```text
coverage                         = 47.4%
inside-domain reliability       = 77.5%
outside-domain reliability      = 53.0%
inside-domain RMSE              = 6.94 mJ/m²
outside-domain RMSE             = 13.20 mJ/m²
```

The pre-specified construction target was 80% reliability at absolute percentage error ≤20%. The nested result of 77.5% is retained without changing the thresholds after inspection.

## Production refit

The production model used by the Streamlit application is fitted on all 441 observations only after model selection and scientific evaluation are complete.

This is a refit of the already-frozen XGBoost specification.

It is not a second model-selection step.

The production model therefore has a different training sample size from the 352-row development-fit model used for the reserved-holdout evaluation, but it has the same:

- model family;
- feature definitions;
- preprocessing;
- hyperparameters.

The production refit is used only for deployment.

## Validation hierarchy

The final evidence should be described with the following hierarchy.

### Final conventional test

Reserved 20% holdout after model freeze.

### Internal robustness and sensitivity analyses

- repeated development-CV model comparison;
- full-dataset row-wise CV;
- exact-composition-grouped CV;
- controlled ablation;
- holdout SHAP interpretation;
- nested exact-composition-grouped applicability-domain analysis.

### Not performed

Independent external validation on a separate dataset, laboratory, or literature source.

No result in this repository should be described as external validation.

## Prohibited post-holdout changes

After the reserved holdout has been evaluated, the following actions would invalidate the clean separation between model development and final evaluation if they were made in response to holdout performance:

- changing model family;
- changing XGBoost hyperparameters;
- deleting or adding predictors;
- changing the SFE cleaning threshold;
- changing the split to obtain better metrics;
- re-ranking candidate models using holdout performance.

None of these changes is part of the final workflow.
