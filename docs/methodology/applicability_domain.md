# Applicability-Domain Methodology

The applicability-domain analysis identifies regions of the represented alloy-composition space in which the frozen XGBoost model tends to produce more reliable predictions.

It is intended as a model-usefulness and caution mechanism for the Streamlit application. It does not establish a universal physical validity range for stacking-fault-energy prediction.

## Motivation

A machine-learning model can return a numerical prediction for almost any syntactically valid composition, including compositions that are poorly represented by its training data.

The applicability-domain procedure therefore asks:

> Does a new composition resemble a sufficiently well represented region of the training composition space, and does the domain-construction procedure identify observations with lower prediction error?

The final rule combines two complementary conditions:

```text
inside visible composition ranges
AND
inside kNN dense-composition region
```

Both conditions must be satisfied for a prediction to be labelled `INSIDE`.

## Why the original same-data evaluation was replaced

An earlier implementation:

1. generated out-of-fold predictions;
2. labelled observations as successful or unsuccessful;
3. searched composition limits using those outcomes;
4. reported reliability on the same OOF observations used to construct the domain.

Although the predictions themselves were out of fold, the domain boundaries were still selected using the same error outcomes later used to evaluate the chosen domain.

That creates an optimistic selection effect.

The final implementation removes this circularity with nested validation.

## Final nested design

The final analysis uses:

```text
outer folds = 5
inner folds = 5
grouping = exact nominal composition
random seed = 42 where applicable
```

Exact composition groups are created from the full numeric tuple of all 16 composition variables.

No rounding or near-duplicate merging is used.

### Outer loop

Each outer fold is treated as an untouched evaluation partition for the applicability-domain procedure.

The outer-validation observations are not used to:

- generate the training-domain success labels;
- select the visible composition ranges;
- fit the scaling transformation;
- fit PCA;
- fit the kNN reference;
- select the kNN distance threshold.

### Inner loop

Within the outer-training partition, exact-composition-grouped OOF predictions are generated using the frozen XGBoost specification.

For each outer-training observation:

```text
APE = |experimental SFE - predicted SFE| / |experimental SFE|
```

A prediction is labelled successful when:

```text
APE ≤ 0.20
```

The domain-construction target is:

```text
minimum reliability = 0.80
```

The 80% value is a construction criterion, not a guaranteed prospective result.

## Visible composition ranges

Candidate axis-aligned composition boxes are constructed from the outer-training data.

The search begins from quantile-trimmed ranges and may adaptively move individual boundaries past unsuccessful observations.

A candidate domain must contain at least:

```text
max(30 observations, 10% of the domain-construction sample)
```

and must satisfy the specified training-domain reliability target.

When more than one valid domain exists, the construction procedure favours broader valid coverage before secondary criteria.

The visible ranges make the operational domain interpretable to users, but rectangular limits alone cannot represent sparse multivariate composition regions.

## PCA-assisted kNN dense core

The second condition assesses multivariate composition density.

The 16 wt.% composition variables are:

1. standardised;
2. transformed by PCA retaining 95% of the variance;
3. represented by a k-nearest-neighbour reference with `k = 10`.

For the training reference, the kth external-neighbour distance is calculated for each observation.

The dense-core threshold is the 90th percentile of these training distances.

A new composition is inside the dense core only when its kth-neighbour distance to the training reference is no larger than the training-derived threshold.

This prevents a composition from being classified as well supported merely because every individual element lies inside a univariate min-max range.

## Complete applicability rule

A composition is classified as inside the applicability domain only when:

```text
inside_visible_composition_ranges == True
AND
inside_knn_dense_core == True
```

Otherwise it is classified as outside.

The application still returns an SFE prediction outside the domain, but it displays a caution message.

## Nested validation result

The final analysis contains:

```text
441 outer-validation predictions
209 classified inside domain
232 classified outside domain
```

Pooled results:

| Metric | Inside domain | Outside domain |
|---|---:|---:|
| Observations | 209 | 232 |
| Reliability at APE ≤20% | 77.5% | 53.0% |
| RMSE | 6.94 mJ/m² | 13.20 mJ/m² |
| MAE | 4.26 mJ/m² | 8.22 mJ/m² |
| R² | 0.704 | 0.500 |

Overall nested reliability across all observations is 64.6%.

Applicability-domain coverage is 47.4%.

The domain therefore identifies a subset with substantially lower RMSE and MAE and higher success frequency than observations classified outside it.

## Fold-level behaviour

Every outer-training fold produced a domain satisfying the approximately 80% training construction target.

Outer-validation inside-domain reliability varied between folds:

```text
Fold 1: 86.5%
Fold 2: 62.8%
Fold 3: 81.4%
Fold 4: 76.7%
Fold 5: 81.4%
```

This variation is important.

It shows why the nested estimate should be reported instead of the higher reliability observed within the data used to construct each domain.

## Interpretation of the 80% target

The pre-specified construction target was:

```text
at least 80% successful predictions
where success means APE ≤20%
```

The pooled nested result is 77.5%, slightly below that target.

The thresholds are not altered after seeing this result.

The correct conclusion is therefore:

- the domain-construction procedure identifies a materially more reliable region;
- the prospective nested reliability estimate is 77.5%;
- the 80% construction threshold should not be presented as an achieved nested-validation guarantee.

## Final operational domain

After nested evaluation, the same construction procedure is refitted using all 441 observations to create the operational rule used by the Streamlit application.

The final visible ranges are:

| Element | Operational range, wt.% |
|---|---:|
| Fe | 52.1098–82.9392 |
| Mn | 0–25.62 |
| Al | 0–3.258 |
| C | 0–0.6164 |
| Ni | 0–24.452 |
| Cr | 0–24.682 |
| Si | 0–2.89 |
| Cu | 0–0.0566 |
| Mo | 0–2.7 |
| Nb | 0 |
| V | 0 |
| Ti | 0 |
| Co | 0 |
| N | 0–0.48 |
| P | 0–0.03 |
| S | 0–0.0116 |

The operational rule also requires the final fitted kNN dense-core criterion.

The ranges must therefore never be interpreted as a sufficient domain definition by themselves.

## Nitrogen lower-bound deployment convention

The data-derived full-fit range originally produced a lower nitrogen limit of:

```text
0.0005 wt.% N
```

The deployed application uses:

```text
0 wt.% N
```

as the operational lower bound.

The original value corresponds to 5 ppm N and is treated as an analytical/reporting-scale lower limit rather than evidence of a distinct metallurgical transition at exactly 0.0005 wt.%.

Allowing zero provides a practical representation for nominal compositions in which nitrogen is unspecified or negligible.

This change is deliberately limited to the operational lower boundary. The upper limit of 0.48 wt.% and the kNN dense-composition requirement are unchanged.

The nested applicability-domain statistics were obtained with the originally derived lower boundary. The extension from 0 to 0.0005 wt.% is therefore described as a pragmatic deployment convention, not as a separately validated expansion of the domain.

## Relationship to the frozen XGBoost model

Applicability-domain analysis occurs after final model selection.

It does not:

- change XGBoost hyperparameters;
- change the 28-feature model representation;
- change the model-selection result;
- replace the reserved-holdout evaluation.

The final full-data production model is a refit of the already-frozen XGBoost specification for deployment.

## Scope

The nested design provides out-of-sample internal evidence that the domain-construction procedure separates more reliable from less reliable regions of the compiled composition space.

It does not demonstrate:

- transfer to an independent experimental campaign;
- transfer to an unseen laboratory;
- transfer to an unseen publication source;
- universal metallurgical validity across all steel composition space.

Independent external validation remains future work.
