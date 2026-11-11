# MLWA Experimental Evidence

This directory contains the additional analyses supporting the manuscript
"Knowledge-Guided Feature Engineering and Generalization-Aware
Evaluation for Stacking-Fault Energy Prediction in Multicomponent Steels".

## Dataset and experimental settings

- Dataset: 441 literature-derived experimental SFE observations
- Development partition: 352 observations
- Reserved holdout: 89 observations
- Predictors: 16 composition variables, four elemental descriptors,
  and eight published SFE-model outputs
- Regressor: XGBoost with frozen hyperparameters
- Grouped evaluations: five-fold GroupKFold
- Radius sensitivity: original radii, Cordero covalent radii,
  and omission of radius descriptors

The original data, frozen hyperparameters, development split manifest,
and nested applicability-domain predictions are retained in the
repository's existing data/ and results/ directories.

## 1. Grouped feature ablation

Script:
paper_audit_tools/02_grouped_generalization_and_ablation.py

Results:
paper_audit/new_experiments/development/

Pooled out-of-fold RMSE (mJ/m2):

Feature set                  Composition grouping    Source grouping
Composition only             11.769                  14.296
Composition + descriptors    11.617                  14.096
Composition + SFE outputs    11.851                  15.221
Full representation          11.793                  15.031

These analyses were performed after the original model-selection
process. The reported bootstrap intervals are conditional on fixed
out-of-fold predictions.

## 2. Radius-convention sensitivity

Script:
paper_audit_tools/04_radius_convention_sensitivity.py

Results:
paper_audit/radius_sensitivity/full/

The experiment independently verifies the original two radius
descriptors and compares the archived radius convention with a
consistent covalent-radius alternative and a no-radius representation.

The original constants are preserved for baseline reproduction.
Predictive performance differences do not establish the physical
superiority of a particular radius convention.

## 3. Applicability-domain comparison

Scripts:
paper_audit_tools/03_applicability_domain_baselines.py
paper_audit_tools/05_domain_coverage_robustness.py

Results:
paper_audit/domain_baselines/
paper_audit/domain_robustness/

The original nested applicability-domain procedure selected
209 of 441 observations (47.4% coverage), with inside-domain
RMSE 6.941 mJ/m2 and APE <= 20% reliability of 77.5%.

The equal-count kNN diagnostic selected the same number of
observations, with RMSE 8.249 mJ/m2.

The equal-count diagnostic uses held-out composition distances
to match selection counts; it is not a deployable threshold rule.
The pooled advantage varied across outer folds.

## 4. Baseline reproducibility

Script:
paper_audit_tools/00_compare_reproduced_results.py

Comparison:
paper_audit/baseline/metric_comparison.csv

Environment:
paper_audit/pip_freeze.txt
environment/requirements-lock.txt

The reproduced original evaluation, ablation, and nested-domain
metrics matched the archived comparison values.

## Reproduction

Install the locked environment using:

python -m pip install -r environment/requirements-lock.txt

Grouped ablation:

python paper_audit_tools/02_grouped_generalization_and_ablation.py --scope development --grouping both --folds 5 --bootstrap 1000 --output paper_audit/new_experiments/development

Radius sensitivity:

python paper_audit_tools/04_radius_convention_sensitivity.py --scope development --grouping both --representations both --folds 5 --bootstrap 1000 --output paper_audit/radius_sensitivity/full

Applicability-domain baselines:

python paper_audit_tools/03_applicability_domain_baselines.py --output paper_audit/domain_baselines

Coverage comparison:

python paper_audit_tools/05_domain_coverage_robustness.py --k 10 --output paper_audit/domain_robustness

## Scope

The grouped ablations are development-only, post-selection
evaluations. Applicability-domain comparisons reuse the original
nested outer-validation predictions. No new independent external
dataset was used to validate the revised research claims.
