# Analysis Scripts

The scripts in this directory reproduce the final analysis pipeline.

They are organised in scientific execution order rather than by file type.

## Structure

```text
scripts/
├── 01_feature_generation/
├── 02_model_screening/
├── 03_hyperparameter_tuning/
├── 04_model_selection/
├── 05_final_evaluation/
├── 06_ablation/
├── 07_interpretability/
└── 08_applicability_domain/
```
## Analysis stages

The final workflow is organised in execution order.

| Stage | Script(s) | Purpose |
|---|---|---|
| `01_feature_generation` | `SFE_with_Descriptors.py`, `SFE_Master.py` | Generate the four composition-derived descriptors and eight retained published SFE-model outputs |
| `02_model_screening` | `SFE_Ensemble.py` | Create the fixed development/holdout split and screen candidate model families on the development partition |
| `03_hyperparameter_tuning` | XGBoost, Gradient Boosting, Extra Trees and HistGradientBoosting tuning scripts | Tune shortlisted candidate configurations using development-only Optuna CV |
| `04_model_selection` | `Repeated_CV_Model_Comparison.py` | Compare frozen candidate configurations using repeated 5-fold × 10-repeat CV and select XGBoost |
| `05_final_evaluation` | `Final_XGBoost_Evaluation.py` | Evaluate the frozen XGBoost specification on the reserved holdout and perform row-wise and exact-composition-grouped robustness analyses |
| `06_ablation` | `Feature_Ablation_XGBoost.py` | Quantify the contribution of composition, descriptors and published SFE outputs using the frozen model configuration |
| `07_interpretability` | `Interpretability_XGBoost.py` | Perform post-hoc SHAP interpretation of the already-frozen model |
| `08_applicability_domain` | `Domain_Applicability_Nested_XGBoost.py` | Evaluate the applicability-domain procedure using nested exact-composition-grouped validation and create the operational deployment-domain reference |

The final XGBoost specification is frozen after Stage 04, before the reserved holdout is evaluated. Stages 05–08 do not alter the selected model family, feature representation, preprocessing or hyperparameters.

Detailed methodology is documented in:

- [`../docs/methodology/pipeline.md`](../docs/methodology/pipeline.md)
- [`../docs/methodology/validation_protocol.md`](../docs/methodology/validation_protocol.md)
- [`../docs/methodology/applicability_domain.md`](../docs/methodology/applicability_domain.md)

## Environment

Use the project environment described in:

```text
environment/requirements.txt
environment/requirements-lock.txt
```

The verified project environment uses Python 3.12.10.

## PowerShell reproduction commands

The commands below reproduce the final analysis workflow.

All commands should be run from the repository root with the project virtual environment activated.

### 1. Generate composition-derived descriptors

```powershell
python .\scripts\01_feature_generation\SFE_with_Descriptors.py `
    --csv ".\data\raw\sfe_literature_dataset.csv" `
    --output ".\data\processed\output_with_descriptors.csv"
```

This produces the 16 composition variables together with the four composition-derived descriptors.

### 2. Generate the eight retained published SFE-model outputs

```powershell
python .\scripts\01_feature_generation\SFE_Master.py `
    --csv ".\data\processed\output_with_descriptors.csv" `
    --output ".\data\processed\output_with_8_sfe_models.csv"
```

The resulting file is the canonical final modelling table used by all subsequent stages.

### 3. Candidate-model screening and fixed development/holdout split

```powershell
python .\scripts\02_model_screening\SFE_Ensemble.py `
    --csv ".\data\processed\output_with_8_sfe_models.csv" `
    --output ".\results\01_screening"
```

This stage creates:

```text
results/01_screening/development_split_manifest.csv
results/01_screening/model_evaluation_results.csv
```

The split manifest must be reused unchanged by all subsequent stages that require the original development/holdout partition.

### 4. Hyperparameter tuning — XGBoost

```powershell
python .\scripts\03_hyperparameter_tuning\Hyperparameter_Tuning_XGB.py `
    --csv ".\data\processed\output_with_8_sfe_models.csv" `
    --output ".\results\02_tuning\xgboost" `
    --split-manifest ".\results\01_screening\development_split_manifest.csv" `
    --n-trials 100 `
    --cv-folds 5
```

### 5. Hyperparameter tuning — Gradient Boosting

```powershell
python .\scripts\03_hyperparameter_tuning\Hyperparameter_Tuning_GB.py `
    --csv ".\data\processed\output_with_8_sfe_models.csv" `
    --output ".\results\02_tuning\gradient_boosting" `
    --split-manifest ".\results\01_screening\development_split_manifest.csv" `
    --n-trials 100 `
    --cv-folds 5
```

### 6. Hyperparameter tuning — Extra Trees

```powershell
python .\scripts\03_hyperparameter_tuning\Hyperparameter_Tuning_ET.py `
    --csv ".\data\processed\output_with_8_sfe_models.csv" `
    --output ".\results\02_tuning\extra_trees" `
    --split-manifest ".\results\01_screening\development_split_manifest.csv" `
    --n-trials 100 `
    --cv-folds 5
```

### 7. Hyperparameter tuning — HistGradientBoosting

```powershell
python .\scripts\03_hyperparameter_tuning\Hyperparameter_Tuning_HGB.py `
    --csv ".\data\processed\output_with_8_sfe_models.csv" `
    --output ".\results\02_tuning\hist_gradient_boosting" `
    --split-manifest ".\results\01_screening\development_split_manifest.csv" `
    --n-trials 100 `
    --cv-folds 5
```

The reserved holdout is not evaluated during any hyperparameter-tuning stage.

### 8. Repeated-CV final model selection

```powershell
python .\scripts\04_model_selection\Repeated_CV_Model_Comparison.py `
    --csv ".\data\processed\output_with_8_sfe_models.csv" `
    --output ".\results\03_repeated_cv_selection" `
    --split-manifest ".\results\01_screening\development_split_manifest.csv" `
    --n-splits 5 `
    --n-repeats 10
```

This performs 5-fold × 10-repeat matched development-only cross-validation.

XGBoost is selected according to the pre-specified primary RMSE, secondary MAE, and supporting R² criteria.

### 9. Final XGBoost evaluation

```powershell
python .\scripts\05_final_evaluation\Final_XGBoost_Evaluation.py `
    --csv ".\data\processed\output_with_8_sfe_models.csv" `
    --output ".\results\04_final_evaluation" `
    --split-manifest ".\results\01_screening\development_split_manifest.csv" `
    --folds 5
```

This is the first stage that evaluates the frozen final model on the reserved 89-row holdout.

It also performs the post-selection row-wise and exact-composition-grouped robustness analyses.

### 10. Feature-group ablation

```powershell
python .\scripts\06_ablation\Feature_Ablation_XGBoost.py `
    --csv ".\data\processed\output_with_8_sfe_models.csv" `
    --output ".\results\05_ablation" `
    --split-manifest ".\results\01_screening\development_split_manifest.csv" `
    --n-splits 5 `
    --n-repeats 10
```

The ablation uses the frozen final XGBoost configuration and excludes the reserved holdout.

### 11. SHAP interpretability

```powershell
python .\scripts\07_interpretability\Interpretability_XGBoost.py `
    --csv ".\data\processed\output_with_8_sfe_models.csv" `
    --output ".\results\06_interpretability" `
    --split-manifest ".\results\01_screening\development_split_manifest.csv" `
    --top-features 10 `
    --dependence-features 6
```

This reproduces the already-evaluated frozen model on the development partition and calculates SHAP explanations on the reserved holdout.

Interpretability results must not be used to alter or reselect the final model.

### 12. Nested applicability-domain analysis

```powershell
python .\scripts\08_applicability_domain\Domain_Applicability_Nested_XGBoost.py `
    --csv ".\data\processed\output_with_8_sfe_models.csv" `
    --model ".\results\04_final_evaluation\final_xgboost_development_pipeline.joblib" `
    --out_dir ".\results\07_applicability_domain" `
    --max_error 0.20 `
    --min_reliability 0.80 `
    --knn_k 10 `
    --knn_quantile 0.90 `
    --outer_cv 5 `
    --inner_cv 5 `
    --absurd_thr 1e6 `
    --suppress_warnings
```

The principal applicability-domain evidence is the nested outer-validation result rather than the full-data internal domain reliability.

### 13. Copy the deployment artefacts

After nested applicability-domain evaluation is complete, the final stage writes the operational domain artefacts and a deployment-only full-data refit of the already-frozen XGBoost specification.

The full-data refit does not alter the model family, feature representation, preprocessing or hyperparameters and is not part of model selection.

Copy the three deployment artefacts into the production-model directory:

```powershell
Copy-Item `
    ".\results\07_applicability_domain\final_xgboost_full_data_pipeline.joblib" `
    ".\models\production\final_xgboost_full_data_pipeline.joblib"

Copy-Item `
    ".\results\07_applicability_domain\composition_domain_reference.joblib" `
    ".\models\production\composition_domain_reference.joblib"

Copy-Item `
    ".\results\07_applicability_domain\domain_summary.json" `
    ".\models\production\domain_summary.json"
```

### 14. Launch the final Streamlit application

```powershell
python -m streamlit run .\app\app.py
```

## Reproducibility rule

Do not alter:

- the fixed split manifest;
- random seed 42;
- final model hyperparameters;
- feature definitions;
- SFE cleaning threshold;
- exact-composition grouping logic;

when attempting to reproduce the reported thesis results.
