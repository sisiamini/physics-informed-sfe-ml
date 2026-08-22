# Software Environment

The final project environment is documented with two requirement files.

```text
environment/
├── requirements.txt
├── requirements-lock.txt
├── release_verification.txt
└── README.md
```

## Python version

The archived final environment uses Python 3.12.10.

Python 3.12.10 environment is recommended for reproducing the thesis artefact and loading the serialized production model.

## requirements.txt

`requirements.txt` contains the direct dependencies used by the final analysis and application.

Pinned direct dependencies:

```text
numpy==2.2.3
pandas==2.2.3
matplotlib==3.10.6
scikit-learn==1.7.2
joblib==1.5.2
xgboost==3.2.0
lightgbm==4.6.0
catboost==1.2.10
optuna==4.9.0
shap==0.52.0
streamlit==1.60.0
```

This file is intended for normal installation and inspection.

## requirements-lock.txt

`requirements-lock.txt` contains the complete fully pinned environment recovered from the archived final project environment, including transitive dependencies.

It is the authoritative dependency record for maximum reproduction fidelity.

## Recommended installation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\environment\requirements.txt
```

For the strict fully pinned environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip==26.2.1
python -m pip install -r .\environment\requirements-lock.txt
```

## Verification

After installation:

```powershell
python -c "import numpy, pandas, sklearn, xgboost, lightgbm, catboost, optuna, shap, streamlit; print('Environment OK')"
```

For the serialized production model, also verify:

```powershell
python -c "import sklearn, xgboost, joblib; print('scikit-learn', sklearn.__version__); print('xgboost', xgboost.__version__); print('joblib', joblib.__version__)"
```

Expected core versions:

```text
scikit-learn 1.7.2
xgboost 3.2.0
joblib 1.5.2
```

## Reproducibility policy

The repository should expose one canonical final environment.

Older environment files with conflicting dependency versions should not remain in the main branch because they create ambiguity about which software stack produced the final artefact.

When updating the project in the future, changes to dependency versions should be accompanied by:

- rerunning the relevant tests;
- checking model deserialization;
- verifying the Streamlit application;
- recording the change in `CHANGELOG.md`.
