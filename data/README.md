# Data

This directory contains the final datasets and metadata used by the SFE prediction workflow.

## Structure

```text
data/
├── raw/
│   └── sfe_literature_dataset.csv
├── processed/
│   └── output_with_8_sfe_models.csv
├── metadata/
│   ├── data_dictionary.csv
│   ├── literature_sources.csv
│   ├── elemental_property_table.csv
│   ├── sfe_model_registry.csv
│   └── dataset_provenance.md
└── README.md
```

## Raw dataset

`sfe_literature_dataset.csv` is the final compiled experimental literature dataset used as the starting point for the modelling workflow.

Final dataset properties:

- 441 rows;
- experimental SFE targets only;
- 16 composition variables;
- all compositions expressed in wt.%;
- Fe treated as the balance element during dataset preparation;
- all final compositions close to 100 wt.% within floating-point tolerance;
- 354 unique exact nominal compositions;
- 47 repeated-composition groups involving 134 observations;
- no exact full-row duplicates;
- no duplicate observations with both identical composition and identical experimental SFE.

The target range is approximately:

```text
3.26–131 mJ/m²
```

The final dataset is maintained at room temperature (300 K).

## Processed modelling table

`output_with_8_sfe_models.csv` is the canonical modelling table used by the final machine-learning scripts.

It contains:

- identifier column;
- 16 composition variables;
- four composition-derived descriptors;
- eight retained published SFE-model outputs;
- experimental SFE target.

This produces a 28-predictor feature matrix.

### Descriptor definitions

Atomic fractions are calculated from wt.% using:

```text
n_i = w_i / A_i
x_i = n_i / Σ n_i
```

The four descriptors are:

```text
R_mean = Σ x_i R_i

R_std = sqrt[Σ x_i (R_i - R_mean)²]

Δχ = sqrt[Σ x_i (χ_i - χ_mean)²]

VEC_mean = Σ x_i VEC_i
```

## Retained SFE-model outputs

The final processed table contains:

- Akbari;
- Allain;
- Dumay;
- Elaraby;
- Wang;
- Xiong;
- Yoo;
- Zambrano.

## Cleaning rule for published SFE outputs

For the retained published SFE-model predictors, values satisfying:

```text
|SFE_model_output| > 1e6
```

are treated as invalid numerical outputs and replaced with missing values before model fitting. Median imputation is then fitted within the relevant training partition through the modelling pipeline.

This rule is part of the final reproducible preprocessing procedure.

## Exact-composition grouping

Where grouped validation is required, the group identifier is created from the exact numeric tuple of all 16 composition columns.

No rounding, tolerance-based clustering, or near-duplicate merging is used for grouped validation.

This prevents repeated observations with the same nominal composition from being split across training and validation folds.

## Provenance

`metadata/literature_sources.csv` maps the 51 dataset source strings to observations and contains completed authors, year, title, publication, volume/issue, pages/article number, DOI URL, reference type and UWE-Harvard-style bibliographic reference.

`metadata/dataset_provenance.md` documents dataset provenance, bibliographic attribution status, formulation references, descriptor-method references and the remaining elemental-constant provenance limitation.

`metadata/data_dictionary.csv` defines the modelling fields and links derived SFE features to their provenance/reference information.

`metadata/elemental_property_table.csv` records the exact constants used by the descriptor-generation script; the original external source of that exact constant set is not established from the submitted material.

`metadata/sfe_model_registry.csv` records the implementation details and full bibliographic references for the eight retained published SFE formulations.

## Redistribution

Only data that can be redistributed legally should be committed to the public repository.

The repository should not include copies of copyrighted journal PDFs or third-party supplementary files unless redistribution rights explicitly allow it. Literature provenance should instead be preserved through DOI and source metadata.
