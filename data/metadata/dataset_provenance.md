# Dataset provenance

## Overview

This repository uses a curated literature dataset of **441 experimental SFE observations** for multicomponent steels. The raw dataset contains **16 composition variables**, the experimental stacking-fault-energy value, and publication/measurement metadata. The processed modelling table contains the same 441 observations and adds four composition-derived descriptors plus eight published SFE-model outputs.

The raw dataset contains 22 columns and the processed modelling table contains 30 columns.

## Experimental target and inclusion scope

Only observations classified as **Experimental** are included. All 441 observations are represented at **300 K** in the curated dataset.

The experimental SFE target spans **3.26 to 131 mJ/m²**.

Experimental techniques represented in the dataset are:

- TEM: 322 observations
- XRD: 102 observations
- Neutron Diffraction: 17 observations

The dataset contains **51 distinct DOI/source strings**. `literature_sources.csv` maps each source string to the relevant dataset observation numbers and experimental technique(s).

## Composition representation

The 16 composition variables are:

Fe, Mn, Al, C, Ni, Cr, Si, Cu, Mo, Nb, V, Ti, Co, N, P, and S.

Compositions are stored in wt.%. Fe is represented as the balance component so that each curated composition closes to 100 wt.%. The maximum absolute closure difference in the supplied dataset is 4.263e-14 wt.%, which is consistent with floating-point representation.

The final dataset contains **354 exact nominal compositions**. **47 exact-composition groups** contain more than one experimental observation, comprising **134 observations** in total. Repeated nominal compositions are retained when they correspond to distinct experimental SFE observations.

For grouped validation, an exact composition group is defined by the complete numerical tuple of all 16 composition variables. No rounding or near-duplicate merging is applied.

## Duplicate handling

Exact duplicate observations with identical nominal composition and identical experimental SFE were removed during curation. Repeated nominal compositions with different experimental SFE measurements were retained to preserve experimental variability.

## Composition-derived descriptors

`SFE_with_Descriptors.py` converts wt.% composition to atomic fractions using

`n_i = w_i / A_i`

and

`x_i = n_i / Σ_j n_j`

where `w_i` is the wt.% amount and `A_i` is atomic weight.

Four descriptors are then calculated:

- **R_mean (pm)**: `Σ_i x_i R_i`
- **R_std (pm)**: `sqrt(Σ_i x_i (R_i - R_mean)^2)`
- **Δχ**: `sqrt(Σ_i x_i (χ_i - χ_mean)^2)`
- **VEC_mean**: `Σ_i x_i VEC_i`

All 16 represented elements are included in the radius, electronegativity, and VEC descriptor calculations. The exact constants encoded by the script are reproduced in `elemental_property_table.csv`.

## Published SFE-model outputs

Eight published SFE formulations are implemented as composition-derived engineered predictors:

- Akbari
- Allain
- Dumay
- Elaraby
- Wang
- Xiong
- Yoo
- Zambrano

The implementation script, function name, required elemental inputs, and output-column name are documented in `sfe_model_registry.csv`.

The processed CSV stores the directly calculated numerical outputs. During machine-learning preprocessing, published SFE-model values with absolute magnitude greater than `1e6 mJ/m²` are treated as missing values, and median imputation is fitted using training data only.

## Raw versus processed datasets

### Raw dataset

Canonical repository location:

`data/raw/sfe_literature_dataset.csv`

Contains:

- observation identifier
- 16 wt.% composition columns
- experimental SFE (`SFE`)
- DOI/source string
- temperature
- method
- experimental technique

### Processed modelling dataset

Canonical repository location:

`data/processed/output_with_8_sfe_models.csv`

Contains:

- observation identifier
- 16 composition predictors
- four composition-derived descriptors
- eight published SFE-model outputs
- canonical target `SFE_Experimental (mJ/m²)`

The final predictor count is **28**: 16 composition variables + 4 descriptors + 8 published SFE-model outputs.

`No.` is an identifier and is not used as a predictor. DOI, temperature, method, and technique are retained in the raw dataset for traceability and are not model predictors.

## Reproducibility

The feature-generation definitions in this metadata package were extracted from the supplied final `SFE_with_Descriptors.py` and `SFE_Master.py` scripts. The raw and processed datasets both contain 441 observations, and the processed target matches the raw experimental SFE values.

## Bibliographic metadata and attribution status

Bibliographic traceability has been completed for the literature-derived dataset and the eight retained published SFE formulations.

### Literature-derived experimental observations

`literature_sources.csv` maps all **51 DOI/source strings** to the dataset observation numbers and also records authors, year, publication title, journal/proceedings title, volume/issue, page or article number, DOI URL, reference type, and a UWE-Harvard-style bibliographic reference. The observation mapping, temperature, method, and technique fields remain those supplied by the project dataset. Bibliographic fields were checked against DOI-linked publisher/index metadata on **22 August 2026**.

These bibliographic records provide source traceability. They do **not**, by themselves, establish permission to redistribute publisher content or any source dataset; copyrighted publications are referenced rather than redistributed.

### Published SFE formulations

`sfe_model_registry.csv` now contains the full bibliographic reference for each retained formulation: Saeed-Akbari et al. (2009), Allain et al. (2004), Dumay et al. (2008), Elaraby et al. (2026), Wang and Xiong (2020), Xiong et al. (2014), Yoo and Park (2008), and Zambrano (2016). These published equations remain prior work. The project contribution is their documented Python implementation, numerical quality control, joint use as engineered predictors, and comparative evaluation.

### Composition-derived descriptor references

The use of composition-derived elemental descriptors is supported by the following references used in the dissertation:

- Liu, C. and Su, H. (2024) Prediction and rational design of stacking fault energy of austenitic alloys based on interpretable machine learning and chemical composition. Steel Research International [online]. 95 (8): 2400225. Available from: https://doi.org/10.1002/srin.202400225 [Accessed 22 August 2026].
- Song, L., Wang, C., Li, Y. and Wei, X. (2025) Predicting stacking fault energy in austenitic stainless steels via physical metallurgy-based machine learning approaches. Journal of Materials Informatics [online]. 5: 2. Available from: https://doi.org/10.20517/jmi.2024.70 [Accessed 22 August 2026].

The exact atomic-weight, atomic-radius, Pauling-electronegativity, and nominal-VEC constants in `elemental_property_table.csv` are reproduced from the final `SFE_with_Descriptors.py` implementation.
