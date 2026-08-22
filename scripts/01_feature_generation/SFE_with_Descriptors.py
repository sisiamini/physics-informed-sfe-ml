#!/usr/bin/env python3
# SFE_with_Descriptors.py
"""
Generate descriptor features from alloy compositions and prepare
an SFE dataset for machine learning.

Input:
    CSV containing composition columns and an "SFE" column.

Output:
    CSV containing:
        - Original composition columns
        - Calculated descriptors:
              R_mean (pm)
              R_std (pm)
              Δχ
              VEC_mean
        - SFE_Experimental (mJ/m²)

Important:
    - Input compositions are assumed to be given in wt.%.
    - Descriptor calculations are performed using atomic fractions.
    - Atomic-radius descriptors are calculated using all 16 elements.
    - Electronegativity spread and VEC are calculated using all elements.

Run:
    python SFE_with_Descriptors.py \
        --csv "data/input.csv" \
        --output "data/output.csv"

Example:
    python SFE_with_Descriptors.py \
        --csv "source folder/source file.csv" \
        --output "destination folder/destination file.csv"
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# -----------------------------
# Input / output files
# -----------------------------
parser = argparse.ArgumentParser()

parser.add_argument(
    "--csv",
    required=True,
    type=Path,
    help="Input CSV file"
)

parser.add_argument(
    "--output",
    required=True,
    type=Path,
    help="Output CSV file"
)

args = parser.parse_args()

INPUT_CSV = args.csv
OUTPUT_CSV = args.output

# -----------------------------
# Required composition columns
# -----------------------------
ELEMENT_COLS = [
    "Fe (%wt)", "Mn (%wt)", "Al (%wt)", "C (%wt)", "Ni (%wt)",
    "Cr (%wt)", "Si (%wt)", "Cu (%wt)", "Mo (%wt)", "Nb (%wt)",
    "V (%wt)", "Ti (%wt)", "Co (%wt)", "N (%wt)", "P (%wt)", "S (%wt)"
]

# -----------------------------
# Elemental properties
# Order must match ELEMENT_COLS
# -----------------------------

# Atomic radius, pm
ATOMIC_R = np.array([
    126, 127, 143, 70, 124,
    128, 111, 128, 139, 146,
    134, 147, 125, 65, 100, 100
], dtype=float)

# Pauling electronegativity
CHI = np.array([
    1.83, 1.55, 1.61, 2.55, 1.91,
    1.66, 1.90, 1.90, 2.16, 1.60,
    1.63, 1.54, 1.88, 3.04, 2.19, 2.58
], dtype=float)

# Valence electron count
VEC = np.array([
    8, 7, 3, 4, 10,
    6, 4, 11, 6, 5,
    5, 4, 9, 5, 5, 6
], dtype=float)

# Standard atomic weights, g/mol
# Used to convert wt.% composition to atomic fraction
ATOMIC_WEIGHT = np.array([
    55.845,       # Fe
    54.938044,    # Mn
    26.9815385,   # Al
    12.011,       # C
    58.6934,      # Ni
    51.9961,      # Cr
    28.085,       # Si
    63.546,       # Cu
    95.95,        # Mo
    92.90637,     # Nb
    50.9415,      # V
    47.867,       # Ti
    58.933194,    # Co
    14.007,       # N
    30.973762,    # P
    32.06         # S
], dtype=float)

# -----------------------------
# Atomic-radius element mask
# -----------------------------
#
# Order must match ELEMENT_COLS.
SUBSTITUTIONAL_MASK = np.array([
    True,   # Fe
    True,   # Mn
    True,   # Al
    True,   # C
    True,   # Ni
    True,   # Cr
    True,   # Si
    True,   # Cu
    True,   # Mo
    True,   # Nb
    True,   # V
    True,   # Ti
    True,   # Co
    True,   # N
    True,   # P
    True    # S
], dtype=bool)

# -----------------------------
# Read dataset
# -----------------------------
df = pd.read_csv(INPUT_CSV)

# -----------------------------
# Ensure all composition columns exist
# Missing element columns are set to 0
# -----------------------------
# This assumes that a completely absent element column means that
# the element is not present in the dataset.
for col in ELEMENT_COLS:
    if col not in df.columns:
        df[col] = 0.0

# Convert composition values to numeric.
# Invalid entries inside existing columns are not silently replaced
# with zero because they may represent missing or corrupted data.
df[ELEMENT_COLS] = df[ELEMENT_COLS].apply(
    pd.to_numeric,
    errors="coerce"
)

if df[ELEMENT_COLS].isna().any().any():
    bad_rows = df.index[
        df[ELEMENT_COLS].isna().any(axis=1)
    ].tolist()

    raise ValueError(
        "Missing or invalid composition values detected "
        f"in dataframe rows: {bad_rows}"
    )

# -----------------------------
# Validate compositions
# -----------------------------
wt_pct = df[ELEMENT_COLS].to_numpy(dtype=float)

# Negative compositions are not physical
if (wt_pct < 0.0).any():
    bad_rows = np.where((wt_pct < 0.0).any(axis=1))[0].tolist()

    raise ValueError(
        "Negative composition values detected "
        f"in dataframe rows: {bad_rows}"
    )

# Rows containing no composition cannot produce physical descriptors
row_sum = wt_pct.sum(axis=1)

if (row_sum <= 0.0).any():
    bad_rows = np.where(row_sum <= 0.0)[0].tolist()

    raise ValueError(
        "Rows with zero total composition detected "
        f"in dataframe rows: {bad_rows}"
    )

# -----------------------------
# Convert wt.% to atomic fraction
# -----------------------------
# Relative number of moles:
#
#     n_i = w_i / A_i
#
# where w_i is the reported mass amount (wt.% is sufficient because
# the common factor of 100 cancels during normalization) and A_i is
# the atomic weight.
relative_moles = wt_pct / ATOMIC_WEIGHT[None, :]

# Atomic fraction:
#
#     x_i = n_i / sum_j(n_j)
atomic_sum = relative_moles.sum(axis=1, keepdims=True)

x = relative_moles / atomic_sum

# -----------------------------
# Calculate descriptors
# -----------------------------

# -----------------------------
# Atomic-radius descriptors
# -----------------------------
# All 16 represented elements are included here.
# The selected atomic fractions are renormalized so that they sum
# to 1 for each alloy.

x_sub = x[:, SUBSTITUTIONAL_MASK]

x_sub_sum = x_sub.sum(axis=1, keepdims=True)

if (x_sub_sum[:, 0] <= 0.0).any():
    bad_rows = np.where(x_sub_sum[:, 0] <= 0.0)[0].tolist()

    raise ValueError(
        "Rows containing no substitutional elements detected "
        f"in dataframe rows: {bad_rows}"
    )

x_sub = x_sub / x_sub_sum

r_sub = ATOMIC_R[SUBSTITUTIONAL_MASK][None, :]

# Composition-weighted mean atomic radius, pm
R_mean = (x_sub * r_sub).sum(axis=1)

# Composition-weighted standard deviation of atomic radius, pm
R_std = np.sqrt(
    (
        x_sub
        * (r_sub - R_mean[:, None]) ** 2
    ).sum(axis=1)
)

# -----------------------------
# Electronegativity descriptor
# -----------------------------
# Electronegativity spread is calculated using all elements,
# including interstitial C and N.

chi = CHI[None, :]

chi_mean = (x * chi).sum(axis=1)

delta_chi = np.sqrt(
    (
        x
        * (chi - chi_mean[:, None]) ** 2
    ).sum(axis=1)
)

# -----------------------------
# Valence-electron descriptor
# -----------------------------
# Mean nominal valence-electron count is calculated using all
# elements and their atomic fractions.

vec = VEC[None, :]

VEC_mean = (x * vec).sum(axis=1)

# -----------------------------
# Add descriptors to dataframe
# -----------------------------
df["R_mean (pm)"] = R_mean
df["R_std (pm)"] = R_std
df["Δχ"] = delta_chi
df["VEC_mean"] = VEC_mean

# -----------------------------
# Rename SFE column
# -----------------------------
if "SFE" not in df.columns:
    raise ValueError("Input CSV must contain a column named 'SFE'.")

df["SFE_Experimental (mJ/m²)"] = df["SFE"]

# -----------------------------
# Add No. column
# -----------------------------
df["No."] = np.arange(1, len(df) + 1)

# -----------------------------
# Final column order
# -----------------------------
OUTPUT_COLS = [
    "No.",
    "Fe (%wt)", "Mn (%wt)", "Al (%wt)", "C (%wt)", "Ni (%wt)",
    "Cr (%wt)", "Si (%wt)", "Cu (%wt)", "Mo (%wt)", "Nb (%wt)",
    "V (%wt)", "Ti (%wt)", "Co (%wt)", "N (%wt)", "P (%wt)", "S (%wt)",
    "R_mean (pm)", "R_std (pm)", "Δχ", "VEC_mean",
    "SFE_Experimental (mJ/m²)"
]

out = df[OUTPUT_COLS]

# -----------------------------
# Save output
# -----------------------------
# Create the output directory if it does not already exist.
OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

out.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print(f"Saved: {OUTPUT_CSV}")