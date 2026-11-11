#!/usr/bin/env python3
"""Fixed-hyperparameter, group-aware sensitivity analysis of the two radius descriptors.

Scientific design:
  - Original (mixed convention), consistent Cordero-2008/ASE covalent, no-radius.
  - The ONLY changed columns in the alternative are R_mean and R_std.
  - Full and composition+descriptor representations are considered separately.
  - Identical fixed GroupKFold splits for all representations.
  - Train-only median imputation; original 352-row development split by default.
  - Never overwrites original thesis artefacts or refits the saved production model.

Cordero table: Beatriz Cordero et al., Dalton Transactions (2008),
DOI 10.1039/B801115J; values as collated in ase.data.covalent_radii.
All 16 values are explicitly frozen below (pm); ASE need not be installed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

ELEMENTS = ["Fe", "Mn", "Al", "C", "Ni", "Cr", "Si", "Cu", "Mo", "Nb", "V", "Ti", "Co", "N", "P", "S"]
COMPOSITION = [f"{e} (%wt)" for e in ELEMENTS]
R_COLS = ["R_mean (pm)", "R_std (pm)"]
DESCRIPTORS = ["R_mean (pm)", "R_std (pm)", "Δχ", "VEC_mean"]
SFE = [
    "SFE_Akbari (mJ/m²)", "SFE_Allain (mJ/m²)", "SFE_Dumay (mJ/m²)",
    "SFE_Elaraby (mJ/m²)", "SFE_Wang (mJ/m²)", "SFE_Xiong (mJ/m²)",
    "SFE_Yoo (mJ/m²)", "SFE_Zambrano (mJ/m²)",
]
# Consistent single-bond covalent radii (Cordero et al., 2008), pm, as used by ASE.
COVALENT_PM = dict(zip(ELEMENTS, [132, 139, 121, 76, 124, 139, 111, 132, 154, 164, 153, 160, 126, 71, 107, 105]))


def norm_source(x):
    s = str(x).strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "doi:"):
        if s.startswith(p):
            s = s[len(p):]
    return s.rstrip("/")


def make_groups(raw, mode):
    if mode == "source":
        return pd.factorize(raw["DOI"].map(norm_source), sort=True)[0]
    return pd.factorize(pd.MultiIndex.from_frame(raw[COMPOSITION]), sort=True)[0]


def metric(y, pred):
    return {
        "n": int(len(y)), "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)), "R2": float(r2_score(y, pred)),
    }


def cluster_ci(y, p1, p2, groups, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    idxs = [np.flatnonzero(groups == g) for g in unique]
    draws = np.empty(n_boot, dtype=float)
    for j in range(n_boot):
        picked = rng.integers(0, len(idxs), size=len(idxs))
        ix = np.concatenate([idxs[k] for k in picked])
        draws[j] = np.sqrt(np.mean((y[ix]-p1[ix])**2)) - np.sqrt(np.mean((y[ix]-p2[ix])**2))
    return np.quantile(draws, [0.025, 0.975])


def descriptor_from_radii(wt_pct, atomic_weights, radii):
    if not np.isfinite(wt_pct).all() or (wt_pct < 0).any():
        raise ValueError("Invalid composition weights")
    moles = wt_pct / atomic_weights[None, :]
    x = moles / moles.sum(axis=1, keepdims=True)
    mean = x @ radii
    spread = np.sqrt(np.sum(x * (radii[None, :] - mean[:, None])**2, axis=1))
    return mean, spread


def model_for(columns, params):
    pre = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), columns)],
        remainder="drop", verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", pre), ("model", XGBRegressor(**params))])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="data/processed/output_with_8_sfe_models.csv")
    p.add_argument("--raw", default="data/raw/sfe_literature_dataset.csv")
    p.add_argument("--properties", default="data/metadata/elemental_property_table.csv")
    p.add_argument("--manifest", default="results/01_screening/development_split_manifest.csv")
    p.add_argument("--hyperparameters", default="results/04_final_evaluation/frozen_xgboost_hyperparameters.json")
    p.add_argument("--scope", choices=["development", "all"], default="development")
    p.add_argument("--grouping", choices=["source", "composition", "both"], default="both")
    p.add_argument("--representations", choices=["full", "descriptors", "both"], default="both")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--smoke", action="store_true", help="Use 10 trees for functional check only")
    p.add_argument("--output", default="paper_audit/radius_sensitivity")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.csv)
    raw = pd.read_csv(a.raw)
    meta = pd.read_csv(a.properties)
    manifest = pd.read_csv(a.manifest).sort_values("Row_Index")
    if not (len(df) == len(raw) == len(manifest) == 441):
        raise ValueError("441-row alignment failed")
    if not np.array_equal(df["No."].astype(str), raw["No."].astype(str)):
        raise ValueError("Raw and processed ID columns differ")
    if not np.array_equal(df["No."].astype(str), manifest["ID"].astype(str)):
        raise ValueError("Manifest row ordering differs")
    if not np.allclose(df["SFE_Experimental (mJ/m²)"], raw["SFE"]):
        raise ValueError("Raw/processed target mismatch")
    meta = meta.set_index("element").loc[ELEMENTS]
    mass = meta["atomic_weight_g_mol"].to_numpy(float)
    baseline_radii = meta["atomic_radius_pm"].to_numpy(float)
    alt_radii = np.array([COVALENT_PM[e] for e in ELEMENTS], float)
    original_mean, original_std = descriptor_from_radii(raw[COMPOSITION].to_numpy(float), mass, baseline_radii)
    # Validate that the independent formula reproduces both existing radius columns.
    if not np.allclose(original_mean, df[R_COLS[0]], rtol=0, atol=1e-7):
        raise ValueError("Archived R_mean does not match independently recalculated original values")
    if not np.allclose(original_std, df[R_COLS[1]], rtol=0, atol=1e-7):
        raise ValueError("Archived R_std does not match independently recalculated original values")
    alt_mean, alt_std = descriptor_from_radii(raw[COMPOSITION].to_numpy(float), mass, alt_radii)
    pd.DataFrame({"Element": ELEMENTS, "Original_radius_pm": baseline_radii,
                  "Cordero_2008_covalent_radius_pm": alt_radii,
                  "Difference_pm": alt_radii-baseline_radii}).to_csv(out / "radius_tables_compared.csv", index=False)
    pd.DataFrame({"ID": df["No."], "R_mean_original": original_mean, "R_std_original": original_std,
                  "R_mean_covalent": alt_mean, "R_std_covalent": alt_std}).to_csv(
        out / "radius_descriptor_comparison_all_441.csv", index=False)
    params = json.loads(Path(a.hyperparameters).read_text(encoding="utf-8"))["hyperparameters"].copy()
    if a.smoke:
        params["n_estimators"] = 10
        print("SMOKE MODE: short execution check only; DO NOT interpret reported scores.")
    X = df[COMPOSITION+DESCRIPTORS+SFE].apply(pd.to_numeric, errors="coerce").copy()
    X[SFE] = X[SFE].mask(X[SFE].abs()>1e6)
    X_alt = X.copy()
    X_alt[R_COLS[0]] = alt_mean
    X_alt[R_COLS[1]] = alt_std
    y = df["SFE_Experimental (mJ/m²)"].to_numpy(float)
    keep = np.arange(len(df)) if a.scope == "all" else np.flatnonzero(manifest["Split"].to_numpy()=="train")
    X = X.iloc[keep].reset_index(drop=True)
    X_alt = X_alt.iloc[keep].reset_index(drop=True)
    raw = raw.iloc[keep].reset_index(drop=True)
    y = y[keep]
    sets = {}
    if a.representations in ("full", "both"):
        sets.update({
            "full_original": (X, COMPOSITION + DESCRIPTORS + SFE),
            "full_covalent": (X_alt, COMPOSITION + DESCRIPTORS + SFE),
            "full_no_radius": (X, COMPOSITION + DESCRIPTORS[2:] + SFE),
        })
    if a.representations in ("descriptors", "both"):
        sets.update({
            "descriptors_original": (X, COMPOSITION + DESCRIPTORS),
            "descriptors_covalent": (X_alt, COMPOSITION + DESCRIPTORS),
            "descriptors_no_radius": (X, COMPOSITION + DESCRIPTORS[2:]),
        })
    modes = ["source", "composition"] if a.grouping == "both" else [a.grouping]
    for mode in modes:
        groups = make_groups(raw, mode)
        splits = list(GroupKFold(n_splits=a.folds).split(X,y,groups))
        folds = np.full(len(y),-1, dtype=int)
        for k,(tr,va) in enumerate(splits,1):
            if set(groups[tr]) & set(groups[va]):
                raise AssertionError("Group leakage in GroupKFold")
            folds[va] = k
        outpred = pd.DataFrame({"ID": raw["No."].to_numpy(), "Fold": folds, "Group_ID": groups, "Actual":y})
        fmetrics = []
        for name,(features,columns) in sets.items():
            preds = np.full(len(y), np.nan)
            for k,(tr,va) in enumerate(splits,1):
                model = model_for(columns, params)
                model.fit(features.iloc[tr][columns],y[tr])
                preds[va] = model.predict(features.iloc[va][columns])
                fmetrics.append({"Grouping": mode, "Scope": a.scope, "Representation": name,
                                 "Fold":k,**metric(y[va], preds[va])})
                print(f"{mode}: {name} fold {k}/{len(splits)} complete", flush=True)
            if not np.isfinite(preds).all():
                raise AssertionError("One or more OOF predictions missing or nonfinite")
            outpred[name] = preds
        outpred.to_csv(out / f"{mode}_{a.scope}_radius_OOF_predictions.csv",index=False)
        pd.DataFrame(fmetrics).to_csv(out / f"{mode}_{a.scope}_radius_fold_metrics.csv",index=False)
        summary = pd.DataFrame([{"Grouping":mode,"Scope":a.scope,"Representation":name,
                                 **metric(y,outpred[name].to_numpy(float))} for name in sets])
        summary.to_csv(out / f"{mode}_{a.scope}_radius_summary.csv",index=False)
        pair_rows = []
        for prefix in ("full", "descriptors"):
            base, alt, drop = [f"{prefix}_{t}" for t in ("original", "covalent", "no_radius")]
            if base not in sets:
                continue
            for contrast, new in [("covalent_minus_original",alt), ("no_radius_minus_original",drop)]:
                p_orig=outpred[base].to_numpy(float)
                p_new=outpred[new].to_numpy(float)
                # POSITIVE => new representation improves RMSE relative to original.
                delta=metric(y,p_orig)["RMSE"]-metric(y,p_new)["RMSE"]
                ci=cluster_ci(y,p_orig,p_new,groups,a.bootstrap)
                pair_rows.append({"Grouping":mode,"Scope":a.scope,"Representation":prefix,
                                  "Comparison":contrast,"RMSE_original_minus_alternative":delta,
                                  "CI_lower":float(ci[0]),"CI_upper":float(ci[1]),
                                  "Bootstrap_groups":mode,
                                  "Caveat":"Conditional on fixed OOF predictions; no model refitting in bootstrap",
                                  "SMOKE_ONLY":a.smoke})
        pd.DataFrame(pair_rows).to_csv(out / f"{mode}_{a.scope}_radius_paired_differences.csv",index=False)
        print(summary.to_string(index=False),flush=True)
        print(pd.DataFrame(pair_rows)[["Representation","Comparison","RMSE_original_minus_alternative","CI_lower","CI_upper"]].to_string(index=False),flush=True)
    metadata = {"scope":a.scope,"grouping":modes,"folds":a.folds,"bootstrap_draws":a.bootstrap,
                "smoke_only":a.smoke,"radius_source":"Cordero et al 2008; ASE covalent_radii",
                "covalent_radius_pm":COVALENT_PM,"original_radius_descriptor_audit":"PASSED",
                "original_holdout_used_for_training":False if a.scope=="development" else "post-selection full-data sensitivity"}
    (out/"radius_experiment_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print("Saved sensitivity outputs:",out.resolve())

if __name__=="__main__":
    main()
