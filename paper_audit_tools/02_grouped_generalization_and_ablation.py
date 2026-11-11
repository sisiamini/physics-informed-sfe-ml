#!/usr/bin/env python3
"""Post-selection grouped CV / four-way ablation; NEVER modifies thesis artefacts.

All feature sets are evaluated with the SAME frozen original full-feature XGBoost
hyperparameters and matched GroupKFold splits. Per-fold median imputation is fitted
on the training fold. Source groups use normalized raw DOI/source strings.

New paper experiment, NOT a reproduction of the thesis' original row-wise ablation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

COMPOSITION=['Fe (%wt)','Mn (%wt)','Al (%wt)','C (%wt)','Ni (%wt)','Cr (%wt)','Si (%wt)','Cu (%wt)','Mo (%wt)','Nb (%wt)','V (%wt)','Ti (%wt)','Co (%wt)','N (%wt)','P (%wt)','S (%wt)']
DESCRIPTORS=['R_mean (pm)','R_std (pm)','Δχ','VEC_mean']
SFE=['SFE_Akbari (mJ/m²)','SFE_Allain (mJ/m²)','SFE_Dumay (mJ/m²)','SFE_Elaraby (mJ/m²)','SFE_Wang (mJ/m²)','SFE_Xiong (mJ/m²)','SFE_Yoo (mJ/m²)','SFE_Zambrano (mJ/m²)']
SETS={'composition':COMPOSITION,'composition_descriptors':COMPOSITION+DESCRIPTORS,'composition_sfe_outputs':COMPOSITION+SFE,'full':COMPOSITION+DESCRIPTORS+SFE}

def norm_source(x):
    s=str(x).strip().lower()
    for p in ('https://doi.org/','http://doi.org/','doi:'):
        if s.startswith(p): s=s[len(p):]
    return s.rstrip('/')

def as_groups(df,mode):
    if mode=='source':
        return pd.factorize(df['DOI'].map(norm_source),sort=True)[0]
    return pd.factorize(pd.MultiIndex.from_frame(df[COMPOSITION]),sort=True)[0]

def build_model(columns,params):
    pre=ColumnTransformer([('numeric',SimpleImputer(strategy='median'),columns)],remainder='drop',verbose_feature_names_out=False)
    return Pipeline([('preprocessor',pre),('model',XGBRegressor(**params))])

def metrics(y,p):
    return {'n':len(y),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'MAE':float(mean_absolute_error(y,p)),'R2':float(r2_score(y,p))}

def group_bootstrap(pred,groups,key1,key2,n=1000,seed=42):
    """Paired cluster-bootstrap CI with fixed OOF predictions; not a model-refitting CI."""
    # pred index = row position with one prediction for each feature set
    gr=np.asarray(groups)
    unique=np.unique(gr)
    clusters=[np.flatnonzero(gr==u) for u in unique]
    rng=np.random.default_rng(seed)
    y=pred['Actual'].to_numpy(); p1=pred[key1].to_numpy(); p2=pred[key2].to_numpy()
    diffs=[]
    for _ in range(n):
        draw=rng.integers(0,len(clusters),len(clusters))
        idx=np.concatenate([clusters[i] for i in draw])
        diffs.append(np.sqrt(np.mean((y[idx]-p1[idx])**2))-np.sqrt(np.mean((y[idx]-p2[idx])**2)))
    return np.quantile(diffs,[0.025,0.5,0.975])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='data/processed/output_with_8_sfe_models.csv')
    p.add_argument('--raw',default='data/raw/sfe_literature_dataset.csv')
    p.add_argument('--manifest',default='results/01_screening/development_split_manifest.csv')
    p.add_argument('--hyperparameters',default='results/04_final_evaluation/frozen_xgboost_hyperparameters.json')
    p.add_argument('--output',default='paper_audit/new_experiments')
    p.add_argument('--grouping',choices=['source','composition','both'],default='both')
    p.add_argument('--scope',choices=['development','all'],default='development')
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--smoke',action='store_true',help='Set n_estimators=10 for a quick FUNCTIONAL smoke test; do not publish smoke metrics')
    a=p.parse_args()
    df=pd.read_csv(a.csv); raw=pd.read_csv(a.raw); manifest=pd.read_csv(a.manifest).sort_values('Row_Index')
    assert len(df)==len(raw)==len(manifest)==441
    assert np.array_equal(df['No.'].astype(str),raw['No.'].astype(str)), 'Raw/processed ID mismatch'
    assert np.array_equal(df['No.'].astype(str),manifest['ID'].astype(str)), 'Manifest ID mismatch'
    assert np.allclose(pd.to_numeric(df['SFE_Experimental (mJ/m²)']),pd.to_numeric(raw['SFE'])), 'Target mismatch'
    assert set(SETS['full']).issubset(df.columns)
    params=json.loads(Path(a.hyperparameters).read_text())['hyperparameters']
    if a.smoke:
        params['n_estimators']=10
        print('SMOKE MODE: 10 trees; these scores are NOT scientific estimates.')
    X=df[SETS['full']].apply(pd.to_numeric,errors='coerce').copy()
    X[SFE]=X[SFE].mask(X[SFE].abs()>1e6)
    y=df['SFE_Experimental (mJ/m²)'].to_numpy(dtype=float)
    data=df[['No.']+COMPOSITION].copy(); data['DOI']=raw['DOI']
    keep=np.arange(len(df)) if a.scope=='all' else np.flatnonzero(manifest['Split'].to_numpy()=='train')
    X=X.iloc[keep].reset_index(drop=True); y=y[keep]; data=data.iloc[keep].reset_index(drop=True)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    modes=['source','composition'] if a.grouping=='both' else [a.grouping]
    for mode in modes:
        print(f'GROUPING: {mode}; SCOPE: {a.scope}; n={len(data)}; {a.folds} folds',flush=True)
        groups=as_groups(data,mode)
        assert len(np.unique(groups)) >= a.folds
        splits=list(GroupKFold(n_splits=a.folds).split(X,y,groups))
        # Fold assignment and explicit disjointness audit.
        assignments=np.empty(len(y),dtype=int)
        fold_rows=[]
        for k,(tr,va) in enumerate(splits,1):
            overlap=set(groups[tr])&set(groups[va]); assert not overlap
            assignments[va]=k
            other_mode = 'composition' if mode=='source' else 'source'
            other_groups = as_groups(data, other_mode)
            other_overlap = len(set(other_groups[tr]) & set(other_groups[va]))
            fold_rows.append({'Fold':k,'Train_n':len(tr),'Validation_n':len(va),'Train_groups':len(np.unique(groups[tr])),'Validation_groups':len(np.unique(groups[va])),'Group_overlap':0,'Other_Grouping':other_mode,'Other_Grouping_Overlap':other_overlap,'Validation_target_mean':float(y[va].mean())})
        result=pd.DataFrame({'ID':data['No.'].to_numpy(),'Fold':assignments,'Group_ID':groups,'Actual':y})
        rows=[]
        for name,columns in SETS.items():
            predicted=np.full(len(y),np.nan)
            for fold,(tr,va) in enumerate(splits,1):
                model=build_model(columns,params)
                model.fit(X.iloc[tr][columns],y[tr])
                v=model.predict(X.iloc[va][columns])
                predicted[va]=v
                rows.append({'Grouping':mode,'Scope':a.scope,'Feature_Set':name,'Fold':fold,**metrics(y[va],v)})
                print(f'  {name}, fold {fold}/{len(splits)} complete',flush=True)
            assert np.all(np.isfinite(predicted))
            result[name]=predicted
        result.to_csv(out/f'{mode}_{a.scope}_OOF_predictions.csv',index=False)
        pd.DataFrame(rows).to_csv(out/f'{mode}_{a.scope}_fold_metrics.csv',index=False)
        pd.DataFrame(fold_rows).to_csv(out/f'{mode}_{a.scope}_fold_audit.csv',index=False)
        summary=[]
        for name in SETS:
            summary.append({'Grouping':mode,'Scope':a.scope,'Feature_Set':name,**metrics(y,result[name].to_numpy())})
        sm=pd.DataFrame(summary)
        sm.to_csv(out/f'{mode}_{a.scope}_pooled_summary.csv',index=False)
        c1=sm.loc[sm.Feature_Set=='composition','RMSE'].item(); c2=sm.loc[sm.Feature_Set=='full','RMSE'].item()
        ci=group_bootstrap(result,groups,'composition','full',n=a.bootstrap)
        pd.DataFrame([{'Grouping':mode,'Scope':a.scope,'RMSE_composition_minus_full':c1-c2,'Percent_RMSE_improvement':100*(c1-c2)/c1,'Group_bootstrap_CI_2_5':ci[0],'Group_bootstrap_median':ci[1],'Group_bootstrap_CI_97_5':ci[2],'CI_note':'Conditional on fixed OOF predictions; clustered by exact group; does not account for model refitting','SMOKE_ONLY':a.smoke}]).to_csv(out/f'{mode}_{a.scope}_paired_difference.csv',index=False)
        print(sm.to_string(index=False),f'\nRMSE improvement (composition - full): {c1-c2:+.4f}; CI {ci[0]:+.4f}, {ci[2]:+.4f}',flush=True)
    print('Complete. DO NOT report smoke-mode scores as scientific results.')

if __name__=='__main__':main()
