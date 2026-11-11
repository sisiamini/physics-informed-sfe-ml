#!/usr/bin/env python3
"""Training-only unsupervised applicability baselines on SAVED original nested OOF predictions.

This is a comparative screening analysis. It reuses the archived, held-out OUTER predictions,
without refitting the XGBoost model. Every baseline's feature scaling, neighbour reference and
cutoff are fitted separately on that outer fold's training rows (no target errors used).
Original combined-domain membership is taken from archived nested OOF evaluation.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

COMPOSITION=['Fe (%wt)','Mn (%wt)','Al (%wt)','C (%wt)','Ni (%wt)','Cr (%wt)','Si (%wt)','Cu (%wt)','Mo (%wt)','Nb (%wt)','V (%wt)','Ti (%wt)','Co (%wt)','N (%wt)','P (%wt)','S (%wt)']

def summ(df,flags,label):
    actual=df.Actual.to_numpy(float);pred=df.Predicted.to_numpy(float)
    inside=np.asarray(flags,bool); n=int(inside.sum())
    def rmse(x):return float(np.sqrt(np.mean(x*x))) if len(x) else np.nan
    err=actual-pred
    ape=np.abs(err)/np.maximum(np.abs(actual),1e-12)
    return {'Method':label,'Total_n':len(df),'Inside_n':n,'Coverage':n/len(df),'Inside_RMSE':rmse(err[inside]),'Outside_RMSE':rmse(err[~inside]),'Inside_MAE':float(np.mean(abs(err[inside]))) if n else np.nan,'Inside_APE_le_20pct':float(np.mean(ape[inside]<=0.20)) if n else np.nan,'Outside_APE_le_20pct':float(np.mean(ape[~inside]<=0.20)) if n<len(df) else np.nan}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',default='data/raw/sfe_literature_dataset.csv')
    p.add_argument('--original',default='results/07_applicability_domain/nested_domain_oof_diagnostics.csv')
    p.add_argument('--output',default='paper_audit/domain_baselines')
    p.add_argument('--k',type=int,default=10)
    a=p.parse_args();o=Path(a.output);o.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(a.raw).reset_index(drop=True)
    orig=pd.read_csv(a.original)
    assert len(raw)==len(orig)==441
    assert orig['Dataset_Row_Index'].is_unique
    orig=orig.sort_values('Dataset_Row_Index').reset_index(drop=True)
    assert np.array_equal(raw['No.'].astype(str),orig['ID'].astype(str))
    assert np.allclose(raw.SFE,orig.Actual)
    raw_X=raw[COMPOSITION].to_numpy(float)
    out=orig.copy()
    for name in ['range_minmax','knn_50pct_training','knn_90pct_training']:
        out[name]=False
    for fold in sorted(out.Outer_Fold.unique()):
        va=np.flatnonzero(out.Outer_Fold.to_numpy()==fold)
        tr=np.flatnonzero(out.Outer_Fold.to_numpy()!=fold)
        xtr=raw_X[tr];xva=raw_X[va]
        # Inclusive min/max baseline: training-only, intentionally simple.
        lower=xtr.min(axis=0);upper=xtr.max(axis=0)
        out.loc[va,'range_minmax']=((xva>=lower)&(xva<=upper)).all(axis=1)
        # Standardization and distance cutoffs are fitted with outer-training rows only.
        scaler=StandardScaler().fit(xtr)
        s_tr=scaler.transform(xtr);s_va=scaler.transform(xva)
        # Self-excluded kth neighbour distance on training; external kth on validation.
        k=min(a.k,len(tr)-1)
        nbr=NearestNeighbors(n_neighbors=k+1).fit(s_tr)
        tr_d=nbr.kneighbors(s_tr,return_distance=True)[0][:,k]
        va_d=nbr.kneighbors(s_va,n_neighbors=k,return_distance=True)[0][:,k-1]
        for label,q in [('knn_50pct_training',0.5),('knn_90pct_training',0.9)]:
            threshold=np.quantile(tr_d,q)
            out.loc[va,label]=(va_d<=threshold)
    methods=['inside_final_applicability_domain','range_minmax','knn_50pct_training','knn_90pct_training']
    results=[summ(out,out[m],m) for m in methods]
    pd.DataFrame(results).to_csv(o/'domain_baseline_pooled_summary.csv',index=False)
    byfold=[]
    for fold,sl in out.groupby('Outer_Fold'):
        for m in methods:
            row=summ(sl,sl[m],m);row['Outer_Fold']=fold;byfold.append(row)
    pd.DataFrame(byfold).to_csv(o/'domain_baseline_fold_summary.csv',index=False)
    out[['Dataset_Row_Index','ID','Outer_Fold','Actual','Predicted']+methods].to_csv(o/'domain_baseline_classifications.csv',index=False)
    print(pd.DataFrame(results).to_string(index=False))
    print('Interpret carefully: the methods have DIFFERENT coverage; this is not a matched-coverage or independently re-tuned-domain comparison.')
    print('All new baseline thresholds are obtained from outer-training COMPOSITIONS, never validation outcomes.')

if __name__=='__main__':main()
