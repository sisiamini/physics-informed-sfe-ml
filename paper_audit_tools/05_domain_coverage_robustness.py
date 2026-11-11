#!/usr/bin/env python3
"""Applicability-domain robustness on archived NESTED grouped outer-fold predictions.

- Keep exact original outer-fold predictions and membership unchanged.
- On each outer train fold, fit StandardScaler and 10-NN distances using only
  composition columns; estimate threshold grid from self-excluded training distances.
- Compare original method with outcome-blind kNN cutoffs over multiple quantiles.
- Additional EQUAL-COUNT DIAGNOSTIC chooses the n nearest validation compositions
  in each fold, where n equals original domain count for that fold; this uses
  held-out covariates to match coverage and must not be claimed as a deployable rule.
- Never use validation SFE targets to select thresholds or membership.
- Explanatory study, not an independent confirmatory benchmark.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COMPOSITION=['Fe (%wt)','Mn (%wt)','Al (%wt)','C (%wt)','Ni (%wt)','Cr (%wt)',
             'Si (%wt)','Cu (%wt)','Mo (%wt)','Nb (%wt)','V (%wt)','Ti (%wt)',
             'Co (%wt)','N (%wt)','P (%wt)','S (%wt)']
ORIGINAL='inside_final_applicability_domain'

def metric_frame(d, flags, name, fold=None):
    y=d['Actual'].to_numpy(float);p=d['Predicted'].to_numpy(float)
    flags=np.asarray(flags,bool)
    err=y-p
    denom=np.maximum(abs(y),1e-12)
    in_n=int(flags.sum())
    def rmse(vals): return float(np.sqrt(np.mean(vals*vals))) if len(vals) else np.nan
    return {'Method':name,'Outer_Fold': fold if fold is not None else 'pooled',
            'Total_n':int(len(d)),'Inside_n':in_n,'Coverage':in_n/len(d),
            'Inside_RMSE':rmse(err[flags]),'Outside_RMSE':rmse(err[~flags]),
            'Inside_MAE':float(np.mean(abs(err[flags]))) if in_n else np.nan,
            'Inside_Reliability_APE_le_20pct':float(np.mean((abs(err[flags])/denom[flags]) <= 0.2)) if in_n else np.nan,
            'Outside_Reliability_APE_le_20pct':float(np.mean((abs(err[~flags])/denom[~flags]) <= 0.2)) if in_n<len(d) else np.nan}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw',default='data/raw/sfe_literature_dataset.csv')
    ap.add_argument('--original',default='results/07_applicability_domain/nested_domain_oof_diagnostics.csv')
    ap.add_argument('--output',default='paper_audit/domain_robustness')
    ap.add_argument('--k',type=int,default=10)
    ap.add_argument('--quantiles',default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',
                    help='Predefined TRAINING-distance quantile grid')
    a=ap.parse_args()
    qs=[float(v.strip()) for v in a.quantiles.split(',')]
    if not qs or any(q<=0 or q>=1 for q in qs) or len(set(qs))!=len(qs):
        raise ValueError('Quantiles must be distinct and in (0,1)')
    if a.k<1:
        raise ValueError('k must be >= 1')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(a.raw).reset_index(drop=True)
    orig=pd.read_csv(a.original).sort_values('Dataset_Row_Index').reset_index(drop=True)
    if not (len(raw)==len(orig)==441):
        raise ValueError('Expected 441 original data and 441 original nested OOF predictions')
    if not orig['Dataset_Row_Index'].is_unique or not np.array_equal(orig['Dataset_Row_Index'].to_numpy(),np.arange(441)):
        raise ValueError('Original OOF diagnostics do not contain unique row indices 0..440')
    if not np.array_equal(raw['No.'].astype(str),orig['ID'].astype(str)):
        raise ValueError('Original ID order does not match raw data')
    if not np.allclose(raw['SFE'],orig['Actual']):
        raise ValueError('Original target does not match raw data')
    folds = orig['Outer_Fold'].to_numpy()
    # Prove that the outer folds are exact-composition-disjoint, as in original validation.
    raw_X=raw[COMPOSITION].to_numpy(float)
    source=pd.factorize(pd.MultiIndex.from_frame(raw[COMPOSITION]),sort=True)[0]
    for fold in np.unique(folds):
        tr=(folds!=fold);va=(folds==fold)
        if set(source[tr]) & set(source[va]):
            raise ValueError('Original outer fold has exact-composition leakage')
    result=orig[['Dataset_Row_Index','ID','Outer_Fold','Actual','Predicted',ORIGINAL]].copy()
    result['knn_kth_distance_new']=np.nan
    result['knn_equal_count_diagnostic']=False
    q_cols={q:f'knn_training_q{int(round(q*100)):02d}' for q in qs}
    if len(set(q_cols.values()))!=len(q_cols):
        raise ValueError('Quantiles collapse to duplicate percentage labels')
    for col in q_cols.values():result[col]=False
    per_fold_cutoff=[]
    per_fold_counts=[]
    for fold in sorted(np.unique(folds)):
        tr = np.flatnonzero(folds!=fold)
        va = np.flatnonzero(folds==fold)
        k = min(a.k,len(tr)-1)
        scaler=StandardScaler().fit(raw_X[tr])
        xs_tr=scaler.transform(raw_X[tr]);xs_va=scaler.transform(raw_X[va])
        neighbors=NearestNeighbors(n_neighbors=k+1).fit(xs_tr)
        tr_self_excluded_kth=neighbors.kneighbors(xs_tr,return_distance=True)[0][:,k]
        va_external_kth=neighbors.kneighbors(xs_va,n_neighbors=k,return_distance=True)[0][:,k-1]
        result.loc[va,'knn_kth_distance_new']=va_external_kth
        for q,col in q_cols.items():
            cutoff=float(np.quantile(tr_self_excluded_kth,q))
            selected = va_external_kth<=cutoff
            result.loc[va,col]=selected
            per_fold_cutoff.append({'Outer_Fold':int(fold),'Threshold_Training_Quantile':q,
                                    'Cutoff_Distance':cutoff,'Validation_n':int(len(va)),
                                    'Inside_n':int(selected.sum()),'Coverage':float(selected.mean()),
                                    'Training_n':int(len(tr))})
        # Strictly descriptive equal-count matching: select n nearest validation
        # compositions based on distance ONLY. No validation target used.
        n_orig=int(orig.loc[va,ORIGINAL].astype(bool).sum())
        order=np.argsort(va_external_kth,kind='stable')
        chosen=va[order[:n_orig]]
        result.loc[chosen,'knn_equal_count_diagnostic']=True
        per_fold_counts.append({'Outer_Fold':int(fold),'Original_Inside_n':n_orig,
                                'Diagnostic_kNN_Inside_n':int(n_orig),'Fold_n':int(len(va)),
                                'Matching_basis':'validation composition distances only; not deployable'})
    result.to_csv(out/'domain_robustness_classifications.csv',index=False)
    pd.DataFrame(per_fold_cutoff).to_csv(out/'knn_training_cutoffs_by_fold.csv',index=False)
    pd.DataFrame(per_fold_counts).to_csv(out/'equal_count_folds.csv',index=False)
    columns=[ORIGINAL]+list(q_cols.values())+['knn_equal_count_diagnostic']
    per_fold = []
    pooled = []
    for col in columns:
        pooled.append(metric_frame(result,result[col],col))
        for fold,sl in result.groupby('Outer_Fold'):
            per_fold.append(metric_frame(sl,sl[col],col,int(fold)))
    pooled_df=pd.DataFrame(pooled)
    fold_df=pd.DataFrame(per_fold)
    pooled_df.to_csv(out/'domain_robustness_pooled.csv',index=False)
    fold_df.to_csv(out/'domain_robustness_by_fold.csv',index=False)
    matched=pooled_df[pooled_df.Method.isin([ORIGINAL,'knn_equal_count_diagnostic'])].copy()
    matched.to_csv(out/'equal_count_pooled_comparison.csv',index=False)
    fold_df[fold_df.Method.isin([ORIGINAL,'knn_equal_count_diagnostic'])].to_csv(out/'equal_count_fold_comparison.csv',index=False)
    curve=pooled_df[pooled_df.Method.isin(q_cols.values())].sort_values('Coverage')
    original=pooled_df[pooled_df.Method==ORIGINAL].iloc[0]
    for ycol,name in [('Inside_RMSE','inside_rmse'),('Inside_Reliability_APE_le_20pct','inside_reliability')]:
        fig,ax=plt.subplots(figsize=(7.2,4.6))
        ax.plot(curve['Coverage']*100,curve[ycol],marker='o',label='kNN training-quantile cutoffs')
        ax.scatter([original['Coverage']*100],[original[ycol]],marker='s',s=70,label='Original nested domain')
        ax.set_xlabel('Inside-domain coverage (%)')
        ax.set_ylabel('RMSE (mJ/m²)' if ycol=='Inside_RMSE' else 'APE ≤ 20% reliability')
        ax.set_title('Applicability-domain comparison (same outer predictions)')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(out/f'{name}_vs_coverage.png',dpi=180)
        plt.close(fig)
    metadata={
        'outer_grouping':'exact-composition groups (archived nested OOF)',
        'model_predictions':'unchanged original outer-fold predictions',
        'k':a.k, 'training_distance_quantiles':qs,
        'threshold_rules':'trained on OUTER-TRAIN composition distances only; no target-based selection',
        'equal_count_diagnostic':'uses outer-validation compositions to select exactly the original count per fold; not a deployable rule or confirmatory test',
        'comparison_scope':'post-hoc descriptive benchmark on previously analyzed dataset; independent new external test not performed'
    }
    (out/'domain_robustness_protocol.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('\nPooled comparison:')
    print(pooled_df[['Method','Inside_n','Coverage','Inside_RMSE','Inside_Reliability_APE_le_20pct']].to_string(index=False))
    print('\nEqual-count descriptive comparison:')
    print(matched[['Method','Inside_n','Coverage','Inside_RMSE','Inside_Reliability_APE_le_20pct']].to_string(index=False))
    print('\nInterpretation: original procedure was tuned using training data; kNN baselines use training-only composition thresholds.')
    print('Equal-count diagnostic uses held-out compositions to align counts; it is NOT a prospective selection rule.')
    print('Outputs saved to:',out.resolve())

if __name__=='__main__':main()
