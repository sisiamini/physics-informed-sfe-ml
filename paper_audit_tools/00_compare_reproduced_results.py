#!/usr/bin/env python3
"""Compare a canonical-environment rerun against thesis archived CSV outputs.

Flags discrepancies for investigation; absolute tolerance below is a *screening
threshold*, not a claim of platform-independent bitwise reproducibility.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',default='paper_audit/baseline')
    p.add_argument('--results',default='results')
    p.add_argument('--output',default='paper_audit/baseline/metric_comparison.csv')
    p.add_argument('--tolerance',type=float,default=0.001)
    a=p.parse_args(); base=Path(a.baseline);res=Path(a.results)
    cases=[('final',res/'04_final_evaluation/final_evaluation_summary.csv',base/'final_evaluation/final_evaluation_summary.csv','Evaluation',['R2','RMSE','MAE']),('ablation',res/'05_ablation/ablation_repeated_cv_summary.csv',base/'original_ablation/ablation_repeated_cv_summary.csv','Feature_Set',['RMSE_mean','MAE_mean','R2_mean'])]
    rows=[]
    for group,orig,reread,key,cols in cases:
        if not reread.exists():
            print(f'MISSING rerun file: {reread}');continue
        a0=pd.read_csv(orig);b0=pd.read_csv(reread)
        for idx,old in a0.set_index(key).iterrows():
            if idx not in b0[key].to_numpy():
                rows.append({'Experiment':group,'Key':idx,'Metric':'ALL','Status':'MISSING_RERUN_ROW'});continue
            new=b0.set_index(key).loc[idx]
            for col in cols:
                prev=float(old[col]); now=float(new[col]); delta=now-prev
                rows.append({'Experiment':group,'Key':idx,'Metric':col,'Archived':prev,'Rerun':now,'Difference':delta,'Status':'MATCH_WITHIN_TOLERANCE' if np.isfinite(delta) and abs(delta)<=a.tolerance else 'INVESTIGATE'})
    before=res/'07_applicability_domain/nested_domain_oof_diagnostics.csv';after=base/'nested_domain/nested_domain_oof_diagnostics.csv'
    if after.exists():
        old=pd.read_csv(before).sort_values('Dataset_Row_Index');new=pd.read_csv(after).sort_values('Dataset_Row_Index')
        if np.array_equal(old.ID.to_numpy(),new.ID.to_numpy()):
            for label,d in [('Archived',old),('Rerun',new)]:
                err=d.Actual.to_numpy()-d.Predicted.to_numpy()
                vals={'Nested_All_RMSE':float(np.sqrt(np.mean(err**2))),'Nested_Inside_RMSE':float(np.sqrt(np.mean(err[d.inside_final_applicability_domain.to_numpy(bool)]**2))),'Nested_Coverage':float(d.inside_final_applicability_domain.mean())}
                if label=='Archived':reference=vals
                else:
                    for k,v in vals.items():
                        delta=v-reference[k];rows.append({'Experiment':'domain','Key':'outer_OOF','Metric':k,'Archived':reference[k],'Rerun':v,'Difference':delta,'Status':'MATCH_WITHIN_TOLERANCE' if abs(delta)<=a.tolerance else 'INVESTIGATE'})
        else: print('WARNING: nested domain old/new ID row alignment differs')
    else: print(f'MISSING nested rerun file (optional until completed): {after}')
    if not rows:
        raise SystemExit('No rerun outputs found; run original scripts first')
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(path,index=False)
    print(df.to_string(index=False));print(f'Comparison saved: {path}')
    print('Differences beyond the screening tolerance need explanation, not silent replacement of archived results.')

if __name__=='__main__':main()
