#!/usr/bin/env python3
"""Create auditable, unfilled source-value and equation-review checklists.

Run from repository root:
  python paper_audit_tools/01_source_and_formula_audit_templates.py --output paper_audit/source_checks
This does not independently validate any published source.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', default='data/raw/sfe_literature_dataset.csv')
    p.add_argument('--registry', default='data/metadata/sfe_model_registry.csv')
    p.add_argument('--constants', default='data/metadata/elemental_property_table.csv')
    p.add_argument('--processed', default='data/processed/output_with_8_sfe_models.csv')
    p.add_argument('--output', default='paper_audit/source_checks')
    a = p.parse_args()
    o = Path(a.output); o.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(a.raw)
    processed = pd.read_csv(a.processed)
    required = ['No.','DOI','SFE','T','Technique']
    assert set(required).issubset(raw), f'Missing raw columns: {set(required)-set(raw)}'
    assert raw['No.'].is_unique and processed['No.'].is_unique
    assert raw['No.'].astype(str).tolist() == processed['No.'].astype(str).tolist()
    rng = np.random.default_rng(42)
    samples = []
    # At least one observation from EACH of the 51 recorded DOI/source groups.
    for _, d in raw.groupby('DOI', sort=True):
        samples.append(d.sample(1, random_state=42))
    # Extra checks for concentration among the three most represented sources.
    top = raw['DOI'].value_counts().head(3).index
    for src in top:
        d = raw.loc[raw['DOI'].eq(src)].sort_values('No.')
        samples.append(d.iloc[np.unique(np.linspace(0,len(d)-1,min(6,len(d)),dtype=int))])
    # Explicitly include tails of the target distribution.
    samples += [raw.nsmallest(6,'SFE'),raw.nlargest(6,'SFE')]
    # Include examples using each reported measurement technique.
    for tech, d in raw.groupby('Technique',sort=True):
        samples.append(d.sample(min(2,len(d)),random_state=42))
    checks = pd.concat(samples).drop_duplicates('No.').sort_values(['DOI','No.']).reset_index(drop=True)
    checks = checks.rename(columns={'No.':'dataset_ID','DOI':'source_as_recorded','SFE':'dataset_SFE_mJ_m2','T':'dataset_T_K','Technique':'dataset_technique'})
    for col in ['source_page_table_figure','source_original_alloy_composition','source_original_SFE','source_original_SFE_unit','source_original_temperature_K','source_measurement_technique','match_composition','match_SFE','match_temperature','copyright_redistribution_checked','reviewer_initials','checked_date','notes']:
        checks[col]=''
    checks['audit_status']='NOT_CHECKED'
    checks.to_csv(o/'source_value_checklist.csv',index=False)
    props = pd.read_csv(a.constants)
    props['independent_reference']=''
    props['radius_convention_confirmed']=''
    props['verified_against_primary_reference']='NOT_CHECKED'
    props['review_notes']=''
    props.to_csv(o/'elemental_constants_checklist.csv',index=False)
    reg = pd.read_csv(a.registry)
    reg['source_equation_number_or_page']=''
    reg['source_applicability_composition_range']=''
    reg['source_temperature_assumptions']=''
    reg['independent_hand_or_spreadsheet_test_ID']=''
    reg['independent_output_mJ_m2']=''
    reg['comparison_with_script']='NOT_CHECKED'
    reg['sign_units_and_reference_state_verified']=''
    reg['review_notes']=''
    reg.to_csv(o/'sfe_formula_checklist.csv',index=False)
    sources = raw.groupby('DOI').agg(n_rows=('No.','size'),min_SFE=('SFE','min'),max_SFE=('SFE','max')).sort_values('n_rows',ascending=False).reset_index()
    sources['source_metadata_status']='NOT_CHECKED'
    sources['data_reuse_or_license_status']='NOT_CHECKED'
    sources['notes']=''
    sources.to_csv(o/'source_register.csv',index=False)
    print(f'Created {len(checks)} sampled observations covering all {len(sources)} source strings in {o}')
    print('Blank review fields intentionally require checking the ORIGINAL publications.')

if __name__ == '__main__':
    main()
