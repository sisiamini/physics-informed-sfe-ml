#!/usr/bin/env python3
# SFE_Master.py
"""
Calculate SFE values from eight empirical SFE model scripts and
combine them with composition, descriptor, and experimental SFE columns.

Input:
    CSV containing composition columns, descriptor columns, and
    "SFE_Experimental (mJ/m²)".

Output:
    CSV containing:
        - Original composition and descriptor columns
        - Calculated SFE values from eight models
        - SFE_Experimental (mJ/m²)

Required:
    The eight SFE model scripts must be in the same folder as this script.

Run:
    python SFE_Master.py \
        --csv "data/output_with_descriptors.csv" \
        --output "data/output_with_8_sfe_models.csv"

Example:
    python SFE_Master.py \
        --csv "source folder/source file.csv" \
        --output "destination folder/destination file.csv"
"""

import csv
import argparse
import importlib.util
import os
import sys
from pathlib import Path

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

# Define model information
# The 'file' value is the exact Python script filename in the same folder as this driver script.
model_info = [
    {
        'name': 'SFE_Akbari',
        'file': 'SFE_Akbari.py',
        'function': 'compute_gamma_akbari',
        'elements': ['Mo', 'Cu', 'V', 'Mn', 'Fe', 'Ni', 'Al', 'Cr', 'Si', 'C']
    },
    {
        'name': 'SFE_Allain',
        'file': 'SFE_Allain.py',
        'function': 'compute_gamma_allain',
        'elements': ['Mn', 'Fe', 'C']
    },
    {
        'name': 'SFE_Dumay',
        'file': 'SFE_Dumay.py',
        'function': 'compute_gamma_dumay',
        'elements': ['Mn', 'Fe', 'C', 'Cu', 'Cr', 'Al', 'Si']
    },
    {
        'name': 'SFE_Elaraby',
        'file': 'SFE_Elaraby.py',
        'function': 'compute_gamma_elaraby',
        'elements': ['C', 'Mn', 'Si', 'Al', 'Mo', 'N', 'Cr', 'Ni', 'Cu', 'Fe']
    },
    {
        'name': 'SFE_Wang',
        'file': 'SFE_Wang.py',
        'function': 'compute_gamma_wang',
        'elements': ['Mn', 'Fe', 'Ni', 'Al', 'C']
    },
    {
        'name': 'SFE_Xiong',
        'file': 'SFE_Xiong.py',
        'function': 'compute_gamma_xiong',
        'elements': ['Mn', 'Fe', 'Si', 'C']
    },
    {
        'name': 'SFE_Yoo',
        'file': 'SFE_Yoo.py',
        'function': 'compute_gamma_yoo',
        'elements': ['Mn', 'Fe', 'Al', 'C']
    },
    {
        'name': 'SFE_Zambrano',
        'file': 'SFE_Zambrano.py',
        'function': 'compute_gamma_zambrano',
        'elements': ['Mn', 'Fe', 'Al', 'Si', 'C']
    }
]

# Column names expected in the input CSV
composition_headers = [
    'No.',
    'Fe (%wt)', 'Mn (%wt)', 'Al (%wt)', 'C (%wt)', 'Ni (%wt)', 'Cr (%wt)',
    'Si (%wt)', 'Cu (%wt)', 'Mo (%wt)', 'Nb (%wt)', 'V (%wt)', 'Ti (%wt)',
    'Co (%wt)', 'N (%wt)', 'P (%wt)', 'S (%wt)', 'R_mean (pm)', 'R_std (pm)',
    'Δχ', 'VEC_mean'
]
experimental_header = 'SFE_Experimental (mJ/m²)'

# Map short element names to CSV headers
csv_element_headers = {
    'Fe': 'Fe (%wt)',
    'Mn': 'Mn (%wt)',
    'Al': 'Al (%wt)',
    'C': 'C (%wt)',
    'Ni': 'Ni (%wt)',
    'Cr': 'Cr (%wt)',
    'Si': 'Si (%wt)',
    'Cu': 'Cu (%wt)',
    'Mo': 'Mo (%wt)',
    'Nb': 'Nb (%wt)',
    'V': 'V (%wt)',
    'Ti': 'Ti (%wt)',
    'Co': 'Co (%wt)',
    'N': 'N (%wt)',
    'P': 'P (%wt)',
    'S': 'S (%wt)'
}


def load_function_from_file(module_file, function_name):
    """Load a function from a Python file, even if the filename contains parentheses."""
    if not os.path.exists(module_file):
        raise FileNotFoundError(f'Model file not found: {module_file}')

    module_name = os.path.splitext(os.path.basename(module_file))[0].replace('(', '_').replace(')', '_')
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def to_float(value):
    """Convert CSV values to float. Empty values are treated as 0.0."""
    if value is None:
        return 0.0
    value = str(value).strip()
    if value == '':
        return 0.0
    return float(value)


# Load model functions
script_dir = os.path.dirname(os.path.abspath(__file__))
for model in model_info:
    try:
        model_path = os.path.join(script_dir, model['file'])
        model['func'] = load_function_from_file(model_path, model['function'])
    except Exception as e:
        print(f"Error loading model {model['name']}: {str(e)}")
        model['func'] = None

# Check input file
input_path = INPUT_CSV
if not input_path.exists():
    print(f'Input file {INPUT_CSV} not found.')
    sys.exit(1)

# Define output headers in the required order
output_headers = composition_headers + [
    'SFE_Akbari (mJ/m²)',
    'SFE_Allain (mJ/m²)',
    'SFE_Dumay (mJ/m²)',
    'SFE_Elaraby (mJ/m²)',
    'SFE_Wang (mJ/m²)',
    'SFE_Xiong (mJ/m²)',
    'SFE_Yoo (mJ/m²)',
    'SFE_Zambrano (mJ/m²)',
    'SFE_Experimental (mJ/m²)'
]

output_rows = []

# Read and process input CSV
try:
    with open(input_path, mode='r', newline='', encoding='utf-8-sig') as f_in:
        reader = csv.DictReader(f_in)

        # Check required columns
        required_headers = composition_headers + [experimental_header]
        missing_headers = [h for h in required_headers if h not in reader.fieldnames]
        if missing_headers:
            print('Missing required input columns:')
            for h in missing_headers:
                print(f'  - {h}')
            sys.exit(1)

        for row_idx, row in enumerate(reader, start=2):
            # Skip completely empty rows
            if not any(str(v).strip() for v in row.values() if v is not None):
                continue

            # Extract all elements and descriptor columns
            try:
                elements = {}
                for elem, header in csv_element_headers.items():
                    elements[elem] = to_float(row.get(header))
            except Exception as e:
                print(f"Row {row_idx}: error reading element values - {str(e)}")
                continue

            # Start output row with original composition/descriptor columns
            output_row = []
            for header in composition_headers:
                output_row.append(row.get(header, ''))

            # Calculate SFE for each model
            model_sfes = {}
            for model in model_info:
                if model['func'] is None:
                    model_sfes[model['name']] = ''
                    continue

                try:
                    values = [elements[elem] for elem in model['elements']]
                    sfe = model['func'](*values)
                    model_sfes[model['name']] = sfe
                except Exception as e:
                    print(f"Row {row_idx}, model {model['name']}: error - {str(e)}")
                    model_sfes[model['name']] = ''

            # Append model results in the required order, then experimental SFE last
            output_row.extend([
                model_sfes['SFE_Akbari'],
                model_sfes['SFE_Allain'],
                model_sfes['SFE_Dumay'],
                model_sfes['SFE_Elaraby'],
                model_sfes['SFE_Wang'],
                model_sfes['SFE_Xiong'],
                model_sfes['SFE_Yoo'],
                model_sfes['SFE_Zambrano'],
                row.get(experimental_header, '')
            ])

            output_rows.append(output_row)

except Exception as e:
    print(f'Error reading input CSV: {str(e)}')
    sys.exit(1)

# Save output CSV
output_path = OUTPUT_CSV
try:
    with open(output_path, mode='w', newline='', encoding='utf-8-sig') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(output_headers)
        writer.writerows(output_rows)
    print(f'Output saved to {OUTPUT_CSV}')
except Exception as e:
    print(f'Error saving output CSV: {str(e)}')
    sys.exit(1)
