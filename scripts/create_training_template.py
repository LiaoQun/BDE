"""
This script generates a CSV template for BDE annotation from a list of SMILES,
based on the faithful port of `alfabet` fragmentation logic.
"""
import os
import argparse
import pandas as pd

# Add project root to path to allow imports from src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.curation.template_generator import generate_fragment_template


def main():
    parser = argparse.ArgumentParser(description="Generate a CSV template for BDE annotation from a list of SMILES.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smiles', nargs='+', help='One or more SMILES strings to process.')
    group.add_argument('--smiles_file', type=str, help='Path to a text file with one SMILES string per line.')

    parser.add_argument('--output_path', type=str, default='fragment_template.csv', help='Path to save the output CSV file.')
    
    args = parser.parse_args()

    # --- Load SMILES ---
    if args.smiles:
        smiles_to_process = args.smiles
    else:
        try:
            with open(args.smiles_file, 'r') as f:
                smiles_to_process = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: Smiles file not found at {args.smi_file}")
            return
            
    if not smiles_to_process:
        print("Error: No SMILES strings to process.")
        return

    # --- Generate Template ---
    print(f"Generating fragment template for {len(smiles_to_process)} molecules...")
    fragment_template_df = generate_fragment_template(smiles_to_process)

    if fragment_template_df.empty:
        print("No valid bonds found for template generation in the provided molecules.")
        return

    # --- Save to CSV ---
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fragment_template_df.to_csv(args.output_path, index=False)

    print(f"\nTemplate successfully saved to: {args.output_path}")
    print("\nDataFrame Head:")
    print(fragment_template_df.head())

    print(f"\nTo use this template:")
    print(f"1. Open '{args.output_path}' in a spreadsheet editor.")
    print(f"2. The file lists all non-ring single bonds and their resulting radical fragments.")
    print(f"3. Fill in the 'bde' column for the bonds you have data for.")
    print(f"4. The completed file can now be used for model training.")


if __name__ == '__main__':
    main()