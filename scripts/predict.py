"""
User-facing script to run BDE predictions on a list of SMILES strings.
"""
import argparse
import os
import json
import pandas as pd
import yaml

# Add project root to path to allow imports from src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.inference.predictor import get_bde_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run BDE predictions using a trained model. "
                    "Provide either a run directory OR the model and vocab paths directly."
    )
    
    # --- Input SMILES ---
    smiles_group = parser.add_mutually_exclusive_group(required=True)
    smiles_group.add_argument('--smiles', nargs='+', help='One or more SMILES strings to predict.')
    smiles_group.add_argument('--smiles_file', type=str, help='Path to a text file with one SMILES string per line.')

    # --- Model Loading ---
    model_group = parser.add_argument_group('Model Loading')
    model_group.add_argument('--run_dir', type=str, help='Path to the training run directory (e.g., training_runs/TIMESTAMP).')
    model_group.add_argument('--model_path', type=str, help='Path to the trained model checkpoint (.pt file).')
    model_group.add_argument('--vocab_path', type=str, help='Path to the corresponding vocabulary file (.json).')
    model_group.add_argument('--featurizer_type', type=str, default=None, 
                             help='Type of featurizer to use (e.g., TokenFeaturizer). ' 
                                  'If --run_dir is provided, this is inferred from config.')

    # --- Other Options ---
    parser.add_argument('--output_path', type=str, default='predictions.csv', help='Path to save the output CSV file.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help="Device to run inference on.")
    parser.add_argument('--keep-duplicates', action='store_true', help="If set, keeps predictions for chemically equivalent bonds (e.g., in symmetric molecules).")
    
    args = parser.parse_args()

    # --- Determine Model, Vocab, and Featurizer ---
    model_path = args.model_path
    vocab_path = args.vocab_path
    featurizer_type = args.featurizer_type or 'TokenFeaturizer' # Default

    if args.run_dir:
        if args.model_path or args.vocab_path:
            parser.error("Cannot specify --run_dir with --model_path or --vocab_path.")
        
        config_path = os.path.join(args.run_dir, 'config.yaml')
        if not os.path.exists(config_path):
            print(f"Error: config.yaml not found in the specified run directory: {args.run_dir}")
            return
            
        cfg = load_config(config_path)
        
        # Construct model path from config
        model_path = os.path.join(args.run_dir, cfg.train.model_save_path)
        featurizer_type = cfg.data.featurizer_type
        num_messages = cfg.model.num_messages

        # Determine vocab path: always run-specific
        run_vocab_path = os.path.join(args.run_dir, 'vocab.json')
        if os.path.exists(run_vocab_path):
            vocab_path = run_vocab_path
        else:
            raise FileNotFoundError(f"vocab.json not found in {args.run_dir}")
            
        # Determine featurizer type from config if not explicitly provided
        if args.featurizer_type is None:
            featurizer_type = cfg.get('data', {}).get('featurizer_type', 'TokenFeaturizer')

        # Get num_messages from config
        num_messages = cfg.model.num_messages
        atom_features = cfg.model.atom_features

        print(f"Loading from run directory '{args.run_dir}':")
        print(f"  -> Model Path: {model_path}")
        print(f"  -> Vocab Path: {vocab_path}")
        print(f"  -> Featurizer: {featurizer_type}")
        print(f"  -> Num Messages: {num_messages}") # Add print for num_messages

    elif not (model_path and vocab_path):
        # Note: If featurizer doesn't use vocab (like ChemProp), vocab_path might be dummy. 
        # But for now we enforce it for safety unless user knows what they are doing.
        if featurizer_type == 'TokenFeaturizer':
            parser.error("If not using --run_dir, both --model_path and --vocab_path are required for TokenFeaturizer.")


    # --- Load SMILES ---
    if args.smiles:
        smiles_list = args.smiles
    else:
        try:
            with open(args.smiles_file, 'r') as f:
                smiles_list = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: Smiles file not found at {args.smiles_file}")
            return
    
    if not smiles_list:
        print("Error: No SMILES strings to process.")
        return

    # --- Run Prediction using the new simple API ---
    print("\nMaking predictions...")
    results_df = get_bde_predictions(
        smiles=smiles_list,
        model_path=model_path,
        vocab_path=vocab_path,
        featurizer_type=featurizer_type,
        atom_features=atom_features,
        num_messages=num_messages, # Pass num_messages here
        drop_duplicates=not args.keep_duplicates,
        device=args.device
    )
    
    # --- Save and Display Results ---
    if results_df.empty:
        print("No valid bonds found for prediction in the provided molecules.")
    else:
        results_df.to_csv(args.output_path, index=False)
        print(f"\nPredictions successfully saved to {args.output_path}")
        print("\nPrediction results head:")
        print(results_df.head().to_string())

if __name__ == '__main__':
    main()
