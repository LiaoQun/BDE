"""
This script runs BDE prediction, extracts bond embeddings from the MPNN model,
and visualizes them using t-SNE.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

# Add project root to path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.inference.predictor import get_bde_predictions_with_embeddings

def main():
    parser = argparse.ArgumentParser(
        description="Run BDE prediction and visualize bond embeddings from MPNN layers."
    )
    
    # --- Input Arguments ---
    parser.add_argument('--run_dir', type=str, required=True, help='Path to the training run directory.')
    parser.add_argument('--smiles_file', type=str, default='examples/test_data.csv.gz', 
                        help='Path to a CSV.gz file with molecule SMILES in a "molecule" column.')
    parser.add_argument('--output_dir', type=str, default='bond_embedding_visualizations',
                        help='Directory to save the output plots.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help="Device to run inference on.")

    args = parser.parse_args()

    # --- Load all SMILES from file ---
    try:
        smiles_df = pd.read_csv(args.smiles_file)
        smiles_list = smiles_df['molecule'].drop_duplicates().sample(100).tolist()
    except FileNotFoundError:
        print(f"Error: SMILES file not found at {args.smiles_file}", file=sys.stderr)
        return
    except KeyError:
        print(f"Error: 'molecule' column not found in {args.smiles_file}", file=sys.stderr)
        return
    
    if not smiles_list:
        print("Error: No SMILES strings to process from the file.", file=sys.stderr)
        return

    # --- Load Config from Run Directory ---
    config_path = os.path.join(args.run_dir, 'config.json')
    if not os.path.exists(config_path):
        print(f"Error: config.json not found in {args.run_dir}", file=sys.stderr)
        return

    cfg = load_config(config_path)

    model_path = os.path.join(args.run_dir, cfg.train.model_save_path)
    vocab_path = os.path.join(args.run_dir, "vocab.json")
    featurizer_type = cfg.data.featurizer_type
    atom_features = cfg.model.atom_features
    num_messages = cfg.model.num_messages

    print("--- Configuration ---")
    print(f"Model Path: {model_path}")
    print(f"Featurizer: {featurizer_type}")
    print(f"Num Messages: {num_messages}")
    print("-" * 21)

    # --- Run Prediction and Get Embeddings ---
    print("\nRunning prediction and extracting embeddings...")
    results_df, embeddings_by_layer = get_bde_predictions_with_embeddings(
        smiles=smiles_list,
        model_path=model_path,
        vocab_path=vocab_path,
        featurizer_type=featurizer_type,
        atom_features=atom_features,
        num_messages=num_messages,
        device=args.device
    )

    if results_df.empty or not embeddings_by_layer:
        print("Could not generate predictions or embeddings.", file=sys.stderr)
        return

    # --- Create output directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Process and Plot Embeddings for EACH Layer ---
    print("\nProcessing and visualizing embeddings for each layer...")
    for layer_idx in sorted(embeddings_by_layer.keys()):
        print(f"  - Layer {layer_idx}...")
        layer_embeddings_df = embeddings_by_layer[layer_idx]

        # Merge with results to get bond types
        full_data_df = pd.merge(results_df, layer_embeddings_df, on=['molecule', 'bond_index'])
        
        if full_data_df.empty or 'embedding' not in full_data_df.columns:
            print(f"  - Failed to merge embeddings with results for layer {layer_idx}.", file=sys.stderr)
            continue

        # Stack embeddings into a numpy array for t-SNE
        X = np.stack(full_data_df['embedding'].values)
        bond_types = full_data_df['bond_type']

        # --- Run t-SNE ---
        # Ensure perplexity is less than number of samples
        perplexity_val = min(30, len(X) - 1)
        if len(X) <= 1: # t-SNE requires at least 2 samples
            print(f"  - Not enough samples for t-SNE in layer {layer_idx}. Skipping.", file=sys.stderr)
            continue
        if perplexity_val <= 0: # Perplexity must be greater than 0
             print(f"  - Perplexity value too low for t-SNE in layer {layer_idx}. Skipping.", file=sys.stderr)
             continue

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
        X_reduced = tsne.fit_transform(X)

        # --- Plotting ---
        plot_df = pd.DataFrame(X_reduced, columns=['dim1', 'dim2'])
        plot_df['bond_type'] = bond_types

        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            data=plot_df,
            x='dim1',
            y='dim2',
            hue='bond_type',
            palette=sns.color_palette("hsv", n_colors=plot_df['bond_type'].nunique()),
            alpha=0.8,
            s=50
        )
        plt.title(f't-SNE Visualization of Bond Embeddings (Layer {layer_idx})')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(title='Bond Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the plot
        output_file_path = os.path.join(args.output_dir, f'bond_embeddings_layer_{layer_idx}.png')
        plt.savefig(output_file_path)
        plt.close() # Close the plot to free memory
        print(f"  - Visualization for layer {layer_idx} saved to {output_file_path}")

    print("\nAll visualizations generated.")


if __name__ == '__main__':
    main()
