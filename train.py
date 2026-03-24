"Main script for training and evaluating the BDE Prediction Model."
import os
import shutil
import sys
import json
from datetime import datetime
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from rdkit import Chem
from tqdm import tqdm
from typing import List
from sklearn.model_selection import train_test_split
import logging # Import logging

from src.config.schema import MainConfig
from src.data.preprocessing import load_and_merge_data, prepare_data

from src.config import load_config, save_flattened_config
from src.features import get_featurizer, get_featurizer_from_vocab
from src.data.dataset import BDEDataset
from src.models.mpnn import BDEModel # Temporarily keep BDEModel
from src.training.trainer import Trainer

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler() # Output to console
                    ])
logger = logging.getLogger(__name__) # Get a logger for this module


def run_training(cfg: MainConfig, config_path: str):
    """
    Main function to set up and run the training and evaluation pipeline.
    """
    # 1. Setup
    torch.manual_seed(cfg.data.random_seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create a unique directory for this run
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg.train.output_dir, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Add FileHandler to save logs in the run directory
    fh = logging.FileHandler(os.path.join(run_dir, "training.log"))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(fh)

    logger.info(f"Saving all artifacts to: {run_dir}")

    # Save the config file for this run for reproducibility
    save_flattened_config(cfg, run_dir)
    logger.info(f"Saved configuration to {run_dir}")
    
    # 2. Load, Merge, and Clean Data
    df = load_and_merge_data(
        cfg.data.data_paths, 
        target_columns=cfg.data.target_columns,
        sample_percentage=cfg.data.sample_percentage,
        random_seed=cfg.data.random_seed
    )

    if df.empty:
        logger.error("Stopping run: No data available after loading and cleaning.")
        return

    processed_smiles_data = prepare_data(df, target_columns=cfg.data.target_columns)
    
    logger.info("Splitting data...")
    train_val_smiles_data, test_smiles_data = train_test_split(processed_smiles_data, test_size=cfg.data.test_size, random_state=cfg.data.random_seed)
    val_split_ratio = cfg.data.val_size / (1.0 - cfg.data.test_size)
    train_smiles_data, val_smiles_data = train_test_split(train_val_smiles_data, test_size=val_split_ratio, random_state=cfg.data.random_seed)

    logger.info(f"Initial splits: Train ({len(train_smiles_data)}), Val ({len(val_smiles_data)}), Test ({len(test_smiles_data)}) unique molecule entries.")

    # 3. Initialize Featurizer, Datasets, and DataLoaders
    vocab_save_path = os.path.join(run_dir, "vocab.json")
    train_smiles = [data[0] for data in train_smiles_data]

    if cfg.data.featurizer_type is None:
        logger.error("Stopping run: 'featurizer_type' is not specified or config failed to load.")
        return

    logger.info(f"Initializing featurizer: {cfg.data.featurizer_type}...")
    featurizer = get_featurizer(
        featurizer_type=cfg.data.featurizer_type,
        smiles_list=train_smiles,
        save_path=vocab_save_path
    )
    logger.info(f"Featurizer built and saved to: {vocab_save_path}")
    effective_vocab_path = vocab_save_path

    logger.info("Initializing datasets...")
    train_dataset = BDEDataset(root=os.path.join(cfg.data.dataset_dir, 'train'), smiles_data=train_smiles_data, featurizer=featurizer)
    val_dataset = BDEDataset(root=os.path.join(cfg.data.dataset_dir, 'val'), smiles_data=val_smiles_data, featurizer=featurizer)
    test_dataset = BDEDataset(root=os.path.join(cfg.data.dataset_dir, 'test'), smiles_data=test_smiles_data, featurizer=featurizer)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    
    # 4. Initialize Model and Optimizer
    logger.info("Initializing model...")
    model = BDEModel(
        atom_input_dim=featurizer.atom_dim,
        bond_input_dim=featurizer.bond_dim,
        atom_features=cfg.model.atom_features,
        num_messages=cfg.model.num_messages,
        inputs_are_discrete=featurizer.is_discrete,
        num_tasks=cfg.model.num_tasks
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # 5. Initialize and run Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        cfg=cfg.train,
        model_cfg=cfg.model,
        run_dir=run_dir,
        # Pass additional data for final evaluation and saving
        full_dataset_df=df,
        data_splits={'train': train_smiles_data, 'val': val_smiles_data, 'test': test_smiles_data},
        vocab_path=effective_vocab_path, # Used by Predictor
        featurizer_type=cfg.data.featurizer_type,
        target_columns=cfg.data.target_columns
    )
    
    trainer.train()
    trainer.evaluate()

    return run_dir # Return the run directory for potential use in testing or further analysis


def main():
    config_path: str = sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/default.yaml"
    logger.info(f"Loading config from: {config_path}")

    cfg: MainConfig = load_config(config_path)

    try:
        run_training(cfg, config_path)
    finally:
        if os.path.exists(cfg.data.dataset_dir):
            logger.info(f"Cleaning up temporary dataset directory: {cfg.data.dataset_dir}")
            shutil.rmtree(cfg.data.dataset_dir)

if __name__ == '__main__':
    main()
