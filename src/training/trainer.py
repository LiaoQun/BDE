import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import r2_score, mean_squared_error
import logging # Import logging

from src.config import TrainConfig, ModelConfig
from src.utils.reporting import save_training_log
from src.utils.plotting import plot_training_curve, plot_parity
from src.inference.predictor import Predictor

logger = logging.getLogger(__name__) # Get a logger for this module

class Trainer:
    """
    Handles the model training, validation, and evaluation pipeline.
    """
    def __init__(self, model, optimizer, train_loader, val_loader, test_loader, 
                 device, cfg: TrainConfig, model_cfg: ModelConfig, run_dir: str, full_dataset_df: pd.DataFrame, 
                 data_splits: Dict[str, List], vocab_path: str, featurizer_type: str = 'TokenFeaturizer',
                 target_columns: List[str] = None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.run_dir = run_dir
        self.model_save_path = os.path.join(run_dir, cfg.model_save_path)
        self.vocab_path = vocab_path
        self.featurizer_type = featurizer_type
        self.full_dataset_df = full_dataset_df
        self.data_splits = data_splits
        self.target_columns = target_columns if target_columns is not None else ['bde']

    def train(self):
        """
        Executes the main training loop, including validation and early stopping.
        """
        logger.info("Starting training...")
        best_val_loss = float('inf')
        patience_counter = 0
        history = []

        for epoch in range(1, self.cfg.epochs + 1):
            # Train one epoch
            avg_train_loss = self._train_epoch(epoch)
            
            # Validate one epoch
            avg_val_loss = self._validate_epoch(epoch)
            
            logger.info(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })

            # Early stopping and model saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), self.model_save_path)
                logger.info(f"  -> New best validation loss: {best_val_loss:.4f}. Model saved to {self.model_save_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"  -> Validation loss did not improve. Patience: {patience_counter}/{self.cfg.early_stopping_patience}")

            if patience_counter >= self.cfg.early_stopping_patience:
                logger.info("\nEarly stopping triggered.")
                break
        
        logger.info("\nTraining finished.")
        
        # Save logs and plots
        history_df = save_training_log(history, self.run_dir)
        plot_training_curve(history_df, self.run_dir)

    def _train_epoch(self, epoch: int) -> float:
        """Handles the training logic for a single epoch."""
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(batch) # [num_edges, num_tasks]
            
            # Mask out missing target elements. Both pred and y have shape [num_edges, num_tasks]
            if batch.mask.sum() > 0:
                loss = F.l1_loss(pred[batch.mask], batch.y[batch.mask])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch.num_graphs
        return total_loss / len(self.train_loader.dataset) if len(self.train_loader.dataset) > 0 else 0

    def _validate_epoch(self, epoch: int) -> float:
        """Handles the validation logic for a single epoch."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                batch = batch.to(self.device)
                pred = self.model(batch) # [num_edges, num_tasks]
                
                if batch.mask.sum() > 0:
                    loss = F.l1_loss(pred[batch.mask], batch.y[batch.mask])
                    total_loss += loss.item() * batch.num_graphs
        return total_loss / len(self.val_loader.dataset) if len(self.val_loader.dataset) > 0 else 0

    def evaluate(self):
        """
        Evaluates the best model on all data splits (train, val, test),
        saves the full predictions, and generates plots.
        """
        logger.info(f"\nLoading best model from {self.model_save_path} for final evaluation...")
        
        try:
            # 1. Initialize predictor with the best model
            predictor = Predictor(
                model_path=self.model_save_path,
                vocab_path=self.vocab_path,
                featurizer_type=self.featurizer_type,
                atom_features=self.model_cfg.atom_features, 
                num_messages=self.model_cfg.num_messages,
                num_tasks=self.model_cfg.num_tasks,
                target_columns=self.target_columns,
                device=self.device
            )
        except FileNotFoundError as e:
            logger.info(f"Could not initialize predictor: {e}. Aborting evaluation.")
            return

        results_for_plotting = {}
        
        # 2. Iterate through each data split, make predictions, and save results
        for split_name, data_list in self.data_splits.items():
            logger.info(f"\n--- Predicting on {split_name} set ---")
            if not data_list:
                logger.info(f"{split_name} set is empty. Skipping.")
                continue

            # Get unique SMILES for the current split
            smiles_list = sorted(list(set([item[0] for item in data_list])))
            
            # Predictor now returns a DataFrame filled with `[task]_pred` columns
            pred_df = predictor.predict(smiles_list, drop_duplicates=False)

            # Retrieve true labels mapping. This assumes target keys exist in full_dataset
            # Predictor merge logic will handle the actual bond index merging per target 
            target_cols = self.target_columns
            
            merged_df = pd.merge(
                pred_df,
                self.full_dataset_df[['molecule', 'bond_index'] + target_cols],
                on=['molecule', 'bond_index'],
                how='inner' 
            )

            # Save the detailed predictions to a CSV file
            output_path = os.path.join(self.run_dir, f'predictions_{split_name}.csv')
            merged_df.to_csv(output_path, index=False)
            logger.info(f"Saved detailed predictions for {split_name} set to {output_path}")

            # Collect results for charting
            if not merged_df.empty:
                # Initialize splitting per target
                if split_name not in results_for_plotting:
                    results_for_plotting[split_name] = {}
                    
                for task in target_cols:
                    pred_col_name = f"{task}_pred"
                    if pred_col_name in merged_df.columns:
                        # Only plot indices where ground truth isn't NaN for THIS task
                        valid_mask = ~merged_df[task].isna()
                        y_true = merged_df.loc[valid_mask, task].values
                        y_pred = merged_df.loc[valid_mask, pred_col_name].values
                        
                        if len(y_true) > 0:
                            results_for_plotting[split_name][task] = (y_true, y_pred)
        
        # 3. Generate parity plot for EACH task
        if results_for_plotting:
            plotted_tasks = set()
            for split, task_dict in results_for_plotting.items():
                for task in task_dict.keys():
                    plotted_tasks.add(task)
            
            for task in plotted_tasks:
                logger.info(f"\nGenerating parity plot for {task}...")
                
                # Reconstruct standard results struct for the plotter: {'train': (true, pred), 'test': (true, pred)}
                task_results = {}
                for split in results_for_plotting.keys():
                    if task in results_for_plotting[split]:
                        task_results[split] = results_for_plotting[split][task]
                
                if task_results:
                    plot_parity(
                        results=task_results,
                        title=f"{task.upper()} Prediction Parity Plot",
                        output_path=os.path.join(self.run_dir, f"parity_plot_{task}.png")
                    )
        else:
            logger.info("No results to plot.")


