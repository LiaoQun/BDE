import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

from src.config import TrainConfig, ModelConfig
from src.utils.reporting import save_training_log
from src.utils.plotting import plot_training_curve, plot_parity
from src.inference.predictor import Predictor

logger = logging.getLogger(__name__)

class Trainer:
    """
    Handles the model training, validation, and evaluation pipeline.
    """

    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        device,
        cfg: TrainConfig,
        model_cfg: ModelConfig,
        run_dir: str,
        full_dataset_df: pd.DataFrame,
        data_splits: Dict[str, List],
        vocab_path: str,
        featurizer_type: str = 'TokenFeaturizer',
        target_columns: List[str] = None,
    ):
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
            avg_train_loss = self._train_epoch(epoch)
            avg_val_loss = self._validate_epoch(epoch)

            logger.info(
                f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            )
            history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            })

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), self.model_save_path)
                logger.info(
                    f"  -> New best validation loss: {best_val_loss:.4f}. "
                    f"Model saved to {self.model_save_path}"
                )
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(
                    f"  -> Validation loss did not improve. "
                    f"Patience: {patience_counter}/{self.cfg.early_stopping_patience}"
                )

            if patience_counter >= self.cfg.early_stopping_patience:
                logger.info("\nEarly stopping triggered.")
                break

        logger.info("Training finished.")

        history_df = save_training_log(history, self.run_dir)
        plot_training_curve(history_df, self.run_dir)

    def _train_epoch(self, epoch: int) -> float:
        """Handles the training logic for a single epoch."""
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(batch)  # [num_edges, num_tasks]

            if batch.mask.sum() > 0:
                loss = F.l1_loss(pred[batch.mask], batch.y[batch.mask])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch.num_graphs

        n = len(self.train_loader.dataset)
        return total_loss / n if n > 0 else 0.0

    def _validate_epoch(self, epoch: int) -> float:
        """Handles the validation logic for a single epoch."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                batch = batch.to(self.device)
                pred = self.model(batch)  # [num_edges, num_tasks]

                if batch.mask.sum() > 0:
                    loss = F.l1_loss(pred[batch.mask], batch.y[batch.mask])
                    total_loss += loss.item() * batch.num_graphs

        n = len(self.val_loader.dataset)
        return total_loss / n if n > 0 else 0.0

    def evaluate(self):
        """
        Evaluates the best model on **train** and **test** splits only,
        saves full-prediction CSVs, and generates a single multi-task
        parity plot (one subplot per task, train/test overlaid).

        The ``val`` split stored in ``self.data_splits`` is intentionally
        skipped here — validation is used only for early-stopping during
        training.
        """
        logger.info(
            f"Loading best model from {self.model_save_path} for final evaluation..."
        )

        try:
            predictor = Predictor(
                model_path=self.model_save_path,
                vocab_path=self.vocab_path,
                featurizer_type=self.featurizer_type,
                atom_features=self.model_cfg.atom_features,
                num_messages=self.model_cfg.num_messages,
                num_tasks=self.model_cfg.num_tasks,
                target_columns=self.target_columns,
                device=self.device,
            )
        except FileNotFoundError as e:
            logger.info(f"Could not initialise predictor: {e}. Aborting evaluation.")
            return

        # results_for_plot structure:
        #   results_for_plot[split][task] = (y_true: np.ndarray, y_pred: np.ndarray)
        results_for_plot: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

        # Only evaluate train and test; skip val
        for split_name in ("train", "test"):
            data_list = self.data_splits.get(split_name, [])
            logger.info(f"--- Predicting on {split_name} set ---")
            if not data_list:
                logger.info(f"{split_name} set is empty. Skipping.")
                continue

            smiles_list = sorted({item[0] for item in data_list})
            pred_df = predictor.predict(smiles_list, drop_duplicates=False)

            # Merge predictions with ground-truth labels
            merged_df = pd.merge(
                pred_df,
                self.full_dataset_df[['molecule', 'bond_index'] + self.target_columns],
                on=['molecule', 'bond_index'],
                how='inner',
            )

            # Persist detailed predictions
            output_path = os.path.join(self.run_dir, f'predictions_{split_name}.csv')
            merged_df.to_csv(output_path, index=False)
            logger.info(f"Saved detailed predictions for {split_name} set to {output_path}")

            if merged_df.empty:
                continue

            results_for_plot[split_name] = {}
            for task in self.target_columns:
                pred_col = f"{task}_pred"
                if pred_col not in merged_df.columns:
                    continue
                valid_mask = ~merged_df[task].isna()
                y_true = merged_df.loc[valid_mask, task].values
                y_pred = merged_df.loc[valid_mask, pred_col].values
                if len(y_true) > 0:
                    results_for_plot[split_name][task] = (y_true, y_pred)

        # ── Single combined parity plot covering all tasks ────────────────────
        if results_for_plot:
            parity_path = os.path.join(self.run_dir, "parity_plot_all.png")
            plot_parity(
                results=results_for_plot,
                task_names=self.target_columns,
                output_path=parity_path,
            )
        else:
            logger.info("No results to plot.")