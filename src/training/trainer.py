import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List, Optional
import logging

from src.config import TrainConfig, ModelConfig
from src.utils.reporting import save_training_log
from src.utils.plotting import plot_training_curve

logger = logging.getLogger(__name__)

class Trainer:
    """
    Handles the model training loop for a single fold.

    Responsibilities:
    - Run training epochs (with or without validation / early stopping)
    - Save the best model checkpoint
    - Persist training log CSV and loss curve plot

    Evaluation against ground-truth labels is intentionally out of scope
    and is handled by EnsembleEvaluator in src/training/ensemble.py.
    """

    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        val_loader,
        device,
        cfg: TrainConfig,
        model_cfg: ModelConfig,
        run_dir: str,
        target_columns: List[str] = None,
        fold_idx: int = 0,
    ):
        """
        Args:
            model: The BDEModel instance to train.
            optimizer: The optimiser (e.g. Adam).
            train_loader: DataLoader for the inner training set.
            val_loader: DataLoader for the inner validation set.
                        Pass None to use Method-A (no early stopping).
            device: torch.device to run training on.
            cfg: TrainConfig with epochs, lr, batch_size, etc.
            model_cfg: ModelConfig with atom_features, num_messages, etc.
            run_dir: Directory to save model checkpoint and training logs.
            target_columns: Names of the prediction targets.
            fold_idx: Zero-based fold index (used for log prefixing).
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.run_dir = run_dir
        self.fold_idx = fold_idx
        self.model_save_path = os.path.join(run_dir, cfg.model_save_path)
        self.target_columns = target_columns if target_columns is not None else ['bde']

    def train(self) -> None:
        """
        Executes the main training loop, including validation and early stopping.

        Behaviour depends on whether ``val_loader`` was provided:

        - **With val_loader** (Method-B / K-Fold): validates each epoch,
          saves the best model (lowest val loss), and applies early stopping.
        - **Without val_loader** (Method-A, ``cv='all'`` + ``val_size=0``):
          trains for exactly ``cfg.epochs`` epochs with no validation step.
          The model is saved after the **last** epoch.
        """
        fold_tag = f"[Fold {self.fold_idx}] " if self.fold_idx > 0 else ""

        if self.val_loader is None:
            # ── Method-A: no validation, fixed epoch count ────────────────────
            logger.info(
                f"{fold_tag}Starting training (Method-A: no validation, "
                f"{self.cfg.epochs} epochs)..."
            )
            history = []
            for epoch in range(1, self.cfg.epochs + 1):
                avg_train_loss = self._train_epoch(epoch)
                logger.info(
                    f"{fold_tag}Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f}"
                )
                history.append({'epoch': epoch, 'train_loss': avg_train_loss, 'val_loss': None})

            # Save the final model (no best-val checkpoint concept here)
            torch.save(self.model.state_dict(), self.model_save_path)
            logger.info(
                f"{fold_tag}Training finished. Final model saved to {self.model_save_path}"
            )

        else:
            # ── Method-B / K-Fold: validate each epoch, early stopping ────────
            logger.info(f"{fold_tag}Starting training with validation...")
            best_val_loss = float('inf')
            patience_counter = 0
            history = []

            for epoch in range(1, self.cfg.epochs + 1):
                avg_train_loss = self._train_epoch(epoch)
                avg_val_loss = self._validate_epoch(epoch)

                logger.info(
                    f"{fold_tag}Epoch {epoch:03d} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f}"
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
                        f"{fold_tag}  -> New best val loss: {best_val_loss:.4f}. "
                        f"Model saved to {self.model_save_path}"
                    )
                    patience_counter = 0
                else:
                    patience_counter += 1
                    logger.info(
                        f"{fold_tag}  -> Val loss did not improve. "
                        f"Patience: {patience_counter}/{self.cfg.early_stopping_patience}"
                    )

                if patience_counter >= self.cfg.early_stopping_patience:
                    logger.info(f"{fold_tag}\nEarly stopping triggered.")
                    break

            logger.info(f"{fold_tag}Training finished.")

        history_df = save_training_log(
            [h for h in history if h.get('val_loss') is not None],
            self.run_dir,
            suffix="",
        )
        if history_df is not None and not history_df.empty:
            plot_training_curve(history_df, self.run_dir, suffix="")

    def _train_epoch(self, epoch: int) -> float:
        """Runs one training epoch and returns the average loss."""
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
        """Runs one validation epoch and returns the average loss."""
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