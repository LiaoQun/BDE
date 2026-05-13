"""
CV fold runner module.

This module is responsible for the **per-fold training lifecycle**:
  1. Receive ``(broad_train, outer_test, fold_idx)`` from the CV generator.
  2. Perform the inner train/val split (early-stopping validation).
  3. Build ``BDEDataset`` objects and ``DataLoader`` instances.
  4. Instantiate a fresh ``BDEModel`` and ``Adam`` optimiser.
  5. Run ``Trainer.train()``.
  6. Immediately evaluate on ``outer_test`` using the just-saved checkpoint.
  7. Append predictions to ``cv_predictions.csv``.
  8. **Explicitly release** all heavy GPU/CPU tensors before the next fold.

After all folds complete, ``run_cv_loop`` draws the CV parity plot from
the accumulated ``cv_predictions.csv``.
"""
import gc
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader

from src.config.schema import MainConfig
from src.data.dataset import BDEDataset
from src.data.splitter import SmilesData, split_inner_val
from src.models.mpnn import BDEModel
from src.training.trainer import Trainer
from src.utils.plotting import plot_parity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FoldResult
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Aggregates the outcomes of a single CV fold.

    Attributes:
        fold_idx:       Zero-based fold index.
        model_path:     Absolute path to the saved model checkpoint.
        n_inner_train:  Number of entries in the inner training set.
        n_inner_val:    Number of entries in the inner validation set
                        (0 when ``val_size == 0.0``).
    """
    fold_idx: int
    model_path: str
    n_inner_train: int = 0
    n_inner_val: int = 0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_model(cfg: MainConfig, featurizer, device: torch.device) -> BDEModel:
    return BDEModel(
        atom_input_dim=featurizer.atom_dim,
        bond_input_dim=featurizer.bond_dim,
        atom_features=cfg.model.atom_features,
        num_messages=cfg.model.num_messages,
        inputs_are_discrete=featurizer.is_discrete,
        num_tasks=cfg.model.num_tasks,
    ).to(device)


def _build_loaders(
    cfg: MainConfig,
    featurizer,
    inner_train: SmilesData,
    inner_val: SmilesData,
    fold_tag: str,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    dataset_base = os.path.join(cfg.data.dataset_dir, fold_tag)

    train_dataset = BDEDataset(
        root=os.path.join(dataset_base, "train"),
        smiles_data=inner_train,
        featurizer=featurizer,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
    )

    val_loader: Optional[DataLoader] = None
    if inner_val:
        val_dataset = BDEDataset(
            root=os.path.join(dataset_base, "val"),
            smiles_data=inner_val,
            featurizer=featurizer,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
        )

    return train_loader, val_loader


def _cleanup_fold(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("CUDA cache cleared after fold cleanup.")


def _append_cv_predictions(pred_df: pd.DataFrame, run_dir: str) -> None:
    """Append fold predictions to cv_predictions.csv.

    Uses header=True on first write, header=False on subsequent appends
    so the file is always a valid single-header CSV.

    Args:
        pred_df:  Predictions DataFrame for this fold (includes 'fold' column).
        run_dir:  Root run directory where cv_predictions.csv lives.
    """
    csv_path = os.path.join(run_dir, "cv_predictions.csv")
    write_header = not os.path.exists(csv_path)
    pred_df.to_csv(csv_path, mode='a', header=write_header, index=False)
    logger.info(
        "Fold %d predictions appended → %s (%d rows)",
        pred_df['fold'].iloc[0], csv_path, len(pred_df),
    )


def _plot_cv_parity(
    run_dir: str,
    full_df: pd.DataFrame,
    cfg: MainConfig,
) -> None:
    """Read cv_predictions.csv and draw a parity plot.

    Each molecule appears exactly once, predicted by the fold that was
    blind to it — statistically honest CV parity plot with no data leakage.

    Args:
        run_dir:  Root run directory.
        full_df:  Complete merged DataFrame for ground-truth label lookup.
        cfg:      Full configuration object.
    """
    csv_path = os.path.join(run_dir, "cv_predictions.csv")
    if not os.path.exists(csv_path):
        logger.info("No cv_predictions.csv found — skipping CV parity plot.")
        return

    combined_df = pd.read_csv(csv_path)

    # Join ground-truth labels
    gt_cols = ["molecule", "bond_index"] + cfg.data.target_columns
    available_gt = [c for c in gt_cols if c in full_df.columns]
    if available_gt:
        combined_df = pd.merge(
            combined_df,
            full_df[available_gt],
            on=["molecule", "bond_index"],
            how="left",
        )

    # Build results dict: results['cv'][task] = (y_true, y_pred)
    plot_results = {"cv": {}}
    for task in cfg.data.target_columns:
        pred_col = f"{task}_pred"
        if task not in combined_df.columns or pred_col not in combined_df.columns:
            continue
        valid = ~combined_df[task].isna()
        y_true = combined_df.loc[valid, task].values
        y_pred = combined_df.loc[valid, pred_col].values
        if len(y_true) == 0:
            continue

        plot_results["cv"][task] = (y_true, y_pred)
        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2   = float(r2_score(y_true, y_pred))
        logger.info(
            "CV overall | %s → MAE=%.4f  RMSE=%.4f  R²=%.4f",
            task, mae, rmse, r2,
        )

    if plot_results["cv"]:
        parity_path = os.path.join(run_dir, "cv_parity_plot.png")
        plot_parity(
            results=plot_results,
            task_names=cfg.data.target_columns,
            output_path=parity_path,
        )
    else:
        logger.info("No ground-truth labels available for CV parity plot.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_cv_loop(
    base_data: SmilesData,
    extra_data: SmilesData,
    cfg: MainConfig,
    featurizer,
    device: torch.device,
    run_dir: str,
    full_df: pd.DataFrame,
) -> List[FoldResult]:
    """Execute the full cross-validation training loop.

    For each fold:
      1. Split ``broad_train`` into ``inner_train`` / ``inner_val``.
      2. Build datasets, loaders, a fresh model, and an optimiser.
      3. Run ``Trainer.train()``.
      4. Immediately load the saved checkpoint and predict on ``outer_test``.
      5. Append predictions to ``cv_predictions.csv``.
      6. Release all heavy tensors.

    After all folds, draw the CV parity plot from ``cv_predictions.csv``.

    Args:
        base_data: Processed smiles data always included in training.
        extra_data: Processed smiles data subject to CV splitting.
        cfg: Full configuration object.
        featurizer: Fitted featurizer (shared across folds, read-only).
        device: Training device.
        run_dir: Root directory for all run artefacts.
        full_df: Complete merged DataFrame for ground-truth label lookup.

    Returns:
        List of ``FoldResult`` objects, one per fold, in fold order.
    """
    from src.data.splitter import generate_cv_splits
    from src.inference.predictor import Predictor

    fold_results: List[FoldResult] = []

    cv_gen = generate_cv_splits(
        base_data=base_data,
        extra_data=extra_data,
        cv_param=cfg.data.cross_validation,
        random_seed=cfg.data.random_seed,
    )

    for broad_train, outer_test, fold_idx in cv_gen:
        fold_tag = f"fold_{fold_idx}"
        fold_run_dir = os.path.join(run_dir, fold_tag)
        os.makedirs(fold_run_dir, exist_ok=True)

        # Copy config.yaml and vocab.json into fold_run_dir so that
        # Predictor.from_run_dir(fold_run_dir) can find them.
        for filename in ("config.yaml", "vocab.json"):
            src = os.path.join(run_dir, filename)
            dst = os.path.join(fold_run_dir, filename)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

        logger.info(
            "\n%s\n  FOLD %d  |  broad_train: %d entries  |  outer_test: %d entries\n%s",
            "=" * 62, fold_idx, len(broad_train), len(outer_test), "=" * 62,
        )

        # ── Inner split ────────────────────────────────────────────────────
        if cfg.data.val_size > 0.0:
            inner_train, inner_val = split_inner_val(
                broad_train=broad_train,
                val_size=cfg.data.val_size,
                random_seed=cfg.data.random_seed,
            )
            logger.info(
                "Inner split (val_size=%.0f%%): inner_train=%d, inner_val=%d",
                cfg.data.val_size * 100, len(inner_train), len(inner_val),
            )
        else:
            inner_train = broad_train
            inner_val = []
            logger.info("Inner split: val_size=0.0 → Method-A (no early stopping).")

        # ── Build data loaders ─────────────────────────────────────────────
        train_loader, val_loader = _build_loaders(
            cfg=cfg,
            featurizer=featurizer,
            inner_train=inner_train,
            inner_val=inner_val,
            fold_tag=fold_tag,
        )

        # ── Instantiate fresh model + optimiser ────────────────────────────
        model = _build_model(cfg, featurizer, device)
        optimizer: Optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.train.lr
        )

        # ── Train ──────────────────────────────────────────────────────────
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            cfg=cfg.train,
            model_cfg=cfg.model,
            run_dir=fold_run_dir,
            target_columns=cfg.data.target_columns,
            fold_idx=fold_idx,
        )
        trainer.train()

        # ── Evaluate on outer_test immediately after training ──────────────
        if outer_test:
            logger.info(
                "Fold %d: evaluating outer_test (%d molecule(s))…",
                fold_idx,
                len({item[0] for item in outer_test}),
            )
            try:
                predictor = Predictor.from_run_dir(fold_run_dir, device=str(device))
                smiles_list = sorted({item[0] for item in outer_test})
                pred_df = predictor.predict(smiles_list, drop_duplicates=False)
                if not pred_df.empty:
                    pred_df['fold'] = fold_idx
                    _append_cv_predictions(pred_df, run_dir)
                else:
                    logger.warning(
                        "Fold %d: outer_test prediction returned empty DataFrame.",
                        fold_idx,
                    )
            except Exception as exc:
                logger.error(
                    "Fold %d: outer_test evaluation failed: %s",
                    fold_idx, exc, exc_info=True,
                )
        else:
            logger.info(
                "Fold %d: no outer_test (cv='none') — skipping fold evaluation.",
                fold_idx,
            )

        # ── Record result ──────────────────────────────────────────────────
        result = FoldResult(
            fold_idx=fold_idx,
            model_path=trainer.model_save_path,
            n_inner_train=len(inner_train),
            n_inner_val=len(inner_val),
        )
        fold_results.append(result)
        logger.info("Fold %d complete. Model saved → %s", fold_idx, result.model_path)

        # ── OOM prevention ─────────────────────────────────────────────────
        _cleanup_fold(train_loader, val_loader, model, optimizer, trainer)

    logger.info(
        "\nAll %d fold(s) finished. Model checkpoints:\n  %s",
        len(fold_results),
        "\n  ".join(r.model_path for r in fold_results),
    )

    # ── Draw CV parity plot from accumulated cv_predictions.csv ───────────
    _plot_cv_parity(run_dir, full_df, cfg)

    return fold_results