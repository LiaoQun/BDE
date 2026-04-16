"""
CV fold runner module.

This module is responsible for the **per-fold training lifecycle**:
  1. Receive ``(broad_train, outer_test, fold_idx)`` from the CV generator.
  2. Perform the inner train/val split (early-stopping validation).
  3. Build ``BDEDataset`` objects and ``DataLoader`` instances.
  4. Instantiate a fresh ``BDEModel`` and ``Adam`` optimiser.
  5. Run ``Trainer.train()``.
  6. Collect and return a ``FoldResult``.
  7. **Explicitly release** all heavy GPU/CPU tensors before the next fold
     to prevent OOM accumulation across folds.

The public entry point is ``run_cv_loop``, which delegates per-fold work to
private helpers and accumulates ``FoldResult`` objects for downstream use.
"""
import gc
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader

from src.config.schema import MainConfig
from src.data.dataset import BDEDataset
from src.data.splitter import SmilesData, split_inner_val
from src.models.mpnn import BDEModel
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FoldResult — data-transfer object returned after each fold
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Aggregates the outcomes of a single CV fold.

    Attributes:
        fold_idx: Zero-based fold index.
        model_path: Absolute path to the saved model checkpoint for this fold.
        outer_test_data: The sealed outer-loop test data for this fold.
            Populated only when the CV strategy produces a non-empty outer
            test set (i.e. not ``cv='none'``).
        n_inner_train: Number of bond entries in the inner training set.
        n_inner_val: Number of bond entries in the inner validation set
            (0 when ``val_size == 0.0``).
    """

    fold_idx: int
    model_path: str
    outer_test_data: SmilesData = field(default_factory=list)
    n_inner_train: int = 0
    n_inner_val: int = 0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_model(cfg: MainConfig, featurizer, device: torch.device) -> BDEModel:
    """Instantiate a fresh, randomly initialised ``BDEModel``.

    A new model is created for every fold so that folds are independent.

    Args:
        cfg: Full configuration object.
        featurizer: Fitted featurizer providing ``atom_dim`` / ``bond_dim``.
        device: Target device (CPU or CUDA).

    Returns:
        A ``BDEModel`` instance moved to *device*.
    """
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
    """Create ``BDEDataset`` objects and wrap them in ``DataLoader`` instances.

    The dataset root directories are scoped per fold so that PyG's on-disk
    cache does not collide between folds.

    Args:
        cfg: Full configuration object.
        featurizer: Fitted featurizer.
        inner_train: Inner training data for this fold.
        inner_val: Inner validation data for this fold (may be empty).
        fold_tag: String tag used in the dataset directory name (e.g.
            ``'fold_0'``).

    Returns:
        Tuple ``(train_loader, val_loader)`` where *val_loader* is ``None``
        when *inner_val* is empty.
    """
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
    """Explicitly release heavy objects at the end of a fold.

    Calling ``del`` on PyTorch / PyG objects and then invoking the garbage
    collector ensures that GPU VRAM and CPU RAM are freed before the next
    fold's tensors are allocated.  Without this, K folds would accumulate
    K sets of graph tensors in memory simultaneously.

    Args:
        *objects: Arbitrary objects to delete (datasets, loaders, model,
            optimiser, trainer, …).
    """
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("CUDA cache cleared after fold cleanup.")


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
) -> List[FoldResult]:
    """Execute the full cross-validation training loop.

    For each fold produced by ``generate_cv_splits``:
      1. Split ``broad_train`` into ``inner_train`` / ``inner_val``.
      2. Build datasets, loaders, a fresh model, and an optimiser.
      3. Run ``Trainer.train()``.
      4. Record the ``FoldResult`` (model path, outer_test, sizes).
      5. Release all heavy tensors (OOM prevention).

    Args:
        base_data: Processed smiles data that is always included in training.
        extra_data: Processed smiles data subject to CV splitting.
        cfg: Full configuration object.
        featurizer: Fitted featurizer (shared across folds, read-only).
        device: Training device.
        run_dir: Root directory for all run artefacts.

    Returns:
        List of ``FoldResult`` objects, one per fold, in fold order.
    """
    # Import here to avoid circular imports with pipeline.py
    from src.data.splitter import generate_cv_splits

    fold_results: List[FoldResult] = []

    cv_gen = generate_cv_splits(
        base_data=base_data,
        extra_data=extra_data,
        cv_param=cfg.data.cross_validation,
        random_seed=cfg.data.random_seed,
    )

    for broad_train, outer_test, fold_idx in cv_gen:
        fold_tag = f"fold_{fold_idx}"
        # Always create a fold_N/ subdirectory, regardless of cv strategy.
        # This ensures a uniform run_dir layout for both cv='none' and K-Fold,
        # making FoldResult.model_path predictable and Predictor maintenance simpler.
        fold_run_dir = os.path.join(run_dir, fold_tag)
        os.makedirs(fold_run_dir, exist_ok=True)

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
                cfg.data.val_size * 100,
                len(inner_train),
                len(inner_val),
            )
        else:
            # Method-A: no inner validation → no early stopping
            inner_train = broad_train
            inner_val = []
            logger.info(
                "Inner split: val_size=0.0 → Method-A (no early stopping)."
            )

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
            val_loader=val_loader,       # None → Method-A (no early stopping)
            test_loader=None,            # Outer-test evaluation in ensemble.py
            device=device,
            cfg=cfg.train,
            model_cfg=cfg.model,
            run_dir=fold_run_dir,
            full_dataset_df=None,        # Not needed here
            data_splits={},              # Not needed here
            vocab_path="",               # Not needed here
            target_columns=cfg.data.target_columns,
            fold_idx=fold_idx,
        )
        trainer.train()

        # ── Record result ──────────────────────────────────────────────────
        result = FoldResult(
            fold_idx=fold_idx,
            model_path=trainer.model_save_path,
            outer_test_data=outer_test,
            n_inner_train=len(inner_train),
            n_inner_val=len(inner_val),
        )
        fold_results.append(result)
        logger.info(
            "Fold %d complete. Model saved → %s", fold_idx, result.model_path
        )

        # ── OOM prevention: release all heavy fold-specific objects ────────
        _cleanup_fold(
            train_loader, val_loader,
            model, optimizer, trainer,
        )

    logger.info(
        "\nAll %d fold(s) finished. Model checkpoints:\n  %s",
        len(fold_results),
        "\n  ".join(r.model_path for r in fold_results),
    )
    return fold_results
