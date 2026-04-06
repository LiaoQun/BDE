"""
Top-level training pipeline orchestrator.

This module provides the single ``run_training`` function that coordinates
the complete training lifecycle:

  1. Load and prepare base and extra datasets.
  2. Build and persist the featurizer vocabulary.
  3. Delegate the cross-validation loop to ``cv_runner.run_cv_loop``.
  4. Delegate post-training inference to ``ensemble.run_ensemble_prediction``.

``run_training`` is intentionally thin: it only wires together the specialised
modules and passes data between them.  It contains no training or inference
logic itself.
"""
import logging
import os
from typing import Optional

import pandas as pd
import torch

from src.config.schema import MainConfig
from src.data.preprocessing import load_and_merge_data, prepare_data
from src.features import get_featurizer
from src.training.cv_runner import run_cv_loop
from src.training.ensemble import run_ensemble_prediction

logger = logging.getLogger(__name__)


def run_training(cfg: MainConfig, run_dir: str) -> Optional[str]:
    """Orchestrate the full CV training and ensemble evaluation pipeline.

    Steps
    -----
    1. Load ``base_data`` (always in training set) from ``cfg.data.base_data_paths``.
    2. Optionally load ``extra_data`` from ``cfg.data.extra_data_paths``.
    3. Build and save the featurizer vocabulary from the UNION of base + extra
       SMILES, so that extra-data molecules are never OOV at featurization time.
    4. Hand off to ``run_cv_loop`` for per-fold training.
    5. Hand off to ``run_ensemble_prediction`` for post-training evaluation.

    Args:
        cfg: Fully populated ``MainConfig`` instance.
        run_dir: Root directory for all run artefacts (logs, checkpoints,
            plots, CSVs).

    Returns:
        *run_dir* on successful completion, or ``None`` if the pipeline
        aborts early due to missing data or misconfiguration.
    """
    torch.manual_seed(cfg.data.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Run directory: %s", run_dir)

    # ── 1. Load base data ──────────────────────────────────────────────────
    logger.info("Loading base data from: %s", cfg.data.base_data_paths)
    base_df = load_and_merge_data(
        cfg.data.base_data_paths,
        target_columns=cfg.data.target_columns,
        random_seed=cfg.data.random_seed,
    )
    if base_df.empty:
        logger.error("Base dataset is empty after loading and cleaning — aborting.")
        return None

    # ── 2. Load extra data (optional) ─────────────────────────────────────
    extra_df = pd.DataFrame()
    if cfg.data.extra_data_paths:
        logger.info("Loading extra data from: %s", cfg.data.extra_data_paths)
        extra_df = load_and_merge_data(
            cfg.data.extra_data_paths,
            target_columns=cfg.data.target_columns,
            random_seed=cfg.data.random_seed,
        )
        if extra_df.empty:
            logger.warning(
                "Extra dataset is empty after loading — proceeding without extra data."
            )
    else:
        logger.info("No extra_data_paths configured — extra_data is empty.")

    # Full merged DataFrame: used for ground-truth label lookup during evaluation
    full_df: pd.DataFrame = (
        pd.concat([base_df, extra_df], ignore_index=True)
        if not extra_df.empty
        else base_df
    )

    # ── 3. Prepare processed smiles data ───────────────────────────────────
    logger.info("Preparing base smiles data…")
    base_smiles_data = prepare_data(base_df, target_columns=cfg.data.target_columns)

    extra_smiles_data = []
    if not extra_df.empty:
        logger.info("Preparing extra smiles data…")
        extra_smiles_data = prepare_data(
            extra_df, target_columns=cfg.data.target_columns
        )

    # ── 4. Build featurizer from base training SMILES ──────────────────────
    if cfg.data.featurizer_type is None:
        logger.error("'featurizer_type' is not configured — aborting.")
        return None

    vocab_save_path = os.path.join(run_dir, "vocab.json")

    # Build featurizer from the UNION of base + extra SMILES so that
    # extra-data molecules are never OOV when the featurizer scans atoms.
    # Deduplication ensures we don't scan the same molecule twice.
    base_train_smiles = [item[0] for item in base_smiles_data]
    extra_train_smiles = [item[0] for item in extra_smiles_data]
    all_train_smiles = list(dict.fromkeys(base_train_smiles + extra_train_smiles))

    logger.info(
        "Building featurizer: %s from %d unique SMILES (%d base + %d extra)…",
        cfg.data.featurizer_type,
        len(all_train_smiles),
        len(base_train_smiles),
        len(extra_train_smiles),
    )
    featurizer = get_featurizer(
        featurizer_type=cfg.data.featurizer_type,
        smiles_list=all_train_smiles,
        save_path=vocab_save_path,
    )
    logger.info("Featurizer saved → %s", vocab_save_path)

    # ── 5. CV training loop ────────────────────────────────────────────────
    fold_results = run_cv_loop(
        base_data=base_smiles_data,
        extra_data=extra_smiles_data,
        cfg=cfg,
        featurizer=featurizer,
        device=device,
        run_dir=run_dir,
    )

    if not fold_results:
        logger.error("CV loop produced no fold results — aborting post-training steps.")
        return None

    # ── 6. Post-training: per-fold evaluation + ensemble ──────────────────
    # Use extra_smiles_data if available, otherwise fall back to base data
    ensemble_target = extra_smiles_data if extra_smiles_data else base_smiles_data

    run_ensemble_prediction(
        fold_results=fold_results,
        extra_smiles_data=ensemble_target,
        full_df=full_df,
        cfg=cfg,
        vocab_path=vocab_save_path,
        device=device,
        run_dir=run_dir,
    )

    logger.info("Pipeline complete. All artefacts saved to: %s", run_dir)
    return run_dir
