"""
Ensemble inference and evaluation module.

This module is responsible for all post-training inference work:

1. **Per-fold evaluation** (``_evaluate_fold``): Each fold's model predicts
   on its own sealed ``outer_test_data``.  The results are collected into a
   ``fold_metrics.csv`` summary.

2. **Cross-fold ensemble** (``_run_cross_fold_ensemble``): Every fold model
   predicts on the *complete* ``extra_smiles_data`` set.  Predictions are
   stacked across folds and the per-sample mean (best estimate) and std
   (uncertainty) are computed.  Results are written to
   ``ensemble_predictions.csv`` and a parity plot with error bars.

The public entry point is ``run_ensemble_prediction``.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config.schema import MainConfig
from src.data.splitter import SmilesData
from src.inference.predictor import Predictor
from src.training.cv_runner import FoldResult
from src.utils.plotting import plot_parity_ensemble

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# results[split_name][task] = (y_true, pred_mean, pred_std)
EnsembleResults = Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_predictor(
    model_path: str,
    vocab_path: str,
    cfg: MainConfig,
    device: torch.device,
) -> Predictor:
    """Instantiate a ``Predictor`` from a saved fold checkpoint.

    Args:
        model_path: Path to the ``.pt`` model checkpoint.
        vocab_path: Path to the featurizer vocabulary JSON.
        cfg: Full configuration object.
        device: Inference device.

    Returns:
        A ``Predictor`` ready for ``.predict()`` calls.
    """
    return Predictor(
        model_path=model_path,
        vocab_path=vocab_path,
        featurizer_type=cfg.data.featurizer_type,
        atom_features=cfg.model.atom_features,
        num_messages=cfg.model.num_messages,
        num_tasks=cfg.model.num_tasks,
        target_columns=cfg.data.target_columns,
        device=device,
    )


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and R² for a single task.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted values.

    Returns:
        Dict with keys ``'MAE'``, ``'RMSE'``, ``'R2'``.
    """
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def _evaluate_fold(
    predictor: Predictor,
    outer_test_data: SmilesData,
    full_df: pd.DataFrame,
    target_columns: List[str],
    fold_idx: int,
) -> Optional[Dict]:
    """Evaluate a fold model on its own sealed outer_test_data.

    Args:
        predictor: Loaded predictor for this fold's model.
        outer_test_data: The fold's sealed test set (never seen during training).
        full_df: Complete merged DataFrame used to look up ground-truth labels.
        target_columns: List of task column names.
        fold_idx: Fold index used for logging.

    Returns:
        A flat dict of metrics ``{fold, task_MAE, task_RMSE, task_R2, …}``
        for constructing a summary row, or ``None`` if evaluation is skipped.
    """
    if not outer_test_data:
        logger.info("Fold %d: outer_test is empty — skipping fold evaluation.", fold_idx)
        return None

    smiles_list = sorted({item[0] for item in outer_test_data})
    pred_df = predictor.predict(smiles_list, drop_duplicates=False)

    if pred_df.empty:
        logger.warning("Fold %d: predictor returned empty DataFrame.", fold_idx)
        return None

    # Join with ground-truth labels
    gt_cols = ["molecule", "bond_index"] + target_columns
    available_gt = [c for c in gt_cols if c in full_df.columns]
    merged = pd.merge(
        pred_df,
        full_df[available_gt],
        on=["molecule", "bond_index"],
        how="inner",
    )

    if merged.empty:
        logger.warning("Fold %d: merge with ground-truth yielded empty DataFrame.", fold_idx)
        return None

    row: Dict = {"fold": fold_idx}
    for task in target_columns:
        pred_col = f"{task}_pred"
        if pred_col not in merged.columns or task not in merged.columns:
            continue
        valid = ~merged[task].isna()
        y_true = merged.loc[valid, task].values
        y_pred = merged.loc[valid, pred_col].values
        if len(y_true) == 0:
            continue
        metrics = _compute_metrics(y_true, y_pred)
        for metric_name, value in metrics.items():
            row[f"{task}_{metric_name}"] = value
        logger.info(
            "Fold %d | outer_test | %s → MAE=%.4f  RMSE=%.4f  R²=%.4f",
            fold_idx, task, metrics["MAE"], metrics["RMSE"], metrics["R2"],
        )

    return row


def _run_cross_fold_ensemble(
    fold_results: List[FoldResult],
    extra_smiles_list: List[str],
    full_df: pd.DataFrame,
    cfg: MainConfig,
    vocab_path: str,
    device: torch.device,
    run_dir: str,
) -> None:
    """Stack fold predictions on the full extra dataset and compute ensemble stats.

    Each fold model predicts on the complete ``extra_smiles_list``.  Predictions
    are stacked along axis 0 (fold dimension) and per-sample mean / std are
    persisted to CSV and plotted with error bars.

    Args:
        fold_results: List of ``FoldResult`` objects from ``run_cv_loop``.
        extra_smiles_list: Unique canonical SMILES strings for the full extra
            dataset (target of ensemble prediction).
        full_df: Complete merged DataFrame for ground-truth label lookup.
        cfg: Full configuration object.
        vocab_path: Path to featurizer vocabulary file.
        device: Inference device.
        run_dir: Directory where output artefacts are written.
    """
    pred_cols = [f"{t}_pred" for t in cfg.data.target_columns]
    stacked: Dict[str, List[np.ndarray]] = {col: [] for col in pred_cols}
    ref_df: Optional[pd.DataFrame] = None

    for result in fold_results:
        logger.info(
            "Ensemble — loading fold %d model: %s",
            result.fold_idx, result.model_path,
        )
        try:
            predictor = _build_predictor(result.model_path, vocab_path, cfg, device)
            pred_df = predictor.predict(extra_smiles_list, drop_duplicates=False)
        except Exception as exc:
            logger.error(
                "Ensemble — fold %d prediction failed: %s", result.fold_idx, exc,
                exc_info=True,
            )
            continue

        if pred_df.empty:
            logger.warning("Ensemble — fold %d returned empty predictions.", result.fold_idx)
            continue

        # Establish reference index from the first successful fold
        if ref_df is None:
            ref_df = pred_df[["molecule", "bond_index"]].copy()

        # Align this fold's predictions onto the reference index
        aligned = pd.merge(
            ref_df,
            pred_df[["molecule", "bond_index"] + pred_cols],
            on=["molecule", "bond_index"],
            how="left",
        )
        for col in pred_cols:
            stacked[col].append(aligned[col].values.astype(float))

    if ref_df is None or not any(stacked.values()):
        logger.error("Ensemble: no valid predictions collected — skipping.")
        return

    # ── Compute mean + std across folds ───────────────────────────────────
    ensemble_df = ref_df.copy()
    for col in pred_cols:
        arr = np.vstack(stacked[col])          # [n_folds, n_bonds]
        ensemble_df[f"{col}_mean"] = arr.mean(axis=0)
        ensemble_df[f"{col}_std"]  = arr.std(axis=0)

    # ── Join ground-truth labels ───────────────────────────────────────────
    gt_cols = ["molecule", "bond_index"] + cfg.data.target_columns
    available_gt = [c for c in gt_cols if c in full_df.columns]
    if available_gt:
        ensemble_df = pd.merge(
            ensemble_df,
            full_df[available_gt],
            on=["molecule", "bond_index"],
            how="left",
        )

    # ── Persist CSV ────────────────────────────────────────────────────────
    csv_path = os.path.join(run_dir, "ensemble_predictions.csv")
    ensemble_df.to_csv(csv_path, index=False)
    logger.info("Ensemble predictions saved → %s", csv_path)

    # ── Build results dict for parity plot ────────────────────────────────
    plot_results: EnsembleResults = {"extra": {}}
    for task in cfg.data.target_columns:
        if task not in ensemble_df.columns:
            continue
        valid = ~ensemble_df[task].isna()
        y_true     = ensemble_df.loc[valid, task].values
        pred_mean  = ensemble_df.loc[valid, f"{task}_pred_mean"].values
        pred_std   = ensemble_df.loc[valid, f"{task}_pred_std"].values
        if len(y_true) > 0:
            plot_results["extra"][task] = (y_true, pred_mean, pred_std)

    if plot_results.get("extra"):
        parity_path = os.path.join(run_dir, "parity_ensemble.png")
        plot_parity_ensemble(
            results=plot_results,
            task_names=cfg.data.target_columns,
            output_path=parity_path,
        )
    else:
        logger.info("Ensemble: no ground-truth labels available for parity plot.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ensemble_prediction(
    fold_results: List[FoldResult],
    extra_smiles_data: SmilesData,
    full_df: pd.DataFrame,
    cfg: MainConfig,
    vocab_path: str,
    device: torch.device,
    run_dir: str,
) -> None:
    """Run per-fold outer_test evaluation and cross-fold ensemble inference.

    This is the single public entry point for all post-training inference.
    It performs two complementary evaluations:

    1. **Per-fold outer_test evaluation**: Each fold model predicts on its own
       sealed test set.  Metrics are aggregated into ``fold_metrics.csv``.

    2. **Cross-fold ensemble**: All fold models predict on the complete
       ``extra_smiles_data``.  Per-sample mean and std are computed and saved
       to ``ensemble_predictions.csv`` and ``parity_ensemble.png``.

    When ``extra_smiles_data`` is empty (e.g. ``cv='none'`` with no extra
    data), the ensemble step is skipped with a log message.

    Args:
        fold_results: List of ``FoldResult`` from ``run_cv_loop``.
        extra_smiles_data: Processed smiles data for the full extra dataset.
        full_df: Complete merged DataFrame (ground-truth labels source).
        cfg: Full configuration object.
        vocab_path: Path to featurizer vocabulary JSON.
        device: Inference device.
        run_dir: Directory for all output artefacts.
    """
    if not fold_results:
        logger.warning("run_ensemble_prediction: no fold results provided — skipping.")
        return

    logger.info(
        "\n%s\n  POST-TRAINING EVALUATION  (%d fold model(s))\n%s",
        "=" * 62, len(fold_results), "=" * 62,
    )

    # ── 1. Per-fold outer_test evaluation ─────────────────────────────────
    fold_metric_rows: List[Dict] = []

    for result in fold_results:
        if not result.outer_test_data:
            continue  # cv='none' → no outer test
        logger.info("Evaluating fold %d on its outer_test…", result.fold_idx)
        try:
            predictor = _build_predictor(
                result.model_path, vocab_path, cfg, device
            )
            row = _evaluate_fold(
                predictor=predictor,
                outer_test_data=result.outer_test_data,
                full_df=full_df,
                target_columns=cfg.data.target_columns,
                fold_idx=result.fold_idx,
            )
            if row is not None:
                fold_metric_rows.append(row)
        except Exception as exc:
            logger.error(
                "Fold %d outer_test evaluation failed: %s",
                result.fold_idx, exc, exc_info=True,
            )

    if fold_metric_rows:
        metrics_path = os.path.join(run_dir, "fold_metrics.csv")
        pd.DataFrame(fold_metric_rows).to_csv(metrics_path, index=False)
        logger.info("Fold metrics saved → %s", metrics_path)

    # ── 2. Cross-fold ensemble ─────────────────────────────────────────────
    if not extra_smiles_data:
        logger.info(
            "No extra_smiles_data provided — skipping cross-fold ensemble."
        )
        return

    extra_smiles_list = sorted({item[0] for item in extra_smiles_data})
    logger.info(
        "Cross-fold ensemble on %d unique extra molecules with %d model(s)…",
        len(extra_smiles_list), len(fold_results),
    )
    _run_cross_fold_ensemble(
        fold_results=fold_results,
        extra_smiles_list=extra_smiles_list,
        full_df=full_df,
        cfg=cfg,
        vocab_path=vocab_path,
        device=device,
        run_dir=run_dir,
    )
