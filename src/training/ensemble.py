"""
Ensemble inference and evaluation module.

This module is responsible for all post-training inference work:

1. **Per-fold evaluation** (``EnsembleEvaluator.run_fold_evaluation``): Each
   fold's model predicts on its own sealed ``outer_test_data``. The results
   are collected into a ``fold_metrics.csv`` summary.

2. **Cross-fold ensemble** (``EnsembleEvaluator.run_cross_fold_ensemble``):
   Every fold model predicts on the *complete* ``extra_smiles_data`` set.
   Predictions are stacked across folds and the per-sample mean (best estimate)
   and std (uncertainty) are computed. Results are written to
   ``ensemble_predictions.csv`` and a parity plot with error bars.

The typical entry point is ``EnsembleEvaluator.run()``, which orchestrates
both steps in sequence.  A module-level shim ``run_ensemble_prediction()`` is
retained for backward compatibility with ``pipeline.py``.
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
from src.inference.predictor import EnsemblePredictor, Predictor
from src.training.cv_runner import FoldResult
from src.utils.plotting import plot_parity_ensemble

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# results[split_name][task] = (y_true, pred_mean, pred_std)
EnsembleResults = Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]


# ---------------------------------------------------------------------------
# EnsembleEvaluator
# ---------------------------------------------------------------------------

class EnsembleEvaluator:
    """Orchestrates post-training fold evaluation and cross-fold ensemble.

    All shared state (config, vocab path, device, output directory, and the
    ground-truth DataFrame) is injected once at construction time.  The two
    evaluation steps are exposed as independent public methods so that callers
    can invoke them separately if needed, and ``run()`` executes both in the
    canonical order.

    Attributes:
        cfg:        Full configuration object.
        vocab_path: Path to the featurizer vocabulary JSON.
        device:     Torch device used for inference.
        run_dir:    Root directory for all output artefacts.
        full_df:    Complete merged DataFrame – source of ground-truth labels.
    """

    def __init__(
        self,
        cfg: MainConfig,
        vocab_path: str,
        device: torch.device,
        run_dir: str,
        full_df: pd.DataFrame,
    ) -> None:
        """Initialise the evaluator with all shared inference context.

        Args:
            cfg:        Full configuration object.
            vocab_path: Path to the featurizer vocabulary JSON.
            device:     Torch device used for inference.
            run_dir:    Root directory where output artefacts are written.
            full_df:    Complete merged DataFrame for ground-truth label lookup.
        """
        self.cfg = cfg
        self.vocab_path = vocab_path
        self.device = device
        self.run_dir = run_dir
        self.full_df = full_df

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def run(
        self,
        fold_results: List[FoldResult],
        extra_smiles_data: SmilesData,
    ) -> None:
        """Run per-fold outer_test evaluation followed by cross-fold ensemble.

        This is the canonical entry point.  It performs two complementary
        evaluations in sequence:

        1. **Per-fold outer_test evaluation** – each fold model predicts on its
           own sealed test set; metrics are aggregated into ``fold_metrics.csv``.
        2. **Cross-fold ensemble** – all fold models predict on the complete
           ``extra_smiles_data``; per-sample mean and std are saved to
           ``ensemble_predictions.csv`` and ``parity_ensemble.png``.

        When ``extra_smiles_data`` is empty the ensemble step is skipped.

        Args:
            fold_results:      List of ``FoldResult`` from ``run_cv_loop``.
            extra_smiles_data: Processed SMILES data for the full extra dataset.
        """
        if not fold_results:
            logger.warning("EnsembleEvaluator.run: no fold results provided — skipping.")
            return

        logger.info(
            "\n%s\n  POST-TRAINING EVALUATION  (%d fold model(s))\n%s",
            "=" * 62, len(fold_results), "=" * 62,
        )

        self.run_fold_evaluation(fold_results)
        self.run_cross_fold_ensemble(fold_results, extra_smiles_data)

    def run_fold_evaluation(self, fold_results: List[FoldResult]) -> None:
        """Evaluate each fold model on its own sealed outer_test set.

        Predictions and ground-truth labels are merged, per-task metrics
        (MAE / RMSE / R²) are computed, and all rows are written to
        ``fold_metrics.csv`` in ``run_dir``.

        Folds with an empty ``outer_test_data`` (e.g. ``cv='none'``) are
        silently skipped.

        Args:
            fold_results: List of ``FoldResult`` from ``run_cv_loop``.
        """
        fold_metric_rows: List[Dict] = []

        for result in fold_results:
            if not result.outer_test_data:
                continue  # cv='none' → no outer test set
            logger.info("Evaluating fold %d on its outer_test…", result.fold_idx)
            try:
                predictor = self._build_predictor(result.model_path)
                row = self._evaluate_fold(
                    predictor=predictor,
                    outer_test_data=result.outer_test_data,
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
            metrics_path = os.path.join(self.run_dir, "fold_metrics.csv")
            pd.DataFrame(fold_metric_rows).to_csv(metrics_path, index=False)
            logger.info("Fold metrics saved → %s", metrics_path)

    def run_cross_fold_ensemble(
        self,
        fold_results: List[FoldResult],
        extra_smiles_data: SmilesData,
    ) -> None:
        """Stack fold predictions on the full extra dataset and compute ensemble stats.

        Each fold model is loaded into an ``EnsemblePredictor`` which handles
        prediction stacking, mean, and std computation internally.  The resulting
        DataFrame (with ``{task}_pred_mean`` / ``{task}_pred_std`` columns) is
        joined with ground-truth labels, saved to CSV, and plotted.

        When ``extra_smiles_data`` is empty the method returns immediately.

        Args:
            fold_results:      List of ``FoldResult`` objects from ``run_cv_loop``.
            extra_smiles_data: Processed SMILES data for the full extra dataset.
        """
        if not extra_smiles_data:
            logger.info("No extra_smiles_data provided — skipping cross-fold ensemble.")
            return

        extra_smiles_list = sorted({item[0] for item in extra_smiles_data})
        logger.info(
            "Cross-fold ensemble on %d unique extra molecules with %d model(s)…",
            len(extra_smiles_list), len(fold_results),
        )

        # ── Build individual Predictor for each fold ───────────────────────
        predictors = []
        for result in fold_results:
            logger.info(
                "Ensemble — loading fold %d model: %s",
                result.fold_idx, result.model_path,
            )
            try:
                predictors.append(self._build_predictor(result.model_path))
            except Exception as exc:
                logger.error(
                    "Ensemble — fold %d model load failed: %s",
                    result.fold_idx, exc, exc_info=True,
                )

        if not predictors:
            logger.error("Ensemble: no predictors could be loaded — skipping.")
            return

        # ── Delegate stacking + mean/std computation to EnsemblePredictor ─
        # EnsemblePredictor returns {task}_pred_mean and {task}_pred_std
        # columns alongside all fragment structural columns.
        ensemble_pred = EnsemblePredictor(predictors)
        ensemble_df = ensemble_pred.predict(extra_smiles_list, drop_duplicates=False)

        if ensemble_df.empty:
            logger.error(
                "Ensemble: EnsemblePredictor returned empty DataFrame — skipping."
            )
            return

        ensemble_df = self._join_ground_truth(ensemble_df)

        # ── Persist CSV ────────────────────────────────────────────────────
        csv_path = os.path.join(self.run_dir, "ensemble_predictions.csv")
        ensemble_df.to_csv(csv_path, index=False)
        logger.info("Ensemble predictions saved → %s", csv_path)

        # ── Parity plot ────────────────────────────────────────────────────
        plot_results = self._build_plot_results(ensemble_df)
        if plot_results.get("extra"):
            parity_path = os.path.join(self.run_dir, "parity_ensemble.png")
            plot_parity_ensemble(
                results=plot_results,
                task_names=self.cfg.data.target_columns,
                output_path=parity_path,
            )
        else:
            logger.info("Ensemble: no ground-truth labels available for parity plot.")

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _build_predictor(self, model_path: str) -> Predictor:
        """Instantiate a ``Predictor`` from a saved fold checkpoint.

        Args:
            model_path: Path to the ``.pt`` model checkpoint.

        Returns:
            A ``Predictor`` ready for ``.predict()`` calls.
        """
        return Predictor(
            model_path=model_path,
            vocab_path=self.vocab_path,
            featurizer_type=self.cfg.data.featurizer_type,
            atom_features=self.cfg.model.atom_features,
            num_messages=self.cfg.model.num_messages,
            num_tasks=self.cfg.model.num_tasks,
            target_columns=self.cfg.data.target_columns,
            device=self.device,
        )

    def _evaluate_fold(
        self,
        predictor: Predictor,
        outer_test_data: SmilesData,
        fold_idx: int,
    ) -> Optional[Dict]:
        """Evaluate a fold model on its own sealed outer_test_data.

        Args:
            predictor:       Loaded predictor for this fold's model.
            outer_test_data: The fold's sealed test set (never seen during training).
            fold_idx:        Fold index used for logging.

        Returns:
            A flat dict of metrics ``{fold, task_MAE, task_RMSE, task_R2, …}``
            for constructing a summary row, or ``None`` if evaluation is skipped.
        """
        if not outer_test_data:
            logger.info(
                "Fold %d: outer_test is empty — skipping fold evaluation.", fold_idx
            )
            return None

        smiles_list = sorted({item[0] for item in outer_test_data})
        pred_df = predictor.predict(smiles_list, drop_duplicates=False)

        if pred_df.empty:
            logger.warning("Fold %d: predictor returned empty DataFrame.", fold_idx)
            return None

        # Join with ground-truth labels
        target_columns = self.cfg.data.target_columns
        gt_cols = ["molecule", "bond_index"] + target_columns
        available_gt = [c for c in gt_cols if c in self.full_df.columns]
        merged = pd.merge(
            pred_df,
            self.full_df[available_gt],
            on=["molecule", "bond_index"],
            how="inner",
        )

        if merged.empty:
            logger.warning(
                "Fold %d: merge with ground-truth yielded empty DataFrame.", fold_idx
            )
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

    # _build_ensemble_df removed: stacking + mean/std logic is now encapsulated
    # inside EnsemblePredictor.predict() (src/inference/predictor.py).

    def _join_ground_truth(self, ensemble_df: pd.DataFrame) -> pd.DataFrame:
        """Left-join ground-truth labels onto the ensemble DataFrame.

        Args:
            ensemble_df: Ensemble results DataFrame (molecule + bond_index keyed).

        Returns:
            DataFrame with ground-truth columns appended where available.
        """
        gt_cols = ["molecule", "bond_index"] + self.cfg.data.target_columns
        available_gt = [c for c in gt_cols if c in self.full_df.columns]
        if not available_gt:
            return ensemble_df
        return pd.merge(
            ensemble_df,
            self.full_df[available_gt],
            on=["molecule", "bond_index"],
            how="left",
        )

    def _build_plot_results(self, ensemble_df: pd.DataFrame) -> EnsembleResults:
        """Extract (y_true, pred_mean, pred_std) tuples for the parity plot.

        Args:
            ensemble_df: Fully joined ensemble DataFrame.

        Returns:
            ``EnsembleResults`` dict keyed by split name (``'extra'``).
        """
        plot_results: EnsembleResults = {"extra": {}}
        for task in self.cfg.data.target_columns:
            if task not in ensemble_df.columns:
                continue
            valid = ~ensemble_df[task].isna()
            y_true    = ensemble_df.loc[valid, task].values
            pred_mean = ensemble_df.loc[valid, f"{task}_pred_mean"].values
            pred_std  = ensemble_df.loc[valid, f"{task}_pred_std"].values
            if len(y_true) > 0:
                plot_results["extra"][task] = (y_true, pred_mean, pred_std)
        return plot_results


# ---------------------------------------------------------------------------
# Private module-level helper (shared metric computation)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Backward-compatible module-level shim
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
    """Module-level shim that delegates to ``EnsembleEvaluator.run()``.

    This function exists solely for backward compatibility with ``pipeline.py``
    and other callers that use the old functional API.  New code should prefer
    constructing an ``EnsembleEvaluator`` directly.

    Args:
        fold_results:      List of ``FoldResult`` from ``run_cv_loop``.
        extra_smiles_data: Processed SMILES data for the full extra dataset.
        full_df:           Complete merged DataFrame (ground-truth labels source).
        cfg:               Full configuration object.
        vocab_path:        Path to featurizer vocabulary JSON.
        device:            Inference device.
        run_dir:           Directory for all output artefacts.
    """
    evaluator = EnsembleEvaluator(
        cfg=cfg,
        vocab_path=vocab_path,
        device=device,
        run_dir=run_dir,
        full_df=full_df,
    )
    evaluator.run(fold_results, extra_smiles_data)
