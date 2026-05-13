"""
Backward-compatible shim for pipeline.py.

CV evaluation (per-fold outer_test prediction + parity plot) is now handled
directly inside ``run_cv_loop`` in ``src/training/cv_runner.py``.

This module retains only the ``run_ensemble_prediction`` function signature
so that ``pipeline.py`` does not need to change yet.  It is a no-op wrapper
that can be removed once ``pipeline.py`` is updated in the new-architecture
refactor.
"""
import logging
from typing import List

import pandas as pd
import torch

from src.config.schema import MainConfig
from src.data.splitter import SmilesData
from src.training.cv_runner import FoldResult

logger = logging.getLogger(__name__)


def run_ensemble_prediction(
    fold_results: List[FoldResult],
    extra_smiles_data: SmilesData,
    full_df: pd.DataFrame,
    cfg: MainConfig,
    vocab_path: str,
    device: torch.device,
    run_dir: str,
) -> None:
    """No-op shim retained for backward compatibility with pipeline.py.

    CV evaluation is now performed inside ``run_cv_loop`` immediately after
    each fold completes. By the time this function is called, both
    ``cv_predictions.csv`` and ``cv_parity_plot.png`` already exist in
    ``run_dir``.

    This function will be removed when ``pipeline.py`` is updated as part
    of the new-architecture refactor.

    Args:
        fold_results:      List of ``FoldResult`` from ``run_cv_loop``.
        extra_smiles_data: Unused. Kept for API compatibility.
        full_df:           Unused. Kept for API compatibility.
        cfg:               Unused. Kept for API compatibility.
        vocab_path:        Unused. Kept for API compatibility.
        device:            Unused. Kept for API compatibility.
        run_dir:           Unused. Kept for API compatibility.
    """
    logger.info(
        "run_ensemble_prediction: CV evaluation already completed inside "
        "run_cv_loop — nothing to do here."
    )