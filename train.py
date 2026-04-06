"""BDE Model Training — Command-line entry point.

Usage::

    python train.py                                     # uses default config
    python train.py configs/experiments/cv_ensemble.yaml

All training logic lives in ``src/training/pipeline.py``.  This file is
intentionally minimal: it handles CLI argument parsing, run-directory
creation, logging setup, and calls ``run_training``.
"""
import logging
import os
import shutil
import sys
from datetime import datetime

from src.config import load_config, save_flattened_config
from src.config.schema import MainConfig
from src.training.pipeline import run_training


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str, log_level: int = logging.INFO) -> None:
    """Configure the root logger with a console handler and a file handler.

    Must be called **once**, before any other module emits log messages.

    Args:
        log_dir: Directory where ``training.log`` will be written.
        log_level: Log level applied to all handlers (default: ``INFO``).
    """
    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(log_level)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.FileHandler(
        os.path.join(log_dir, "training.log"), encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point.

    Reads the config path from ``sys.argv[1]``; falls back to
    ``configs/experiments/default.yaml`` when no argument is provided.
    """
    config_path: str = (
        sys.argv[1] if len(sys.argv) > 1
        else "configs/experiments/default.yaml"
    )

    cfg: MainConfig = load_config(config_path)

    # Create a unique directory for this run
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg.train.output_dir, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)

    setup_logger(run_dir)
    logger.info(f"Loading config from: {config_path}")

    # Save the config file for this run for reproducibility
    save_flattened_config(cfg, run_dir)
    logger.info(f"Saved configuration to {run_dir}")

    try:
        run_training(cfg, run_dir)
    finally:
        if os.path.exists(cfg.data.dataset_dir):
            logger.info(f"Cleaning up temporary dataset directory: {cfg.data.dataset_dir}")
            shutil.rmtree(cfg.data.dataset_dir)

if __name__ == '__main__':
    main()
