import os
import json
import pandas as pd
from typing import Any, Dict, List, Optional

def save_training_log(
    history: List[Dict[str, Any]],
    output_dir: str,
    suffix: str = "",
) -> Optional[pd.DataFrame]:
    """Saves the training history to a CSV file.

    Args:
        history: A list of dicts representing per-epoch metrics.
        output_dir: The directory to save the log file in.
        suffix: Optional filename suffix (e.g. ``'_fold_0'``).

    Returns:
        The history as a DataFrame, or ``None`` if *history* is empty.
    """
    if not history:
        return None
    history_df = pd.DataFrame(history)
    log_path = os.path.join(output_dir, f'training_log{suffix}.csv')
    history_df.to_csv(log_path, index=False)
    print(f"Training log saved to {log_path}")
    return history_df

def save_test_metrics(metrics: Dict[str, float], output_dir: str):
    """
    Saves the final test metrics to a JSON file.

    Args:
        metrics (Dict[str, float]): A dictionary of metric names and their values.
        output_dir (str): The directory to save the metrics file in.
    """
    print("\nFinal Test Metrics:")
    for k, v in metrics.items():
        print(f"  - {k.upper()}: {v:.4f}")

    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Test metrics saved to {metrics_path}")
