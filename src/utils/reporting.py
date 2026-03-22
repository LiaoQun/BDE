import os
import json
import pandas as pd
from typing import List, Dict, Any

def save_training_log(history: List[Dict[str, Any]], output_dir: str) -> pd.DataFrame:
    """
    Saves the training history to a CSV file.

    Args:
        history (List[Dict[str, Any]]): A list of dictionaries, where each dict
                                        represents an epoch's metrics.
        output_dir (str): The directory to save the log file in.
    
    Returns:
        pd.DataFrame: The history converted to a DataFrame.
    """
    history_df = pd.DataFrame(history)
    log_path = os.path.join(output_dir, 'training_log.csv')
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
