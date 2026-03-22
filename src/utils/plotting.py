"""This module contains utility functions for plotting and visualization."""
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def plot_training_curve(history_df: pd.DataFrame, output_dir: str) -> None:
    """
    Generates and saves a plot of training and validation loss curves.

    Args:
        history_df (pd.DataFrame): DataFrame containing 'epoch', 'train_loss',
                                   and 'val_loss' columns.
        output_dir (str): Directory to save the plot image.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Training Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
    plt.title('Training & Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(output_dir, 'training_curve.png')
    plt.savefig(output_path, dpi=300)
    print(f"Training curve saved to {output_path}")
    plt.close() # Close the figure to free memory


def plot_parity(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Parity Plot",
    output_path: str = None,
) -> None:
    """
    Generates and saves a parity plot with separate subplots for train,
    validation, and test sets.

    Args:
        results (Dict[str, Tuple[np.ndarray, np.ndarray]]):
            A dictionary where keys are dataset names ('train', 'validation', 'test')
            and values are tuples of (y_true, y_pred).
        title (str, optional): The suptitle for the entire figure.
        output_path (str, optional): If provided, the plot will be saved.
    """
    # Check if there's anything to plot
    if not any(name in results and results[name][0].size > 0 for name in ["train", "validation", "test"]):
        print("Warning: No data found for any dataset to generate parity plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(title, fontsize=16, y=1.02)

    dataset_names = ["train", "validation", "test"]
    
    # Find the global min/max for all data points to set axis limits consistently
    all_values_list = [arr for name in dataset_names if name in results for arr in results[name] if arr.size > 0]
    if not all_values_list:
        print("Warning: No data in specified datasets to plot.")
        plt.close(fig)
        return
        
    all_values = np.concatenate(all_values_list)
    min_val, max_val = np.min(all_values), np.max(all_values)
    buffer = (max_val - min_val) * 0.05

    for i, name in enumerate(dataset_names):
        ax = axes[i]
        
        target_name = title.split()[0] if title else "Target"
        if name not in results or results[name][0].size == 0:
            ax.set_title(f"{name.capitalize()} Set (No Data)", fontsize=14)
            ax.set_xlabel(f"Actual {target_name} (kcal/mol)", fontsize=12)
            ax.set_ylabel(f"Predicted {target_name} (kcal/mol)", fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(min_val - buffer, max_val + buffer)
            ax.set_ylim(min_val - buffer, max_val + buffer)
            continue

        y_true, y_pred = results[name]

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, color=['blue', 'green', 'red'][i])

        # Statistics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        stats_str = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, stats_str, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

        # y=x line
        ax.plot([min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer], 'k--', lw=2, label="y=x")

        ax.set_title(f"{name.capitalize()} Set", fontsize=14)
        ax.set_xlabel(f"Actual {target_name} (kcal/mol)", fontsize=12)
        ax.set_ylabel(f"Predicted {target_name} (kcal/mol)", fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(loc="lower right")
        ax.set_xlim(min_val - buffer, max_val + buffer)
        ax.set_ylim(min_val - buffer, max_val + buffer)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Parity plot saved to {output_path}")
    
    plt.close(fig)