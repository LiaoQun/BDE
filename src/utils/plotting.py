"""This module contains utility functions for plotting and visualization."""
import os
from typing import Dict, List, Tuple

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
    plt.close()


# ── Visual style for each split ─────────────────────────────────────────────
_SPLIT_STYLES: Dict[str, Dict] = {
    "train": {"color": "#2196F3", "alpha": 0.45, "label": "Train"},
    "test":  {"color": "#F44336", "alpha": 0.55, "label": "Test"},
}


def _compute_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Returns MAE, RMSE, and R² for a pair of arrays."""
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R²":   float(r2_score(y_true, y_pred)),
    }


def plot_parity(
    results: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    task_names: List[str],
    output_path: str = None,
) -> None:
    """
    Generates a multi-task parity plot.

    Layout
    ------
    * One **column per task** (side-by-side).
    * Each subplot overlays **train** (blue) and **test** (red) scatter points
      on the same axes so the two splits can be compared at a glance.
    * Per-split statistics (MAE / RMSE / R²) are shown in **separate,
      colour-coded text boxes** inside each subplot.

    Args:
        results: Nested dict  ``results[split][task] = (y_true, y_pred)``.
                 Recognised split keys: ``"train"``, ``"test"``.
                 Tasks not present in a split are silently skipped.
        task_names: Ordered list of task names (determines column order).
        output_path: If provided, the figure is saved to this path.

    Example::

        plot_parity(
            results={
                "train": {"bde": (y_true_tr, y_pred_tr)},
                "test":  {"bde": (y_true_te, y_pred_te)},
            },
            task_names=["bde"],
            output_path="training_runs/run_X/parity_plot_all.png",
        )
    """
    n_tasks = len(task_names)
    if n_tasks == 0:
        print("Warning: No tasks provided for parity plot.")
        return

    # ── Collect all values to set shared global axis limits ─────────────────
    all_values: List[np.ndarray] = []
    for split_dict in results.values():
        for task in task_names:
            if task in split_dict:
                y_true, y_pred = split_dict[task]
                if y_true.size > 0:
                    all_values.extend([y_true, y_pred])

    if not all_values:
        print("Warning: No data found for parity plot.")
        return

    all_concat = np.concatenate(all_values)
    global_min, global_max = float(np.min(all_concat)), float(np.max(all_concat))
    buffer = (global_max - global_min) * 0.05
    lim_lo, lim_hi = global_min - buffer, global_max + buffer

    # ── Figure setup ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, n_tasks,
        figsize=(7 * n_tasks, 6),
        squeeze=False,
    )
    fig.suptitle("Parity Plot — All Tasks", fontsize=15, y=1.01)

    # Vertical anchors for the two stats boxes (train above, test below)
    _BOX_YPOS = {"train": 0.97, "test": 0.68}

    for col_idx, task in enumerate(task_names):
        ax = axes[0][col_idx]
        has_data = False

        for split in ("train", "test"):
            split_dict = results.get(split, {})
            if task not in split_dict:
                continue
            y_true, y_pred = split_dict[task]
            if y_true.size == 0:
                continue

            style = _SPLIT_STYLES[split]
            ax.scatter(
                y_true, y_pred,
                color=style["color"],
                alpha=style["alpha"],
                s=18,
                label=style["label"],
                rasterized=True,
            )

            stats = _compute_stats(y_true, y_pred)
            stats_str = (
                f"{style['label']}\n"
                f"  MAE : {stats['MAE']:.2f}\n"
                f"  RMSE: {stats['RMSE']:.2f}\n"
                f"  R²  : {stats['R²']:.3f}"
            )
            props = dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor=style["color"],
                alpha=0.88,
            )
            ax.text(
                0.04, _BOX_YPOS[split],
                stats_str,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                color=style["color"],
                bbox=props,
            )
            has_data = True

        # ── Diagonal reference line y = x ────────────────────────────────────
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1.5, label="y = x")

        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(f"Task: {task}", fontsize=13)
        ax.set_xlabel("Actual (kcal/mol)", fontsize=11)
        ax.set_ylabel("Predicted (kcal/mol)", fontsize=11)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(loc="lower right", fontsize=9)

        if not has_data:
            ax.text(
                0.5, 0.5, "No Data",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=14, color="grey",
            )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Parity plot saved to {output_path}")

    plt.close(fig)