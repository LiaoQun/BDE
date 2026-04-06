"""
Data splitting module for cross-validation strategies.

This module provides two public functions:

- ``generate_cv_splits``: A **lazy generator** that implements the *outer loop*
  of nested cross-validation.  It yields one fold at a time so that only a
  single fold's data ever lives in memory simultaneously (OOM-safe).

- ``split_inner_val``: A helper that implements the *inner loop* split within a
  single fold — separating ``broad_train`` into ``inner_train`` (fed to the
  optimiser) and ``inner_val`` (used for early stopping).

Splitting granularity is always at the **molecule level**: all bonds that
belong to the same molecule are kept together in the same split, which
prevents data leakage across splits.
"""
import logging
from dataclasses import dataclass
from typing import Generator, List, Tuple, Union

import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# Each element is a (SMILES, labels_dict) tuple produced by prepare_data().
SmilesData = List[Tuple]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _unique_smiles(data: SmilesData) -> np.ndarray:
    """Return a sorted array of unique SMILES strings found in *data*.

    Args:
        data: Processed smiles data as a list of ``(smiles, labels_dict)``.

    Returns:
        1-D NumPy array of unique SMILES strings (sorted for reproducibility).
    """
    return np.array(sorted({item[0] for item in data}))


def _keep(data: SmilesData, smiles_set: set) -> SmilesData:
    """Filter *data* to only entries whose SMILES is in *smiles_set*.

    Args:
        data: Processed smiles data.
        smiles_set: Set of SMILES strings to keep.

    Returns:
        Filtered list (shallow copy of matching elements).
    """
    return [item for item in data if item[0] in smiles_set]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_cv_splits(
    base_data: SmilesData,
    extra_data: SmilesData,
    cv_param: Union[str, int],
    random_seed: int = 42,
) -> Generator[Tuple[SmilesData, SmilesData, int], None, None]:
    """Lazy generator that yields ``(broad_train, outer_test, fold_idx)`` tuples.

    This function implements the **outer loop** of nested cross-validation.
    Each call to ``next()`` builds only one fold's data, keeping memory usage
    to the footprint of a single fold at any given time.

    Strategies
    ----------
    ``'none'``
        No cross-validation.  All data (``base_data + extra_data``) is yielded
        as ``broad_train`` with an empty ``outer_test``.  Yields exactly once
        (``fold_idx = 0``).  Whether a validation set is created from
        ``broad_train`` is controlled by the caller via ``split_inner_val``.

    ``'leave_one_out'``
        ``sklearn.model_selection.LeaveOneOut`` applied at the **molecule**
        level of ``extra_data`` (or ``base_data`` when ``extra_data`` is
        empty).  Each fold holds out exactly one unique molecule.

    ``int`` (≥ 2)
        ``sklearn.model_selection.KFold(n_splits=cv_param, shuffle=True)``
        applied at the **molecule** level.  Each fold holds out roughly
        ``1/cv_param`` of the unique extra molecules.

    In K-Fold and LOO modes, ``broad_train`` for each fold always contains
    the *full* ``base_data`` plus the portion of ``extra_data`` not in
    ``outer_test``.

    Args:
        base_data: Processed smiles data that is always included in
            ``broad_train`` (never appears in ``outer_test``).
        extra_data: Processed smiles data subject to CV splitting.  When
            empty and the strategy is K-Fold / LOO, the split falls back to
            ``base_data``.
        cv_param: Cross-validation strategy.  One of ``'none'``,
            ``'leave_one_out'``, or a positive integer ≥ 2.
        random_seed: Random seed for reproducible splits (used by K-Fold
            shuffle; LOO is deterministic).

    Yields:
        Tuple ``(broad_train, outer_test, fold_idx)`` where:

        - ``broad_train``: Data the model *may* see during training for this
          fold (inner split is the caller's responsibility).
        - ``outer_test``: Molecules held out as the fold's sealed evaluation
          set.  **Must not be seen by the model until after training.**
        - ``fold_idx``: Zero-based fold index.

    Raises:
        ValueError: For unrecognised *cv_param* or integer values < 2.
    """
    all_data: SmilesData = base_data + extra_data

    # ── Strategy: 'none' (no CV, single training run) ─────────────────────
    if cv_param == 'none':
        logger.info(
            "CV strategy: 'none' — single training run, no outer test split."
        )
        yield all_data, [], 0
        return

    # ── Determine pool of molecules to split ──────────────────────────────
    split_pool: SmilesData = extra_data if extra_data else all_data
    smiles_arr: np.ndarray = _unique_smiles(split_pool)
    n_mols = len(smiles_arr)

    # ── Strategy: 'leave_one_out' ──────────────────────────────────────────
    if cv_param == 'leave_one_out':
        logger.info(
            f"CV strategy: LeaveOneOut — {n_mols} folds "
            "(one molecule per outer_test)."
        )
        for fold_idx, (train_idx, test_idx) in enumerate(
            LeaveOneOut().split(smiles_arr)
        ):
            outer_test_smiles = set(smiles_arr[test_idx].tolist())
            broad_pool_smiles = set(smiles_arr[train_idx].tolist())

            outer_test = _keep(split_pool, outer_test_smiles)
            broad_pool  = _keep(split_pool, broad_pool_smiles)
            broad_train = (base_data + broad_pool) if extra_data else broad_pool

            logger.debug(
                f"Fold {fold_idx}: outer_test mol = "
                f"{list(outer_test_smiles)[0][:40]!r}"
            )
            yield broad_train, outer_test, fold_idx

    # ── Strategy: K-Fold (integer) ─────────────────────────────────────────
    elif isinstance(cv_param, int):
        if cv_param < 2:
            raise ValueError(
                f"K-Fold requires at least 2 splits, got {cv_param!r}."
            )
        if cv_param > n_mols:
            raise ValueError(
                f"Cannot create {cv_param} folds from only {n_mols} "
                "unique molecules.  Reduce cv_param or add more data."
            )
        logger.info(
            f"CV strategy: {cv_param}-Fold — {n_mols} molecules, "
            f"{cv_param} outer folds."
        )
        kf = KFold(n_splits=cv_param, shuffle=True, random_state=random_seed)
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(smiles_arr)):
            outer_test_smiles = set(smiles_arr[test_idx].tolist())
            broad_pool_smiles  = set(smiles_arr[train_idx].tolist())

            outer_test  = _keep(split_pool, outer_test_smiles)
            broad_pool  = _keep(split_pool, broad_pool_smiles)
            broad_train = (base_data + broad_pool) if extra_data else broad_pool

            logger.info(
                f"Fold {fold_idx}: broad_train mols ≈ {len(broad_pool_smiles)}, "
                f"outer_test mols = {len(outer_test_smiles)}"
            )
            yield broad_train, outer_test, fold_idx

    else:
        raise ValueError(
            f"Unrecognised cv_param={cv_param!r}. "
            "Expected 'none', 'leave_one_out', or an integer ≥ 2."
        )


def split_inner_val(
    broad_train: SmilesData,
    val_size: float,
    random_seed: int = 42,
) -> Tuple[SmilesData, SmilesData]:
    """Split ``broad_train`` into ``inner_train`` and ``inner_val``.

    This function implements the **inner loop** split within a single CV fold.
    ``inner_val`` is used exclusively for early stopping; it never influences
    the outer evaluation.

    Splitting is done at the **molecule level** to prevent leakage: all bonds
    of the same molecule always land in the same partition.

    Args:
        broad_train: Combined training data for the current fold (the part
            that the model is allowed to see during training).
        val_size: Fraction of unique molecules to reserve for ``inner_val``.
            Must be in the open interval ``(0, 1)``.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple ``(inner_train, inner_val)`` where both elements are lists of
        ``(smiles, labels_dict)`` tuples.

    Raises:
        ValueError: If *val_size* is not in ``(0, 1)``, or if ``broad_train``
            contains fewer than 2 unique molecules.
    """
    if not 0.0 < val_size < 1.0:
        raise ValueError(
            f"val_size must be in (0, 1), got {val_size!r}."
        )

    unique_smiles = _unique_smiles(broad_train)
    n_total = len(unique_smiles)

    if n_total < 2:
        raise ValueError(
            f"broad_train has only {n_total} unique molecule(s); "
            "cannot split into inner_train and inner_val."
        )

    n_val = max(1, int(n_total * val_size))
    rng = np.random.default_rng(random_seed)
    val_smiles = set(
        rng.choice(unique_smiles, size=n_val, replace=False).tolist()
    )
    train_smiles = set(unique_smiles.tolist()) - val_smiles

    inner_train = _keep(broad_train, train_smiles)
    inner_val   = _keep(broad_train, val_smiles)

    logger.debug(
        f"Inner split: inner_train mols = {len(train_smiles)}, "
        f"inner_val mols = {len(val_smiles)}"
    )
    return inner_train, inner_val
