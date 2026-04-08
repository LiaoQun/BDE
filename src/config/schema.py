"""
This file contains dataclasses for managing model, training, and data configurations.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class DataConfig:
    """Configuration for data loading and processing.

    Attributes:
        base_data_paths: Paths to base datasets. These are always included in
            the training split across all CV folds.
        extra_data_paths: Paths to extra datasets. These are the subject of
            cross-validation splitting. When empty, ``base_data_paths`` data
            is used for splitting instead.
        cross_validation: Validation strategy. Supported values:
            - ``'none'``: No cross-validation. All data is used for a single
              training run. No outer test split is created.
            - ``'leave_one_out'``: sklearn LeaveOneOut on molecule level of
              extra data (outer loop). Each unique molecule is held out once.
            - ``int`` (e.g. ``5``): sklearn KFold with that many splits on
              molecule level of extra data (outer loop).
        vocab_path: Path to the featurizer vocabulary file.
        dataset_dir: Temporary directory for cached graph datasets.
        target_columns: Column names for prediction targets.
        test_size: Fraction of base data held out as a fixed test set.
            Currently reserved for future use.
        val_size: Fraction of the *inner* training data used for validation
            (early stopping).  Applied **after** the outer CV split:
            - When ``cross_validation='none'`` and ``val_size > 0``:
              splits ``all_data`` into inner_train / inner_val (Method-B).
            - When ``cross_validation='none'`` and ``val_size == 0``:
              trains on full data with no validation (Method-A).
            - When ``cross_validation=K``: splits ``broad_train`` (the
              ``K-1/K`` portion of extra_data + base_data) into
              inner_train / inner_val.
        random_seed: Global random seed for reproducibility.
        featurizer_type: Featurizer class name. Options: ``'TokenFeaturizer'``,
            ``'ChemPropFeaturizer'``.
    """

    base_data_paths: List[str]
    extra_data_paths: List[str]
    cross_validation: Union[str, int]
    dataset_dir: str
    target_columns: List[str]
    test_size: float
    val_size: float
    random_seed: int
    featurizer_type: Optional[str]

@dataclass
class ModelConfig:
    """Configuration for the BDEModel."""
    atom_features: int
    num_messages: int
    num_tasks: int = field(init=False)  # derived from target_columns; do not set manually

@dataclass
class TrainConfig:
    """Configuration for the training process."""
    epochs: int
    lr: float
    batch_size: int
    model_save_path: str
    output_dir: str
    early_stopping_patience: int
    
@dataclass
class MainConfig:
    """Main configuration holding all sub-configurations."""
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    def __post_init__(self) -> None:
        """Enforce cross-field invariants after initialisation.

        Derives ``model.num_tasks`` from ``len(data.target_columns)``.
        """
        self.model.num_tasks = len(self.data.target_columns)
