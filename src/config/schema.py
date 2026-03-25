"""
This file contains dataclasses for managing model, training, and data configurations.
"""
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    data_paths: List[str] = field(default_factory=lambda: ['examples/test_data.csv.gz'])
    vocab_path: str = 'etc/preprocessor.json'
    dataset_dir: str = 'temp_dataset'
    target_columns: List[str] = field(default_factory=lambda: ['bde'])
    test_size: float = 0.1
    val_size: float = 0.1 # Percentage of the original data, not the training set
    random_seed: int = 42
    sample_percentage: float = 1.0
    featurizer_type: Optional[str] = None  # Options: 'TokenFeaturizer', 'ChemPropFeaturizer'

@dataclass
class ModelConfig:
    """Configuration for the BDEModel."""
    atom_features: int = 128
    num_messages: int = 6
    num_tasks: int = field(init=False, default=0)  # derived from target_columns; do not set manually

@dataclass
class TrainConfig:
    """Configuration for the training process."""
    # device: str = 'cuda'
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 64
    model_save_path: str = 'bde_model.pt'
    output_dir: str = 'training_runs'
    early_stopping_patience: int = 10
    
@dataclass
class MainConfig:
    """Main configuration holding all sub-configurations."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        """Enforce cross-field invariants after initialisation.

        Derives ``model.num_tasks`` from ``len(data.target_columns)``.
        """
        self.model.num_tasks = len(self.data.target_columns)
