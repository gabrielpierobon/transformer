from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

@dataclass
class DataConfig:
    """Configuration for data processing pipeline."""
    train_data_path: Path
    test_data_path: Path
    processed_data_path: Path
    extension: int = 61  # Sequence length
    batch_size: int = 32
    random_seed: int = 42
    validation_split: float = 0.2
    feature_columns: Optional[List[str]] = None  # If None, use all columns
    target_column: str = "V1"  # Default target column
    normalize_data: bool = True
    max_sequence_length: Optional[int] = None  # If None, use all available timesteps
    
    # Detrending configuration
    detrend_data: bool = False  # Whether to detrend series before creating subsamples
    min_points_stl: int = 12  # Minimum points required for STL decomposition
    min_points_linear: int = 5  # Minimum points required for linear regression
    force_linear_detrend: bool = False  # Force using linear detrending even if STL is possible
    dtype_precision: str = "float16"  # New parameter

    def __post_init__(self):
        self.dtype_precision = self.dtype_precision  # New attribute