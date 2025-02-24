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