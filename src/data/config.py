from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

@dataclass
class DataConfig:
    """Configuration for data processing pipeline."""
    raw_data_path: Path
    processed_data_path: Path
    extension: int = 61  # Sequence length
    batch_size: int = 32
    random_seed: int = 42
    validation_split: float = 0.2
    test_split: float = 0.1