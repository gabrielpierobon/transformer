import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from .data_validator import DataValidator

class DatasetLoader:
    """Handles loading and initial processing of datasets."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.validator = DataValidator()
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from specified path.
        
        Returns:
            DataFrame containing raw data
        """
        try:
            data = pd.read_csv(self.config.raw_data_path)
            if not self.validator.validate_time_series(data):
                raise ValueError("Data validation failed")
            return data
        except Exception as e:
            raise Exception(f"Error loading raw data: {str(e)}")
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of train, validation, and test DataFrames
        """
        # Calculate split indices
        train_end = int(len(data) * (1 - self.config.validation_split - self.config.test_split))
        val_end = int(len(data) * (1 - self.config.test_split))
        
        # Split data
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return train_data, val_data, test_data