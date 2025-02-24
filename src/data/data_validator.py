from typing import Dict, List
import pandas as pd
import numpy as np

class DataValidator:
    """Validates data quality and structure."""
    
    @staticmethod
    def validate_time_series(data: pd.DataFrame) -> bool:
        """
        Validate time series data requirements.
        
        Args:
            data: Input DataFrame containing time series data
            
        Returns:
            bool: True if data meets requirements
        """
        if data.empty:
            return False
            
        # Check if we have any columns
        if len(data.columns) < 2:  # Need at least V1 and one numeric column
            return False
            
        # Check if column names follow the expected pattern
        if not all(col.startswith('V') for col in data.columns):
            return False
            
        # Check if numeric columns (all except V1) can be converted to float
        numeric_cols = data.columns[1:]  # All columns except V1
        try:
            for col in numeric_cols:
                pd.to_numeric(data[col], errors='raise')
        except:
            return False
            
        return True
    
    @staticmethod
    def check_sequence_validity(sequence: np.ndarray, min_length: int) -> bool:
        """
        Validate sequence requirements.
        
        Args:
            sequence: Input sequence
            min_length: Minimum required length
            
        Returns:
            bool: True if sequence meets requirements
        """
        if len(sequence) < min_length:
            return False
            
        if np.isnan(sequence).any():
            return False
            
        return True