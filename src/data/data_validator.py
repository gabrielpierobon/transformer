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
            
        # Check for required columns
        required_cols = ['V1']  # Add your required columns
        if not all(col in data.columns for col in required_cols):
            return False
            
        # Check for missing values
        if data.isnull().any().any():
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