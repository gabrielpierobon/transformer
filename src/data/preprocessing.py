from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from .config import DataConfig

class TimeSeriesPreprocessor:
    """Handles time series preprocessing and feature engineering."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def preprocess_series(self, series: pd.Series) -> np.ndarray:
        """
        Preprocess a single time series.
        
        Args:
            series: Input time series
            
        Returns:
            Preprocessed numpy array
        """
        try:
            # Ensure numeric values
            series = pd.to_numeric(series, errors='coerce')
            
            # Handle any NaN values
            series = series.fillna(method='ffill').fillna(method='bfill')
            
            # Convert to numpy and reshape for scaling
            values = series.values.reshape(-1, 1)
            
            # Scale the values
            scaled_values = self.scaler.fit_transform(values).flatten()
            
            return scaled_values
        except Exception as e:
            raise Exception(f"Error preprocessing series: {str(e)}")
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        try:
            # Convert all columns to numeric
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Forward fill followed by backward fill
            processed_data = data.fillna(method='ffill').fillna(method='bfill')
            
            return processed_data
        except Exception as e:
            raise Exception(f"Error handling missing values: {str(e)}")
    
    def process_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of time series data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        try:
            # Handle missing values
            cleaned_data = self.handle_missing_values(data)
            
            # Process each column
            processed_data = pd.DataFrame()
            for column in cleaned_data.columns:
                processed_data[column] = self.preprocess_series(cleaned_data[column])
                
            return processed_data
        except Exception as e:
            raise Exception(f"Error processing batch: {str(e)}")