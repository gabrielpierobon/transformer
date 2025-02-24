from sklearn.preprocessing import MinMaxScaler
import numpy as np
from typing import List, Dict, Tuple

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
        # Convert to numpy and reshape for scaling
        values = series.values.reshape(-1, 1)
        
        # Scale the values
        scaled_values = self.scaler.fit_transform(values).flatten()
        
        return scaled_values
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        # Forward fill followed by backward fill
        processed_data = data.ffill().bfill()
        
        return processed_data
    
    def process_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of time series data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Processed DataFrame
        """
        # Handle missing values
        cleaned_data = self.handle_missing_values(data)
        
        # Process each series
        processed_series = {}
        for series_id in cleaned_data['V1'].unique():
            series = cleaned_data[cleaned_data['V1'] == series_id]
            processed_series[series_id] = self.preprocess_series(series)
            
        return pd.DataFrame(processed_series)