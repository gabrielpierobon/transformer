import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import MinMaxScaler
from .data_validator import DataValidator
from .config import DataConfig

class DatasetLoader:
    """Handles loading and initial processing of datasets."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.validator = DataValidator()
        self.sequence_length = 60  # Length of input sequences
        
    def load_data(
        self, 
        start_series: Optional[int] = None, 
        end_series: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess the training and test data with proper padding and masking.
        
        Args:
            start_series: Optional starting index for series (e.g., 1 for M1)
            end_series: Optional ending index for series (e.g., 50 for M50)
            verbose: Whether to print progress messages
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_val, y_val
        """
        if verbose:
            print("Loading training data...")
            
        # Read CSV and convert numeric columns efficiently
        train_df = pd.read_csv(
            self.config.train_data_path,
            quoting=1,  # QUOTE_ALL
            quotechar='"',
            sep=',',
            encoding='utf-8'
        )
        
        if verbose:
            print(f"Training data shape: {train_df.shape}")
        
        # Filter series if start_series and end_series are provided
        if start_series is not None and end_series is not None:
            series_ids = [f"M{i}" for i in range(start_series, end_series + 1)]
            train_df = train_df[train_df['V1'].isin(series_ids)]
            if verbose:
                print(f"Filtered to series M{start_series} through M{end_series}")
                print(f"Filtered data shape: {train_df.shape}")
        
        # Create sequences for each time series
        X_sequences = []
        y_sequences = []
        
        # Process each series
        total_series = len(train_df['V1'].unique())
        for idx, series_id in enumerate(train_df['V1'].unique(), 1):
            if verbose:
                print(f"Processing series {series_id} ({idx}/{total_series})")
            
            # Get series data
            series_data = train_df[train_df['V1'] == series_id]
            
            # Melt the series data to long format
            time_columns = [f'V{i}' for i in range(2, train_df.shape[1] + 1)]
            series_long = series_data.melt(id_vars='V1', value_vars=time_columns, 
                                         var_name='Time', value_name='Value')
            series_long['Time'] = series_long['Time'].str[1:].astype(int)
            series_long = series_long.sort_values('Time').dropna()
            
            # Skip if not enough data points
            if len(series_long) <= self.sequence_length:
                if verbose:
                    print(f"Skipping series {series_id} due to insufficient data points")
                continue
            
            # Scale the values for this series
            scaler = MinMaxScaler(feature_range=(0, 1))
            values = series_long['Value'].values.reshape(-1, 1)
            scaled_values = scaler.fit_transform(values).flatten()
            
            # Create subsequences
            for start_idx in range(len(scaled_values) - self.sequence_length):
                subsequence = scaled_values[start_idx:start_idx + self.sequence_length]
                target = scaled_values[start_idx + self.sequence_length]
                
                # Create padded versions with proper masking
                for i in range(1, len(subsequence) + 1):
                    # Create padded sequence with zeros (which will be masked)
                    padded_subsequence = np.zeros(self.sequence_length)
                    # Fill from the end with actual values
                    padded_subsequence[-i:] = subsequence[-i:]
                    
                    # Add the sequence and target
                    X_sequences.append(padded_subsequence)
                    y_sequences.append(target)
        
        if not X_sequences:
            raise ValueError("No valid sequences were created. Check the input data.")
        
        # Convert to numpy arrays with proper dtype for masking
        X = np.array(X_sequences, dtype=np.float32)
        y = np.array(y_sequences, dtype=np.float32)
        
        # Reshape X to have a feature dimension
        X = X.reshape((X.shape[0], X.shape[1], 1))
        y = y.reshape(-1, 1)
        
        # Split into train and validation
        validation_split = self.config.validation_split
        split_idx = int(len(X) * (1 - validation_split))
        
        # Shuffle before splitting to ensure random distribution of series
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]
        
        if verbose:
            print(f"\nFinal shapes:")
            print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
            print(f"Total sequences created: {len(X)}")
            print(f"Value ranges - X: [{X.min():.4f}, {X.max():.4f}], y: [{y.min():.4f}, {y.max():.4f}]")
            print(f"Padding ratio: {np.mean(X == 0):.2%}")
        
        return X_train, y_train, X_val, y_val
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare feature columns for the model."""
        # Use all columns except V1 as features
        feature_columns = [col for col in data.columns if col != 'V1']
        return data[feature_columns].values.astype(np.float32)
        
    def get_target(self, data: pd.DataFrame) -> np.ndarray:
        """Get target values (next value in the sequence)."""
        # Use V2 as target (next value after V1)
        return data['V2'].values.astype(np.float32)