import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import MinMaxScaler
from .data_validator import DataValidator
from .config import DataConfig
from .series_detrending import SeriesDetrending
import logging

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Handles loading and initial processing of datasets."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.validator = DataValidator()
        self.sequence_length = 60  # Length of input sequences
        # Set numpy dtype based on configuration
        self.dtype = np.float16 if config.dtype_precision == "float16" else np.float32
        if config.detrend_data:
            self.detrending = SeriesDetrending(
                min_points_stl=config.min_points_stl,
                min_points_linear=config.min_points_linear
            )
        
    def process_series(
        self,
        series_long: pd.DataFrame,
        series_id: str,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single time series, including optional detrending.
        
        Args:
            series_long: DataFrame containing the time series in long format
            series_id: Identifier for the series
            verbose: Whether to print progress messages
            
        Returns:
            Tuple of X sequences and y targets
        """
        X_sequences = []
        y_sequences = []
        
        # Get the raw values first as float32 to avoid overflow
        values = series_long['Value'].values.reshape(-1, 1).astype(np.float32)
        
        # Apply detrending if enabled (using float32 for calculations)
        if self.config.detrend_data:
            try:
                series = pd.Series(values.flatten(), index=series_long.index)
                detrended_series, trend_params = self.detrending.remove_trend(
                    series,
                    force_linear=self.config.force_linear_detrend
                )
                values = detrended_series.values.reshape(-1, 1)
                logger.info(
                    f"Detrended series {series_id} using {trend_params['type']} method. "
                    f"Original range: [{series.min():.4f}, {series.max():.4f}], "
                    f"Detrended range: [{detrended_series.min():.4f}, {detrended_series.max():.4f}]"
                )
            except Exception as e:
                logger.warning(f"Failed to detrend series {series_id}: {str(e)}")
                if verbose:
                    print(f"Warning: Failed to detrend series {series_id}")
        
        # Scale the values while still in float32
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(values).flatten()
        
        # Now it's safe to convert to float16 as values are between 0 and 1
        scaled_values = scaled_values.astype(self.dtype)
        
        # Create subsequences
        for start_idx in range(len(scaled_values) - self.sequence_length):
            subsequence = scaled_values[start_idx:start_idx + self.sequence_length]
            target = scaled_values[start_idx + self.sequence_length]
            
            # Create padded versions with proper masking
            for i in range(1, len(subsequence) + 1):
                # Create padded sequence with zeros
                padded_subsequence = np.zeros(self.sequence_length, dtype=self.dtype)
                # Fill from the end with actual values
                padded_subsequence[-i:] = subsequence[-i:]
                
                # Add the sequence and target
                X_sequences.append(padded_subsequence)
                y_sequences.append(target)
        
        X = np.array(X_sequences, dtype=self.dtype)
        y = np.array(y_sequences, dtype=self.dtype)
        
        if verbose:
            logger.info(f"Created sequences with dtype {X.dtype} - X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
        
    def load_data(
        self, 
        start_series: Optional[int] = None, 
        end_series: Optional[int] = None,
        sample_size: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess the training and test data with proper padding and masking.
        
        Args:
            start_series: Optional starting index for series (e.g., 1 for M1)
            end_series: Optional ending index for series (e.g., 50 for M50)
            sample_size: Optional number of series to randomly sample from the range
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
            
            # If sample_size is provided, randomly sample from the series
            if sample_size is not None and sample_size < len(series_ids):
                # Set seed for reproducibility
                random.seed(42)
                sampled_indices = random.sample(range(len(series_ids)), sample_size)
                series_ids = [series_ids[i] for i in sampled_indices]
                if verbose:
                    print(f"Randomly sampled {sample_size} series from range M{start_series} through M{end_series}")
                    print(f"First 10 sampled series: {series_ids[:10] if len(series_ids) > 10 else series_ids}")
            
            train_df = train_df[train_df['V1'].isin(series_ids)]
            if verbose:
                if sample_size is None:
                    print(f"Filtered to series M{start_series} through M{end_series}")
                print(f"Filtered data shape: {train_df.shape}")
                print(f"Number of unique series: {len(train_df['V1'].unique())}")
        
        # Process each series
        all_X = []
        all_y = []
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
            
            # Process the series
            X_sequences, y_sequences = self.process_series(series_long, series_id, verbose)
            all_X.append(X_sequences)
            all_y.append(y_sequences)
        
        if not all_X:
            raise ValueError("No valid sequences were created. Check the input data.")
        
        # Combine all sequences
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        
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