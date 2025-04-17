#!/usr/bin/env python
# scripts/create_tourism_dataset.py

import sys
import os
from pathlib import Path
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
import traceback
import gc
from typing import Dict, Any, Tuple, List, Optional
from sklearn.preprocessing import MinMaxScaler

# Add the project root directory to the Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TourismDataConfig:
    """Configuration for tourism dataset processing."""
    def __init__(
        self,
        train_data_path: str,
        test_data_path: str,
        processed_data_path: str,
        input_seq_len: int = 60,
        output_seq_len: int = 24,
        batch_size: int = 32,
        validation_split: float = 0.2,
        dtype_precision: str = 'float32'
    ):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.processed_data_path = processed_data_path
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.dtype_precision = dtype_precision

class TourismDatasetLoader:
    """Loader for tourism dataset processing."""
    def __init__(self, config: TourismDataConfig):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def _identify_columns(self, df: pd.DataFrame) -> Tuple[str, str, str]:
        """Identify the key columns in the dataset."""
        # ID column
        id_column = next((col for col in ['series_id', 'id', 'series', 'unique_id'] 
                         if col in df.columns), df.columns[0])
        
        # Date column
        date_column = next((col for col in ['date', 'time', 'timestamp', 'ds'] 
                          if col in df.columns), df.columns[1])
        
        # Value column
        value_column = next((col for col in ['value', 'y', 'target', 'values', 'data'] 
                           if col in df.columns), df.columns[-1])
        
        logger.info(f"Using columns - ID: {id_column}, Date: {date_column}, Value: {value_column}")
        return id_column, date_column, value_column

    def _create_sequences(
        self,
        values: np.ndarray,
        test_values: np.ndarray,
        series_id: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Create sequences for a single series."""
        X_sequences, y_sequences = [], []
        
        # Exclude the last output_seq_len points (24 points) from training
        train_values = values[:-self.config.output_seq_len]
        
        # Create samples with data augmentation
        for start_idx in range(len(train_values) - self.config.input_seq_len):
            subsequence = train_values[start_idx:start_idx + self.config.input_seq_len]
            target_idx = start_idx + self.config.input_seq_len
            
            # Skip if we don't have enough values for the target
            if target_idx >= len(train_values):
                continue
            
            target_value = train_values[target_idx]  # Single target value
            
            # Create padded versions with proper masking
            for i in range(1, len(subsequence) + 1):
                padded_subsequence = np.zeros(self.config.input_seq_len)
                padded_subsequence[-i:] = subsequence[-i:]
                X_sequences.append(padded_subsequence)
                y_sequences.append(target_value)
        
        return X_sequences, y_sequences

    def load_data(
        self,
        log_transform: bool = False,
        random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and process tourism data."""
        np.random.seed(random_seed)
        
        # Load data
        train_df = pd.read_csv(self.config.train_data_path)
        test_df = pd.read_csv(self.config.test_data_path)
        
        # Identify columns
        id_column, date_column, value_column = self._identify_columns(train_df)
        series_ids = train_df[id_column].unique()
        
        logger.info(f"Processing {len(series_ids)} series from tourism dataset")
        
        # Process each series
        X_all, y_all = [], []
        
        for series_id in series_ids:
            # Get series data
            series_train = train_df[train_df[id_column] == series_id].sort_values(date_column)
            series_test = test_df[test_df[id_column] == series_id].sort_values(date_column)
            
            # Extract and reshape values
            values = series_train[value_column].values.reshape(-1, 1)
            test_values = series_test[value_column].values.reshape(-1, 1)
            
            # Apply log transform if requested
            if log_transform:
                values = np.log1p(values)
                test_values = np.log1p(test_values)
            
            # Scale the values
            values = self.scaler.fit_transform(values).flatten()
            test_values = self.scaler.transform(test_values).flatten()
            
            logger.info(f"Series {series_id} scaled to range [{values.min():.4f}, {values.max():.4f}]")
            
            # Create sequences
            X_series, y_series = self._create_sequences(values, test_values, series_id)
            
            X_all.extend(X_series)
            y_all.extend(y_series)
        
        # Convert to arrays
        X_all = np.array(X_all, dtype=self.config.dtype_precision)
        y_all = np.array(y_all, dtype=self.config.dtype_precision)
        
        # Random shuffle and split into train/validation
        indices = np.arange(len(X_all))
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * (1 - self.config.validation_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Split the data
        X_train = X_all[train_indices]
        y_train = y_all[train_indices]
        X_val = X_all[val_indices]
        y_val = y_all[val_indices]
        
        # Reshape to add channel dimension for X, keep y as single values
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        # Reshape y to be (n_samples, 1)
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        
        logger.info(f"Created dataset with validation split {self.config.validation_split}:")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        
        return X_train, y_train, X_val, y_val

def load_config(config_path: str) -> TourismDataConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return TourismDataConfig(
        train_data_path=config['tourism_train_data_path'],
        test_data_path=config['tourism_test_data_path'],
        processed_data_path=config['processed_data_path'],
        input_seq_len=config.get('input_seq_len', 60),
        output_seq_len=config.get('output_seq_len', 24),
        batch_size=config.get('batch_size', 32),
        validation_split=config.get('validation_split', 0.2),
        dtype_precision=config.get('dtype', {}).get('precision', 'float32')
    )

def create_tourism_dataset(config: TourismDataConfig, log_transform: bool = False, random_seed: int = 42) -> None:
    """Create processed dataset from tourism data."""
    try:
        logger.info("Initializing tourism data pipeline components...")
        loader = TourismDatasetLoader(config)
        
        # Load and process data
        logger.info(f"Loading and processing tourism data with random seed: {random_seed}...")
        X_train, y_train, X_val, y_val = loader.load_data(
            log_transform=log_transform,
            random_seed=random_seed
        )
        
        # Create output directory
        output_dir = Path(config.processed_data_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define suffix
        suffix = "_tourism"
        if log_transform:
            suffix += "_log"
        
        # Log memory usage
        logger.info(f"Memory usage - X_train: {X_train.nbytes / (1024**2):.2f} MB")
        logger.info(f"Memory usage - y_train: {y_train.nbytes / (1024**2):.2f} MB")
        logger.info(f"Memory usage - X_val: {X_val.nbytes / (1024**2):.2f} MB")
        logger.info(f"Memory usage - y_val: {y_val.nbytes / (1024**2):.2f} MB")
        
        # Save data with memory management
        try:
            for name, data in [
                ("X_train", X_train),
                ("y_train", y_train),
                ("X_val", X_val),
                ("y_val", y_val)
            ]:
                logger.info(f"Saving {name}{suffix}.npy (shape: {data.shape})...")
                np.save(output_dir / f"{name}{suffix}.npy", data)
                logger.info(f"Successfully saved {name}{suffix}.npy")
                del data
            gc.collect()
            
        except Exception as save_error:
            logger.error(f"Error during save: {str(save_error)}")
            logger.error(traceback.format_exc())
            raise

        logger.info(f"Tourism dataset creation completed successfully!")
        logger.info(f"To use this dataset for finetuning, run:")
        logger.info(f"python scripts/finetune.py --pretrained-model <MODEL_NAME> --dataset-suffix {suffix}")

    except Exception as e:
        logger.error(f"Error creating tourism dataset: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create processed tourism dataset')
    parser.add_argument('--config', type=str, default='config/data_config.yaml',
                      help='Path to data configuration file')
    parser.add_argument('--log-transform', action='store_true',
                        help='Apply log transformation to the data')
    parser.add_argument('--random-seed', type=int, default=42,
                      help='Random seed for reproducible sampling')
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path
    
    config = load_config(str(config_path))
    
    # Create tourism dataset
    create_tourism_dataset(
        config=config,
        log_transform=args.log_transform,
        random_seed=args.random_seed
    )

if __name__ == '__main__':
    main() 