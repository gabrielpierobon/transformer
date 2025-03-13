#!/usr/bin/env python
"""
Script to create a balanced dataset by:
1. Processing each series individually
2. Sampling a fixed number of subsequences from each series
3. Saving these subsequences to separate files
4. Optionally combining them into a single dataset
"""

import sys
import os
from pathlib import Path
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import gc
import shutil

# Add the project root directory to the Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.data.config import DataConfig
from src.data.dataset_loader import DatasetLoader
from src.data.series_detrending import SeriesDetrending

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> DataConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return DataConfig(
        train_data_path=config['train_data_path'],
        test_data_path=config['test_data_path'],
        processed_data_path=config['processed_data_path'],
        extension=config['extension'],
        batch_size=config['batch_size'],
        validation_split=config['validation_split'],
        detrend_data=config.get('detrending', {}).get('enabled', False),
        min_points_stl=config.get('detrending', {}).get('min_points_stl', 12),
        min_points_linear=config.get('detrending', {}).get('min_points_linear', 5),
        force_linear_detrend=config.get('detrending', {}).get('force_linear', False),
        dtype_precision=config.get('dtype', {}).get('precision', 'float16')
    )

def process_single_series(
    config: DataConfig,
    series_id: str,
    train_df: pd.DataFrame,
    output_dir: Path,
    subsequences_per_series: int,
    random_seed: int,
    verbose: bool = False
) -> bool:
    """
    Process a single series, sample subsequences, and save to a file.
    
    Args:
        config: Data processing configuration
        series_id: ID of the series to process
        train_df: DataFrame containing all training data
        output_dir: Directory to save the output file
        subsequences_per_series: Number of subsequences to sample from this series
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress messages
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Set random seed for reproducibility
        np.random.seed(random_seed + int(series_id[1:]))  # Use series ID as part of seed
        
        # Get series data
        series_data = train_df[train_df['V1'] == series_id]
        
        # Skip if no data
        if len(series_data) == 0:
            if verbose:
                logger.info(f"Skipping series {series_id}: No data found")
            return False
        
        # Melt the series data to long format
        time_columns = [f'V{i}' for i in range(2, train_df.shape[1] + 1)]
        series_long = series_data.melt(id_vars='V1', value_vars=time_columns, 
                                     var_name='Time', value_name='Value')
        series_long['Time'] = series_long['Time'].str[1:].astype(int)
        series_long = series_long.sort_values('Time').dropna()
        
        # Set sequence length
        sequence_length = 60  # Default sequence length
        
        # Skip if not enough data points
        if len(series_long) <= sequence_length:
            if verbose:
                logger.info(f"Skipping series {series_id}: Insufficient data points")
            return False
        
        # Set numpy dtype based on configuration
        dtype = np.float16 if config.dtype_precision == "float16" else np.float32
        
        # Get the raw values
        values = series_long['Value'].values.reshape(-1, 1).astype(np.float32)
        
        # Apply detrending if enabled
        if config.detrend_data:
            try:
                # Initialize detrending
                detrending = SeriesDetrending(
                    min_points_stl=config.min_points_stl,
                    min_points_linear=config.min_points_linear
                )
                
                series = pd.Series(values.flatten(), index=series_long.index)
                detrended_series, trend_params = detrending.remove_trend(
                    series,
                    force_linear=config.force_linear_detrend
                )
                values = detrended_series.values.reshape(-1, 1)
            except Exception as e:
                if verbose:
                    logger.warning(f"Failed to detrend series {series_id}: {str(e)}")
        
        # Scale the values
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(values).flatten()
        
        # Convert to specified dtype
        scaled_values = scaled_values.astype(dtype)
        
        # Create subsequences
        all_X = []
        all_y = []
        
        for start_idx in range(len(scaled_values) - sequence_length):
            subsequence = scaled_values[start_idx:start_idx + sequence_length]
            target = scaled_values[start_idx + sequence_length]
            
            # Create padded versions with proper masking
            for i in range(1, len(subsequence) + 1):
                # Create padded sequence with zeros
                padded_subsequence = np.zeros(sequence_length, dtype=dtype)
                # Fill from the end with actual values
                padded_subsequence[-i:] = subsequence[-i:]
                
                # Add the sequence and target
                all_X.append(padded_subsequence)
                all_y.append(target)
        
        # Check if we have enough subsequences
        if len(all_X) < subsequences_per_series:
            if verbose:
                logger.warning(f"Series {series_id} has only {len(all_X)} subsequences, " 
                              f"which is less than the required {subsequences_per_series}")
            
            # If we don't have enough, use all available with replacement
            indices = np.random.choice(len(all_X), size=subsequences_per_series, replace=True)
        else:
            # Randomly sample subsequences
            indices = np.random.choice(len(all_X), size=subsequences_per_series, replace=False)
        
        # Select the sampled subsequences
        X = np.array([all_X[i] for i in indices], dtype=dtype)
        y = np.array([all_y[i] for i in indices], dtype=dtype)
        
        # Reshape X to have a feature dimension
        X = X.reshape((X.shape[0], X.shape[1], 1))
        y = y.reshape(-1, 1)
        
        # Save to file
        output_file = output_dir / f"{series_id}.npz"
        np.savez_compressed(output_file, X=X, y=y)
        
        # Clear memory
        del X, y, all_X, all_y
        gc.collect()
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing series {series_id}: {str(e)}")
        return False

def combine_series_files(
    series_dir: Path,
    output_dir: Path,
    output_suffix: str,
    validation_split: float = 0.2,
    random_seed: int = 42,
    dtype_precision: str = 'float16',
    verbose: bool = True
) -> None:
    """
    Combine individual series files into a single dataset.
    
    Args:
        series_dir: Directory containing individual series files
        output_dir: Directory to save the combined dataset
        output_suffix: Suffix to add to the output files
        validation_split: Fraction of data to use for validation
        random_seed: Random seed for reproducibility
        dtype_precision: Data type precision ('float16' or 'float32')
        verbose: Whether to print progress messages
    """
    # Set random seed
    np.random.seed(random_seed)
    
    # Set numpy dtype
    dtype = np.float16 if dtype_precision == "float16" else np.float32
    
    # Get all series files
    series_files = list(series_dir.glob("*.npz"))
    
    if verbose:
        logger.info(f"Found {len(series_files)} series files")
    
    # Calculate total subsequences
    total_subsequences = 0
    for file in tqdm(series_files, desc="Counting subsequences"):
        with np.load(file) as data:
            total_subsequences += len(data['X'])
    
    if verbose:
        logger.info(f"Total subsequences: {total_subsequences}")
    
    # Calculate split sizes
    train_size = int(total_subsequences * (1 - validation_split))
    val_size = total_subsequences - train_size
    
    if verbose:
        logger.info(f"Training subsequences: {train_size}")
        logger.info(f"Validation subsequences: {val_size}")
    
    # Initialize arrays for training and validation
    X_train = np.zeros((train_size, 60, 1), dtype=dtype)
    y_train = np.zeros((train_size, 1), dtype=dtype)
    X_val = np.zeros((val_size, 60, 1), dtype=dtype)
    y_val = np.zeros((val_size, 1), dtype=dtype)
    
    # Combine files
    train_idx = 0
    val_idx = 0
    
    for file in tqdm(series_files, desc="Combining files"):
        with np.load(file) as data:
            X = data['X']
            y = data['y']
            
            # Determine split for this file
            file_train_size = int(len(X) * (1 - validation_split))
            file_val_size = len(X) - file_train_size
            
            # Shuffle indices
            indices = np.random.permutation(len(X))
            train_indices = indices[:file_train_size]
            val_indices = indices[file_train_size:]
            
            # Add to training set
            X_train[train_idx:train_idx + file_train_size] = X[train_indices]
            y_train[train_idx:train_idx + file_train_size] = y[train_indices]
            train_idx += file_train_size
            
            # Add to validation set
            X_val[val_idx:val_idx + file_val_size] = X[val_indices]
            y_val[val_idx:val_idx + file_val_size] = y[val_indices]
            val_idx += file_val_size
    
    # Shuffle the combined datasets
    train_indices = np.random.permutation(train_size)
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    
    val_indices = np.random.permutation(val_size)
    X_val = X_val[val_indices]
    y_val = y_val[val_indices]
    
    # Save the combined datasets
    if verbose:
        logger.info(f"Saving combined datasets with shapes: X_train {X_train.shape}, y_train {y_train.shape}, X_val {X_val.shape}, y_val {y_val.shape}")
    
    np.save(output_dir / f"X_train_{output_suffix}.npy", X_train)
    np.save(output_dir / f"y_train_{output_suffix}.npy", y_train)
    np.save(output_dir / f"X_val_{output_suffix}.npy", X_val)
    np.save(output_dir / f"y_val_{output_suffix}.npy", y_val)
    
    if verbose:
        logger.info("Combined datasets created successfully!")

def create_balanced_dataset(
    config: DataConfig,
    start_series: int = 1,
    end_series: int = 48000,
    subsequences_per_series: int = 375,
    random_seed: int = 42,
    temp_dir: str = None,
    combine: bool = True,
    verbose: bool = True
) -> None:
    """
    Create a balanced dataset by processing each series individually.
    
    Args:
        config: Data processing configuration
        start_series: Starting index for series (e.g., 1 for M1)
        end_series: Ending index for series (e.g., 48000 for M48000)
        subsequences_per_series: Number of subsequences to sample from each series
        random_seed: Random seed for reproducibility
        temp_dir: Directory to store temporary files
        combine: Whether to combine the individual files into a single dataset
        verbose: Whether to print progress messages
    """
    try:
        # Set up temporary directory
        if temp_dir is None:
            temp_dir = Path(config.processed_data_path) / "series_samples"
        else:
            temp_dir = Path(temp_dir)
        
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Load training data
        if verbose:
            logger.info("Loading training data...")
        
        train_df = pd.read_csv(
            config.train_data_path,
            quoting=1,  # QUOTE_ALL
            quotechar='"',
            sep=',',
            encoding='utf-8'
        )
        
        if verbose:
            logger.info(f"Training data shape: {train_df.shape}")
        
        # Generate series IDs
        series_ids = [f"M{i}" for i in range(start_series, end_series + 1)]
        
        # Process each series
        successful_series = 0
        
        for series_id in tqdm(series_ids, desc="Processing series"):
            success = process_single_series(
                config=config,
                series_id=series_id,
                train_df=train_df,
                output_dir=temp_dir,
                subsequences_per_series=subsequences_per_series,
                random_seed=random_seed,
                verbose=verbose
            )
            
            if success:
                successful_series += 1
        
        if verbose:
            logger.info(f"Successfully processed {successful_series} out of {len(series_ids)} series")
            logger.info(f"Total subsequences: {successful_series * subsequences_per_series}")
        
        # Combine files if requested
        if combine:
            if verbose:
                logger.info("Combining individual files into a single dataset...")
            
            # Create output suffix
            if start_series == 1 and end_series == 48000:
                output_suffix = f"balanced_sampled{successful_series * subsequences_per_series}_seed{random_seed}"
            else:
                output_suffix = f"M{start_series}_M{end_series}_balanced_sampled{successful_series * subsequences_per_series}_seed{random_seed}"
            
            combine_series_files(
                series_dir=temp_dir,
                output_dir=Path(config.processed_data_path),
                output_suffix=output_suffix,
                validation_split=config.validation_split,
                random_seed=random_seed,
                dtype_precision=config.dtype_precision,
                verbose=verbose
            )
            
            if verbose:
                logger.info(f"Dataset creation completed successfully!")
                logger.info(f"Output files saved with suffix: {output_suffix}")
        
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        raise

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create balanced dataset from all series')
    parser.add_argument(
        '--config',
        type=str,
        default=str(ROOT_DIR / 'config' / 'data_config.yaml'),
        help='Path to configuration file'
    )
    parser.add_argument(
        '--start-series',
        type=int,
        default=1,
        help='Starting series index (e.g., 1 for M1)'
    )
    parser.add_argument(
        '--end-series',
        type=int,
        default=48000,
        help='Ending series index (e.g., 48000 for M48000)'
    )
    parser.add_argument(
        '--subsequences-per-series',
        type=int,
        default=375,
        help='Number of subsequences to sample from each series'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        required=True,
        help='Random seed for reproducible sampling'
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        help='Directory to store temporary files'
    )
    parser.add_argument(
        '--no-combine',
        action='store_true',
        help='Do not combine the individual files into a single dataset'
    )
    return parser.parse_args()

def main():
    """Main function."""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Log detrending configuration
        if config.detrend_data:
            logger.info(
                "Detrending enabled with configuration: "
                f"min_points_stl={config.min_points_stl}, "
                f"min_points_linear={config.min_points_linear}, "
                f"force_linear={config.force_linear_detrend}"
            )
        else:
            logger.info("Detrending disabled - using original series")
        
        # Create dataset
        create_balanced_dataset(
            config=config,
            start_series=args.start_series,
            end_series=args.end_series,
            subsequences_per_series=args.subsequences_per_series,
            random_seed=args.random_seed,
            temp_dir=args.temp_dir,
            combine=not args.no_combine,
            verbose=True
        )
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()