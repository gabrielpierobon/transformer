# scripts/create_dataset.py

import sys
import os
from pathlib import Path
import logging
import argparse
import yaml
import numpy as np
import traceback
import gc
from typing import Dict, Any

# Add the project root directory to the Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.data.config import DataConfig
from src.data.dataset_loader import DatasetLoader

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

def create_dataset(config: DataConfig, start_series: int = None, end_series: int = None, sample_size: int = None, random_seed: int = 42) -> None:
    """
    Create processed dataset from raw data.
    
    Args:
        config: Data processing configuration
        start_series: Optional starting index for series (e.g., 1 for M1)
        end_series: Optional ending index for series (e.g., 50 for M50)
        sample_size: Optional number of series to randomly sample
        random_seed: Random seed for reproducible sampling
    """
    try:
        logger.info("Initializing data pipeline components...")
        loader = DatasetLoader(config)

        # Load and process data
        logger.info(f"Loading and processing data with random seed: {random_seed}...")
        X_train, y_train, X_val, y_val = loader.load_data(
            start_series=start_series,
            end_series=end_series,
            sample_size=sample_size,
            random_seed=random_seed
        )

        # Create output directory if it doesn't exist
        output_dir = Path(config.processed_data_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define output paths with series range in filename if specified
        series_suffix = f"_M{start_series}_M{end_series}" if start_series and end_series else ""
        
        # Add sample information to the filename if sampling was used
        if sample_size is not None:
            series_suffix += f"_sampled{sample_size}"
        
        # Log memory usage before saving
        logger.info(f"Memory usage before saving - X_train: {X_train.nbytes / (1024**2):.2f} MB, y_train: {y_train.nbytes / (1024**2):.2f} MB")
        logger.info(f"Memory usage before saving - X_val: {X_val.nbytes / (1024**2):.2f} MB, y_val: {y_val.nbytes / (1024**2):.2f} MB")
        logger.info(f"Total memory usage: {(X_train.nbytes + y_train.nbytes + X_val.nbytes + y_val.nbytes) / (1024**2):.2f} MB")
        
        # Save processed data with detailed error handling
        logger.info(f"Saving processed data to {output_dir}")
        
        try:
            logger.info(f"Saving X_train{series_suffix}.npy (shape: {X_train.shape})...")
            np.save(output_dir / f"X_train{series_suffix}.npy", X_train)
            logger.info(f"Successfully saved X_train{series_suffix}.npy")
            
            # Force garbage collection after each save
            del X_train
            gc.collect()
            
            logger.info(f"Saving y_train{series_suffix}.npy (shape: {y_train.shape})...")
            np.save(output_dir / f"y_train{series_suffix}.npy", y_train)
            logger.info(f"Successfully saved y_train{series_suffix}.npy")
            
            del y_train
            gc.collect()
            
            logger.info(f"Saving X_val{series_suffix}.npy (shape: {X_val.shape})...")
            np.save(output_dir / f"X_val{series_suffix}.npy", X_val)
            logger.info(f"Successfully saved X_val{series_suffix}.npy")
            
            del X_val
            gc.collect()
            
            logger.info(f"Saving y_val{series_suffix}.npy (shape: {y_val.shape})...")
            np.save(output_dir / f"y_val{series_suffix}.npy", y_val)
            logger.info(f"Successfully saved y_val{series_suffix}.npy")
            
            del y_val
            gc.collect()
            
        except Exception as save_error:
            logger.error(f"Error during save operation: {str(save_error)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        logger.info("Dataset creation completed successfully!")

    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create processed dataset from raw data')
    parser.add_argument('--config', type=str, default='config/data_config.yaml',
                        help='Path to data configuration file')
    parser.add_argument('--start-series', type=int,
                        help='Starting index for series (e.g., 1 for M1)')
    parser.add_argument('--end-series', type=int,
                        help='Ending index for series (e.g., 50 for M50)')
    parser.add_argument('--sample-size', type=int,
                        help='Number of series to randomly sample')
    parser.add_argument('--random-seed', type=int, required=True,
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
    
    # Create dataset
    create_dataset(
        config=config,
        start_series=args.start_series,
        end_series=args.end_series,
        sample_size=args.sample_size,
        random_seed=args.random_seed
    )

if __name__ == '__main__':
    main()