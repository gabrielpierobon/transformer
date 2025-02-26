# scripts/create_dataset.py

import sys
import os
from pathlib import Path
import logging
import argparse
import yaml
import numpy as np
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

def create_dataset(config: DataConfig, start_series: int = None, end_series: int = None, sample_size: int = None) -> None:
    """
    Create processed dataset from raw data.
    
    Args:
        config: Data processing configuration
        start_series: Optional starting index for series (e.g., 1 for M1)
        end_series: Optional ending index for series (e.g., 50 for M50)
        sample_size: Optional number of series to randomly sample
    """
    try:
        logger.info("Initializing data pipeline components...")
        loader = DatasetLoader(config)

        # Load and process data
        logger.info("Loading and processing data...")
        X_train, y_train, X_val, y_val = loader.load_data(
            start_series=start_series,
            end_series=end_series,
            sample_size=sample_size
        )

        # Create output directory if it doesn't exist
        output_dir = Path(config.processed_data_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define output paths with series range in filename if specified
        series_suffix = f"_M{start_series}_M{end_series}" if start_series and end_series else ""
        
        # Add sample information to the filename if sampling was used
        if sample_size is not None:
            series_suffix += f"_sampled{sample_size}"
        
        # Save processed data
        logger.info(f"Saving processed data to {output_dir}")
        np.save(output_dir / f"X_train{series_suffix}.npy", X_train)
        np.save(output_dir / f"y_train{series_suffix}.npy", y_train)
        np.save(output_dir / f"X_val{series_suffix}.npy", X_val)
        np.save(output_dir / f"y_val{series_suffix}.npy", y_val)

        logger.info("Dataset creation completed successfully!")

    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        raise

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create processed dataset from raw data')
    parser.add_argument(
        '--config',
        type=str,
        default=str(ROOT_DIR / 'config' / 'data_config.yaml'),
        help='Path to configuration file'
    )
    parser.add_argument(
        '--start-series',
        type=int,
        help='Starting series index (e.g., 1 for M1)'
    )
    parser.add_argument(
        '--end-series',
        type=int,
        help='Ending series index (e.g., 50 for M50)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Number of series to randomly sample from the range (optional)'
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
        create_dataset(
            config,
            start_series=args.start_series,
            end_series=args.end_series,
            sample_size=args.sample_size
        )
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()