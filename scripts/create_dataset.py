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

def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def create_dataset(config: DataConfig, start_series: int = None, end_series: int = None) -> None:
    """
    Create processed dataset from raw data.
    
    Args:
        config: Data processing configuration
        start_series: Optional starting index for series (e.g., 1 for M1)
        end_series: Optional ending index for series (e.g., 50 for M50)
    """
    try:
        logger.info("Initializing data pipeline components...")
        loader = DatasetLoader(config)

        # Load and process data
        logger.info("Loading and processing data...")
        X_train, y_train, X_val, y_val = loader.load_data(
            start_series=start_series,
            end_series=end_series
        )

        # Create output directory if it doesn't exist
        output_dir = Path(config.processed_data_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define output paths with series range in filename if specified
        series_suffix = f"_M{start_series}_M{end_series}" if start_series and end_series else ""
        
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
    return parser.parse_args()

def main():
    """Main function."""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config_dict = load_config(Path(args.config))
        
        # Create DataConfig object
        config = DataConfig(
            train_data_path=Path(config_dict['train_data_path']),
            test_data_path=Path(config_dict['test_data_path']),
            processed_data_path=Path(config_dict['processed_data_path']),
            extension=config_dict.get('extension', 61),
            batch_size=config_dict.get('batch_size', 32),
            random_seed=config_dict.get('random_seed', 42),
            validation_split=config_dict.get('validation_split', 0.2),
            feature_columns=config_dict.get('feature_columns', None),
            target_column=config_dict.get('target_column', 'V1'),
            normalize_data=config_dict.get('normalize_data', True),
            max_sequence_length=config_dict.get('max_sequence_length', None)
        )
        
        # Create dataset
        create_dataset(
            config,
            start_series=args.start_series,
            end_series=args.end_series
        )
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()