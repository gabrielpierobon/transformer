# scripts/create_dataset.py

import sys
import os
from pathlib import Path
import logging
import argparse
import yaml
from typing import Dict, Any

# Add the project root directory to the Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.data.config import DataConfig
from src.data.dataset_loader import DatasetLoader
from src.data.preprocessing import TimeSeriesPreprocessor
from src.data.sequence_generator import SequenceGenerator

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

def create_dataset(config: DataConfig) -> None:
    """
    Create processed dataset from raw data.
    
    Args:
        config: Data processing configuration
    """
    try:
        logger.info("Initializing data pipeline components...")
        loader = DatasetLoader(config)
        preprocessor = TimeSeriesPreprocessor(config)
        sequence_gen = SequenceGenerator(config)

        # Load raw data
        logger.info(f"Loading raw data from {config.raw_data_path}")
        raw_data = loader.load_raw_data()
        logger.info(f"Loaded data shape: {raw_data.shape}")

        # Split data
        logger.info("Splitting data into train/validation/test sets...")
        train_data, val_data, test_data = loader.split_data(raw_data)
        logger.info(f"Train shape: {train_data.shape}")
        logger.info(f"Validation shape: {val_data.shape}")
        logger.info(f"Test shape: {test_data.shape}")

        # Process each set
        logger.info("Processing training data...")
        processed_train = preprocessor.process_batch(train_data)
        logger.info("Processing validation data...")
        processed_val = preprocessor.process_batch(val_data)
        logger.info("Processing test data...")
        processed_test = preprocessor.process_batch(test_data)

        # Generate sequences
        logger.info("Generating sequences for training...")
        X_train, y_train = sequence_gen.create_sequences(
            processed_train.values,
            sequence_length=config.extension - 1
        )
        
        logger.info("Generating sequences for validation...")
        X_val, y_val = sequence_gen.create_sequences(
            processed_val.values,
            sequence_length=config.extension - 1
        )
        
        logger.info("Generating sequences for testing...")
        X_test, y_test = sequence_gen.create_sequences(
            processed_test.values,
            sequence_length=config.extension - 1
        )

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(config.processed_data_path), exist_ok=True)

        # Save processed data
        logger.info(f"Saving processed data to {config.processed_data_path}")
        np.save(f"{config.processed_data_path}/X_train.npy", X_train)
        np.save(f"{config.processed_data_path}/y_train.npy", y_train)
        np.save(f"{config.processed_data_path}/X_val.npy", X_val)
        np.save(f"{config.processed_data_path}/y_val.npy", y_val)
        np.save(f"{config.processed_data_path}/X_test.npy", X_test)
        np.save(f"{config.processed_data_path}/y_test.npy", y_test)

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
            raw_data_path=Path(config_dict['raw_data_path']),
            processed_data_path=Path(config_dict['processed_data_path']),
            extension=config_dict.get('extension', 61),
            batch_size=config_dict.get('batch_size', 32),
            random_seed=config_dict.get('random_seed', 42),
            validation_split=config_dict.get('validation_split', 0.2),
            test_split=config_dict.get('test_split', 0.1)
        )
        
        # Create dataset
        create_dataset(config)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()