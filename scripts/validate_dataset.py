import numpy as np
import os
from pathlib import Path
import logging
import yaml
import argparse
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_sequence(sequence: np.ndarray, target: float, title: str = "Sequence Plot"):
    """Plot a single sequence with its target value."""
    plt.figure(figsize=(10, 4))
    plt.plot(sequence.flatten(), label='Sequence')
    plt.scatter(len(sequence), target, color='red', label='Target')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def validate_dataset(series_suffix: str = ""):
    """
    Validate and inspect the preprocessed dataset.
    
    Args:
        series_suffix: Optional suffix for files (e.g., '_M1_M50')
    """
    # Load config
    config_path = Path(__file__).resolve().parents[1] / 'config' / 'data_config.yaml'
    config = load_config(config_path)
    processed_dir = Path(config['processed_data_path'])
    
    # List of splits to check
    splits = ['train', 'val']
    
    for split in splits:
        # Load X and y arrays
        X_path = processed_dir / f'X_{split}{series_suffix}.npy'
        y_path = processed_dir / f'y_{split}{series_suffix}.npy'
        
        if not X_path.exists() or not y_path.exists():
            logger.error(f"Files for {split} split not found")
            continue
            
        X = np.load(X_path)
        y = np.load(y_path)
        
        # Print shapes and basic info
        logger.info(f"\n{split.upper()} split:")
        logger.info(f"X shape: {X.shape}")
        logger.info(f"y shape: {y.shape}")
        
        # Print sample statistics
        logger.info(f"\nX statistics:")
        logger.info(f"  Mean: {np.nanmean(X):.4f}")
        logger.info(f"  Std: {np.nanstd(X):.4f}")
        logger.info(f"  Min: {np.nanmin(X):.4f}")
        logger.info(f"  Max: {np.nanmax(X):.4f}")
        
        logger.info(f"\ny statistics:")
        logger.info(f"  Mean: {np.nanmean(y):.4f}")
        logger.info(f"  Std: {np.nanstd(y):.4f}")
        logger.info(f"  Min: {np.nanmin(y):.4f}")
        logger.info(f"  Max: {np.nanmax(y):.4f}")
        
        # Check padding
        zero_counts = (X == 0).sum(axis=(1, 2))  # Count zeros in each sequence
        logger.info(f"\nPadding statistics:")
        logger.info(f"  Average zeros per sequence: {zero_counts.mean():.2f}")
        logger.info(f"  Max zeros in a sequence: {zero_counts.max()}")
        logger.info(f"  Min zeros in a sequence: {zero_counts.min()}")
        
        # Print and plot a few sample sequences
        n_samples = min(3, len(X))
        logger.info(f"\nFirst {n_samples} sequences:")
        for i in range(n_samples):
            seq = X[i].flatten()
            non_zero_start = np.nonzero(seq)[0][0] if np.any(seq) else len(seq)
            logger.info(f"\nSequence {i+1}:")
            logger.info(f"  Padding zeros: {non_zero_start}")
            logger.info(f"  Non-zero values: {seq[non_zero_start:]}")
            logger.info(f"  Target value: {y[i][0]}")
            
            # Plot the sequence
            plot_sequence(seq, y[i][0], f"{split} - Sequence {i+1}")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate processed dataset')
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

if __name__ == '__main__':
    args = parse_args()
    series_suffix = f"_M{args.start_series}_M{args.end_series}" if args.start_series and args.end_series else ""
    validate_dataset(series_suffix) 