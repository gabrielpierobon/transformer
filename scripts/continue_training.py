#!/usr/bin/env python
"""
Script to continue training a saved transformer model.
This is a convenience wrapper around train.py with the --continue-from flag.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Continue training a saved transformer model')
    
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to the saved model to continue training from'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of additional epochs to train (default: 10)'
    )
    
    parser.add_argument(
        '--initial-epoch',
        type=int,
        help='Initial epoch to start from (default: auto-detect from model name or 0)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for training (default: use the same as in train.py)'
    )
    
    parser.add_argument(
        '--memory-limit',
        type=int,
        help='GPU memory limit in MB (default: use the same as in train.py)'
    )
    
    # Dataset parameters
    parser.add_argument(
        '--start-series',
        type=int,
        help='Starting series index (e.g., 1 for M1)'
    )
    
    parser.add_argument(
        '--end-series',
        type=int,
        help='Ending series index (e.g., 500 for M500)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Number of series used in sampling (if the dataset was sampled)'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        help='Random seed used for dataset creation (needed for balanced datasets)'
    )
    
    # Allow overriding model type
    parser.add_argument(
        '--probabilistic',
        action='store_true',
        help='Force probabilistic model (overrides detection from model name)'
    )
    
    # Allow overriding loss type
    parser.add_argument(
        '--loss-type',
        type=str,
        choices=['gaussian_nll', 'smape', 'hybrid', 'mse'],
        help='Force specific loss type (overrides detection from model name)'
    )
    
    # Add option to specify if the original model used gaussian_nll loss
    parser.add_argument(
        '--original-gaussian-loss',
        action='store_true',
        help='Specify that the original model was trained with gaussian_nll loss (for point models)'
    )
    
    parser.add_argument(
        '--loss-alpha',
        type=float,
        help='Weight for sMAPE in hybrid loss (1-alpha for Gaussian NLL)'
    )
    
    # Memory optimization parameters
    parser.add_argument(
        '--disable-memory-growth',
        action='store_true',
        help='Disable memory growth (can help with some DirectML issues)'
    )
    
    parser.add_argument(
        '--mixed-precision',
        action='store_true',
        help='Enable mixed precision training (can speed up training on some GPUs)'
    )
    
    parser.add_argument(
        '--aggressive-cleanup',
        action='store_true',
        help='Perform aggressive memory cleanup between epochs'
    )
    
    return parser.parse_args()

def main():
    """Main function to continue training."""
    args = parse_args()
    
    # Check if model path exists
    model_path = Path(args.model_path)
    index_file = Path(f"{args.model_path}.index")
    
    if not model_path.exists() and not index_file.exists():
        print(f"Error: Model path {args.model_path} does not exist.")
        
        # List available models
        models_dir = Path('models/final')
        if models_dir.exists():
            print("\nAvailable models:")
            for item in models_dir.iterdir():
                if item.is_dir():
                    print(f"  - {item.name} (full model)")
                elif item.name.endswith('.index'):
                    print(f"  - {item.name[:-6]} (weights-only)")
        
        sys.exit(1)
    
    # Build command to continue training
    cmd = [sys.executable, "scripts/train.py", "--continue-from", str(model_path), "--epochs", str(args.epochs)]
    
    # Add initial epoch if provided
    if args.initial_epoch is not None:
        cmd.extend(["--initial-epoch", str(args.initial_epoch)])
    
    # Add batch size if provided
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    
    # Add memory limit if provided
    if args.memory_limit is not None:
        cmd.extend(["--memory-limit", str(args.memory_limit)])
    
    # Add dataset parameters if provided
    if args.start_series is not None:
        cmd.extend(["--start-series", str(args.start_series)])
    
    if args.end_series is not None:
        cmd.extend(["--end-series", str(args.end_series)])
    
    if args.sample_size is not None:
        cmd.extend(["--sample-size", str(args.sample_size)])
    
    if args.random_seed is not None:
        cmd.extend(["--random-seed", str(args.random_seed)])
    
    # Add probabilistic flag if provided
    if args.probabilistic:
        cmd.append("--probabilistic")
    
    # Add loss type if provided
    if args.loss_type:
        cmd.extend(["--loss-type", args.loss_type])
    
    # Add original-gaussian-loss flag if provided
    if args.original_gaussian_loss:
        cmd.append("--original-gaussian-loss")
    
    # Add loss alpha if provided
    if args.loss_alpha is not None:
        cmd.extend(["--loss-alpha", str(args.loss_alpha)])
    
    # Add memory optimization flags if provided
    if args.disable_memory_growth:
        cmd.append("--disable-memory-growth")
    
    if args.mixed_precision:
        cmd.append("--mixed-precision")
    
    if args.aggressive_cleanup:
        cmd.append("--aggressive-cleanup")
    
    # Print command
    print("Running command:", " ".join(cmd))
    
    # Execute command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main() 