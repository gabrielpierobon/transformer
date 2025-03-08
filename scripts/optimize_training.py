#!/usr/bin/env python
"""
Script to optimize training performance for transformer models.

This script applies several optimizations to improve training speed,
particularly for continued training sessions that might experience slowdowns.
"""

import os
import sys
import argparse
import tensorflow as tf
import gc
import psutil
import numpy as np
from pathlib import Path
import subprocess

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optimize training performance')
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to the model to optimize (e.g., models/final/transformer_1.0_directml_point_M1_M48000_sampled2001)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training (default: 16)'
    )
    
    parser.add_argument(
        '--memory-limit',
        type=int,
        default=4096,  # 4GB default
        help='GPU memory limit in MB (default: 4096)'
    )
    
    parser.add_argument(
        '--disable-memory-growth',
        action='store_true',
        help='Disable memory growth (can help with some DirectML issues)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs to train (default: 10)'
    )
    
    parser.add_argument(
        '--mixed-precision',
        action='store_true',
        help='Enable mixed precision training (can speed up training on some GPUs)'
    )
    
    return parser.parse_args()

def optimize_gpu_settings(args):
    """Apply optimized GPU settings."""
    print("\nOptimizing GPU settings...")
    
    # Clear any existing TensorFlow session
    tf.keras.backend.clear_session()
    
    # Force garbage collection
    gc.collect()
    
    # Configure GPU memory settings
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Configure memory growth if not disabled
            if not args.disable_memory_growth:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        print(f"Enabled memory growth for {gpu}")
                    except RuntimeError as e:
                        print(f"Warning: {e}")
            else:
                print("Memory growth disabled as requested")
            
            # Set memory limit based on argument
            for gpu in gpus:
                try:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=args.memory_limit)]
                    )
                    print(f"Set memory limit to {args.memory_limit}MB for {gpu}")
                except RuntimeError as e:
                    print(f"Warning: Could not set memory limit: {e}")
            
            # Print GPU information
            print("\nGPU Information:")
            print(f"Number of GPUs available: {len(gpus)}")
            print(f"GPU devices: {gpus}")
            
        except RuntimeError as e:
            print(f"\nWarning when configuring GPU: {e}")
    else:
        print("\nNo GPU devices found. Training will proceed on CPU.")
    
    # Enable mixed precision if requested
    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("\nEnabled mixed precision training (float16)")
    
    # Print memory status
    process = psutil.Process(os.getpid())
    print(f"\nCurrent memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def run_optimized_training(args):
    """Run training with optimized settings."""
    # Build command to continue training
    cmd = [
        sys.executable, 
        "scripts/train.py", 
        "--continue-from", 
        args.model_path, 
        "--epochs", 
        str(args.epochs),
        "--batch-size", 
        str(args.batch_size),
        "--memory-limit", 
        str(args.memory_limit)
    ]
    
    # Add flag to disable memory growth if specified
    if args.disable_memory_growth:
        cmd.append("--disable-memory-growth")
    
    # Add flag for mixed precision if specified
    if args.mixed_precision:
        cmd.append("--mixed-precision")
    
    # Print command
    print("\nRunning optimized training with command:")
    print(" ".join(cmd))
    
    # Execute command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)

def main():
    """Main function to optimize training."""
    args = parse_args()
    
    # Check if model path exists
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists() and not Path(f"{args.model_path}.index").exists():
            print(f"Error: Model path {args.model_path} does not exist.")
            sys.exit(1)
    
    # Apply GPU optimizations
    optimize_gpu_settings(args)
    
    # Run optimized training if model path is provided
    if args.model_path:
        run_optimized_training(args)
    else:
        print("\nNo model path provided. Applied GPU optimizations only.")
        print("To run training, provide a model path with --model-path.")

if __name__ == "__main__":
    main() 