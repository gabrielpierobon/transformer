#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create datasets in smaller batches to avoid memory issues in Docker.
This script processes data in manageable chunks and then combines them.
"""

import os
import gc
import time
import logging
import argparse
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import psutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB

def create_dataset_docker(config, start_series, end_series, sample_size, max_batch_size, random_seed=42):
    """
    Create a dataset by processing data in smaller batches to avoid memory issues.
    
    Parameters:
    -----------
    config : dict
        Configuration parameters
    start_series : int
        Starting series number
    end_series : int
        Ending series number
    sample_size : int
        Total number of samples to include in the dataset
    max_batch_size : int
        Maximum number of samples to process in a single batch
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    bool
        True if dataset creation was successful, False otherwise
    """
    try:
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Calculate number of batches needed
        num_batches = (sample_size + max_batch_size - 1) // max_batch_size
        logger.info(f"Processing {sample_size} samples in {num_batches} batches of max {max_batch_size} samples each")
        
        # Create temporary directory for batch files
        temp_dir = Path(config['processed_data_path']) / "temp_batches"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Track processed samples
        total_processed = 0
        batch_files = []
        
        # Process data in batches
        for batch_idx in range(num_batches):
            batch_start_time = time.time()
            
            # Calculate batch size (last batch may be smaller)
            remaining = sample_size - total_processed
            current_batch_size = min(max_batch_size, remaining)
            
            logger.info(f"Processing batch {batch_idx+1}/{num_batches} with {current_batch_size} samples")
            logger.info(f"Memory usage before batch: {get_memory_usage():.2f} MB")
            
            # Calculate series range for this batch
            batch_series_range = end_series - start_series + 1
            
            # Randomly select series for this batch
            selected_series = np.random.randint(start_series, end_series + 1, size=current_batch_size)
            
            # Process the selected series for this batch
            try:
                # Initialize arrays for this batch
                X_batch = []
                y_batch = []
                
                # Process each series in this batch
                for i, series_id in enumerate(selected_series):
                    if i % 100 == 0:
                        logger.info(f"  Batch {batch_idx+1}: Processing series {i+1}/{current_batch_size}")
                    
                    # Load the raw data file
                    file_path = Path(config['raw_data_path']) / f"{series_id}.csv"
                    if not file_path.exists():
                        logger.warning(f"File not found: {file_path}")
                        continue
                    
                    # Read and process the data
                    df = pd.read_csv(file_path)
                    
                    # Extract features and target
                    X = df[config['feature_columns']].values
                    y = df[config['target_column']].values
                    
                    # Add to batch arrays
                    X_batch.append(X)
                    y_batch.append(y)
                
                # Convert lists to numpy arrays
                X_batch = np.array(X_batch)
                y_batch = np.array(y_batch)
                
                # Save this batch to a temporary file
                batch_file = temp_dir / f"batch_{batch_idx}.npz"
                logger.info(f"Saving batch {batch_idx+1} to {batch_file}")
                logger.info(f"Memory usage before saving: {get_memory_usage():.2f} MB")
                
                try:
                    np.savez_compressed(
                        batch_file,
                        X=X_batch,
                        y=y_batch
                    )
                    batch_files.append(batch_file)
                    logger.info(f"Successfully saved batch {batch_idx+1}")
                except Exception as e:
                    logger.error(f"Error saving batch {batch_idx+1}: {str(e)}")
                    logger.error(traceback.format_exc())
                
                # Update processed count
                total_processed += current_batch_size
                
                # Force garbage collection
                del X_batch, y_batch
                gc.collect()
                
                logger.info(f"Memory usage after batch: {get_memory_usage():.2f} MB")
                logger.info(f"Batch {batch_idx+1} completed in {time.time() - batch_start_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx+1}: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Combine all batches into final dataset
        logger.info("Combining all batches into final dataset...")
        logger.info(f"Memory usage before combining: {get_memory_usage():.2f} MB")
        
        # Initialize arrays for combined data
        X_combined = []
        y_combined = []
        
        # Load and combine each batch
        for batch_file in batch_files:
            try:
                logger.info(f"Loading batch file: {batch_file}")
                data = np.load(batch_file)
                
                # Append data from this batch
                X_combined.append(data['X'])
                y_combined.append(data['y'])
                
                # Close the file and force garbage collection
                data.close()
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error loading batch file {batch_file}: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Concatenate all batches
        try:
            X_final = np.concatenate(X_combined, axis=0)
            y_final = np.concatenate(y_combined, axis=0)
            
            logger.info(f"Final dataset shape: X={X_final.shape}, y={y_final.shape}")
            logger.info(f"Memory usage before saving final dataset: {get_memory_usage():.2f} MB")
            
            # Save the final combined dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(config['processed_data_path']) / f"dataset_{sample_size}_{timestamp}.npz"
            
            logger.info(f"Saving final dataset to {output_file}")
            try:
                np.savez_compressed(
                    output_file,
                    X=X_final,
                    y=y_final
                )
                logger.info(f"Successfully saved final dataset to {output_file}")
            except Exception as e:
                logger.error(f"Error saving final dataset: {str(e)}")
                logger.error(traceback.format_exc())
                return False
            
        except Exception as e:
            logger.error(f"Error combining batches: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
        # Clean up temporary files
        logger.info("Cleaning up temporary batch files...")
        for batch_file in batch_files:
            try:
                os.remove(batch_file)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {batch_file}: {str(e)}")
        
        try:
            os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Could not remove temporary directory {temp_dir}: {str(e)}")
        
        logger.info("Dataset creation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in dataset creation: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to parse arguments and create the dataset."""
    parser = argparse.ArgumentParser(description='Create a dataset in batches for Docker environments')
    parser.add_argument('--start-series', type=int, default=1, help='Starting series number')
    parser.add_argument('--end-series', type=int, default=48000, help='Ending series number')
    parser.add_argument('--sample-size', type=int, default=2000, help='Total number of samples to include')
    parser.add_argument('--max-batch-size', type=int, default=500, help='Maximum samples per batch')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'raw_data_path': 'data/raw',
        'processed_data_path': 'data/processed',
        'feature_columns': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'target_column': 'Close'
    }
    
    # Log start of dataset creation
    logger.info(f"Starting dataset creation with parameters:")
    logger.info(f"  Start series: {args.start_series}")
    logger.info(f"  End series: {args.end_series}")
    logger.info(f"  Sample size: {args.sample_size}")
    logger.info(f"  Max batch size: {args.max_batch_size}")
    logger.info(f"  Random seed: {args.random_seed}")
    
    # Create dataset
    success = create_dataset_docker(
        config=config,
        start_series=args.start_series,
        end_series=args.end_series,
        sample_size=args.sample_size,
        max_batch_size=args.max_batch_size,
        random_seed=args.random_seed
    )
    
    if success:
        logger.info("Dataset creation completed successfully")
    else:
        logger.error("Dataset creation failed")
        exit(1)

if __name__ == '__main__':
    main() 