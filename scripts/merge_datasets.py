import numpy as np
import os
from pathlib import Path

def load_dataset(start_series, end_series, sample_size):
    """Load dataset for a specific range of series."""
    prefix = f"M{start_series}_M{end_series}_sampled{sample_size}"
    data_dir = "data/processed"
    
    X_train = np.load(os.path.join(data_dir, f"X_train_{prefix}.npy"))
    y_train = np.load(os.path.join(data_dir, f"y_train_{prefix}.npy"))
    X_val = np.load(os.path.join(data_dir, f"X_val_{prefix}.npy"))
    y_val = np.load(os.path.join(data_dir, f"y_val_{prefix}.npy"))
    
    return X_train, y_train, X_val, y_val

def save_arrays(arrays, names, output_dir):
    """Save arrays to disk with given names."""
    os.makedirs(output_dir, exist_ok=True)
    for arr, name in zip(arrays, names):
        np.save(os.path.join(output_dir, name), arr)

def main():
    print("Starting dataset merge process...")
    
    # Define dataset parameters
    ranges = [
        (1, 12000),
        (12001, 24000),
        (24001, 36000),
        (36001, 48000)
    ]
    sample_size = 2500
    output_dir = "data/processed/merged"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize empty arrays for first dataset
    print(f"Loading first dataset (series {ranges[0][0]} to {ranges[0][1]})...")
    X_train_merged, y_train_merged, X_val_merged, y_val_merged = load_dataset(*ranges[0], sample_size)
    
    # Save first batch
    print("Saving first batch...")
    save_arrays(
        [X_train_merged, y_train_merged, X_val_merged, y_val_merged],
        ["merged_X_train.npy", "merged_y_train.npy", "merged_X_val.npy", "merged_y_val.npy"],
        output_dir
    )
    
    # Process remaining datasets
    for start, end in ranges[1:]:
        print(f"Processing series {start} to {end}...")
        
        # Load current dataset
        X_train, y_train, X_val, y_val = load_dataset(start, end, sample_size)
        
        # Load existing merged data
        X_train_merged = np.load(os.path.join(output_dir, "merged_X_train.npy"))
        y_train_merged = np.load(os.path.join(output_dir, "merged_y_train.npy"))
        X_val_merged = np.load(os.path.join(output_dir, "merged_X_val.npy"))
        y_val_merged = np.load(os.path.join(output_dir, "merged_y_val.npy"))
        
        # Concatenate
        print("Concatenating arrays...")
        X_train_merged = np.concatenate([X_train_merged, X_train], axis=0)
        y_train_merged = np.concatenate([y_train_merged, y_train], axis=0)
        X_val_merged = np.concatenate([X_val_merged, X_val], axis=0)
        y_val_merged = np.concatenate([y_val_merged, y_val], axis=0)
        
        # Save intermediate results
        print("Saving merged arrays...")
        save_arrays(
            [X_train_merged, y_train_merged, X_val_merged, y_val_merged],
            ["merged_X_train.npy", "merged_y_train.npy", "merged_X_val.npy", "merged_y_val.npy"],
            output_dir
        )
    
    # Print final shapes
    print("\nFinal dataset shapes:")
    print(f"X_train: {X_train_merged.shape}")
    print(f"y_train: {y_train_merged.shape}")
    print(f"X_val: {X_val_merged.shape}")
    print(f"y_val: {y_val_merged.shape}")
    print("\nMerge completed successfully!")

if __name__ == "__main__":
    main() 