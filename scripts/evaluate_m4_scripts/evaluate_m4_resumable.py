#!/usr/bin/env python
"""
Script to evaluate the transformer model on the full M4 test set (48,000 series).
This version supports resuming from interruptions by tracking progress in a state file.
Processes series in random order and tracks running average sMAPE.
Results are saved in evaluation/full_set directory.
"""

import argparse
import logging
import os
import sys
import random
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.classes import TransformerModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE)."""
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denominator > 0
    if np.any(mask):
        return np.mean(np.abs(predicted[mask] - actual[mask]) / denominator[mask]) * 100
    return np.nan

def naive2_forecast(history: np.ndarray, h: int, seasonal_period: int = 12) -> np.ndarray:
    """Generate Naïve2 forecast with seasonal adjustment and drift."""
    n = len(history)
    if n <= seasonal_period:
        drift = (history[-1] - history[0]) / (n - 1) if n > 1 else 0
        return np.array([history[-1] + (i + 1) * drift for i in range(h)])
    
    forecast = np.zeros(h)
    drift = (history[-1] - history[-seasonal_period - 1]) / seasonal_period
    
    for i in range(h):
        seasonal_idx = n - seasonal_period + (i % seasonal_period)
        forecast[i] = history[seasonal_idx] + ((i + 1) // seasonal_period + (i + 1) % seasonal_period / seasonal_period) * drift
    
    return forecast

def setup_directories() -> Tuple[str, str]:
    """Create necessary directories for evaluation results and return paths."""
    eval_dir = Path("evaluation/full_set")
    state_dir = eval_dir / "state"
    
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "plots").mkdir(exist_ok=True)
    state_dir.mkdir(exist_ok=True)
    
    logger.info("Created evaluation directories")
    return str(eval_dir), str(state_dir)

def load_m4_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load M4 training and test data."""
    train_path = Path("data/raw/Monthly-train.csv")
    test_path = Path("data/raw/Monthly-test.csv")
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "M4 data files not found. Please ensure Monthly-train.csv and "
            "Monthly-test.csv are in the data/raw directory."
        )
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Loaded training data: {train_df.shape[0]} series")
    logger.info(f"Loaded test data: {test_df.shape[0]} series")
    
    return train_df, test_df

def plot_series_forecast(
    train_values: List[float],
    forecast_values: np.ndarray,
    actual_values: np.ndarray,
    series_id: str,
    model_name: str,
    timestamp: str
) -> None:
    """Create a plot for a single series showing history, forecast, and ground truth."""
    plt.figure(figsize=(12, 6))
    
    # Create x-axis points
    history_x = np.arange(1, len(train_values) + 1)
    forecast_x = np.arange(len(train_values) + 1, len(train_values) + len(forecast_values) + 1)
    
    # Plot historical data
    plt.plot(history_x, train_values, label="Historical", color="blue")
    
    # Plot forecast
    plt.plot(forecast_x, forecast_values, label="Forecast", color="red")
    
    # Plot ground truth
    actual_x = forecast_x[:len(actual_values)]
    plt.plot(actual_x, actual_values, label="Ground Truth", color="green", linestyle="--")
    
    # Add vertical line to separate history from forecast
    plt.axvline(x=len(train_values) + 0.5, color='gray', linestyle='--')
    
    # Add labels and title
    plt.title(f"Series {series_id}: History, Forecast, and Ground Truth")
    plt.xlabel("Time Period")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_path = f"evaluation/full_set/plots/{model_name}_{series_id}_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory

def evaluate_series(
    model: TransformerModel,
    train_series: pd.Series,
    test_series: pd.Series,
    forecast_horizon: int = 18
) -> Dict[str, Any]:
    """Evaluate a single series and return metrics."""
    series_id = train_series['V1']
    
    # Get all values from training series (for plotting)
    all_train_values = train_series.dropna().tolist()[1:]  # Skip series ID
    
    # Get the last 60 values from training series
    last_60_values = all_train_values[-60:] if len(all_train_values) >= 60 else all_train_values
    
    # Prepare series for prediction
    dates = pd.date_range(start="2019-01-01", periods=len(last_60_values), freq="MS")
    df = pd.DataFrame({
        "ds": dates,
        "y": last_60_values,
        "unique_id": series_id
    })
    df.set_index("ds", inplace=True)
    
    # Generate forecast
    forecast = model.predict(
        series_df=df,
        n=forecast_horizon,
        num_samples=1000,
        low_bound_conf=30,
        high_bound_conf=70,
    )
    
    # Extract predicted values and actual values
    predicted_values = forecast['q_0.5'].values
    actual_values = test_series.iloc[1:forecast_horizon+1].values
    
    # Generate Naïve2 forecast
    naive2_values = naive2_forecast(
        history=np.array(last_60_values),
        h=forecast_horizon
    )
    
    # Calculate metrics
    smape = calculate_smape(actual_values, predicted_values)
    naive2_smape = calculate_smape(actual_values, naive2_values)
    
    return {
        'series_id': series_id,
        'smape': smape,
        'naive2_smape': naive2_smape,
        'actual': actual_values,
        'predicted': predicted_values,
        'naive2': naive2_values,
        'history': all_train_values,  # Add historical values for plotting
        'processed_time': time.time()  # Add timestamp
    }

def plot_running_metrics(
    metrics_history: List[Dict],
    model_name: str,
    save_path: str
) -> None:
    """Plot running average of metrics."""
    series_count = range(1, len(metrics_history) + 1)
    running_smape = [np.nanmean([m['smape'] for m in metrics_history[:i+1]]) 
                    for i in range(len(metrics_history))]
    running_naive2 = [np.nanmean([m['naive2_smape'] for m in metrics_history[:i+1]]) 
                     for i in range(len(metrics_history))]
    
    plt.figure(figsize=(12, 6))
    plt.plot(series_count, running_smape, label='Model sMAPE')
    plt.plot(series_count, running_naive2, label='Naïve2 sMAPE')
    plt.xlabel('Number of Series Evaluated')
    plt.ylabel('Running Average sMAPE')
    plt.title(f'Running Average sMAPE - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def save_state(
    state_dir: str,
    model_name: str,
    timestamp: str,
    processed_indices: Set[int],
    metrics_history: List[Dict]
) -> None:
    """Save current evaluation state to disk."""
    state_file = Path(state_dir) / f"state_{model_name}_{timestamp}.json"
    
    # Convert numpy arrays to lists in metrics_history
    serializable_metrics = []
    for metrics in metrics_history:
        metrics_copy = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_copy[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                metrics_copy[key] = value.item()
            else:
                metrics_copy[key] = value
        serializable_metrics.append(metrics_copy)
    
    state = {
        'timestamp': timestamp,
        'processed_indices': list(processed_indices),
        'metrics_history': serializable_metrics
    }
    
    # Save to temporary file first
    temp_file = state_file.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(state, f)
    
    # Rename temporary file to actual state file
    temp_file.replace(state_file)

def find_latest_state(state_dir: str, model_name: str) -> str:
    """Find the most recent state file for the given model and return its timestamp."""
    state_dir = Path(state_dir)
    state_files = list(state_dir.glob(f"state_{model_name}_*.json"))
    
    if not state_files:
        return time.strftime("%Y%m%d_%H%M%S")
        
    # Extract timestamps and find the most recent one
    timestamps = []
    for state_file in state_files:
        # Extract timestamp from filename (state_modelname_timestamp.json)
        try:
            timestamp = state_file.stem.split('_', 2)[2]
            timestamps.append(timestamp)
        except IndexError:
            continue
    
    if timestamps:
        return max(timestamps)  # Return the most recent timestamp
    
    return time.strftime("%Y%m%d_%H%M%S")

def load_state(state_dir: str, model_name: str, timestamp: str = None) -> Tuple[Set[int], List[Dict], str]:
    """
    Load evaluation state from disk or initialize new state.
    If timestamp is None, will look for the most recent state file.
    """
    if timestamp is None:
        timestamp = find_latest_state(state_dir, model_name)
        
    state_file = Path(state_dir) / f"state_{model_name}_{timestamp}.json"
    
    if state_file.exists():
        logger.info(f"Found existing state file: {state_file}")
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            processed_indices = set(state['processed_indices'])
            metrics_history = state['metrics_history']
            
            # Convert lists back to numpy arrays
            for metrics in metrics_history:
                for key in ['actual', 'predicted', 'naive2', 'history']:
                    if key in metrics:
                        metrics[key] = np.array(metrics[key])
                        
            timestamp = state['timestamp']
            logger.info(f"Resuming from {len(processed_indices)} processed series")
            
            # Also check for and load running metrics file
            running_metrics_file = Path(state_dir).parent / f"running_metrics_{model_name}_{timestamp}.csv"
            if running_metrics_file.exists():
                logger.info(f"Found existing running metrics file with {len(pd.read_csv(running_metrics_file))} entries")
                
        except Exception as e:
            logger.error(f"Error loading state file: {e}")
            logger.info("Starting fresh evaluation")
            processed_indices = set()
            metrics_history = []
            timestamp = time.strftime("%Y%m%d_%H%M%S")
    else:
        logger.info("No existing state found, starting fresh evaluation")
        processed_indices = set()
        metrics_history = []
        
    return processed_indices, metrics_history, timestamp

def save_results(
    eval_dir: str,
    model_name: str,
    timestamp: str,
    metrics: Dict,
    mode: str = 'a'
) -> None:
    """
    Save results for a single series or append to existing results.
    
    Args:
        eval_dir: Directory to save results
        model_name: Name of the model
        timestamp: Timestamp for the file
        metrics: Metrics dictionary for a single series
        mode: File open mode ('a' for append, 'w' for write)
    """
    results_file = Path(eval_dir) / f"results_{model_name}_{timestamp}.csv"
    
    # Convert numpy values to Python scalars
    row_dict = {}
    for key, value in metrics.items():
        if key in ['series_id', 'smape', 'naive2_smape', 'processed_time']:
            if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                row_dict[key] = value.item()
            else:
                row_dict[key] = value
    
    # Create DataFrame with single row
    df = pd.DataFrame([row_dict])
    
    # Write header only if file doesn't exist or mode is 'w'
    write_header = not results_file.exists() or mode == 'w'
    
    # Save to CSV
    df.to_csv(results_file, mode=mode, header=write_header, index=False)

def save_running_metrics(
    eval_dir: str,
    model_name: str,
    timestamp: str,
    metrics_history: List[Dict],
    current_series_id: str,
    mode: str = 'a'
) -> None:
    """
    Save running metrics to CSV after processing each series.
    
    Args:
        eval_dir: Directory to save results
        model_name: Name of the model
        timestamp: Timestamp for the file
        metrics_history: List of all metrics so far
        current_series_id: ID of the series just processed
        mode: File open mode ('a' for append, 'w' for write)
    """
    running_metrics_file = Path(eval_dir) / f"running_metrics_{model_name}_{timestamp}.csv"
    
    # Calculate running metrics
    running_smape = np.nanmean([m['smape'] for m in metrics_history])
    running_naive2 = np.nanmean([m['naive2_smape'] for m in metrics_history])
    relative_performance = running_smape / running_naive2 if running_naive2 != 0 else np.nan
    
    # Create row dictionary
    row_dict = {
        'series_id': current_series_id,
        'series_count': len(metrics_history),
        'running_smape': running_smape,
        'running_naive2_smape': running_naive2,
        'running_relative_performance': relative_performance,
        'timestamp': time.time()
    }
    
    # Create DataFrame with single row
    df = pd.DataFrame([row_dict])
    
    # Write header only if file doesn't exist or mode is 'w'
    write_header = not running_metrics_file.exists() or mode == 'w'
    
    # Save to CSV
    df.to_csv(running_metrics_file, mode=mode, header=write_header, index=False)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate transformer model on full M4 test set with resume capability"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="transformer_1.0_directml_point_M1_M48000_sampled2000",
        help="Name of the model directory in models/final"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=10,
        help="Save state every N series processed"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        help="Specific timestamp to resume from (optional)"
    )
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Setup directories
    eval_dir, state_dir = setup_directories()
    
    # Load data
    train_df, test_df = load_m4_data()
    
    # Initialize model
    model = TransformerModel(
        model_name=args.model_name,
        input_series_length=60
    )
    
    # Load or initialize state (will find most recent state if timestamp not provided)
    processed_indices, metrics_history, timestamp = load_state(
        state_dir, args.model_name, args.timestamp
    )
    
    # Create shuffled indices for remaining series
    all_indices = set(range(len(train_df)))
    remaining_indices = list(all_indices - processed_indices)
    random.shuffle(remaining_indices)
    
    # Process remaining series
    start_time = time.time()
    total_series = len(train_df)
    
    try:
        for count, idx in enumerate(tqdm(remaining_indices, desc="Processing series")):
            try:
                # Get series
                train_series = train_df.iloc[idx]
                test_series = test_df.iloc[idx]
                
                # Evaluate series
                metrics = evaluate_series(model, train_series, test_series)
                metrics_history.append(metrics)
                processed_indices.add(idx)
                
                # Save individual series results
                save_results(eval_dir, args.model_name, timestamp, metrics, mode='a')
                
                # Save running metrics
                save_running_metrics(
                    eval_dir,
                    args.model_name,
                    timestamp,
                    metrics_history,
                    metrics['series_id'],
                    mode='a'
                )
                
                # Create plot for this series
                plot_series_forecast(
                    train_values=metrics['history'],
                    forecast_values=metrics['predicted'],
                    actual_values=metrics['actual'],
                    series_id=metrics['series_id'],
                    model_name=args.model_name,
                    timestamp=timestamp
                )
                
                # Calculate running average sMAPE
                current_smape = np.nanmean([m['smape'] for m in metrics_history])
                current_naive2 = np.nanmean([m['naive2_smape'] for m in metrics_history])
                
                # Print progress
                total_processed = len(processed_indices)
                logger.info(
                    f"Series {total_processed}/{total_series} "
                    f"({total_processed/total_series*100:.1f}%) | "
                    f"ID: {metrics['series_id']} | "
                    f"Current sMAPE: {metrics['smape']:.2f} | "
                    f"Running Avg sMAPE: {current_smape:.2f}"
                )
                
                # Save state periodically
                if (total_processed % args.checkpoint_frequency == 0) or (count == len(remaining_indices) - 1):
                    # Save state
                    save_state(state_dir, args.model_name, timestamp, processed_indices, metrics_history)
                    
                    # Plot running metrics
                    plot_running_metrics(
                        metrics_history,
                        args.model_name,
                        f"{eval_dir}/running_metrics_plot_{args.model_name}_{timestamp}.png"
                    )
                    
                    # Print detailed progress
                    elapsed_time = time.time() - start_time
                    series_per_second = total_processed / elapsed_time
                    remaining_series = total_series - total_processed
                    eta = remaining_series / series_per_second if series_per_second > 0 else float('inf')
                    
                    logger.info(
                        f"\nDetailed Progress Report:"
                        f"\n------------------------"
                        f"\nProcessed: {total_processed}/{total_series} series"
                        f"\nCompletion: {total_processed/total_series*100:.1f}%"
                        f"\nRunning sMAPE: {current_smape:.2f}"
                        f"\nRunning Naïve2 sMAPE: {current_naive2:.2f}"
                        f"\nSpeed: {series_per_second:.2f} series/sec"
                        f"\nETA: {eta/3600:.1f}h"
                        f"\n------------------------"
                    )
            
            except Exception as e:
                logger.error(f"Error processing series at index {idx}: {str(e)}")
                continue
    
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user. Saving current state...")
        save_state(state_dir, args.model_name, timestamp, processed_indices, metrics_history)
        logger.info("State saved. You can resume evaluation by running the script again.")
        sys.exit(0)
    
    # Calculate final metrics
    final_smape = np.nanmean([m['smape'] for m in metrics_history])
    final_naive2 = np.nanmean([m['naive2_smape'] for m in metrics_history])
    relative_performance = final_smape / final_naive2
    
    # Save final results
    final_metrics = {
        'model_name': args.model_name,
        'total_series': total_series,
        'processed_series': len(processed_indices),
        'final_smape': final_smape,
        'final_naive2_smape': final_naive2,
        'relative_performance': relative_performance,
        'timestamp': timestamp
    }
    
    with open(f"{eval_dir}/final_metrics_{args.model_name}_{timestamp}.txt", 'w') as f:
        for key, value in final_metrics.items():
            f.write(f"{key}: {value}\n")
    
    logger.info("Evaluation completed successfully")
    logger.info(f"Final sMAPE: {final_smape:.2f}")
    logger.info(f"Final Naïve2 sMAPE: {final_naive2:.2f}")
    logger.info(f"Relative Performance: {relative_performance:.2f}")

if __name__ == "__main__":
    main() 