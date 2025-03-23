#!/usr/bin/env python
"""
Script to evaluate the transformer model on the full M4 test set (48,000 series).
Processes series in random order and tracks running average sMAPE.
Results are saved in evaluation/full_set directory.
"""

import argparse
import logging
import os
import sys
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
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

def setup_directories() -> None:
    """Create necessary directories for evaluation results."""
    os.makedirs("evaluation/full_set", exist_ok=True)
    os.makedirs("evaluation/full_set/plots", exist_ok=True)
    logger.info("Created evaluation directories")

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
    """
    Create a plot for a single series showing history, forecast, and ground truth.
    
    Args:
        train_values: Historical values from training data
        forecast_values: Predicted values
        actual_values: Actual values from test data
        series_id: Identifier for the series
        model_name: Name of the model used
        timestamp: Timestamp for the filename
    """
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
        'history': all_train_values  # Add historical values for plotting
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

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate transformer model on full M4 test set"
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
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Setup directories
    setup_directories()
    
    # Load data
    train_df, test_df = load_m4_data()
    
    # Initialize model
    model = TransformerModel(
        model_name=args.model_name,
        input_series_length=60
    )
    
    # Create shuffled indices for all series
    indices = list(range(len(train_df)))
    random.shuffle(indices)
    
    # Initialize metrics storage
    metrics_history = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Process all series
    start_time = time.time()
    total_series = len(indices)
    
    for count, idx in enumerate(tqdm(indices, desc="Processing series")):
        try:
            # Get series
            train_series = train_df.iloc[idx]
            test_series = test_df.iloc[idx]
            
            # Evaluate series
            metrics = evaluate_series(model, train_series, test_series)
            metrics_history.append(metrics)
            
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
            
            # Print progress after each series
            logger.info(
                f"Series {count + 1}/{total_series} ({(count + 1)/total_series*100:.1f}%) | "
                f"ID: {metrics['series_id']} | "
                f"Current sMAPE: {metrics['smape']:.2f} | "
                f"Running Avg sMAPE: {current_smape:.2f}"
            )
            
            # Save detailed results and plots every 100 series
            if (count + 1) % 100 == 0 or count == 0 or count == total_series - 1:
                elapsed_time = time.time() - start_time
                series_per_second = (count + 1) / elapsed_time
                eta = (total_series - (count + 1)) / series_per_second
                
                logger.info(
                    f"\nDetailed Progress Report:"
                    f"\n------------------------"
                    f"\nProcessed: {count + 1}/{total_series} series"
                    f"\nCompletion: {(count + 1)/total_series*100:.1f}%"
                    f"\nRunning sMAPE: {current_smape:.2f}"
                    f"\nRunning Naïve2 sMAPE: {current_naive2:.2f}"
                    f"\nSpeed: {series_per_second:.2f} series/sec"
                    f"\nETA: {eta/3600:.1f}h"
                    f"\n------------------------"
                )
                
                # Save intermediate results
                results_df = pd.DataFrame(metrics_history)
                results_df.to_csv(f"evaluation/full_set/results_{args.model_name}_{timestamp}.csv", index=False)
                
                # Plot running metrics
                plot_running_metrics(
                    metrics_history,
                    args.model_name,
                    f"evaluation/full_set/running_metrics_{args.model_name}_{timestamp}.png"
                )
        
        except Exception as e:
            logger.error(f"Error processing series at index {idx}: {str(e)}")
    
    # Calculate final metrics
    final_smape = np.nanmean([m['smape'] for m in metrics_history])
    final_naive2 = np.nanmean([m['naive2_smape'] for m in metrics_history])
    relative_performance = final_smape / final_naive2
    
    # Save final results
    final_metrics = {
        'model_name': args.model_name,
        'total_series': total_series,
        'final_smape': final_smape,
        'final_naive2_smape': final_naive2,
        'relative_performance': relative_performance,
        'timestamp': timestamp
    }
    
    with open(f"evaluation/full_set/final_metrics_{args.model_name}_{timestamp}.txt", 'w') as f:
        for key, value in final_metrics.items():
            f.write(f"{key}: {value}\n")
    
    logger.info("Evaluation completed successfully")
    logger.info(f"Final sMAPE: {final_smape:.2f}")
    logger.info(f"Final Naïve2 sMAPE: {final_naive2:.2f}")
    logger.info(f"Relative Performance: {relative_performance:.2f}")

if __name__ == "__main__":
    main()