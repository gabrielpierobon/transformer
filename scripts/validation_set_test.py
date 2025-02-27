"""
Script to test the transformer model with a random series from the validation set.

This script demonstrates how to:
1. Load validation data from processed files
2. Select and prepare a random series
3. Generate and compare forecasts with ground truth
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.classes import TransformerModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_model_exists(model_name: str) -> None:
    """
    Validate that the specified model exists in models/final directory.

    Args:
        model_name: Name of the model directory

    Raises:
        ValueError: If model directory doesn't exist
    """
    model_path = Path("models/final") / model_name
    if not os.path.exists(model_path):
        raise ValueError(
            f"Model directory not found: {model_path}\n"
            f"Please ensure the model exists in the models/final directory."
        )


def load_validation_data() -> tuple:
    """
    Load X and y validation data from processed files.

    Returns:
        Tuple of (X_val, y_val) arrays
    """
    data_dir = Path("data/processed")
    X_val = np.load(data_dir / "X_val_M1_M48000_sampled1000.npy")
    y_val = np.load(data_dir / "y_val_M1_M48000_sampled1000.npy")
    logger.info(f"Loaded validation data: X_val shape {X_val.shape}, y_val shape {y_val.shape}")
    return X_val, y_val


def prepare_series_df(series: np.ndarray, start_date: str = "2019-01-01") -> pd.DataFrame:
    """
    Prepare a series for the transformer model.

    Args:
        series: Input series array
        start_date: Start date for the time series

    Returns:
        DataFrame with proper format for the model
    """
    dates = pd.date_range(start=start_date, periods=len(series), freq="MS")
    df = pd.DataFrame({
        "ds": dates,
        "y": series,
        "unique_id": "validation_series"
    })
    df.set_index("ds", inplace=True)
    return df


def plot_comparison(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    ground_truth: np.ndarray,
    title: str = "Forecast vs Ground Truth"
) -> None:
    """
    Plot comparison between historical data, forecast, and ground truth.
    
    Args:
        historical_df: DataFrame with historical data
        forecast_df: DataFrame with forecast data
        ground_truth: Ground truth values
        title: Plot title
    """
    plt.figure(figsize=(12, 6))

    # Plot historical data
    plt.plot(historical_df.index, historical_df["y"], label="Historical", color="blue")

    # Plot forecast
    plt.plot(forecast_df["ds"], forecast_df["q_0.5"], label="Forecast", color="red")

    # Plot confidence interval
    plt.fill_between(
        forecast_df["ds"],
        forecast_df["q_0.30"],
        forecast_df["q_0.70"],
        color="red",
        alpha=0.2,
        label="Confidence Interval"
    )

    # Plot ground truth (only for available points)
    future_dates = pd.date_range(
        start=historical_df.index[-1],
        periods=len(ground_truth) + 1,
        freq="MS"
    )[1:]
    plt.plot(future_dates, ground_truth, label="Ground Truth", color="green", linestyle="--")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Calculate error metrics between predictions and ground truth.

    Args:
        predictions: Predicted values
        ground_truth: Ground truth values

    Returns:
        Dictionary with error metrics
    """
    # Only compare metrics for available ground truth points
    predictions = predictions[:len(ground_truth)]
    
    mse = np.mean((predictions - ground_truth) ** 2)
    mae = np.mean(np.abs(predictions - ground_truth))
    mape = np.mean(np.abs((ground_truth - predictions) / ground_truth)) * 100
    
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mae,
        "MAPE": mape
    }


def main():
    """
    Main function to demonstrate transformer model usage with validation data.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test transformer model with validation data"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model directory in models/final"
    )
    parser.add_argument(
        "--series_index",
        type=int,
        help="Index of series to test (random if not provided)"
    )
    args = parser.parse_args()

    # Validate model exists
    validate_model_exists(args.model_name)
    
    # Load validation data
    X_val, y_val = load_validation_data()
    
    # Select random series if index not provided
    if args.series_index is None:
        series_index = np.random.randint(0, len(X_val))
    else:
        series_index = args.series_index
    
    logger.info(f"Using series index: {series_index}")
    
    # Prepare input series
    input_series = X_val[series_index, :, 0]  # Take first feature
    ground_truth = y_val[series_index]
    series_df = prepare_series_df(input_series)
    
    print("\nInput Series:\n", series_df)

    # Initialize Model
    model = TransformerModel(
        model_name=args.model_name,
        input_series_length=len(input_series)
    )
    logger.info(f"Model initialized successfully")

    # Generate forecasts (always 36 points)
    predictions = model.predict(
        series_df=series_df,
        n=36,  # Fixed forecast horizon
        num_samples=1000,
        low_bound_conf=30,
        high_bound_conf=70,
    )

    # Calculate metrics (only for available ground truth points)
    metrics = calculate_metrics(
        predictions["q_0.5"].values,
        ground_truth
    )
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Date': predictions['ds'][:len(ground_truth)],
        'Predicted': predictions['q_0.5'][:len(ground_truth)].round(4),
        'Actual': ground_truth.round(4),
        'Difference': (predictions['q_0.5'][:len(ground_truth)] - ground_truth).round(4),
        'Error %': ((predictions['q_0.5'][:len(ground_truth)] - ground_truth) / ground_truth * 100).round(2)
    })
    
    print("\nPrediction vs Actual Values:")
    print(comparison_df.to_string(index=False))
    
    print("\nPrediction Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Plot results with ground truth
    plot_comparison(
        series_df,
        predictions,
        ground_truth,
        title=f"Forecast vs Ground Truth (Series {series_index})"
    )


if __name__ == "__main__":
    main() 