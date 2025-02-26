"""
Script to test the transformer model with dummy data.

This script demonstrates how to:
1. Generate synthetic time series data
2. Initialize and use the transformer model
3. Generate and visualize forecasts

example ussage:
python scripts/dummy_data_test.py --model_name transformer_1.0_directml_point_M1_M48000_sampled1000
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

from src.classes import TransformerModel, DummyDataframeInitializer

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


def generate_dummy_data(input_series_length: int = 60) -> pd.DataFrame:
    """
    Generate dummy time series data for testing.

    Args:
        input_series_length: Length of each time series.

    Returns:
        DataFrame containing synthetic time series data.
    """
    # Create synthetic time series with different patterns
    t = np.linspace(0, 4*np.pi, input_series_length)
    
    # Series 1: Sine wave with trend
    series1 = 10 + 0.5*t + 5*np.sin(t)
    
    # Series 2: Exponential growth
    series2 = 5 * np.exp(0.05*t)
    
    # Series 3: Damped oscillation
    series3 = 20 * np.exp(-0.1*t) * np.cos(t)

    # Configure dummy data generator
    array_config = {
        "sine_trend": series1.tolist(),
        "exp_growth": series2.tolist(),
        "damped_osc": series3.tolist(),
    }
    
    forecast_config = {
        "input_series_length": input_series_length
    }

    # Initialize and generate dummy data
    initializer = DummyDataframeInitializer(array_config, forecast_config)
    series_df = initializer.initialize_series_df()
    
    logger.info(f"Generated {len(array_config)} synthetic time series")
    return series_df


def main():
    """
    Main function to demonstrate transformer model usage with dummy data.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test transformer model with dummy data"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model directory in models/final"
    )
    parser.add_argument(
        "--input_length",
        type=int,
        default=60,
        help="Length of input time series (default: 60)"
    )
    args = parser.parse_args()

    # Validate model exists
    validate_model_exists(args.model_name)
    
    # Generate dummy data
    series_df = generate_dummy_data(args.input_length)
    print("\nGenerated Time Series:\n", series_df)

    # Initialize Model
    model = TransformerModel(
        model_name=args.model_name,
        input_series_length=args.input_length
    )
    logger.info(f"Model initialized successfully")

    # Generate forecasts
    predictions = model.predict(
        series_df=series_df,
        n=24,  # forecast horizon (months)
        num_samples=1000,  # for probabilistic forecast
        low_bound_conf=25,
        high_bound_conf=75,
    )

    # Print results for each series
    for unique_id in series_df["unique_id"].unique():
        series_pred = predictions[predictions["unique_id"] == unique_id]
        print(f"\nPredictions for {unique_id}:")
        print(series_pred[["ds", "q_0.5", "unique_id"]].head())
        
        # Plot individual series
        series_hist = series_df[series_df["unique_id"] == unique_id]
        model.plot_forecast(series_hist, series_pred)


if __name__ == "__main__":
    main() 