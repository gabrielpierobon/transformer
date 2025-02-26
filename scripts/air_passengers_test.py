"""
Script to test the transformer model with Air Passengers dataset.

This script demonstrates how to:
1. Load and prepare the Air Passengers dataset
2. Initialize and use the transformer model
3. Generate and visualize forecasts

Usage:
    python scripts/air_passengers_test.py --model_name transformer_1.0_directml_point_M1_M48000_sampled1000
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

import pandas as pd

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


def main():
    """
    Main function to demonstrate transformer model usage with Air Passengers dataset.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test transformer model with Air Passengers dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model directory in models/final"
    )
    args = parser.parse_args()

    # Validate model exists
    validate_model_exists(args.model_name)
    
    # Determine model type from name
    is_probabilistic = "proba" in args.model_name.lower()
    logger.info(
        f"Using {'probabilistic' if is_probabilistic else 'point'} "
        f"model: {args.model_name}"
    )

    # Load and prepare data
    logger.info("Loading Air Passengers dataset")
    df = pd.read_csv(
        "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv",
    )
    df["ds"] = pd.to_datetime(df["timestamp"])  # Convert 'timestamp' to datetime format
    df["y"] = df["value"]  # Rename 'value' to 'y'
    df["unique_id"] = "series_1"  # Create a unique id for this time series
    df.set_index("ds", inplace=True)
    series_df = df[["unique_id", "y"]]

    print("\nTime Series to predict:\n", series_df)

    # Initialize Model
    model = TransformerModel(model_name=args.model_name)
    logger.info(f"Model initialized successfully")

    if is_probabilistic:
        # Make predictions with probabilistic model
        predictions = model.predict(
            series_df=series_df,
            n=36,  # forecast horizon (months)
            num_samples=1000,  # for probabilistic forecast
            low_bound_conf=25,
            high_bound_conf=75,
        )
        print("\nProbabilistic Predictions:")
        print("- Point forecast (q_0.5):")
        print(predictions[["ds", "q_0.5", "unique_id"]])
        print("\n- Confidence Intervals:")
        print(predictions[["ds", "q_0.25", "q_0.75", "unique_id"]])
    else:
        # Make predictions with point model
        predictions = model.predict(
            series_df=series_df,
            n=36,  # forecast horizon (months)
        )
        print("\nPoint Predictions:")
        print(predictions[["ds", "q_0.5", "unique_id"]])

    # Plot results
    logger.info("Plotting forecasts")
    model.plot_forecast(series_df, predictions)


if __name__ == "__main__":
    main() 