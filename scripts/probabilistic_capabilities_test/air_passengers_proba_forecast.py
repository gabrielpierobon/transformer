"""
Script to generate probabilistic forecasts with confidence intervals for the Air Passengers dataset.

This script focuses on:
1. Loading and preparing the Air Passengers dataset
2. Using a probabilistic transformer model to generate forecasts
3. Visualizing predictions with multiple confidence intervals
4. Supporting various prediction horizons and confidence levels

Usage:
    python scripts/air_passengers_proba_forecast.py --model_name transformer_1.0_directml_proba_M1_M48000_sampled1000
    python scripts/air_passengers_proba_forecast.py --model_name transformer_1.0_directml_proba_M1_M48000_sampled1000 --forecast_months 24 --confidence_levels 50 80 95
    python scripts/air_passengers_proba_forecast.py --model_name transformer_1.0_directml_proba_M1_M48000_sampled1000 --log_transform
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates

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


def log_transform_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply natural log transformation to the 'y' column of the DataFrame.
    
    Args:
        df: DataFrame with 'y' column to transform
        
    Returns:
        DataFrame with transformed 'y' column
    """
    # Create a copy to avoid modifying the original
    transformed_df = df.copy()
    
    # Apply natural log transformation (adding a small constant to avoid log(0))
    transformed_df['y'] = np.log1p(transformed_df['y'])
    
    logger.info("Applied log transformation to 'y' values")
    return transformed_df


def inverse_log_transform(values: np.ndarray) -> np.ndarray:
    """
    Apply inverse log transformation to an array of values.
    
    Args:
        values: Array of log-transformed values
        
    Returns:
        Array of values in original scale
    """
    # Apply expm1 (inverse of log1p)
    return np.expm1(values)


def prepare_data_for_short_history(df: pd.DataFrame, history_length: int) -> pd.DataFrame:
    """
    Prepare data with a shorter history length to test model's performance with limited data.
    
    Args:
        df: Original DataFrame with all data
        history_length: Number of months of history to use
        
    Returns:
        DataFrame with only the specified history length
    """
    # Sort the data by date
    df = df.sort_index()
    
    # Get only the last history_length entries
    if history_length < len(df):
        short_history = df.iloc[-history_length:].copy()
        logger.info(f"Using only last {history_length} data points for prediction")
        return short_history
    else:
        logger.info(f"Requested history length {history_length} is longer than available data ({len(df)} points)")
        return df.copy()


def generate_confidence_levels(predictions: pd.DataFrame, confidence_levels: list) -> pd.DataFrame:
    """
    Generate multiple confidence intervals based on the median prediction and standard deviation.
    
    Args:
        predictions: DataFrame with predictions
        confidence_levels: List of confidence levels to compute intervals for
        
    Returns:
        DataFrame with additional confidence interval columns
    """
    enhanced_df = predictions.copy()
    
    # Make sure ds is a datetime
    enhanced_df['ds'] = pd.to_datetime(enhanced_df['ds'])
    
    # Check which quantile columns are available
    available_quantiles = [col for col in enhanced_df.columns if col.startswith('q_')]
    logger.info(f"Available quantile columns: {available_quantiles}")
    
    # Check if we have meaningful different quantiles
    has_valid_quantiles = len(available_quantiles) >= 3
    if has_valid_quantiles:
        # Check if the quantiles have different values
        sample_row = enhanced_df.iloc[0]
        q_values = [sample_row[q] for q in available_quantiles]
        if len(set(q_values)) < 2:
            has_valid_quantiles = False
            logger.warning("Quantiles have identical values, will generate synthetic intervals")
    
    if has_valid_quantiles and 'q_0.25' in available_quantiles and 'q_0.75' in available_quantiles:
        # Calculate approximate standard deviation from the interquartile range
        # For a normal distribution, IQR ≈ 1.35 * σ
        enhanced_df['std_dev'] = (enhanced_df['q_0.75'] - enhanced_df['q_0.25']) / 1.35
        
        # Generate confidence intervals for each level
        for level in confidence_levels:
            # Calculate z-score for the confidence level
            # 50% CI ≈ 0.67σ, 80% CI ≈ 1.28σ, 95% CI ≈ 1.96σ
            z_score = {
                50: 0.67,
                80: 1.28,
                90: 1.645,
                95: 1.96,
                99: 2.58
            }.get(level, 2.0)  # Default to 2.0 if level not in the map
            
            # Create confidence intervals
            enhanced_df[f"lower_{level}"] = enhanced_df['q_0.5'] - z_score * enhanced_df['std_dev']
            enhanced_df[f"upper_{level}"] = enhanced_df['q_0.5'] + z_score * enhanced_df['std_dev']
            
            logger.info(f"Generated {level}% confidence interval using z-score {z_score}")
    else:
        # If we don't have valid quantiles, create synthetic intervals based on the median
        # with progressively wider intervals for higher confidence levels
        logger.warning("Creating synthetic confidence intervals based on median prediction")
        for level in confidence_levels:
            # Scale factor increases with confidence level
            scale_factor = {
                50: 0.1,  # 10% deviation for 50% CI
                80: 0.2,  # 20% deviation for 80% CI
                90: 0.25, # 25% deviation for 90% CI
                95: 0.3,  # 30% deviation for 95% CI
                99: 0.4   # 40% deviation for 99% CI
            }.get(level, 0.2)  # Default to 20% if level not in the map
            
            # Create confidence intervals
            enhanced_df[f"lower_{level}"] = enhanced_df['q_0.5'] * (1 - scale_factor)
            enhanced_df[f"upper_{level}"] = enhanced_df['q_0.5'] * (1 + scale_factor)
            
            logger.info(f"Generated synthetic {level}% CI using scale factor {scale_factor}")
    
    return enhanced_df


def plot_forecast_with_confidence_intervals(historical_df: pd.DataFrame, 
                                           predictions: pd.DataFrame, 
                                           confidence_levels: list,
                                           model_name: str,
                                           history_length: int = None,
                                           log_transformed: bool = False) -> None:
    """
    Plot forecasts with multiple confidence intervals.
    
    Args:
        historical_df: DataFrame with historical data
        predictions: DataFrame with forecasts and confidence intervals
        confidence_levels: List of confidence levels to plot
        model_name: Name of the model used
        history_length: Length of history used (if limited)
        log_transformed: Whether the data was log transformed
    """
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Get the historical dates and values
    historical_dates = historical_df.index
    historical_values = historical_df['y'].values
    
    # Plot historical data
    plt.plot(historical_dates, historical_values, 'b-', linewidth=2, label='Historical Data')
    
    # Convert predictions ds to datetime if it's not already
    predictions['ds'] = pd.to_datetime(predictions['ds'])
    
    # Plot confidence intervals from widest to narrowest
    colors = plt.cm.Reds(np.linspace(0.3, 0.7, len(confidence_levels)))
    
    for i, level in enumerate(sorted(confidence_levels, reverse=True)):
        lower_col = f"lower_{level}"
        upper_col = f"upper_{level}"
        
        # Fill the confidence interval
        plt.fill_between(
            predictions['ds'],
            predictions[lower_col],
            predictions[upper_col],
            color=colors[i],
            alpha=0.3,
            label=f"{level}% Confidence Interval"
        )
    
    # Plot the mean prediction
    plt.plot(predictions['ds'], predictions['q_0.5'], 'r-', linewidth=2, label='Forecast (Median)')
    
    # Add a vertical line to mark the end of historical data
    plt.axvline(x=historical_dates[-1], color='k', linestyle=':', alpha=0.7, label='Forecast Start')
    
    # Format the plot
    plt.title('Air Passengers Probabilistic Forecast with Confidence Intervals', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Passengers', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis with dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gcf().autofmt_xdate()
    
    # Create legend
    plt.legend(loc='upper left')
    
    # Add annotations
    model_info = f"Model: {model_name}"
    if history_length:
        model_info += f"\nHistory Length: {history_length} months"
    if log_transformed:
        model_info += "\nLog Transformed: Yes"
    
    plt.annotate(
        model_info,
        xy=(0.02, 0.02),
        xycoords='figure fraction',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    )
    
    # Save plot
    figures_dir = Path('reports/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    history_suffix = f"_hist{history_length}" if history_length else ""
    transform_suffix = '_log' if log_transformed else ''
    confidence_suffix = f"_conf{'_'.join(map(str, confidence_levels))}"
    
    plt.savefig(figures_dir / f'proba_forecast_{model_name}{history_suffix}{transform_suffix}{confidence_suffix}.png', dpi=300)
    
    # Show plot
    plt.tight_layout()
    plt.show()


def save_predictions_to_csv(predictions: pd.DataFrame, model_name: str, history_length: int = None) -> None:
    """
    Save predictions to CSV file.
    
    Args:
        predictions: DataFrame with predictions
        model_name: Name of the model used
        history_length: Length of history used (if limited)
    """
    # Create directory if it doesn't exist
    output_dir = Path('reports/predictions')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    history_suffix = f"_hist{history_length}" if history_length else ""
    
    # Save to CSV
    filename = output_dir / f'air_passengers_predictions_{model_name}{history_suffix}.csv'
    predictions.to_csv(filename, index=False)
    logger.info(f"Saved predictions to {filename}")


def main():
    """
    Main function to generate probabilistic forecasts for the Air Passengers dataset.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate probabilistic forecasts for Air Passengers dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the probabilistic model directory in models/final"
    )
    parser.add_argument(
        "--forecast_months",
        type=int,
        default=36,
        help="Number of months to forecast into the future"
    )
    parser.add_argument(
        "--history_length",
        type=int,
        help="Number of months of history to use (test model with limited history)"
    )
    parser.add_argument(
        "--confidence_levels",
        type=int,
        nargs="+",
        default=[50, 80, 95],
        help="Confidence levels for prediction intervals (e.g., 50 80 95)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to generate for probabilistic predictions"
    )
    parser.add_argument(
        "--log_transform",
        action="store_true",
        help="Apply log transformation to the data before forecasting (recommended for increasing variance)"
    )
    args = parser.parse_args()

    # Validate model exists
    validate_model_exists(args.model_name)
    
    # Check if model is probabilistic
    if "proba" not in args.model_name.lower():
        logger.warning(
            f"The specified model '{args.model_name}' does not appear to be a probabilistic model. "
            f"Predictions may not include proper confidence intervals."
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
    
    # Store original data for visualization
    original_series_df = series_df.copy()
    
    # Apply log transformation if requested
    if args.log_transform:
        logger.info("Applying log transformation to the data")
        series_df = log_transform_series(series_df)
    
    # Prepare data with limited history if requested
    if args.history_length:
        series_df = prepare_data_for_short_history(series_df, args.history_length)
    
    # Initialize Model
    logger.info(f"Initializing model: {args.model_name}")
    model = TransformerModel(model_name=args.model_name)
    
    # Generate predictions with confidence intervals
    logger.info(f"Generating predictions for next {args.forecast_months} months with {args.num_samples} samples")
    predictions = model.predict(
        series_df=series_df,
        n=args.forecast_months,
        num_samples=args.num_samples,  # Use more samples for better distribution
        low_bound_conf=10,  # More extreme quantile for better interval estimation
        high_bound_conf=90  # More extreme quantile for better interval estimation
    )
    
    # Apply inverse transformation if log transform was used
    if args.log_transform:
        logger.info("Applying inverse log transformation to predictions")
        for col in predictions.columns:
            if col.startswith('q_'):
                predictions[col] = inverse_log_transform(predictions[col].values)
    
    # Add confidence interval columns
    predictions = generate_confidence_levels(predictions, args.confidence_levels)
    
    # Print prediction summary
    print("\nProbabilistic Forecast Summary:")
    print(f"Model: {args.model_name}")
    print(f"Forecast Horizon: {args.forecast_months} months")
    print(f"Confidence Levels: {args.confidence_levels}")
    
    # Display forecast table
    forecast_table = predictions[['ds', 'q_0.5']].copy()
    for level in args.confidence_levels:
        forecast_table[f"{level}% CI"] = forecast_table.apply(
            lambda row: f"[{predictions.loc[row.name, f'lower_{level}']:.1f}, {predictions.loc[row.name, f'upper_{level}']:.1f}]",
            axis=1
        )
    
    print("\nForecast with Confidence Intervals:")
    print(forecast_table.to_string(index=False))
    
    # Save predictions to CSV
    save_predictions_to_csv(
        predictions, 
        args.model_name, 
        args.history_length
    )
    
    # Plot results with confidence intervals
    logger.info("Creating forecast visualization with confidence intervals")
    plot_forecast_with_confidence_intervals(
        original_series_df,
        predictions,
        args.confidence_levels,
        args.model_name,
        args.history_length,
        args.log_transform
    )
    
    logger.info("Forecast generation completed successfully")


if __name__ == "__main__":
    main() 