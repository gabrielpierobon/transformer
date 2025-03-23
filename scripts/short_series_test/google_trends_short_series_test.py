"""
Script to demonstrate model performance on short Google Trends 'iPhone' time series.

This script evaluates how well the transformer model performs with limited
historical data by testing it on the Google Trends iPhone data using only
the last N months of data to predict 12 months into the future.

Usage:
    python scripts/google_trends_short_series_test.py --model_name your_model_name
    python scripts/google_trends_short_series_test.py --model_name your_model_name --log_transform
    python scripts/google_trends_short_series_test.py --model_name your_model_name --short_series_months 3 6 12 24
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
from matplotlib.gridspec import GridSpec
from datetime import datetime

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


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """
    Calculate various error metrics between actual and predicted values.

    Args:
        actual: Series of actual values
        predicted: Series of predicted values

    Returns:
        dict: Dictionary containing various error metrics
    """
    # Remove any NaN values
    mask = ~(actual.isna() | predicted.isna())
    actual = actual[mask]
    predicted = predicted[mask]
    
    # Calculate metrics
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    
    # Calculate MAPE safely (avoiding division by zero)
    non_zero_mask = actual != 0
    if non_zero_mask.any():
        mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
    else:
        mape = np.nan
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


def get_data_subset(df: pd.DataFrame, last_n_months: int, pad_to_length: int = 60) -> pd.DataFrame:
    """
    Get a subset of the data containing only the last N months.
    If N is less than pad_to_length, pad with zeros at the beginning.
    
    Args:
        df: DataFrame with time series data
        last_n_months: Number of last months to include
        pad_to_length: Minimum length to pad to if series is shorter
        
    Returns:
        DataFrame with only the last N months of data, padded if necessary
    """
    # Sort the data by date
    df = df.sort_index()
    
    # Get only the last N months
    subset = df.iloc[-last_n_months:].copy()
    
    # If the series is shorter than pad_to_length, add padding
    if last_n_months < pad_to_length:
        logger.info(f"Series length {last_n_months} is less than {pad_to_length}, adding zero padding")
        
        # Create padding dataframe with zeros
        padding_length = pad_to_length - last_n_months
        first_date = subset.index[0]
        
        # Create dates for padding (monthly, before the start of the series)
        padding_dates = pd.date_range(
            end=first_date - pd.Timedelta(days=1),
            periods=padding_length,
            freq='M'
        )
        
        # Create padding dataframe
        padding_df = pd.DataFrame({
            'unique_id': ['iphone_trends'] * padding_length,
            'y': [0.0] * padding_length
        }, index=padding_dates)
        
        # Combine padding with subset
        padded_subset = pd.concat([padding_df, subset])
        
        logger.info(f"Padded series shape: {padded_subset.shape}")
        return padded_subset
    
    return subset


def load_google_trends_data():
    """
    Load Google Trends 'iPhone' data from a CSV file or create from the provided data.
    If the CSV doesn't exist, create it first.
    
    Returns:
        pd.DataFrame: DataFrame with the Google Trends data
    """
    # Define the data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the file path
    file_path = data_dir / "google_trends_iphone.csv"
    
    # Check if the file exists
    if not file_path.exists():
        logger.info("Creating Google Trends iPhone data file")
        
        # Hard-coded data from the query
        data = """Month,google_trends
2019-01,38
2019-02,36
2019-03,34
2019-04,32
2019-05,33
2019-06,35
2019-07,34
2019-08,35
2019-09,45
2019-10,42
2019-11,43
2019-12,43
2020-01,39
2020-02,38
2020-03,35
2020-04,38
2020-05,37
2020-06,37
2020-07,38
2020-08,37
2020-09,38
2020-10,44
2020-11,43
2020-12,44
2021-01,41
2021-02,37
2021-03,36
2021-04,35
2021-05,36
2021-06,34
2021-07,35
2021-08,36
2021-09,45
2021-10,44
2021-11,42
2021-12,43
2022-01,40
2022-02,36
2022-03,36
2022-04,34
2022-05,35
2022-06,37
2022-07,38
2022-08,37
2022-09,47
2022-10,39
2022-11,38
2022-12,40
2023-01,40
2023-02,38
2023-03,36
2023-04,38
2023-05,35
2023-06,37
2023-07,40
2023-08,39
2023-09,48
2023-10,40
2023-11,41
2023-12,41
2024-01,41
2024-02,40
2024-03,39
2024-04,38
2024-05,39
2024-06,39
2024-07,40
2024-08,41
2024-09,52
2024-10,44
2024-11,45
2024-12,48"""
        
        # Write to file
        with open(file_path, "w") as f:
            f.write(data)
        
        logger.info(f"Created Google Trends data file at {file_path}")
    
    # Load the data
    logger.info(f"Loading Google Trends data from {file_path}")
    df = pd.read_csv(file_path)
    df["ds"] = pd.to_datetime(df["Month"])  # Convert 'Month' to datetime format
    df["y"] = df["google_trends"]  # Rename 'google_trends' to 'y'
    df["unique_id"] = "iphone_trends"  # Create a unique id for this time series
    df.set_index("ds", inplace=True)
    return df[["unique_id", "y"]]


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test transformer model on short Google Trends iPhone time series"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model directory in models/final"
    )
    parser.add_argument(
        "--log_transform",
        action="store_true",
        help="Apply log transformation to the data before forecasting"
    )
    parser.add_argument(
        "--forecast_months",
        type=int,
        default=12,
        help="Number of months to forecast into the future"
    )
    parser.add_argument(
        "--short_series_months",
        nargs='+',
        type=int,
        default=[3, 6, 9, 12, 18, 24, 36],
        help="List of short series lengths to test (in months)"
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

    # Load the complete Google Trends dataset
    logger.info("Loading Google Trends iPhone dataset")
    full_series_df = load_google_trends_data()
    
    # Initialize the model once (to reuse for all tests)
    logger.info(f"Initializing model: {args.model_name}")
    model = TransformerModel(model_name=args.model_name)
    
    # Get model input sequence length
    # Default to 60 if not available
    input_sequence_length = getattr(model, 'input_series_length', (60,))[0]
    logger.info(f"Model input sequence length: {input_sequence_length}")
    
    # Store all results for comparison
    all_series_lengths = sorted(args.short_series_months)
    all_predictions = {}
    all_metrics = {}
    
    # Create a directory for figures
    figures_dir = Path('reports/figures/short_series')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a multi-plot figure
    plt.figure(figsize=(20, 15))
    gs = GridSpec(len(all_series_lengths) // 2 + len(all_series_lengths) % 2, 2)
    
    # Test with different short series lengths
    for i, n_months in enumerate(all_series_lengths):
        logger.info(f"Testing with last {n_months} months of data")
        
        # Get the subset of data
        subset_df = get_data_subset(full_series_df, n_months, pad_to_length=input_sequence_length)
        logger.info(f"Subset shape: {subset_df.shape}")
        
        # Create a copy for potential transformation
        input_df = subset_df.copy()
        
        # Apply log transformation if requested
        if args.log_transform:
            input_df = log_transform_series(input_df)
        
        # Generate predictions
        predictions = model.predict(
            series_df=input_df,
            n=args.forecast_months,
            num_samples=1000 if is_probabilistic else None,
            low_bound_conf=25,
            high_bound_conf=75,
        )
        
        # Apply inverse transformation if log transform was used
        if args.log_transform:
            logger.info("Applying inverse log transformation to predictions")
            # Transform prediction columns
            predictions['q_0.5'] = inverse_log_transform(predictions['q_0.5'].values)
            if is_probabilistic:
                for col in ['q_0.25', 'q_0.75']:
                    if col in predictions.columns:
                        predictions[col] = inverse_log_transform(predictions[col].values)
        
        # Store predictions
        all_predictions[n_months] = predictions
        
        # Calculate metrics for the portion that overlaps with actual data
        predictions['ds'] = pd.to_datetime(predictions['ds'])
        overlap_predictions = predictions[predictions['ds'].isin(full_series_df.index)]
        
        if not overlap_predictions.empty:
            actual_values = []
            predicted_values = []
            
            for date in overlap_predictions['ds']:
                if date in full_series_df.index:
                    actual_values.append(full_series_df.loc[date, 'y'])
                    idx = overlap_predictions['ds'] == date
                    predicted_values.append(overlap_predictions.loc[idx, 'q_0.5'].values[0])
            
            if actual_values and predicted_values:
                metrics = calculate_metrics(
                    pd.Series(actual_values),
                    pd.Series(predicted_values)
                )
                all_metrics[n_months] = metrics
                logger.info(f"Metrics for {n_months} months: {metrics}")
        
        # Create individual plots
        row, col = i // 2, i % 2
        ax = plt.subplot(gs[row, col])
        
        # Determine if padding was used
        padding_used = subset_df.shape[0] > n_months
        
        if padding_used:
            # Split into padding and actual data
            padding_df = subset_df.iloc[:-n_months]
            actual_subset_df = subset_df.iloc[-n_months:]
            
            # Plot padding as gray dots
            if not padding_df.empty:
                ax.plot(padding_df.index, padding_df['y'], 'o', color='gray', alpha=0.3, label='Zero Padding')
            
            # Plot actual input data
            ax.plot(actual_subset_df.index, actual_subset_df['y'], 'b-', label=f'Input ({n_months} months)')
        else:
            # Plot input data (no padding)
            ax.plot(subset_df.index, subset_df['y'], 'b-', label=f'Input ({n_months} months)')
        
        # Plot future actual data (if exists)
        last_date = subset_df.index[-1]
        future_actual = full_series_df[full_series_df.index > last_date]
        if not future_actual.empty:
            ax.plot(future_actual.index, future_actual['y'], 'g-', label='Actual Future')
        
        # Plot predictions
        ax.plot(predictions['ds'], predictions['q_0.5'], 'r--', label='Predictions')
        
        # Add confidence intervals for probabilistic models
        if is_probabilistic and 'q_0.25' in predictions.columns and 'q_0.75' in predictions.columns:
            ax.fill_between(
                predictions['ds'], 
                predictions['q_0.25'], 
                predictions['q_0.75'], 
                color='r', 
                alpha=0.2, 
                label='50% CI'
            )
        
        # Add vertical line at the end of input data
        ax.axvline(x=last_date, color='k', linestyle=':', alpha=0.7)
        
        # If padding was used, mark the start of actual data
        if padding_used:
            first_actual_date = actual_subset_df.index[0]
            ax.axvline(x=first_actual_date, color='gray', linestyle='--', alpha=0.5)
        
        # Add title with metrics
        title = f"Last {n_months} months → {args.forecast_months} months forecast"
        if padding_used:
            title += f" (padded to {subset_df.shape[0]})"
        if n_months in all_metrics:
            title += f"\nMAPE: {all_metrics[n_months]['MAPE']:.2f}%, RMSE: {all_metrics[n_months]['RMSE']:.2f}"
        ax.set_title(title)
        
        # Make the plot look nice
        ax.set_xlabel('Date')
        ax.set_ylabel('Search Volume Index')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    # Add overall title
    plt.suptitle(
        f"Google Trends 'iPhone' Forecasting with Limited History\n"
        f"Model: {args.model_name}, "
        f"{'Log-transformed' if args.log_transform else 'Original scale'}, "
        f"Forecast: {args.forecast_months} months",
        fontsize=16
    )
    
    # Save and show multi-plot figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    transform_suffix = '_log' if args.log_transform else ''
    plt.savefig(figures_dir / f'google_trends_short_series_{args.model_name}{transform_suffix}.png', dpi=300)
    plt.show()
    
    # Save summary metrics to CSV
    if all_metrics:
        # Convert metrics dict to DataFrame
        metrics_df = pd.DataFrame.from_dict({
            k: {metric: value for metric, value in v.items()}
            for k, v in all_metrics.items()
        }, orient='index')
        
        # Add a column for series length
        metrics_df.index.name = 'series_length_months'
        metrics_df = metrics_df.reset_index()
        
        # Save to CSV
        metrics_df.to_csv(figures_dir / f'google_trends_short_series_metrics_{args.model_name}{transform_suffix}.csv', index=False)
        logger.info(f"Saved metrics to: {figures_dir / f'google_trends_short_series_metrics_{args.model_name}{transform_suffix}.csv'}")
        
        # Print summary
        print("\nSummary of Metrics by Series Length:")
        print(metrics_df.set_index('series_length_months'))
    
    logger.info("Analysis completed successfully!")


if __name__ == "__main__":
    main() 