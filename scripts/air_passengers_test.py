"""
Script to test the transformer model with Air Passengers dataset.

This script demonstrates how to:
1. Load and prepare the Air Passengers dataset
2. Initialize and use the transformer model
3. Generate and visualize forecasts
4. Compare predictions with actual values (backtesting)

Usage:
    python scripts/air_passengers_test.py --model_name transformer_1.0_directml_point_M1_M48000_sampled1000
    python scripts/air_passengers_test.py --model_name transformer_1.0_directml_point_M1_M48000_sampled1000 --backtest_months 12 --forecast_months 24
    python scripts/air_passengers_test.py --model_name transformer_1.0_directml_point_M1_M48000_sampled1000 --log_transform
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
import numpy as np
from datetime import datetime, timedelta
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


def prepare_data_for_backtesting(df: pd.DataFrame, backtest_months: int, forecast_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for backtesting by splitting it into training and test sets.

    Args:
        df: Original DataFrame with all data
        backtest_months: Number of months to use for backtesting
        forecast_months: Number of months to forecast into the future

    Returns:
        tuple: (training_data, test_data)
    """
    # Sort the data by date
    df = df.sort_index()
    
    # Calculate the split point
    split_date = df.index[-backtest_months]
    
    # Split the data
    training_data = df[df.index < split_date].copy()
    test_data = df[df.index >= split_date].copy()
    
    # Add future dates for forecasting
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                               periods=forecast_months, 
                               freq='M')
    
    # Create future data frame with NaN values
    future_data = pd.DataFrame({
        'unique_id': ['series_1'] * len(future_dates),
        'y': [np.nan] * len(future_dates)
    }, index=future_dates)
    
    # Combine test data with future data
    test_data = pd.concat([test_data, future_data])
    
    return training_data, test_data


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
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


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
    parser.add_argument(
        "--backtest_months",
        type=int,
        default=0,
        help="Number of months to use for backtesting (comparing predictions with actuals)"
    )
    parser.add_argument(
        "--forecast_months",
        type=int,
        default=36,
        help="Number of months to forecast into the future"
    )
    parser.add_argument(
        "--log_transform",
        action="store_true",
        help="Apply log transformation to the data before forecasting (to handle increasing variance)"
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
    
    # Store original data for later comparison
    original_series_df = series_df.copy()
    
    # Apply log transformation if requested
    if args.log_transform:
        logger.info("Applying log transformation to the data")
        series_df = log_transform_series(series_df)

    # Prepare data for backtesting if requested
    if args.backtest_months > 0:
        logger.info(f"Preparing data for {args.backtest_months} months of backtesting")
        training_data, test_data = prepare_data_for_backtesting(
            series_df, 
            args.backtest_months, 
            args.forecast_months
        )
        print("\nTraining data (excluding backtest period):\n", training_data)
        print("\nTest data (including backtest period and future):\n", test_data)
    else:
        training_data = series_df
        test_data = None

    # Initialize Model
    model = TransformerModel(model_name=args.model_name)
    logger.info(f"Model initialized successfully")

    # Generate predictions
    if args.backtest_months > 0:
        # Generate predictions for both backtest period and future
        predictions = model.predict(
            series_df=training_data,
            n=args.backtest_months + args.forecast_months,
            num_samples=1000 if is_probabilistic else None,
            low_bound_conf=25,
            high_bound_conf=75,
        )
        
        # Apply inverse transformation if log transform was used
        if args.log_transform:
            logger.info("Applying inverse log transformation to predictions")
            # Transform prediction columns (q_0.5, q_0.25, q_0.75 if present)
            for col in ['q_0.5']:
                if col in predictions.columns:
                    predictions[col] = inverse_log_transform(predictions[col].values)
            
            if is_probabilistic:
                for col in ['q_0.25', 'q_0.75']:
                    if col in predictions.columns:
                        predictions[col] = inverse_log_transform(predictions[col].values)
        
        # Split predictions into backtest and future periods
        predictions['ds'] = pd.to_datetime(predictions['ds'])
        last_historical_date = original_series_df.index[-1]
        
        # Split based on dates, not index
        backtest_predictions = predictions[predictions['ds'] <= last_historical_date]
        future_predictions = predictions[predictions['ds'] > last_historical_date]
        
        # Calculate metrics for backtest period using original scale data
        backtest_dates = backtest_predictions['ds'].values
        actual_series = pd.Series([original_series_df.loc[date, 'y'] for date in backtest_dates if date in original_series_df.index], 
                                 index=backtest_dates)
        predicted_series = pd.Series(backtest_predictions['q_0.5'].values, index=backtest_dates)
        
        metrics = calculate_metrics(actual_series, predicted_series)
        
        print("\nBacktesting Results:")
        print("="*50)
        print(f"Period: {backtest_predictions['ds'].min().strftime('%Y-%m')} to {backtest_predictions['ds'].max().strftime('%Y-%m')}")
        print(f"Log Transformation: {'Applied' if args.log_transform else 'Not Applied'}")
        print("\nError Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nBacktest Predictions vs Actuals:")
        # Create a better comparison dataframe with aligned dates
        comparison_df = pd.DataFrame({
            'Date': backtest_dates,
            'Actual': actual_series.values,
            'Predicted': predicted_series.values
        })
        comparison_df['Error'] = comparison_df['Actual'] - comparison_df['Predicted']
        comparison_df['Error %'] = (comparison_df['Error'] / comparison_df['Actual'] * 100).round(2)
        comparison_df.set_index('Date', inplace=True)
        print(comparison_df)
        
        if is_probabilistic:
            print("\nProbabilistic Forecasts (Future):")
            future_df = future_predictions[["ds", "q_0.25", "q_0.5", "q_0.75"]]
            future_df.set_index('ds', inplace=True)
            print(future_df)
        else:
            print("\nPoint Forecasts (Future):")
            future_df = future_predictions[["ds", "q_0.5"]]
            future_df.set_index('ds', inplace=True)
            print(future_df)
    else:
        # Original behavior - just forecast future
        if is_probabilistic:
            predictions = model.predict(
                series_df=series_df,
                n=args.forecast_months,
                num_samples=1000,
                low_bound_conf=25,
                high_bound_conf=75,
            )
            
            # Apply inverse transformation if log transform was used
            if args.log_transform:
                logger.info("Applying inverse log transformation to predictions")
                for col in ['q_0.5', 'q_0.25', 'q_0.75']:
                    if col in predictions.columns:
                        predictions[col] = inverse_log_transform(predictions[col].values)
            
            print("\nProbabilistic Predictions:")
            print("- Point forecast (q_0.5):")
            print(predictions[["ds", "q_0.5", "unique_id"]])
            print("\n- Confidence Intervals:")
            print(predictions[["ds", "q_0.25", "q_0.75", "unique_id"]])
        else:
            predictions = model.predict(
                series_df=series_df,
                n=args.forecast_months,
            )
            
            # Apply inverse transformation if log transform was used
            if args.log_transform:
                logger.info("Applying inverse log transformation to predictions")
                predictions['q_0.5'] = inverse_log_transform(predictions['q_0.5'].values)
            
            print("\nPoint Predictions:")
            print(predictions[["ds", "q_0.5", "unique_id"]])

    # Plot results
    logger.info("Plotting forecasts")
    if args.backtest_months > 0:
        # Custom plotting to distinguish backtest vs forward predictions
        
        # Get the historical data (using original scale)
        historical_dates = original_series_df.index[:-args.backtest_months]
        historical_values = original_series_df.loc[historical_dates, 'y'].values
        
        # Get backtest data
        predictions['ds'] = pd.to_datetime(predictions['ds'])
        last_historical_date = original_series_df.index[-1]
        
        # Split predictions into backtest and future portions
        backtest_predictions = predictions[predictions['ds'] <= last_historical_date]
        future_predictions = predictions[predictions['ds'] > last_historical_date]
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(historical_dates, historical_values, 'b-', label='Historical Data')
        
        # Plot actual data for backtest period
        backtest_actual_dates = original_series_df.index[-args.backtest_months:]
        backtest_actual_values = original_series_df.loc[backtest_actual_dates, 'y'].values
        plt.plot(backtest_actual_dates, backtest_actual_values, 'g-', label='Actual (Test Period)')
        
        # Plot backtest predictions
        plt.plot(backtest_predictions['ds'], backtest_predictions['q_0.5'], 'r--', label='Backtest Predictions')
        
        # Plot future predictions
        plt.plot(future_predictions['ds'], future_predictions['q_0.5'], 'r-', label='Future Predictions')
        
        # Add vertical line to mark split between historical and forecast
        split_date = original_series_df.index[-(args.backtest_months + 1)]
        plt.axvline(x=split_date, color='k', linestyle='--', alpha=0.7)
        
        # Add vertical line to mark end of actual data
        plt.axvline(x=last_historical_date, color='k', linestyle=':', alpha=0.7)
        
        # Add labels
        title = 'Time Series Forecast with Backtesting'
        if args.log_transform:
            title += ' (Log Transformed)'
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Passengers')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        figures_dir = Path('reports/figures')
        figures_dir.mkdir(parents=True, exist_ok=True)
        transform_suffix = '_log' if args.log_transform else ''
        plt.savefig(figures_dir / f'forecast_with_backtest_{args.model_name}{transform_suffix}.png')
        
        # Show plot
        plt.tight_layout()
        plt.show()
    else:
        # For simple forecasting, we need to transform the data back for plotting
        if args.log_transform:
            # We need to create a custom plot as the model's plot_forecast doesn't handle transformed data
            
            # Get the historical data (original scale)
            historical_dates = original_series_df.index
            historical_values = original_series_df['y'].values
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            # Plot historical data
            plt.plot(historical_dates, historical_values, 'b-', label='Historical Data')
            
            # Plot future predictions
            predictions['ds'] = pd.to_datetime(predictions['ds'])
            plt.plot(predictions['ds'], predictions['q_0.5'], 'r-', label='Predictions')
            
            # Add vertical line to mark end of historical data
            plt.axvline(x=historical_dates[-1], color='k', linestyle=':', alpha=0.7)
            
            # Add labels
            plt.title('Time Series Forecast (Log Transformed)')
            plt.xlabel('Date')
            plt.ylabel('Passengers')
            plt.grid(True)
            plt.legend()
            
            # Save plot
            figures_dir = Path('reports/figures')
            figures_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(figures_dir / f'forecast_{args.model_name}_log.png')
            
            # Show plot
            plt.tight_layout()
            plt.show()
        else:
            model.plot_forecast(original_series_df, predictions)


if __name__ == "__main__":
    main() 