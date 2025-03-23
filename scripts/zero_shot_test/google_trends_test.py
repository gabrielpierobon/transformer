"""
Script to test the transformer model with Google Trends 'iPhone' data.

This script demonstrates how to:
1. Load and prepare Google Trends data
2. Initialize and use the transformer model
3. Generate and visualize forecasts
4. Compare predictions with actual values (backtesting)

Usage:
    python scripts/google_trends_test.py --model_name your_model_name
    python scripts/google_trends_test.py --model_name your_model_name --backtest_months 12 --forecast_months 24
    python scripts/google_trends_test.py --model_name your_model_name --log_transform
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import io

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
from datetime import datetime
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
        'unique_id': ['iphone_trends'] * len(future_dates),
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
2010-01,51
2010-02,47
2010-03,46
2010-04,49
2010-05,50
2010-06,74
2010-07,66
2010-08,65
2010-09,71
2010-10,67
2010-11,67
2010-12,73
2011-01,70
2011-02,68
2011-03,64
2011-04,62
2011-05,63
2011-06,62
2011-07,62
2011-08,64
2011-09,70
2011-10,94
2011-11,72
2011-12,78
2012-01,74
2012-02,67
2012-03,66
2012-04,62
2012-05,58
2012-06,60
2012-07,63
2012-08,63
2012-09,100
2012-10,69
2012-11,66
2012-12,72
2013-01,66
2013-02,65
2013-03,63
2013-04,58
2013-05,58
2013-06,59
2013-07,58
2013-08,58
2013-09,79
2013-10,63
2013-11,60
2013-12,66
2014-01,61
2014-02,59
2014-03,56
2014-04,52
2014-05,52
2014-06,51
2014-07,55
2014-08,56
2014-09,88
2014-10,68
2014-11,61
2014-12,62
2015-01,61
2015-02,54
2015-03,55
2015-04,55
2015-05,52
2015-06,53
2015-07,58
2015-08,59
2015-09,69
2015-10,65
2015-11,59
2015-12,62
2016-01,59
2016-02,55
2016-03,63
2016-04,55
2016-05,52
2016-06,54
2016-07,55
2016-08,57
2016-09,74
2016-10,61
2016-11,55
2016-12,57
2017-01,53
2017-02,50
2017-03,48
2017-04,48
2017-05,46
2017-06,46
2017-07,49
2017-08,47
2017-09,68
2017-10,55
2017-11,61
2017-12,58"""
        
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
    """
    Main function to demonstrate transformer model usage with Google Trends data.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test transformer model with Google Trends iPhone data"
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
        default=12,
        help="Number of months to use for backtesting (comparing predictions with actuals)"
    )
    parser.add_argument(
        "--forecast_months",
        type=int,
        default=12,
        help="Number of months to forecast into the future"
    )
    parser.add_argument(
        "--log_transform",
        action="store_true",
        help="Apply log transformation to the data before forecasting"
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
    logger.info("Loading Google Trends iPhone data")
    series_df = load_google_trends_data()
    
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
        
        # Add shaded area for probabilistic predictions if available
        if is_probabilistic and 'q_0.25' in future_predictions.columns and 'q_0.75' in future_predictions.columns:
            plt.fill_between(
                future_predictions['ds'], 
                future_predictions['q_0.25'], 
                future_predictions['q_0.75'],
                color='r', alpha=0.2, label='50% Confidence Interval'
            )
        
        # Add vertical line to mark split between historical and forecast
        split_date = original_series_df.index[-(args.backtest_months + 1)]
        plt.axvline(x=split_date, color='k', linestyle='--', alpha=0.7)
        
        # Add vertical line to mark end of actual data
        plt.axvline(x=last_historical_date, color='k', linestyle=':', alpha=0.7)
        
        # Add labels
        title = 'Google Trends "iPhone" Search Volume Forecast with Backtesting'
        if args.log_transform:
            title += ' (Log Transformed)'
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Search Volume Index')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        figures_dir = Path('reports/figures')
        figures_dir.mkdir(parents=True, exist_ok=True)
        transform_suffix = '_log' if args.log_transform else ''
        plt.savefig(figures_dir / f'google_trends_iphone_forecast_{args.model_name}{transform_suffix}.png')
        
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
            
            # Add shaded area for probabilistic predictions if available
            if is_probabilistic and 'q_0.25' in predictions.columns and 'q_0.75' in predictions.columns:
                plt.fill_between(
                    predictions['ds'], 
                    predictions['q_0.25'], 
                    predictions['q_0.75'],
                    color='r', alpha=0.2, label='50% Confidence Interval'
                )
            
            # Add vertical line to mark end of historical data
            plt.axvline(x=historical_dates[-1], color='k', linestyle=':', alpha=0.7)
            
            # Add labels
            plt.title('Google Trends "iPhone" Search Volume Forecast (Log Transformed)')
            plt.xlabel('Date')
            plt.ylabel('Search Volume Index')
            plt.grid(True)
            plt.legend()
            
            # Save plot
            figures_dir = Path('reports/figures')
            figures_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(figures_dir / f'google_trends_iphone_forecast_{args.model_name}_log.png')
            
            # Show plot
            plt.tight_layout()
            plt.show()
        else:
            # Custom plotting for better labeling
            model.plot_forecast(
                original_series_df, 
                predictions, 
                title='Google Trends "iPhone" Search Volume Forecast',
                ylabel='Search Volume Index'
            )


if __name__ == "__main__":
    main() 