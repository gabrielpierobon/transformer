#!/usr/bin/env python
"""
Script to evaluate the transformer model on the M4 test set.

This script:
1. Loads the M4 training and test data
2. Randomly samples series from the dataset
3. Takes the last 60 points of each training series
4. Generates forecasts for the next 18 months
5. Compares forecasts with actual values from the test set
6. Calculates performance metrics (MAE, MAPE, SMAPE, etc.)
7. Saves results to CSV files in the evaluation/ directory
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
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Formula: SMAPE = (100%/n) * sum(|predicted - actual| / ((|actual| + |predicted|) / 2))
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        SMAPE value as a percentage
    """
    # Calculate denominator
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    
    # Create mask for valid values (denominator > 0)
    mask = denominator > 0
    
    # Calculate SMAPE if there are valid values
    if np.any(mask):
        return np.mean(np.abs(predicted[mask] - actual[mask]) / denominator[mask]) * 100
    else:
        return np.nan


def setup_directories() -> None:
    """Create necessary directories for evaluation results."""
    os.makedirs("evaluation", exist_ok=True)
    os.makedirs("evaluation/plots", exist_ok=True)
    logger.info("Created evaluation directories")


def load_m4_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load M4 training and test data.
    
    Returns:
        Tuple of (train_df, test_df)
    """
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


def sample_series(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_size: int,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly sample series from the dataset.
    
    Args:
        train_df: Training data DataFrame
        test_df: Test data DataFrame
        sample_size: Number of series to sample
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (sampled_train_df, sampled_test_df)
    """
    random.seed(random_seed)
    
    # Get all series IDs
    series_ids = train_df['V1'].tolist()
    
    # Sample series IDs
    if sample_size >= len(series_ids):
        sampled_ids = series_ids
        logger.warning(
            f"Sample size {sample_size} is larger than available series "
            f"({len(series_ids)}). Using all series."
        )
    else:
        sampled_ids = random.sample(series_ids, sample_size)
    
    # Filter dataframes to only include sampled series
    sampled_train_df = train_df[train_df['V1'].isin(sampled_ids)]
    sampled_test_df = test_df[test_df['V1'].isin(sampled_ids)]
    
    logger.info(f"Sampled {len(sampled_ids)} series for evaluation")
    
    return sampled_train_df, sampled_test_df


def prepare_series_for_prediction(
    series_values: List[float],
    series_id: str
) -> pd.DataFrame:
    """
    Prepare a series for prediction with the transformer model.
    
    Args:
        series_values: List of values for the series
        series_id: Identifier for the series
        
    Returns:
        DataFrame formatted for the transformer model
    """
    # Create a date range (doesn't matter for prediction, just for format)
    dates = pd.date_range(start="2019-01-01", periods=len(series_values), freq="MS")
    
    # Create DataFrame
    df = pd.DataFrame({
        "ds": dates,
        "y": series_values,
        "unique_id": series_id
    })
    
    # Set index
    df.set_index("ds", inplace=True)
    
    return df


def get_last_n_values(
    series: pd.Series,
    n: int = 60
) -> List[float]:
    """
    Get the last n non-NaN values from a series.
    
    Args:
        series: Series to extract values from
        n: Number of values to extract
        
    Returns:
        List of the last n values
    """
    # Drop NaN values and convert to list, skipping the series ID
    values = series.dropna().tolist()[1:]
    
    # Return the last n values
    return values[-n:] if len(values) >= n else values


def naive2_forecast(
    history: np.ndarray,
    h: int,
    seasonal_period: int = 12  # Monthly data has a seasonal period of 12
) -> np.ndarray:
    """
    Generate a Naïve2 forecast for the given history.
    
    The Naïve2 method combines a seasonal naïve approach with a drift component.
    For monthly data, it uses the value from the same month in the previous year,
    adjusted by the average trend observed in the historical data.
    
    Args:
        history: Historical values
        h: Forecast horizon
        seasonal_period: Seasonal period (12 for monthly data)
        
    Returns:
        Array of forecasted values
    """
    n = len(history)
    
    # If we don't have enough history for seasonal component, use simple drift
    if n <= seasonal_period:
        # Calculate drift (average change)
        drift = (history[-1] - history[0]) / (n - 1) if n > 1 else 0
        return np.array([history[-1] + (i + 1) * drift for i in range(h)])
    
    # Initialize forecast array
    forecast = np.zeros(h)
    
    # Calculate drift (average change per period)
    drift = (history[-1] - history[-seasonal_period - 1]) / seasonal_period
    
    # Generate forecasts
    for i in range(h):
        # Get the value from the same season in the last year
        seasonal_idx = n - seasonal_period + (i % seasonal_period)
        # Add drift component
        forecast[i] = history[seasonal_idx] + ((i + 1) // seasonal_period + (i + 1) % seasonal_period / seasonal_period) * drift
    
    return forecast


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    series_id: str,
    horizon: int,
    history: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate performance metrics for a single prediction.
    
    Metrics calculated:
    - MAE: Mean Absolute Error
    - MAPE: Mean Absolute Percentage Error
    - SMAPE: Symmetric Mean Absolute Percentage Error 
             Formula: SMAPE = (100%/n) * sum(|predicted - actual| / ((|actual| + |predicted|) / 2))
             Note: Values where both actual and predicted are zero are excluded from the calculation
    - RMSE: Root Mean Squared Error
    - WMAPE: Weighted Mean Absolute Percentage Error
    - MASE: Mean Absolute Scaled Error
    
    Args:
        actual: Actual values
        predicted: Predicted values
        series_id: Identifier for the series
        horizon: Forecast horizon (1-18)
        history: Historical values (needed for MASE calculation)
        
    Returns:
        Dictionary of metrics
    """
    # For horizon-specific metrics, we only need the first value
    # This is because we're calculating metrics for a specific horizon
    # and the arrays passed in are already sliced to start at that horizon
    actual_h = actual[0:1]  # Just the first value for this horizon
    predicted_h = predicted[0:1]  # Just the first value for this horizon
    
    # For overall metrics (across all available points), use all values
    # Ensure arrays are the same length
    min_len = min(len(actual), len(predicted))
    actual_all = actual[:min_len]
    predicted_all = predicted[:min_len]
    
    # Handle zero values in actual for MAPE calculation
    non_zero_mask = actual_all != 0
    
    # Calculate metrics
    mae = np.mean(np.abs(predicted_all - actual_all))
    
    # MAPE (Mean Absolute Percentage Error)
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((actual_all[non_zero_mask] - predicted_all[non_zero_mask]) / actual_all[non_zero_mask])) * 100
    else:
        mape = np.nan
    
    # SMAPE (Symmetric Mean Absolute Percentage Error) for the specific horizon
    # Formula: SMAPE = (100%/n) * sum(|predicted - actual| / ((|actual| + |predicted|) / 2))
    # For a single horizon point, this simplifies to:
    # SMAPE = 100% * |predicted - actual| / ((|actual| + |predicted|) / 2)
    denominator_h = (np.abs(actual_h) + np.abs(predicted_h)) / 2
    
    # Avoid division by zero
    if np.all(denominator_h > 0):
        smape = calculate_smape(actual_h, predicted_h)
    else:
        smape = np.nan
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((predicted_all - actual_all) ** 2))
    
    # WMAPE (Weighted Mean Absolute Percentage Error)
    if np.sum(np.abs(actual_all)) > 0:
        wmape = np.sum(np.abs(predicted_all - actual_all)) / np.sum(np.abs(actual_all)) * 100
    else:
        wmape = np.nan
    
    # MASE (Mean Absolute Scaled Error)
    mase = np.nan
    if history is not None and len(history) > 1:
        # For monthly data, use seasonal=12
        seasonal_period = 12
        
        # Calculate errors of the naive seasonal forecast in the historical data
        naive_errors = []
        for i in range(seasonal_period, len(history)):
            naive_errors.append(abs(history[i] - history[i - seasonal_period]))
        
        # Calculate mean of naive errors
        if naive_errors and np.mean(naive_errors) > 0:
            # Scale the forecast errors by the mean naive error
            mase = np.mean(np.abs(predicted_all - actual_all)) / np.mean(naive_errors)
        
    return {
        "series_id": series_id,
        "horizon": horizon,
        "actual": float(actual_h[0]),  # For the specific horizon
        "predicted": float(predicted_h[0]),  # For the specific horizon
        "mae": mae,
        "mape": mape,
        "smape": smape,
        "rmse": rmse,
        "wmape": wmape,
        "mase": mase
    }


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
    plot_path = f"evaluation/plots/{model_name}_{series_id}_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory


def evaluate_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast_horizon: int = 18,
    random_seed: int = 42  # Add random seed parameter
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Evaluate the model on the sampled series.
    
    Args:
        model_name: Name of the model to evaluate
        train_df: Training data DataFrame
        test_df: Test data DataFrame
        forecast_horizon: Number of periods to forecast
        random_seed: Random seed for shuffling series order
        
    Returns:
        Tuple of (detailed_results_df, summary_results_df, naive2_metrics)
    """
    # Initialize model
    model = TransformerModel(
        model_name=model_name,
        input_series_length=60  # We'll use the last 60 points
    )
    
    # Initialize results list
    detailed_results = []
    naive2_results = []
    series_summary_results = []
    all_series_naive2_metrics = []  # Store all series naive2 metrics
    
    # Create timestamp for filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create a list of indices and shuffle them to process series in random order
    random.seed(random_seed)
    indices = list(range(len(train_df)))
    random.shuffle(indices)
    
    # Get train and test data as lists for random access
    train_data = [row for _, row in train_df.iterrows()]
    test_data = [row for _, row in test_df.iterrows()]
    
    # Calculate total number of series for progress tracking
    total_series = len(indices)
    start_time = time.time()
    
    # Process each series in shuffled order
    for count, idx in enumerate(tqdm(indices, desc="Processing series", unit="series")):
        train_series = train_data[idx]
        test_series = test_data[idx]
        
        series_id = train_series['V1']
        
        # Get all non-NaN values from the training series (for plotting)
        all_train_values = train_series.dropna().tolist()[1:]  # Skip series ID
        
        # Get the last 60 values from the training series (for prediction)
        last_60_values = get_last_n_values(train_series, 60)
        
        # Prepare series for prediction
        series_df = prepare_series_for_prediction(last_60_values, series_id)
        
        try:
            # Generate forecast
            forecast = model.predict(
                series_df=series_df,
                n=forecast_horizon,
                num_samples=1000,
                low_bound_conf=30,
                high_bound_conf=70,
            )
            
            # Extract predicted values (median forecast)
            predicted_values = forecast['q_0.5'].values
            
            # Extract actual values from test data (skip series ID)
            actual_values = test_series.iloc[1:forecast_horizon+1].values
            
            # Create plot for this series
            plot_series_forecast(
                train_values=all_train_values,
                forecast_values=predicted_values,
                actual_values=actual_values,
                series_id=series_id,
                model_name=model_name,
                timestamp=timestamp
            )
            
            # Generate Naïve2 forecast for comparison
            naive2_forecast_values = naive2_forecast(
                history=np.array(last_60_values),
                h=forecast_horizon
            )
            
            # Lists to store metrics for this series across all horizons
            series_metrics = []
            series_naive2_metrics = []
            
            # Calculate metrics for each horizon
            for h in range(1, forecast_horizon + 1):
                # Calculate metrics for transformer model
                horizon_metrics = calculate_metrics(
                    actual=actual_values[h-1:],
                    predicted=predicted_values[h-1:],
                    series_id=series_id,
                    horizon=h,
                    history=np.array(last_60_values)
                )
                
                detailed_results.append(horizon_metrics)
                series_metrics.append(horizon_metrics)
                
                # Calculate metrics for Naïve2 model
                naive2_metrics = calculate_metrics(
                    actual=actual_values[h-1:],
                    predicted=naive2_forecast_values[h-1:],
                    series_id=series_id,
                    horizon=h,
                    history=np.array(last_60_values)
                )
                
                naive2_results.append(naive2_metrics)
                series_naive2_metrics.append(naive2_metrics)
            
            # Calculate average metrics across all horizons for this series
            series_df = pd.DataFrame(series_metrics)
            
            # For SMAPE, we need to recalculate it across all horizons
            # First, get all actual and predicted values for this series
            all_actuals = actual_values
            all_predicted = predicted_values[:len(all_actuals)]
            
            # Calculate overall SMAPE for the series
            series_smape = calculate_smape(all_actuals, all_predicted[:len(all_actuals)])
            
            series_avg_metrics = {
                'series_id': series_id,
                'mae': series_df['mae'].mean(),
                'mape': series_df['mape'].mean(),
                'smape': series_smape,  # Use the recalculated SMAPE
                'rmse': series_df['rmse'].mean(),
                'wmape': series_df['wmape'].mean(),
                'mase': series_df['mase'].mean()
            }
            
            # Calculate Naïve2 metrics for this series
            naive2_series_df = pd.DataFrame(series_naive2_metrics)
            
            # Calculate overall SMAPE for Naïve2
            all_naive2_predicted = naive2_forecast_values[:len(all_actuals)]
            naive2_series_smape = calculate_smape(all_actuals, all_naive2_predicted)
            
            naive2_series_avg_metrics = {
                'series_id': series_id,  # Add series_id to naive2 metrics
                'smape': naive2_series_smape,
                'mase': naive2_series_df['mase'].mean()
            }
            
            # Calculate OWA for this series
            if not np.isnan(naive2_series_avg_metrics['smape']) and naive2_series_avg_metrics['smape'] > 0 and not np.isnan(naive2_series_avg_metrics['mase']) and naive2_series_avg_metrics['mase'] > 0:
                relative_smape = series_avg_metrics['smape'] / naive2_series_avg_metrics['smape']
                relative_mase = series_avg_metrics['mase'] / naive2_series_avg_metrics['mase']
                owa = (relative_smape + relative_mase) / 2
            else:
                relative_smape = np.nan
                relative_mase = np.nan
                owa = np.nan
            
            # Add OWA metrics to the series summary
            series_avg_metrics['relative_smape'] = relative_smape
            series_avg_metrics['relative_mase'] = relative_mase
            series_avg_metrics['owa'] = owa
            
            # Add to series summary results
            series_summary_results.append(series_avg_metrics)
            all_series_naive2_metrics.append(naive2_series_avg_metrics)  # Store naive2 metrics for this series
            
            # Print progress information every 10 series or when requested
            if (count + 1) % 10 == 0 or (count + 1) == total_series:
                elapsed_time = time.time() - start_time
                series_per_second = (count + 1) / elapsed_time if elapsed_time > 0 else 0
                estimated_total_time = total_series / series_per_second if series_per_second > 0 else 0
                remaining_time = estimated_total_time - elapsed_time if estimated_total_time > 0 else 0
                
                logger.info(
                    f"Progress: {count + 1}/{total_series} series ({(count + 1)/total_series*100:.1f}%) | "
                    f"Speed: {series_per_second:.2f} series/sec | "
                    f"Elapsed: {elapsed_time/60:.1f} min | "
                    f"Remaining: {remaining_time/60:.1f} min | "
                    f"Current series: {series_id}"
                )
                
        except Exception as e:
            logger.error(f"Error processing series {series_id}: {str(e)}")
    
    # Convert detailed results to DataFrame
    detailed_df = pd.DataFrame(detailed_results)
    naive2_df = pd.DataFrame(naive2_results)
    
    # Convert series summary results to DataFrame
    summary_df = pd.DataFrame(series_summary_results)
    
    # Calculate overall metrics across all series
    # For SMAPE, we take the mean of the series-level SMAPE values
    # This is the correct approach for the M4 competition
    overall_metrics = {
        'series_id': 'Overall',
        'mae': detailed_df['mae'].mean(),
        'mape': detailed_df['mape'].mean(),
        'smape': summary_df['smape'].mean(),  # Use series-level SMAPE
        'rmse': detailed_df['rmse'].mean(),
        'wmape': detailed_df['wmape'].mean(),
        'mase': detailed_df['mase'].mean()
    }
    
    # For Naïve2, we need to collect all the series-level SMAPE values
    naive2_smape_values = []
    for series_id in summary_df['series_id'].values:
        if series_id != 'Overall':  # Skip the overall row
            # Find the corresponding naive2 SMAPE for this series
            matching_metrics = [m for m in all_series_naive2_metrics if m['series_id'] == series_id]
            if matching_metrics:
                naive2_smape_values.append(matching_metrics[0]['smape'])
    
    naive2_overall_metrics = {
        'smape': np.mean(naive2_smape_values) if naive2_smape_values else np.nan,
        'mase': naive2_df['mase'].mean()
    }
    
    # Calculate overall OWA
    if not np.isnan(naive2_overall_metrics['smape']) and naive2_overall_metrics['smape'] > 0 and not np.isnan(naive2_overall_metrics['mase']) and naive2_overall_metrics['mase'] > 0:
        relative_smape = overall_metrics['smape'] / naive2_overall_metrics['smape']
        relative_mase = overall_metrics['mase'] / naive2_overall_metrics['mase']
        owa = (relative_smape + relative_mase) / 2
    else:
        relative_smape = np.nan
        relative_mase = np.nan
        owa = np.nan
    
    # Add OWA metrics to the overall metrics
    overall_metrics['relative_smape'] = relative_smape
    overall_metrics['relative_mase'] = relative_mase
    overall_metrics['owa'] = owa
    
    # Add overall summary row
    summary_df = pd.concat([summary_df, pd.DataFrame([overall_metrics])], ignore_index=True)
    
    # Add OWA metrics to the overall metrics dictionary for separate reporting
    owa_metrics = {
        'relative_smape': relative_smape,
        'relative_mase': relative_mase,
        'owa': owa
    }
    
    return detailed_df, summary_df, owa_metrics


def save_results(
    detailed_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    owa_metrics: Dict[str, float],
    model_name: str,
    sample_size: int
) -> None:
    """
    Save evaluation results to Excel (XLSX) files.
    
    Args:
        detailed_df: DataFrame with detailed results
        summary_df: DataFrame with summary results
        owa_metrics: Dictionary with OWA metrics
        model_name: Name of the model evaluated
        sample_size: Number of series sampled
    """
    # Create timestamp for filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Round numeric columns in detailed_df to 2 decimal places
    detailed_numeric_columns = ['actual', 'predicted', 'mae', 'mape', 'smape', 'rmse', 'wmape', 'mase']
    for col in detailed_numeric_columns:
        if col in detailed_df.columns:
            detailed_df[col] = detailed_df[col].round(2)
    
    # Round numeric columns in summary_df to 2 decimal places
    summary_numeric_columns = ['mae', 'mape', 'smape', 'rmse', 'wmape', 'mase', 'relative_smape', 'relative_mase', 'owa']
    for col in summary_numeric_columns:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(2)
    
    # Create OWA DataFrame
    owa_df = pd.DataFrame([{
        'metric': 'Relative sMAPE',
        'value': round(owa_metrics['relative_smape'], 2)
    }, {
        'metric': 'Relative MASE',
        'value': round(owa_metrics['relative_mase'], 2)
    }, {
        'metric': 'OWA (Overall Weighted Average)',
        'value': round(owa_metrics['owa'], 2)
    }])
    
    # Save all results to a single Excel file with multiple sheets
    excel_path = f"evaluation/{model_name}_results_{sample_size}series_{timestamp}.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        detailed_df.to_excel(writer, sheet_name='Detailed', index=False)
        owa_df.to_excel(writer, sheet_name='OWA Metrics', index=False)
    
    logger.info(f"Saved all results to Excel file: {excel_path}")
    
    # Print OWA metrics
    print("\nOverall Weighted Average (OWA) Metrics:")
    print(f"Relative sMAPE: {owa_metrics['relative_smape']:.2f}")
    print(f"Relative MASE: {owa_metrics['relative_mase']:.2f}")
    print(f"OWA: {owa_metrics['owa']:.2f}")
    print(f"Note: OWA < 1 means the model outperforms the Naïve2 benchmark")
    
    # Print summary table with rounded values
    print("\nSummary Metrics by Series (showing first 10 series + Overall):")
    display_df = pd.concat([summary_df.head(10), summary_df.tail(1)])
    print(display_df.to_string(index=False))


def plot_summary_metrics(
    summary_df: pd.DataFrame,
    model_name: str,
    sample_size: int,
    owa_metrics: Dict[str, float]
) -> None:
    """
    Plot summary metrics by horizon.
    
    Args:
        summary_df: DataFrame with summary metrics
        model_name: Name of the model evaluated
        sample_size: Number of series sampled
        owa_metrics: Dictionary with OWA metrics
    """
    # Filter out the 'Overall' row for distribution plots
    plot_df = summary_df[summary_df['series_id'] != 'Overall'].copy()
    
    # Create figure with subplots for metric distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot MAE distribution
    axes[0, 0].hist(plot_df['mae'].dropna(), bins=20)
    axes[0, 0].set_title('MAE Distribution')
    axes[0, 0].set_xlabel('MAE')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True)
    
    # Plot MAPE distribution
    axes[0, 1].hist(plot_df['mape'].dropna(), bins=20)
    axes[0, 1].set_title('MAPE Distribution')
    axes[0, 1].set_xlabel('MAPE (%)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True)
    
    # Plot SMAPE distribution
    axes[0, 2].hist(plot_df['smape'].dropna(), bins=20)
    axes[0, 2].set_title('SMAPE Distribution')
    axes[0, 2].set_xlabel('SMAPE (%)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].grid(True)
    
    # Plot RMSE distribution
    axes[1, 0].hist(plot_df['rmse'].dropna(), bins=20)
    axes[1, 0].set_title('RMSE Distribution')
    axes[1, 0].set_xlabel('RMSE')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(True)
    
    # Plot MASE distribution
    axes[1, 1].hist(plot_df['mase'].dropna(), bins=20)
    axes[1, 1].set_title('MASE Distribution')
    axes[1, 1].set_xlabel('MASE')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True)
    
    # Plot OWA distribution
    axes[1, 2].hist(plot_df['owa'].dropna(), bins=20)
    axes[1, 2].set_title('OWA Distribution')
    axes[1, 2].set_xlabel('OWA')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].axvline(x=1.0, color='red', linestyle='--', label='Benchmark')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    # Add overall title with OWA
    plt.suptitle(
        f'Forecast Performance Metrics Distribution\n'
        f'{model_name} ({sample_size} series)\n'
        f'Overall OWA: {owa_metrics["owa"]:.2f} '
        f'({"Better than" if owa_metrics["owa"] < 1 else "Worse than"} Naïve2 benchmark)',
        fontsize=16
    )
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = f"evaluation/{model_name}_metrics_distribution_{sample_size}series_{timestamp}.png"
    plt.savefig(plot_path)
    
    # Create a second plot for the percentage of series that beat the benchmark
    plt.figure(figsize=(10, 6))
    
    # Calculate percentage of series that beat the benchmark
    better_than_benchmark = (plot_df['owa'] < 1).mean() * 100
    
    # Create a bar chart
    plt.bar(['Better than Naïve2', 'Worse than Naïve2'], 
            [better_than_benchmark, 100 - better_than_benchmark])
    
    plt.title(f'Percentage of Series vs. Naïve2 Benchmark\n{model_name} ({sample_size} series)')
    plt.ylabel('Percentage of Series (%)')
    plt.ylim(0, 100)
    
    # Add percentage labels on bars
    plt.text(0, better_than_benchmark + 2, f'{better_than_benchmark:.1f}%', 
             ha='center', va='bottom')
    plt.text(1, 100 - better_than_benchmark + 2, f'{100 - better_than_benchmark:.1f}%', 
             ha='center', va='bottom')
    
    plt.grid(axis='y')
    
    # Save figure
    benchmark_plot_path = f"evaluation/{model_name}_benchmark_comparison_{sample_size}series_{timestamp}.png"
    plt.savefig(benchmark_plot_path)
    
    logger.info(f"Saved benchmark comparison plot to {benchmark_plot_path}")


def main():
    """Main function to run the evaluation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate transformer model on M4 test set"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="transformer_1.0_directml_point_M1_M48000_sampled2000",
        help="Name of the model directory in models/final"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of series to sample for evaluation"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Load data
    train_df, test_df = load_m4_data()
    
    # Sample series
    sampled_train_df, sampled_test_df = sample_series(
        train_df, test_df, args.sample_size, args.random_seed
    )
    
    # Evaluate model
    detailed_df, summary_df, owa_metrics = evaluate_model(
        model_name=args.model_name,
        train_df=sampled_train_df,
        test_df=sampled_test_df,
        random_seed=args.random_seed  # Pass random seed to evaluate_model
    )
    
    # Save results
    save_results(
        detailed_df=detailed_df,
        summary_df=summary_df,
        owa_metrics=owa_metrics,
        model_name=args.model_name,
        sample_size=args.sample_size
    )
    
    # Plot summary metrics
    plot_summary_metrics(
        summary_df=summary_df,
        model_name=args.model_name,
        sample_size=args.sample_size,
        owa_metrics=owa_metrics
    )
    
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main() 