#!/usr/bin/env python
"""
Script to evaluate the transformer model on the Tourism monthly dataset.

This script:
1. Loads the Tourism training and test data
2. Generates forecasts for each series
3. Compares forecasts with actual values from the test set
4. Calculates performance metrics (MAE, MAPE, SMAPE, etc.) following the methodology in the paper
5. Saves results to CSV files in the evaluation/tourism/ directory
"""

import argparse
import logging
import os
import sys
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[1])
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


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Formula: MAPE = (100%/n) * sum(|predicted - actual| / |actual|)
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        MAPE value as a percentage
    """
    # Avoid division by zero by only including non-zero actual values
    non_zero_mask = actual != 0
    
    if np.any(non_zero_mask):
        return np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
    else:
        return np.nan


def calculate_mase(actual: np.ndarray, predicted: np.ndarray, history: np.ndarray, seasonality: int = 12) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).
    
    MASE = MAE / MAE_naive
    MAE_naive is calculated using the seasonal naive forecast (t-m)
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        history: Historical values used for calculating the naïve forecast denominator
        seasonality: Seasonality period (12 for monthly data)
        
    Returns:
        MASE value
    """
    mae = np.mean(np.abs(predicted - actual))
    
    if len(history) <= seasonality:
        # Not enough history for seasonal naïve
        naive_errors = np.abs(np.diff(history))
    else:
        # Calculate errors for the seasonal naïve forecast
        naive_predictions = history[:-seasonality]
        naive_actual = history[seasonality:]
        naive_errors = np.abs(naive_predictions - naive_actual)
    
    # Avoid division by zero
    if len(naive_errors) == 0 or np.mean(naive_errors) == 0:
        return np.nan
    
    return mae / np.mean(naive_errors)


def calculate_metrics(
    actual: np.ndarray, 
    predicted: np.ndarray, 
    series_id: str, 
    horizon: int,
    history: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate various error metrics for a forecast.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values (must be same length as actual)
        series_id: Identifier for the series
        horizon: Forecast horizon
        history: Historical values (for MASE calculation)
        
    Returns:
        Dictionary containing error metrics
    """
    # Ensure arrays are same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    # Handle empty arrays
    if min_len == 0:
        return {
            'series_id': series_id,
            'horizon': horizon,
            'mae': np.nan,
            'mape': np.nan,
            'smape': np.nan,
            'rmse': np.nan,
            'wmape': np.nan,
            'mase': np.nan
        }
    
    # Calculate MAE
    mae = np.mean(np.abs(predicted - actual))
    
    # Calculate MAPE
    mape = calculate_mape(actual, predicted)
    
    # Calculate SMAPE
    smape = calculate_smape(actual, predicted)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    
    # Calculate WMAPE (Weighted MAPE)
    wmape = np.sum(np.abs(predicted - actual)) / np.sum(np.abs(actual)) * 100 if np.sum(np.abs(actual)) > 0 else np.nan
    
    # Calculate MASE
    mase = np.nan
    if history is not None and len(history) > 0:
        mase = calculate_mase(actual, predicted, history, seasonality=12)
    
    return {
        'series_id': series_id,
        'horizon': horizon,
        'mae': mae,
        'mape': mape,
        'smape': smape,
        'rmse': rmse,
        'wmape': wmape,
        'mase': mase
    }


def setup_directories() -> None:
    """Create necessary directories for evaluation results."""
    os.makedirs("evaluation/tourism", exist_ok=True)
    os.makedirs("evaluation/tourism/plots", exist_ok=True)
    logger.info("Created evaluation directories")


def load_tourism_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Tourism training and test data.
    
    Returns:
        Tuple of (train_df, test_df)
    """
    train_path = Path("data/processed/tourism_monthly_dataset.csv")
    test_path = Path("data/processed/tourism_monthly_test.csv")
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Tourism data files not found. Please run convert_tourism_tsf_to_csv.py first."
        )
    
    train_df = pd.read_csv(train_path, parse_dates=['ds'])
    test_df = pd.read_csv(test_path, parse_dates=['ds'])
    
    logger.info(f"Loaded {len(train_df)} training data points")
    logger.info(f"Loaded {len(test_df)} test data points")
    
    # Count unique series
    train_series_count = train_df['unique_id'].nunique()
    test_series_count = test_df['unique_id'].nunique()
    
    logger.info(f"Training data contains {train_series_count} unique series")
    logger.info(f"Test data contains {test_series_count} unique series")
    
    return train_df, test_df


def sample_series(train_df: pd.DataFrame, test_df: pd.DataFrame, sample_size: int, random_seed: int = 42) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """
    Randomly sample series from the dataset.
    
    Args:
        train_df: Training data DataFrame
        test_df: Test data DataFrame
        sample_size: Number of series to sample
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (sampled_series_ids, sampled_train_df, sampled_test_df)
    """
    random.seed(random_seed)
    
    # Get all unique series IDs
    all_series_ids = train_df['unique_id'].unique().tolist()
    
    # Ensure we only include series that exist in both train and test sets
    test_series_ids = set(test_df['unique_id'].unique())
    valid_series_ids = [s_id for s_id in all_series_ids if s_id in test_series_ids]
    
    logger.info(f"Found {len(valid_series_ids)} series that exist in both train and test sets")
    
    # Sample series IDs
    if sample_size >= len(valid_series_ids):
        sampled_ids = valid_series_ids
        logger.warning(
            f"Sample size {sample_size} is larger than available series "
            f"({len(valid_series_ids)}). Using all series."
        )
    else:
        sampled_ids = random.sample(valid_series_ids, sample_size)
    
    # Filter dataframes to only include sampled series
    sampled_train_df = train_df[train_df['unique_id'].isin(sampled_ids)]
    sampled_test_df = test_df[test_df['unique_id'].isin(sampled_ids)]
    
    logger.info(f"Sampled {len(sampled_ids)} series for evaluation")
    
    return sampled_ids, sampled_train_df, sampled_test_df


def calculate_naive2_forecast(series: np.ndarray, forecast_horizon: int) -> np.ndarray:
    """
    Generate Naive2 forecasts (average of last 2 observations)
    
    Args:
        series: Array of historical values
        forecast_horizon: Number of periods to forecast
        
    Returns:
        Array of forecasts
    """
    if len(series) < 2:
        # Fall back to Naive if not enough data
        return np.repeat(series[-1], forecast_horizon)
    
    # Average last 2 observations
    last_value = np.mean(series[-2:])
    
    # Repeat for the forecast horizon
    return np.repeat(last_value, forecast_horizon)


def evaluate_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    series_ids: List[str],
    forecast_horizon: int,
    input_length: int,
    random_seed: int,
    log_transform: bool = False,
    include_naive2: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate the model on the sampled series.
    
    Args:
        model_name: Name of the model to evaluate
        train_df: DataFrame with training data
        test_df: DataFrame with test data
        series_ids: List of series IDs to evaluate
        forecast_horizon: Number of periods to forecast
        input_length: Length of input sequence for the model
        random_seed: Random seed for reproducibility
        log_transform: Whether to apply log transformation to the data
        include_naive2: Whether to include Naive2 benchmark (averaging last 2 values)
        
    Returns:
        Tuple of DataFrames with detailed and summary results
    """
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Initialize model
    logger.info(f"Initializing model {model_name}")
    model = TransformerModel(model_name=model_name)
    
    # Prepare results dataframes
    detailed_results = []
    naive2_results = [] if include_naive2 else None
    
    # Store original data for metrics calculation if using log transform
    original_train_df = train_df.copy() if log_transform else None
    original_test_df = test_df.copy() if log_transform else None
    
    # Apply log transformation if requested
    if log_transform:
        logger.info("Applying log transformation to the data")
        train_df = log_transform_series(train_df)
        test_df = log_transform_series(test_df)
    
    # Process each series
    for i, series_id in enumerate(tqdm(series_ids, desc="Evaluating series")):
        # Extract series data
        series_train = train_df[train_df['unique_id'] == series_id].copy()
        series_test = test_df[test_df['unique_id'] == series_id].copy()
        
        # Skip if not enough data
        if len(series_train) < input_length:
            logger.warning(f"Series {series_id} has insufficient data ({len(series_train)} < {input_length}), skipping")
            continue
        
        # Generate forecast
        try:
            # Convert 'ds' column to datetime if it's not already
            series_train['ds'] = pd.to_datetime(series_train['ds'])
            
            # Set the 'ds' column as the index before passing to the model
            series_train_indexed = series_train.set_index('ds')
            
            forecast = model.predict(
                series_df=series_train_indexed,
                n=forecast_horizon
            )
            
            # Apply inverse transformation if log transform was used
            if log_transform:
                forecast['q_0.5'] = inverse_log_transform(forecast['q_0.5'].values)
                
                # Get original scale data for metrics calculation
                series_test_orig = original_test_df[original_test_df['unique_id'] == series_id].copy()
            
            # Extract actual values from test set
            actual_values = []
            actual_dates = []
            for date in forecast['ds']:
                date_str = pd.to_datetime(date)
                actual_dates.append(date_str)
                matching_rows = series_test if not log_transform else series_test_orig
                matching_rows = matching_rows[matching_rows['ds'] == date_str]
                if not matching_rows.empty:
                    actual_values.append(matching_rows['y'].values[0])
                else:
                    actual_values.append(np.nan)
            
            # Calculate metrics for transformer model
            forecast_values = forecast['q_0.5'].values
            metrics = calculate_metrics(
                actual=np.array(actual_values),
                predicted=forecast_values,
                series_id=series_id,
                horizon=forecast_horizon,
                history=series_train['y'].values
            )
            detailed_results.append(metrics)
            
            # Calculate Naive2 benchmark if requested
            if include_naive2:
                # Use last 2 observations to calculate Naive2 forecast
                train_series = series_train['y'].values
                naive2_forecast = calculate_naive2_forecast(train_series, forecast_horizon)
                
                # Apply inverse transformation if log transform was used
                if log_transform:
                    naive2_forecast = inverse_log_transform(naive2_forecast)
                
                # Calculate metrics for Naive2
                naive2_metrics = calculate_metrics(
                    actual=np.array(actual_values),
                    predicted=naive2_forecast,
                    series_id=series_id,
                    horizon=forecast_horizon,
                    history=series_train['y'].values
                )
                naive2_metrics['method'] = 'naive2'
                naive2_results.append(naive2_metrics)
            
            # Plot forecast for every series
            plot_series_forecast(
                series_id=series_id,
                train_df=original_train_df if log_transform else train_df,
                test_df=original_test_df if log_transform else test_df,
                forecast=forecast,
                model_name=model_name,
                naive2_forecast=naive2_forecast if include_naive2 else None,
                actual_dates=actual_dates if include_naive2 else None
            )
                
        except Exception as e:
            logger.error(f"Error processing series {series_id}: {str(e)}")
            continue
    
    # Convert detailed results to DataFrame
    detailed_df = pd.DataFrame(detailed_results)
    
    # Add Naive2 results if included
    if include_naive2:
        naive2_df = pd.DataFrame(naive2_results)
        detailed_df['method'] = 'transformer'
        combined_df = pd.concat([detailed_df, naive2_df])
    else:
        combined_df = detailed_df
    
    # Calculate summary metrics
    summary_metrics = {}
    
    # Group by method if Naive2 is included
    if include_naive2:
        for method in ['transformer', 'naive2']:
            method_df = combined_df[combined_df['method'] == method]
            method_prefix = f"{method}_"
            for metric in ['mae', 'mape', 'smape', 'rmse', 'wmape', 'mase']:
                if metric in method_df.columns:
                    # Calculate mean
                    mean_value = method_df[metric].mean()
                    # Calculate median
                    median_value = method_df[metric].median()
                    # Store in summary dict
                    summary_metrics[f'{method_prefix}mean_{metric}'] = mean_value
                    summary_metrics[f'{method_prefix}median_{metric}'] = median_value
    else:
        for metric in ['mae', 'mape', 'smape', 'rmse', 'wmape', 'mase']:
            if metric in combined_df.columns:
                # Calculate mean
                mean_value = combined_df[metric].mean()
                # Calculate median
                median_value = combined_df[metric].median()
                # Store in summary dict
                summary_metrics[f'mean_{metric}'] = mean_value
                summary_metrics[f'median_{metric}'] = median_value
    
    # Convert summary metrics to DataFrame
    summary_df = pd.DataFrame([summary_metrics])
    
    return combined_df, summary_df


def plot_series_forecast(
    series_id: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast: pd.DataFrame,
    model_name: str,
    naive2_forecast: np.ndarray = None,
    actual_dates: List[datetime] = None
) -> None:
    """
    Plot the forecast for a single series.
    
    Args:
        series_id: Identifier for the series
        train_df: DataFrame with training data
        test_df: DataFrame with test data
        forecast: DataFrame with forecast data
        model_name: Name of the model used
        naive2_forecast: Optional array with Naive2 forecasts
        actual_dates: Optional list of datetime objects for the forecast dates
    """
    # Filter data for this series
    series_train = train_df[train_df['unique_id'] == series_id].copy()
    series_test = test_df[test_df['unique_id'] == series_id].copy()
    
    # Convert dates to datetime if needed
    series_train['ds'] = pd.to_datetime(series_train['ds'])
    series_test['ds'] = pd.to_datetime(series_test['ds'])
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.plot(series_train['ds'], series_train['y'], 'b-', label='Training')
    
    # Plot test data
    plt.plot(series_test['ds'], series_test['y'], 'g-', label='Actual')
    
    # Plot forecast
    plt.plot(forecast['ds'], forecast['q_0.5'], 'r--', label='Transformer Forecast')
    
    # Plot Naive2 forecast if provided
    if naive2_forecast is not None and actual_dates is not None:
        plt.plot(actual_dates, naive2_forecast, 'm:', label='Naive2 Forecast')
    
    # Add vertical line at the end of training data
    if not series_train.empty:
        last_train_date = series_train['ds'].max()
        plt.axvline(x=last_train_date, color='k', linestyle='--', alpha=0.5)
    
    # Add labels
    plt.title(f'Series: {series_id}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = f"evaluation/tourism/plots/{model_name}_{series_id}.png"
    plt.savefig(plot_path)
    plt.close()


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


def save_results(
    detailed_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    model_name: str,
    sample_size: int,
    include_naive2: bool = False
) -> None:
    """
    Save evaluation results to CSV files.
    
    Args:
        detailed_df: DataFrame with detailed results for each series
        summary_df: DataFrame with summary metrics
        model_name: Name of the model used for evaluation
        sample_size: Number of series sampled for evaluation
        include_naive2: Whether Naive2 benchmark was included
    """
    # Create timestamp for filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    detailed_path = f"evaluation/tourism/{model_name}_{sample_size}series_detailed_{timestamp}.csv"
    detailed_df.to_csv(detailed_path, index=False)
    logger.info(f"Saved detailed results to {detailed_path}")
    
    # Calculate the weighted average MAPE as done in the paper
    # Equation (6): MAPE_Average = (N_Year/N_Tot * MAPE_Year + N_Quart/N_Tot * MAPE_Quart + N_Month/N_Tot * MAPE_Month)
    # Since we're only using monthly data, this simplifies to just the monthly MAPE
    
    if include_naive2:
        transformer_df = detailed_df[detailed_df['method'] == 'transformer']
        naive2_df = detailed_df[detailed_df['method'] == 'naive2']
        
        if 'mape' in transformer_df.columns and 'mape' in naive2_df.columns:
            transformer_mape = transformer_df['mape'].mean()
            naive2_mape = naive2_df['mape'].mean()
            improvement = (naive2_mape - transformer_mape) / naive2_mape * 100 if naive2_mape > 0 else 0
            
            logger.info(f"TOURISM Monthly MAPE - Transformer (366 series, 24 horizon): {transformer_mape:.2f}")
            logger.info(f"TOURISM Monthly MAPE - Naive2 (366 series, 24 horizon): {naive2_mape:.2f}")
            logger.info(f"Improvement over Naive2: {improvement:.2f}%")
    else:
        if 'mape' in detailed_df.columns:
            monthly_mape = detailed_df['mape'].mean()
            logger.info(f"TOURISM Monthly MAPE (366 series, 24 horizon): {monthly_mape:.2f}")
    
    # Save summary results
    summary_path = f"evaluation/tourism/{model_name}_{sample_size}series_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary results to {summary_path}")
    
    # Also save as Excel for easier viewing
    excel_path = f"evaluation/tourism/{model_name}_{sample_size}series_results_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        detailed_df.to_excel(writer, sheet_name='Detailed', index=False)
    logger.info(f"Saved combined results to {excel_path}")


def plot_summary_metrics(
    summary_df: pd.DataFrame, 
    model_name: str, 
    sample_size: int,
    include_naive2: bool = False
) -> None:
    """
    Create bar plots of summary metrics.
    
    Args:
        summary_df: DataFrame with summary metrics
        model_name: Name of the model used for evaluation
        sample_size: Number of series sampled for evaluation
        include_naive2: Whether Naive2 benchmark was included
    """
    # Create timestamp for filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if include_naive2:
        # Create comparison plots between transformer and Naive2
        metrics = ['mape', 'smape', 'rmse', 'mae', 'wmape', 'mase']
        display_names = ['MAPE', 'SMAPE', 'RMSE', 'MAE', 'WMAPE', 'MASE']
        
        for metric, display_name in zip(metrics, display_names):
            transformer_mean = summary_df.get(f'transformer_mean_{metric}', None)
            naive2_mean = summary_df.get(f'naive2_mean_{metric}', None)
            
            if transformer_mean is not None and naive2_mean is not None:
                plt.figure(figsize=(8, 6))
                methods = ['Transformer', 'Naive2']
                values = [transformer_mean.values[0], naive2_mean.values[0]]
                
                bars = plt.bar(methods, values)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2.,
                        height,
                        f'{height:.4f}',
                        ha='center',
                        va='bottom',
                        rotation=0
                    )
                
                # Add labels and title
                plt.title(f'{display_name} Comparison for {model_name}')
                plt.ylabel(display_name)
                plt.grid(True, alpha=0.3, axis='y')
                
                # Save the plot
                filename = f'evaluation/tourism/{model_name}_{sample_size}series_{metric}_comparison_{timestamp}.png'
                plt.savefig(filename)
                plt.close()
                
                logger.info(f"Saved {metric} comparison plot to {filename}")
    else:
        # Define metrics to plot (with 'mean_' prefix)
        metrics = ['mean_mae', 'mean_mape', 'mean_smape', 'mean_rmse', 'mean_wmape', 'mean_mase']
        display_names = ['MAE', 'MAPE', 'SMAPE', 'RMSE', 'WMAPE', 'MASE']
        
        # Filter to only include metrics that exist in the DataFrame
        available_metrics = [m for m in metrics if m in summary_df.columns]
        display_names = [display_names[metrics.index(m)] for m in available_metrics]
        
        if not available_metrics:
            logger.warning("No metrics available for plotting")
            return
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Create bar plot
        bars = plt.bar(display_names, summary_df[available_metrics].values[0])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.4f}',
                ha='center',
                va='bottom',
                rotation=0
            )
        
        # Add labels and title
        plt.title(f'Summary Metrics for {model_name} ({sample_size} series)')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        filename = f'evaluation/tourism/{model_name}_{sample_size}series_metrics_{timestamp}.png'
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"Saved summary metrics plot to {filename}")


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate the transformer model on Tourism data')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name of the model to evaluate (should be a directory in models/final/)')
    parser.add_argument('--sample-size', type=int, default=366,
                        help='Number of series to sample for evaluation')
    parser.add_argument('--forecast-horizon', type=int, default=24,
                        help='Number of periods to forecast (24 for monthly as per the paper)')
    parser.add_argument('--input-length', type=int, default=60,
                        help='Length of input sequence for the model')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log-transform', action='store_true',
                        help='Apply log transformation to the data before forecasting (to handle increasing variance)')
    parser.add_argument('--include-naive2', action='store_true',
                        help='Include Naive2 benchmark (average of last 2 values)')
    args = parser.parse_args()
    
    # Create necessary directories
    setup_directories()
    
    # Load data
    train_df, test_df = load_tourism_data()
    
    # For TOURISM monthly dataset, paper reports 366 series
    logger.info(f"Paper reports 366 monthly series with forecast horizon of 24")
    
    # Sample series
    series_ids, sampled_train_df, sampled_test_df = sample_series(
        train_df=train_df,
        test_df=test_df,
        sample_size=args.sample_size,
        random_seed=args.random_seed
    )
    
    # Evaluate model
    detailed_df, summary_df = evaluate_model(
        model_name=args.model_name,
        train_df=sampled_train_df,
        test_df=sampled_test_df,
        series_ids=series_ids,
        forecast_horizon=args.forecast_horizon,
        input_length=args.input_length,
        random_seed=args.random_seed,
        log_transform=args.log_transform,
        include_naive2=args.include_naive2
    )
    
    # Save results
    save_results(
        detailed_df=detailed_df,
        summary_df=summary_df,
        model_name=args.model_name,
        sample_size=args.sample_size,
        include_naive2=args.include_naive2
    )
    
    # Plot summary metrics
    plot_summary_metrics(
        summary_df=summary_df,
        model_name=args.model_name,
        sample_size=args.sample_size,
        include_naive2=args.include_naive2
    )
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main() 