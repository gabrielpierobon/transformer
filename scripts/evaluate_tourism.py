#!/usr/bin/env python
"""
Script to evaluate the transformer model on the Tourism monthly dataset.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pytorch_lightning as pl
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.losses.pytorch import DistributionLoss, MAE, MAPE
from neuralforecast.tsdataset import TimeSeriesDataset

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


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    non_zero_mask = actual != 0
    if np.any(non_zero_mask):
        return np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
    return np.nan


def calculate_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denominator > 0
    if np.any(mask):
        return np.mean(np.abs(predicted[mask] - actual[mask]) / denominator[mask]) * 100
        return np.nan


def calculate_naive2_forecast(values: np.ndarray, horizon: int) -> np.ndarray:
    """Generate Naive2 forecast (average of last 2 observations)."""
    if len(values) < 2:
        return np.repeat(values[-1], horizon)
    return np.repeat(np.mean(values[-2:]), horizon)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate various error metrics."""
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    if min_len == 0:
        return {
            'mae': np.nan,
            'mape': np.nan,
            'smape': np.nan,
            'rmse': np.nan
        }
    
    mae = np.mean(np.abs(predicted - actual))
    mape = calculate_mape(actual, predicted)
    smape = calculate_smape(actual, predicted)
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    
    return {
        'mae': mae,
        'mape': mape,
        'smape': smape,
        'rmse': rmse
    }


def load_tourism_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Tourism training and test data."""
    train_path = Path("data/processed/tourism_monthly_dataset.csv")
    test_path = Path("data/processed/tourism_monthly_test.csv")
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Tourism data files not found. Please run create_tourism_dataset.py first."
        )
    
    train_df = pd.read_csv(train_path, parse_dates=['ds'])
    test_df = pd.read_csv(test_path, parse_dates=['ds'])
    
    logger.info(f"Loaded {len(train_df)} training records")
    logger.info(f"Loaded {len(test_df)} test records")
    
    return train_df, test_df


def train_global_nbeats_model(train_df: pd.DataFrame, forecast_horizon: int) -> NeuralForecast:
    """Train a global NBEATS model on all series at once.
    
    Args:
        train_df: DataFrame with 'unique_id', 'ds', and 'y' columns for all series
        forecast_horizon: Number of steps to forecast
        
    Returns:
        Trained NeuralForecast model with NBEATS
    """
    logger.info("Training global NBEATS model on all series...")
    
    # Initialize NBEATS model with early stopping
    nbeats_model = NBEATS(
        h=forecast_horizon,             # Forecast horizon
        input_size=60,                  # Keep input window size at 60
        max_steps=3000,                 # Maximum number of training steps
        early_stop_patience_steps=50,   # Stop after 10 validation steps without improvement
        val_check_steps=100,             # Check validation every 20 steps
        learning_rate=0.0001,            # Learning rate
        loss=MAPE(),                    # Use MAPE loss
        #loss=MAE(),                     # Use MAE loss
        random_seed=42                  # For reproducibility
    )
    
    # Create NeuralForecast model
    nf = NeuralForecast(
        models=[nbeats_model],
        freq='M'  # Monthly frequency
    )
    
    try:
        # Train the model with early stopping
        nf.fit(df=train_df, val_size=24)  # Use 24 months (2 years) for validation
        #nf.fit(df=train_df)
        logger.info("Global NBEATS model training complete")
        return nf
    except Exception as e:
        logger.error(f"Error training global NBEATS model: {str(e)}")
        raise


def plot_forecast(
    series_id: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast: pd.DataFrame,
    nbeats_forecast: pd.DataFrame,
    naive2_forecast: np.ndarray,
    model_name: str,
    was_log_transformed: bool = False
) -> None:
    """Plot and save forecast for a single series."""
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.plot(train_df['ds'], train_df['y'], 'b-', label='Training', alpha=0.7)
    
    # Highlight the last 60 points used for prediction
    if len(train_df) >= 60:
        inference_data = train_df.iloc[-60:]
        plt.plot(inference_data['ds'], inference_data['y'], 'b-', alpha=1.0)
        plt.axvspan(
            inference_data['ds'].iloc[0],
            inference_data['ds'].iloc[-1],
            color='lightblue',
            alpha=0.3,
            label='Last 60 Points'
        )
    
    # Plot actuals from test set
    actuals = []
    for date in forecast['ds']:
        date_str = pd.to_datetime(date)
        matching_rows = test_df[test_df['ds'] == date_str]
        if not matching_rows.empty:
            actuals.append((date_str, matching_rows.iloc[0]['y']))
    
    if actuals:
        dates, values = zip(*actuals)
        plt.plot(dates, values, 'g-', label='Actual', linewidth=2.5)
    
    # Plot Transformer forecast
    transformer_label = 'Transformer'
    if was_log_transformed:
        transformer_label += ' (log transformed)'
    plt.plot(forecast['ds'], forecast['q_0.5'], 'r--', label=transformer_label, linewidth=1.5)
    
    # Plot NBEATS forecast if available
    if nbeats_forecast is not None and not nbeats_forecast.empty:
        nbeats_column = None
        
        # Try to find a usable column for NBEATS forecast
        if 'NBEATS' in nbeats_forecast.columns:
            nbeats_column = 'NBEATS'
    else:
            # Look for any numeric column that's not ds or unique_id
            for col in nbeats_forecast.columns:
                if col != 'ds' and col != 'unique_id' and pd.api.types.is_numeric_dtype(nbeats_forecast[col]):
                    nbeats_column = col
                    break
        
        # Only plot if we found a usable column
        if nbeats_column is not None:
            plt.plot(nbeats_forecast['ds'], nbeats_forecast[nbeats_column], 'c--', 
                    label='NBEATS (global)', linewidth=1.5)
    
    # Plot Naive2 forecast
    forecast_dates = [pd.to_datetime(d) for d in forecast['ds']]
    plt.plot(forecast_dates, naive2_forecast, 'k:', label='Naive2', linewidth=1.5)
    
    # Add a vertical line to separate train and test
    if len(train_df) > 0:
        last_train_date = train_df['ds'].iloc[-1]
        plt.axvline(x=last_train_date, color='gray', linestyle='--', linewidth=1.5)
    
    # Add grid, title, and labels
    plt.grid(True, alpha=0.3)
    plt.title(f'Series: {series_id}' + (' (Log Transformed)' if was_log_transformed else ''), fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(loc='upper left')
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plot_dir = Path(f"evaluation/tourism/plots/{model_name}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f"{series_id}.png")
    plt.close()


def should_log_transform(series: pd.DataFrame, inference_window: int = 60) -> bool:
    """Determine if the series should be log-transformed based on increasing variance in inference window.
    
    Args:
        series: DataFrame with 'ds' and 'y' columns
        inference_window: Number of recent points to analyze (default: 60)
        
    Returns:
        Boolean indicating whether log transformation should be applied
    """
    # Make sure we have enough data for the test
    if len(series) < inference_window or inference_window < 10:
        return False
    
    # Get the last inference_window points
    y_values = series['y'].values[-inference_window:]
    
    # Check 1: Test for variance increase using first half vs second half of the window
    half_size = inference_window // 2
    first_half_std = np.std(y_values[:half_size])
    second_half_std = np.std(y_values[half_size:])
    
    # Calculate variance ratio
    variance_ratio = second_half_std / first_half_std if first_half_std > 0 else 1.0
    
    # Check 2: Test correlation between level and volatility
    indices = np.arange(len(y_values))
    rolling_std = pd.Series(y_values).rolling(window=5).std().fillna(method='bfill')
    level_vol_corr = np.corrcoef(y_values, rolling_std)[0, 1]
    
    # Check 3: Test if data increases in magnitude significantly
    first_quarter_mean = np.mean(y_values[:inference_window//4]) if inference_window >= 4 else np.mean(y_values[:5])
    last_quarter_mean = np.mean(y_values[-inference_window//4:]) if inference_window >= 4 else np.mean(y_values[-5:])
    level_increase_ratio = last_quarter_mean / first_quarter_mean if first_quarter_mean > 0 else 1.0
    
    # Check 4: Check range ratio (max/min) to detect high amplitude seasonality
    range_ratio = np.max(y_values) / np.min(y_values) if np.min(y_values) > 0 else 1.0
    
    # Check 5: Calculate coefficient of variation (CV) to measure relative dispersion
    cv = np.std(y_values) / np.mean(y_values) if np.mean(y_values) > 0 else 0
    
    # Decision rule: Apply log transform if:
    # 1. Strong correlation between level and volatility (primary indicator)
    # 2. Either variance is increasing OR level is increasing OR range is wide OR CV is high
    logger.info(f"Variance test results - Ratio: {variance_ratio:.2f}, Level-Vol Corr: {level_vol_corr:.2f}, " +
                f"Level Increase: {level_increase_ratio:.2f}, Range Ratio: {range_ratio:.2f}, CV: {cv:.2f}")
    
    should_transform = (
        level_vol_corr > 0.3 and  # Strong correlation between level and volatility
        (
            variance_ratio > 1.1 or  # Slightly increasing variance (reduced threshold)
            level_increase_ratio > 1.05 or  # Slightly increasing level (reduced threshold)
            range_ratio > 2.5 or  # Wide range between min and max values
            cv > 0.3  # High coefficient of variation
        )
    )
    
    if should_transform:
        logger.info(f"Log transform recommended for series based on: " +
                    f"Level-Vol Corr={level_vol_corr:.2f}, CV={cv:.2f}, Range Ratio={range_ratio:.2f}")
    
    return should_transform


def apply_log_transform(series: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Apply log transformation to series if positive.
    
    Args:
        series: DataFrame with 'ds' and 'y' columns
        
    Returns:
        Tuple of (transformed DataFrame, was_transformed flag)
    """
    # Check if all values are positive (required for log transform)
    if np.all(series['y'] > 0):
        transformed = series.copy()
        transformed['y'] = np.log(transformed['y'])
        return transformed, True
    else:
        # Log warning that log transform couldn't be applied due to non-positive values
        logger.warning("Log transform not applied: series contains zero or negative values")
        return series.copy(), False


def evaluate_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    series_ids: List[str],
    forecast_horizon: int = 24,
    specific_series: str = None
) -> None:
    """Evaluate the model on tourism data using global forecasting approach."""
    # Initialize transformer model
    logger.info(f"Loading transformer model {model_name}")
    transformer_model = TransformerModel(model_name=model_name)
    
    # Start timing
    start_time = time.time()
    
    # Filter for specific series if provided
    if specific_series:
        if specific_series not in series_ids:
            raise ValueError(f"Series {specific_series} not found in dataset")
        series_to_evaluate = [specific_series]
        logger.info(f"Evaluating only series {specific_series}")
    else:
        series_to_evaluate = series_ids
        logger.info(f"Evaluating {len(series_to_evaluate)} series")
    
    # Filter the train and test dataframes to include only the series we're evaluating
    filtered_train_df = train_df[train_df['unique_id'].isin(series_to_evaluate)]
    filtered_test_df = test_df[test_df['unique_id'].isin(series_to_evaluate)]
    
    # Report on the data we're using
    logger.info(f"Using {len(filtered_train_df)} training records and {len(filtered_test_df)} test records")
    
    # Prepare results storage
    results = []
    
    # Create output directory
    os.makedirs("evaluation/tourism", exist_ok=True)
    excel_path = f"evaluation/tourism/{model_name}_evaluation.xlsx"
    
    # Train global NBEATS model on all series at once
    global_nbeats_model = None
    try:
        global_nbeats_model = train_global_nbeats_model(filtered_train_df, forecast_horizon)
        logger.info("Global NBEATS model training complete")
    except Exception as e:
        logger.error(f"Failed to train global NBEATS model: {str(e)}")
        logger.warning("Will proceed with evaluation using only Transformer and Naive2 models")
    
    # Process each series for transformer model and naive2 forecasts
    for series_idx, series_id in enumerate(tqdm(series_to_evaluate, desc="Evaluating series")):
        try:
            # Get series data
            series_train = filtered_train_df[filtered_train_df['unique_id'] == series_id]
            series_test = filtered_test_df[filtered_test_df['unique_id'] == series_id]
            
            if len(series_train) == 0:
                logger.warning(f"No training data found for series {series_id}, skipping")
                continue
                
            if len(series_test) == 0:
                logger.warning(f"No test data found for series {series_id}, skipping")
                continue
            
            # Save original data for plotting
            original_series_train = series_train.copy()
            
            # Check if log transformation is needed for Transformer (based on variance pattern)
            needs_log_transform = should_log_transform(series_train)
            was_transformed = False
            
            # Generate transformer forecast
            if needs_log_transform:
                logger.info(f"Applying log transform for series {series_id} (Transformer model only)")
                transformed_series, was_transformed = apply_log_transform(series_train)
                transformed_series_indexed = transformed_series.set_index('ds')
                
                # Generate forecast in log space
                log_forecast = transformer_model.predict(
                    series_df=transformed_series_indexed,
                    n=forecast_horizon
                )
                
                # Back-transform forecast
                if was_transformed:
                    for col in log_forecast.columns:
                        if col != 'ds' and pd.api.types.is_numeric_dtype(log_forecast[col]):
                            log_forecast[col] = np.exp(log_forecast[col])
                
                forecast = log_forecast
            else:
                # No transformation needed
                logger.info(f"No log transform needed for series {series_id}")
                series_train_indexed = series_train.set_index('ds')
                forecast = transformer_model.predict(
                    series_df=series_train_indexed,
                    n=forecast_horizon
                )
            
            # Generate Naive2 forecast
            naive2_forecast = calculate_naive2_forecast(
                values=series_train['y'].values,
                horizon=forecast_horizon
            )
            
            # Store the results for this series
            results.append({
                'series_id': series_id,
                'log_transformed': was_transformed,
                'transformer_forecast': forecast,
                'naive2_forecast': naive2_forecast,
                'actual_values': series_test
            })
            
            # Log progress
            logger.info(f"Completed Transformer & Naive2 for series {series_idx+1}/{len(series_to_evaluate)} ({(series_idx+1)/len(series_to_evaluate)*100:.1f}%)")
                
        except Exception as e:
            # Convert exception to string to handle any type of error object
            error_message = f"{type(e).__name__}: {e}"
            logger.error(f"Error processing series {series_id}: {error_message}")
            continue
    
    # Generate global NBEATS forecasts for all series at once
    nbeats_forecasts = {}
    if global_nbeats_model is not None:
        try:
            # Generate forecasts for all series at once
            all_nbeats_forecasts = predict_global_nbeats(global_nbeats_model, filtered_train_df)
            
            # Extract forecasts for each series
            for series_id in series_to_evaluate:
                series_forecast = all_nbeats_forecasts[all_nbeats_forecasts['unique_id'] == series_id]
                if len(series_forecast) > 0:
                    nbeats_forecasts[series_id] = series_forecast
    else:
                    logger.warning(f"No NBEATS forecast generated for series {series_id}")
                    
            logger.info(f"Generated NBEATS forecasts for {len(nbeats_forecasts)} series")
        except Exception as e:
            logger.error(f"Error generating global NBEATS forecasts: {str(e)}")
    
    # Calculate metrics and generate plots for all models
    final_results = []
    for result in results:
        series_id = result['series_id']
        transformer_forecast = result['transformer_forecast']
        naive2_forecast = result['naive2_forecast']
        actual_values_df = result['actual_values']
        was_transformed = result['log_transformed']
        
        # Extract actual values as array
        actual_values = []
        for date in transformer_forecast['ds']:
            date_str = pd.to_datetime(date)
            matching_rows = actual_values_df[actual_values_df['ds'] == date_str]
            if not matching_rows.empty:
                actual_values.append(matching_rows.iloc[0]['y'])
    else:
                actual_values.append(np.nan)
        actual_values = np.array(actual_values)
        
        # Get NBEATS forecast for this series if available
        nbeats_forecast = None
        if series_id in nbeats_forecasts:
            nbeats_forecast = nbeats_forecasts[series_id]
            nbeats_column = None
            if 'NBEATS' in nbeats_forecast.columns:
                nbeats_column = 'NBEATS'
            else:
                for col in nbeats_forecast.columns:
                    if col != 'ds' and col != 'unique_id' and pd.api.types.is_numeric_dtype(nbeats_forecast[col]):
                        nbeats_column = col
                        break
            
            if nbeats_column is not None:
                nbeats_metrics = calculate_metrics(actual_values, nbeats_forecast[nbeats_column].values)
            else:
                logger.warning(f"No NBEATS forecast column found for series {series_id}")
                nbeats_metrics = {
                    'mae': np.nan,
                    'mape': np.nan,
                    'smape': np.nan,
                    'rmse': np.nan
                }
        else:
            nbeats_metrics = {
                'mae': np.nan,
                'mape': np.nan,
                'smape': np.nan,
                'rmse': np.nan
            }
        
        # Calculate metrics for transformer and naive2
        transformer_metrics = calculate_metrics(actual_values, transformer_forecast['q_0.5'].values)
        naive2_metrics = calculate_metrics(actual_values, naive2_forecast)
        
        # Store results
        final_results.append({
            'series_id': series_id,
            'log_transformed': was_transformed,
            'transformer_mae': transformer_metrics['mae'],
            'transformer_mape': transformer_metrics['mape'],
            'transformer_smape': transformer_metrics['smape'],
            'transformer_rmse': transformer_metrics['rmse'],
            'nbeats_mae': nbeats_metrics['mae'],
            'nbeats_mape': nbeats_metrics['mape'],
            'nbeats_smape': nbeats_metrics['smape'],
            'nbeats_rmse': nbeats_metrics['rmse'],
            'naive2_mae': naive2_metrics['mae'],
            'naive2_mape': naive2_metrics['mape'],
            'naive2_smape': naive2_metrics['smape'],
            'naive2_rmse': naive2_metrics['rmse']
        })
        
        # Plot results
        series_train = filtered_train_df[filtered_train_df['unique_id'] == series_id]
        if series_id in nbeats_forecasts:
            plot_forecast(
                series_id=series_id,
                train_df=series_train,
                test_df=actual_values_df,
                forecast=transformer_forecast,
                nbeats_forecast=nbeats_forecasts[series_id],
                naive2_forecast=naive2_forecast,
                model_name=model_name,
                was_log_transformed=was_transformed
            )
    
    # Create final results DataFrame
    results_df = pd.DataFrame(final_results)
    
    # Calculate summary statistics
    summary_stats = {
        'Metric': ['MAPE', 'SMAPE', 'MAE', 'RMSE'],
        'Transformer': [
            results_df['transformer_mape'].mean(),
            results_df['transformer_smape'].mean(),
            results_df['transformer_mae'].mean(),
            results_df['transformer_rmse'].mean()
        ],
        'NBEATS (global)': [
            results_df['nbeats_mape'].mean(),
            results_df['nbeats_smape'].mean(),
            results_df['nbeats_mae'].mean(),
            results_df['nbeats_rmse'].mean()
        ],
        'Naive2': [
            results_df['naive2_mape'].mean(),
            results_df['naive2_smape'].mean(),
            results_df['naive2_mae'].mean(),
            results_df['naive2_rmse'].mean()
        ]
    }
    summary_df = pd.DataFrame(summary_stats)
    
    # Save results to Excel
    with pd.ExcelWriter(excel_path) as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        results_df.to_excel(writer, sheet_name='Detailed', index=False)
    
    logger.info(f"Results saved to {excel_path}")
    
    # Log summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"Number of series evaluated: {len(results_df)}")
    
    logger.info("\nMean MAPE:")
    logger.info(f"Transformer: {results_df['transformer_mape'].mean():.2f}%")
    logger.info(f"NBEATS (global): {results_df['nbeats_mape'].mean():.2f}%")
    logger.info(f"Naive2: {results_df['naive2_mape'].mean():.2f}%")
    logger.info("\nMean SMAPE:")
    logger.info(f"Transformer: {results_df['transformer_smape'].mean():.2f}%")
    logger.info(f"NBEATS (global): {results_df['nbeats_smape'].mean():.2f}%")
    logger.info(f"Naive2: {results_df['naive2_smape'].mean():.2f}%")


def predict_global_nbeats(model: NeuralForecast, test_series: pd.DataFrame) -> pd.DataFrame:
    """Generate forecasts for all series at once using the global NBEATS model.
    
    Args:
        model: Trained NeuralForecast model
        test_series: DataFrame containing all test series
        
    Returns:
        DataFrame with forecasts for all series
    """
    logger.info("Generating forecasts using global NBEATS model...")
    
    try:
        # Use the correct predict method with the DataFrame directly
        forecasts = model.predict(df=test_series)
        logger.info(f"Generated forecasts for {len(forecasts['unique_id'].unique())} series")
        return forecasts
    except Exception as e:
        logger.error(f"Error generating forecasts with global NBEATS model: {str(e)}")
        raise


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate transformer model on Tourism data')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Name of the model to evaluate')
    parser.add_argument('--series-id', type=str,
                       help='Optional: evaluate only this specific series')
    parser.add_argument('--series-list', type=str,
                       help='Optional: comma-separated list of series IDs to evaluate (e.g., "T1,T2,T3")')
    args = parser.parse_args()
    
    # Load data
    train_df, test_df = load_tourism_data()
    
    # Get unique series IDs
    all_series_ids = train_df['unique_id'].unique().tolist()
    
    # Determine which series to evaluate
    if args.series_list:
        # Parse the comma-separated list
        series_ids_to_evaluate = [s.strip() for s in args.series_list.split(',')]
        # Verify all specified series exist in the dataset
        for series_id in series_ids_to_evaluate:
            if series_id not in all_series_ids:
                logger.warning(f"Series {series_id} not found in dataset, will be skipped")
        # Only keep valid series IDs
        series_ids_to_evaluate = [s for s in series_ids_to_evaluate if s in all_series_ids]
        if not series_ids_to_evaluate:
            logger.error("None of the specified series were found in the dataset")
            return
        logger.info(f"Evaluating {len(series_ids_to_evaluate)} series: {', '.join(series_ids_to_evaluate)}")
        
        # Run evaluation with the specified list of series
        evaluate_model(
            model_name=args.model_name,
        train_df=train_df,
        test_df=test_df,
            series_ids=series_ids_to_evaluate,
            specific_series=None  # Not using specific_series since we're using series_ids
        )
    elif args.series_id:
        # Single series evaluation
        if args.series_id not in all_series_ids:
            logger.error(f"Series {args.series_id} not found in dataset")
            return
        logger.info(f"Evaluating single series: {args.series_id}")
        
        # Run evaluation with the specific series
        evaluate_model(
        model_name=args.model_name,
            train_df=train_df,
            test_df=test_df,
            series_ids=all_series_ids,
            specific_series=args.series_id
        )
    else:
        # Evaluate all series
        logger.info(f"Evaluating all {len(all_series_ids)} series")
        
        # Run evaluation with all series
        evaluate_model(
        model_name=args.model_name,
            train_df=train_df,
            test_df=test_df,
            series_ids=all_series_ids,
            specific_series=None
    )
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main() 