#!/usr/bin/env python
"""
Script to evaluate the transformer model on the Tourism monthly dataset.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.losses.pytorch import DistributionLoss, MAE

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


def detect_and_clean_outliers(time_series: pd.DataFrame, threshold: float = 8.0, focus_last_n: int = 60) -> Tuple[pd.DataFrame, np.ndarray]:
    """Detect and replace outliers in a time series.
    
    Args:
        time_series: DataFrame with 'ds' and 'y' columns
        threshold: Z-score threshold to identify outliers (default: 8.0)
        focus_last_n: Focus outlier detection on the last N points (default: 60)
        
    Returns:
        Tuple of (cleaned DataFrame with outliers replaced, numpy array of outlier indices)
    """
    try:
        # Create a copy to avoid modifying the original data
        cleaned_series = time_series.copy()
        
        # A seasonal pattern is normal in tourism data
        # Only flag extremely unusual values (like drops to near zero or massive irregular spikes)
        
        # Calculate year-over-year ratios where possible (tourism data is often seasonal)
        # This helps detect when a value is extreme compared to the same month in previous years
        if len(cleaned_series) >= 12:
            # Calculate the ratio to the same month in the previous year
            monthly_values = cleaned_series['y'].values
            yearly_ratios = []
            for i in range(12, len(monthly_values)):
                yearly_ratios.append(monthly_values[i] / max(monthly_values[i-12], 1.0))  # Avoid division by zero
            
            # Pad the beginning where we couldn't calculate ratios
            yearly_ratios = [1.0] * 12 + yearly_ratios
            
            # Identify extreme ratios (more than 2x increase or less than 50% of previous year)
            extreme_yearly_ratios = []
            for i in range(len(yearly_ratios)):
                ratio = yearly_ratios[i]
                if ratio > 2.0 or ratio < 0.5:
                    # Only flag if it's not part of a consistent pattern
                    if i >= 24:
                        # Check if this is a recurring pattern (seasonal changes can be large but consistent)
                        prev_ratio = yearly_ratios[i-12]
                        if abs(ratio - prev_ratio) > 0.5:  # If ratio differs by more than 50% from previous year's ratio
                            extreme_yearly_ratios.append(i)
                    else:
                        extreme_yearly_ratios.append(i)
        else:
            extreme_yearly_ratios = []
        
        # Calculate rolling median and std with a very large window (global approach)
        window_size = min(len(cleaned_series) // 4, 24)  # Use up to 2 years of data, but at least 25% of time series
        window_size = max(window_size, 5)  # Ensure at least 5 points for statistics
            
        rolling_median = cleaned_series['y'].rolling(window=window_size, center=True).median()
        rolling_median = rolling_median.fillna(method='ffill').fillna(method='bfill')
        
        rolling_std = cleaned_series['y'].rolling(window=window_size, center=True).std()
        rolling_std = rolling_std.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure non-zero standard deviation
        min_std = cleaned_series['y'].std() * 0.1 if cleaned_series['y'].std() > 0 else 1.0
        rolling_std = rolling_std.clip(lower=min_std)
        
        # Calculate z-scores
        z_scores = (cleaned_series['y'] - rolling_median) / rolling_std
        
        # Only flag points with extremely high z-scores (truly exceptional points)
        outliers_zscore = abs(z_scores) > threshold
        
        # Flag huge drops or spikes (80%+ change in a single month)
        pct_changes = cleaned_series['y'].pct_change().fillna(0)
        extreme_monthly_changes = abs(pct_changes) > 0.8
        
        # Combine outlier detection methods
        outliers = np.zeros(len(cleaned_series), dtype=bool)
        
        # Convert extreme yearly ratios to boolean mask
        for idx in extreme_yearly_ratios:
            if idx < len(outliers):
                outliers[idx] = True
        
        # Combine with other methods
        outliers = outliers | outliers_zscore.values | extreme_monthly_changes.values
        
        # Only consider the focus window if specified
        if focus_last_n < len(outliers):
            focus_mask = np.zeros(len(outliers), dtype=bool)
            focus_mask[-focus_last_n:] = True
            outliers = outliers & focus_mask
        
        outlier_indices = np.where(outliers)[0]
        
        # Replace detected outliers with the rolling median
        if len(outlier_indices) > 0:
            cleaned_series.loc[outliers, 'y'] = rolling_median[outliers]
        
            # Log detected outliers
            logger.info(f"Detected {len(outlier_indices)} outliers in series {time_series['unique_id'].iloc[0]}")
            for idx in outlier_indices:
                if idx < len(time_series):
                    logger.info(f"Outlier at index {idx}: value={time_series['y'].iloc[idx]}, "
                               f"date={time_series['ds'].iloc[idx]}, z-score={z_scores.iloc[idx]:.2f}")
        
        return cleaned_series, outlier_indices
    
    except Exception as e:
        logger.error(f"Error in outlier detection: {type(e).__name__}: {e}")
        # Return original series and empty outlier list on error
        return time_series.copy(), np.array([])


def plot_forecast(
    series_id: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast: pd.DataFrame,
    nbeats_forecast_60: pd.DataFrame,
    nbeats_forecast_full: pd.DataFrame,
    naive2_forecast: np.ndarray,
    model_name: str,
    outlier_indices: np.ndarray = None,
    was_log_transformed: bool = False
) -> None:
    """Plot and save forecast for a single series."""
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.plot(train_df['ds'], train_df['y'], 'b-', label='Training', alpha=0.7)
    
    # Highlight outliers with red markers if provided
    if outlier_indices is not None and len(outlier_indices) > 0:
        plt.scatter(
            train_df['ds'].iloc[outlier_indices], 
            train_df['y'].iloc[outlier_indices],
            color='red', s=80, marker='o', label='Outliers', zorder=10
        )
    
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
    
    # Plot forecasts
    transformer_label = 'Transformer'
    if was_log_transformed:
        transformer_label += ' (log transformed)'
    plt.plot(forecast['ds'], forecast['q_0.5'], 'r--', label=transformer_label, linewidth=1.5)
    
    # Find column for NBEATS 60 forecast
    nbeats_column_60 = None
    if 'NBEATS_60' in nbeats_forecast_60.columns:
        nbeats_column_60 = 'NBEATS_60'
    elif 'NBEATS' in nbeats_forecast_60.columns:
        nbeats_column_60 = 'NBEATS'
    else:
        for col in nbeats_forecast_60.columns:
            if col != 'ds' and col != 'unique_id' and pd.api.types.is_numeric_dtype(nbeats_forecast_60[col]):
                nbeats_column_60 = col
                break
    
    # Find column for NBEATS full forecast
    nbeats_column_full = None
    if 'NBEATS_Full' in nbeats_forecast_full.columns:
        nbeats_column_full = 'NBEATS_Full'
    elif 'NBEATS' in nbeats_forecast_full.columns:
        nbeats_column_full = 'NBEATS'
    else:
        for col in nbeats_forecast_full.columns:
            if col != 'ds' and col != 'unique_id' and pd.api.types.is_numeric_dtype(nbeats_forecast_full[col]):
                nbeats_column_full = col
                break
    
    # Plot NBEATS forecasts if columns were found
    if nbeats_column_60:
        plt.plot(nbeats_forecast_60['ds'], nbeats_forecast_60[nbeats_column_60], 'c--', 
                 label='NBEATS (60 points)', linewidth=1.5)
    
    if nbeats_column_full:
        plt.plot(nbeats_forecast_full['ds'], nbeats_forecast_full[nbeats_column_full], 'm--', 
                 label='NBEATS (full history)', linewidth=1.5)
    
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


def generate_nbeats_forecast(train_df: pd.DataFrame, horizon: int, use_full_history: bool = False) -> pd.DataFrame:
    """Generate forecast using enhanced NBEATS model configuration.
    
    Args:
        train_df: Training data
        horizon: Forecast horizon
        use_full_history: Whether to use the entire history (True) or just the last 60 points (False)
    """
    # Calculate available data length
    available_data_length = len(train_df)
    
    # Determine input size based on whether to use full history
    if use_full_history:
        # Use full history, but make sure to leave enough data for training windows
        # NBEATS requires at least 2 windows
        input_size = min(available_data_length - horizon - 1, available_data_length)
        model_prefix = "NBEATS_Full"
    else:
        # Use 60 points or less if not enough data
        input_size = min(60, available_data_length - horizon - 1)
        model_prefix = "NBEATS_60"
    
    # Ensure input_size is positive
    input_size = max(input_size, 1)
    
    logger.info(f"Using input_size={input_size} for {model_prefix} model (series length: {available_data_length})")
    
    # Initialize NBEATS model with a more powerful configuration
    model = NBEATS(
        h=horizon,                     # forecast horizon
        input_size=input_size,         # Either 60 or full history
        loss=MAE(),                    # simple MAE loss
        stack_types=['trend', 'seasonality'],  # Use both trend and seasonality
        n_blocks=[3, 3],               # More blocks for better representation
        mlp_units=[[256, 256], [256, 256]],  # Larger network for more capacity
        n_harmonics=2,                 # More harmonics for better seasonality modeling
        n_polynomials=3,               # Higher degree for more flexible trend
        max_steps=200,                 # More training steps
        learning_rate=0.001,           # Standard learning rate
        batch_size=32,                 # Standard batch size
        scaler_type='standard'         # Scale data for better training
    )

    # Initialize NeuralForecast with NBEATS
    nf = NeuralForecast(
        models=[model],
        freq='M'  # Monthly frequency
    )

    try:
        # Fit model and generate forecast
        nf.fit(df=train_df, val_size=0)  # No validation
        forecast = nf.predict()
        
        # Debug: Print column names
        logger.info(f"{model_prefix} forecast columns: {forecast.columns.tolist()}")
        
        # Add a column to identify this forecast version
        if 'NBEATS' in forecast.columns:
            forecast[model_prefix] = forecast['NBEATS'].values
        
        return forecast
    except Exception as e:
        logger.error(f"Error training {model_prefix} model: {str(e)}")
        # Create an empty forecast with the same structure as transformer forecast
        empty_forecast = pd.DataFrame({
            'ds': train_df['ds'].iloc[-1:].values[0] + pd.DateOffset(months=range(1, horizon+1)),
            'unique_id': [train_df['unique_id'].iloc[0]] * horizon
        })
        # Add a column with zeros for the forecast
        empty_forecast['NBEATS'] = np.zeros(horizon)
        # Add identification column
        empty_forecast[model_prefix] = np.zeros(horizon)
        
        logger.warning(f"Using fallback zero forecast for {model_prefix}")
        return empty_forecast


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
    """Evaluate the model on tourism data."""
    # Initialize model
    logger.info(f"Loading model {model_name}")
    model = TransformerModel(model_name=model_name)
    
    # Filter for specific series if provided
    if specific_series:
        if specific_series not in series_ids:
            raise ValueError(f"Series {specific_series} not found in dataset")
        series_ids = [specific_series]
        logger.info(f"Evaluating only series {specific_series}")
    
    # Prepare results storage
    results = []
    
    # Process each series
    for series_id in tqdm(series_ids, desc="Evaluating series"):
        try:
            # Get series data
            series_train = train_df[train_df['unique_id'] == series_id]
            series_test = test_df[test_df['unique_id'] == series_id]
            
            # Save original data for plotting
            original_series_train = series_train.copy()
            
            # Use an extremely conservative threshold to detect only true anomalies
            cleaned_series_train, outlier_indices = detect_and_clean_outliers(series_train, threshold=8.0)
            
            # Check if log transformation is needed for Transformer (based on variance pattern)
            needs_log_transform = should_log_transform(cleaned_series_train)
            was_transformed = False
            
            # Generate transformer forecast using cleaned data (with log transform if needed)
            if needs_log_transform:
                logger.info(f"Applying log transform for series {series_id} (Transformer model only)")
                transformed_series, was_transformed = apply_log_transform(cleaned_series_train)
                transformed_series_indexed = transformed_series.set_index('ds')
                
                # Generate forecast in log space
                log_forecast = model.predict(
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
                cleaned_series_train_indexed = cleaned_series_train.set_index('ds')
                forecast = model.predict(
                    series_df=cleaned_series_train_indexed,
                    n=forecast_horizon
                )
            
            # Generate NBEATS forecasts - both 60-point and full history versions
            # (No log transform for NBEATS as requested)
            nbeats_forecast_60 = generate_nbeats_forecast(cleaned_series_train, forecast_horizon, use_full_history=False)
            nbeats_forecast_full = generate_nbeats_forecast(cleaned_series_train, forecast_horizon, use_full_history=True)
            
            # Generate Naive2 forecast from cleaned data (no log transform)
            naive2_forecast = calculate_naive2_forecast(
                values=cleaned_series_train['y'].values,
                horizon=forecast_horizon
            )
            
            # Get actual values
            actual_values = []
            for date in forecast['ds']:
                date_str = pd.to_datetime(date)
                matching_rows = series_test[series_test['ds'] == date_str]
                if not matching_rows.empty:
                    actual_values.append(matching_rows.iloc[0]['y'])
                else:
                    actual_values.append(np.nan)
            actual_values = np.array(actual_values)
            
            # Find column for NBEATS 60 forecast
            nbeats_column_60 = None
            # First check for our added column
            if 'NBEATS_60' in nbeats_forecast_60.columns:
                nbeats_column_60 = 'NBEATS_60'
            # Then check for NBEATS column
            elif 'NBEATS' in nbeats_forecast_60.columns:
                nbeats_column_60 = 'NBEATS'
            # Finally, try any numeric column that's not ds or unique_id
            else:
                for col in nbeats_forecast_60.columns:
                    if col != 'ds' and col != 'unique_id' and pd.api.types.is_numeric_dtype(nbeats_forecast_60[col]):
                        logger.info(f"Using column '{col}' for NBEATS_60 forecast")
                        nbeats_column_60 = col
                        break
            
            # Find column for NBEATS full forecast
            nbeats_column_full = None
            # First check for our added column
            if 'NBEATS_Full' in nbeats_forecast_full.columns:
                nbeats_column_full = 'NBEATS_Full'
            # Then check for NBEATS column
            elif 'NBEATS' in nbeats_forecast_full.columns:
                nbeats_column_full = 'NBEATS'
            # Finally, try any numeric column that's not ds or unique_id
            else:
                for col in nbeats_forecast_full.columns:
                    if col != 'ds' and col != 'unique_id' and pd.api.types.is_numeric_dtype(nbeats_forecast_full[col]):
                        logger.info(f"Using column '{col}' for NBEATS_Full forecast")
                        nbeats_column_full = col
                        break
            
            if nbeats_column_60 is None or nbeats_column_full is None:
                logger.error(f"NBEATS_60 columns: {nbeats_forecast_60.columns.tolist()}")
                logger.error(f"NBEATS_Full columns: {nbeats_forecast_full.columns.tolist()}")
                raise ValueError("No usable NBEATS forecast column found")
            
            logger.info(f"Using NBEATS_60 column: {nbeats_column_60}")
            logger.info(f"Using NBEATS_Full column: {nbeats_column_full}")
            
            # Calculate metrics for each model
            transformer_metrics = calculate_metrics(actual_values, forecast['q_0.5'].values)
            nbeats_60_metrics = calculate_metrics(actual_values, nbeats_forecast_60[nbeats_column_60].values)
            nbeats_full_metrics = calculate_metrics(actual_values, nbeats_forecast_full[nbeats_column_full].values)
            naive2_metrics = calculate_metrics(actual_values, naive2_forecast)
            
            # Add information about number of outliers detected and log transform status
            outlier_count = len(outlier_indices) if outlier_indices is not None else 0
            
            # Store results with proper column names
            results.append({
                'series_id': series_id,
                'outliers_count': outlier_count,
                'log_transformed': was_transformed,
                'transformer_mae': transformer_metrics['mae'],
                'transformer_mape': transformer_metrics['mape'],
                'transformer_smape': transformer_metrics['smape'],
                'transformer_rmse': transformer_metrics['rmse'],
                'nbeats_60_mae': nbeats_60_metrics['mae'],
                'nbeats_60_mape': nbeats_60_metrics['mape'],
                'nbeats_60_smape': nbeats_60_metrics['smape'],
                'nbeats_60_rmse': nbeats_60_metrics['rmse'],
                'nbeats_full_mae': nbeats_full_metrics['mae'],
                'nbeats_full_mape': nbeats_full_metrics['mape'],
                'nbeats_full_smape': nbeats_full_metrics['smape'],
                'nbeats_full_rmse': nbeats_full_metrics['rmse'],
                'naive2_mae': naive2_metrics['mae'],
                'naive2_mape': naive2_metrics['mape'],
                'naive2_smape': naive2_metrics['smape'],
                'naive2_rmse': naive2_metrics['rmse']
            })
            
            # Plot results using original data for visualization
            plot_forecast(
                series_id=series_id,
                train_df=original_series_train,  # Use original data for plotting
                test_df=series_test,
                forecast=forecast,
                nbeats_forecast_60=nbeats_forecast_60,
                nbeats_forecast_full=nbeats_forecast_full,
                naive2_forecast=naive2_forecast,
                model_name=model_name,
                outlier_indices=outlier_indices,
                was_log_transformed=was_transformed
            )
                
        except Exception as e:
            # Convert exception to string to handle any type of error object
            error_message = f"{type(e).__name__}: {e}"
            logger.error(f"Error processing series {series_id}: {error_message}")
            # Add empty results to maintain DataFrame structure
            results.append({
                'series_id': series_id,
                'outliers_count': 0,
                'log_transformed': False,
                'transformer_mae': np.nan,
                'transformer_mape': np.nan,
                'transformer_smape': np.nan,
                'transformer_rmse': np.nan,
                'nbeats_60_mae': np.nan,
                'nbeats_60_mape': np.nan,
                'nbeats_60_smape': np.nan,
                'nbeats_60_rmse': np.nan,
                'nbeats_full_mae': np.nan,
                'nbeats_full_mape': np.nan,
                'nbeats_full_smape': np.nan,
                'nbeats_full_rmse': np.nan,
                'naive2_mae': np.nan,
                'naive2_mape': np.nan,
                'naive2_smape': np.nan,
                'naive2_rmse': np.nan
            })
            continue
    
    # Save results to Excel
    results_df = pd.DataFrame(results)
    os.makedirs("evaluation/tourism", exist_ok=True)
    excel_path = f"evaluation/tourism/{model_name}_evaluation.xlsx"
    
    # Calculate summary statistics
    summary_stats = {
        'Metric': ['MAPE', 'SMAPE', 'MAE', 'RMSE'],
        'Transformer': [
            results_df['transformer_mape'].mean(),
            results_df['transformer_smape'].mean(),
            results_df['transformer_mae'].mean(),
            results_df['transformer_rmse'].mean()
        ],
        'NBEATS (60 points)': [
            results_df['nbeats_60_mape'].mean(),
            results_df['nbeats_60_smape'].mean(),
            results_df['nbeats_60_mae'].mean(),
            results_df['nbeats_60_rmse'].mean()
        ],
        'NBEATS (full history)': [
            results_df['nbeats_full_mape'].mean(),
            results_df['nbeats_full_smape'].mean(),
            results_df['nbeats_full_mae'].mean(),
            results_df['nbeats_full_rmse'].mean()
        ],
        'Naive2': [
            results_df['naive2_mape'].mean(),
            results_df['naive2_smape'].mean(),
            results_df['naive2_mae'].mean(),
            results_df['naive2_rmse'].mean()
        ]
    }
    summary_df = pd.DataFrame(summary_stats)
    
    # Create outlier summary
    total_outliers = results_df['outliers_count'].sum()
    series_with_outliers = results_df[results_df['outliers_count'] > 0].shape[0]
    max_outliers = results_df['outliers_count'].max()
    avg_outliers = results_df['outliers_count'].mean()
    
    outlier_summary = pd.DataFrame({
        'Metric': ['Total Outliers', 'Series with Outliers', 'Max Outliers per Series', 'Avg Outliers per Series'],
        'Value': [total_outliers, series_with_outliers, max_outliers, avg_outliers]
    })
    
    # Save both detailed and summary results
    with pd.ExcelWriter(excel_path) as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        outlier_summary.to_excel(writer, sheet_name='Outlier Summary', index=False)
        results_df.to_excel(writer, sheet_name='Detailed', index=False)
    
    logger.info(f"Results saved to {excel_path}")
    logger.info("\nSummary Statistics:")
    logger.info(f"Number of series evaluated: {len(results_df)}")
    
    logger.info("\nOutlier Statistics:")
    logger.info(f"Total outliers detected: {total_outliers}")
    logger.info(f"Series with outliers: {series_with_outliers} ({series_with_outliers/len(results_df)*100:.1f}%)")
    logger.info(f"Maximum outliers in a single series: {max_outliers}")
    logger.info(f"Average outliers per series: {avg_outliers:.2f}")
    
    logger.info("\nMean MAPE:")
    logger.info(f"Transformer: {results_df['transformer_mape'].mean():.2f}%")
    logger.info(f"NBEATS (60 points): {results_df['nbeats_60_mape'].mean():.2f}%")
    logger.info(f"NBEATS (full history): {results_df['nbeats_full_mape'].mean():.2f}%")
    logger.info(f"Naive2: {results_df['naive2_mape'].mean():.2f}%")
    logger.info("\nMean SMAPE:")
    logger.info(f"Transformer: {results_df['transformer_smape'].mean():.2f}%")
    logger.info(f"NBEATS (60 points): {results_df['nbeats_60_smape'].mean():.2f}%")
    logger.info(f"NBEATS (full history): {results_df['nbeats_full_smape'].mean():.2f}%")
    logger.info(f"Naive2: {results_df['naive2_smape'].mean():.2f}%")


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate transformer model on Tourism data')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Name of the model to evaluate')
    parser.add_argument('--series-id', type=str,
                       help='Optional: evaluate only this specific series')
    args = parser.parse_args()
    
    # Load data
    train_df, test_df = load_tourism_data()
    
    # Get unique series IDs
    series_ids = train_df['unique_id'].unique().tolist()
    
    # Run evaluation
    evaluate_model(
        model_name=args.model_name,
        train_df=train_df,
        test_df=test_df,
        series_ids=series_ids,
        specific_series=args.series_id
    )
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main() 