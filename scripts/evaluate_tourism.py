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


def plot_forecast(
    series_id: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast: pd.DataFrame,
    nbeats_forecast_60: pd.DataFrame,
    nbeats_forecast_full: pd.DataFrame,
    naive2_forecast: np.ndarray,
    model_name: str
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
    
    # Plot test data
    plt.plot(test_df['ds'], test_df['y'], 'g-', label='Actual')
    
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
                logger.info(f"Using column '{col}' for NBEATS_60 forecast in plot")
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
                logger.info(f"Using column '{col}' for NBEATS_Full forecast in plot")
                nbeats_column_full = col
                break
    
    # Plot forecasts
    plt.plot(forecast['ds'], forecast['q_0.5'], 'r--', label='Transformer')
    
    if nbeats_column_60:
        plt.plot(nbeats_forecast_60['ds'], nbeats_forecast_60[nbeats_column_60], 'c--', 
                 label='NBEATS (60 points)')
    
    if nbeats_column_full:
        plt.plot(nbeats_forecast_full['ds'], nbeats_forecast_full[nbeats_column_full], 'm--', 
                 label='NBEATS (full history)')
    
    plt.plot(forecast['ds'], naive2_forecast, 'k:', label='Naive2')
    
    # Add vertical line at forecast start
    plt.axvline(x=train_df['ds'].max(), color='k', linestyle='--', alpha=0.5)
    
    # Customize plot
    plt.title(f'Series: {series_id}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs("evaluation/tourism/plots", exist_ok=True)
    plot_path = f"evaluation/tourism/plots/{model_name}_{series_id}.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
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
            
            # Generate transformer forecast
            series_train_indexed = series_train.set_index('ds')
            forecast = model.predict(
                series_df=series_train_indexed,
                n=forecast_horizon
            )
            
            # Generate NBEATS forecasts - both 60-point and full history versions
            nbeats_forecast_60 = generate_nbeats_forecast(series_train, forecast_horizon, use_full_history=False)
            nbeats_forecast_full = generate_nbeats_forecast(series_train, forecast_horizon, use_full_history=True)
            
            # Generate Naive2 forecast
            naive2_forecast = calculate_naive2_forecast(
                values=series_train['y'].values,
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
            
            # Store results with proper column names
            results.append({
                'series_id': series_id,
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
            
            # Plot results
            plot_forecast(
                series_id=series_id,
                train_df=series_train,
                test_df=series_test,
                forecast=forecast,
                nbeats_forecast_60=nbeats_forecast_60,
                nbeats_forecast_full=nbeats_forecast_full,
                naive2_forecast=naive2_forecast,
                model_name=model_name
            )
            
        except Exception as e:
            logger.error(f"Error processing series {series_id}: {str(e)}")
            # Add empty results to maintain DataFrame structure
            results.append({
                'series_id': series_id,
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
    
    # Save both detailed and summary results
    with pd.ExcelWriter(excel_path) as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        results_df.to_excel(writer, sheet_name='Detailed', index=False)
    
    logger.info(f"Results saved to {excel_path}")
    logger.info("\nSummary Statistics:")
    logger.info(f"Number of series evaluated: {len(results_df)}")
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