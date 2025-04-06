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
    
    # Plot forecasts
    plt.plot(forecast['ds'], forecast['q_0.5'], 'r--', label='Transformer')
    plt.plot(forecast['ds'], naive2_forecast, 'm:', label='Naive2')
    
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
        # Get series data
        series_train = train_df[train_df['unique_id'] == series_id]
        series_test = test_df[test_df['unique_id'] == series_id]
        
        try:
            # Generate transformer forecast
            series_train_indexed = series_train.set_index('ds')
            forecast = model.predict(
                series_df=series_train_indexed,
                n=forecast_horizon
            )
            
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
                    actual_values.append(matching_rows['y'].values[0])
                else:
                    actual_values.append(np.nan)
            actual_values = np.array(actual_values)
            
            # Calculate metrics for both methods
            transformer_metrics = calculate_metrics(actual_values, forecast['q_0.5'].values)
            naive2_metrics = calculate_metrics(actual_values, naive2_forecast)
            
            # Store results
            results.append({
                'series_id': series_id,
                'transformer_mae': transformer_metrics['mae'],
                'transformer_mape': transformer_metrics['mape'],
                'transformer_smape': transformer_metrics['smape'],
                'transformer_rmse': transformer_metrics['rmse'],
                'naive2_mae': naive2_metrics['mae'],
                'naive2_mape': naive2_metrics['mape'],
                'naive2_smape': naive2_metrics['smape'],
                'naive2_rmse': naive2_metrics['rmse']
            })
            
            # Plot and save forecast
            plot_forecast(
                series_id=series_id,
                train_df=series_train,
                test_df=series_test,
                forecast=forecast,
                naive2_forecast=naive2_forecast,
                model_name=model_name
            )
            
        except Exception as e:
            logger.error(f"Error processing series {series_id}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary_stats = {
        'Metric': ['MAE', 'MAPE', 'SMAPE', 'RMSE'],
        'Transformer (Mean)': [
            results_df['transformer_mae'].mean(),
            results_df['transformer_mape'].mean(),
            results_df['transformer_smape'].mean(),
            results_df['transformer_rmse'].mean()
        ],
        'Transformer (Median)': [
            results_df['transformer_mae'].median(),
            results_df['transformer_mape'].median(),
            results_df['transformer_smape'].median(),
            results_df['transformer_rmse'].median()
        ],
        'Naive2 (Mean)': [
            results_df['naive2_mae'].mean(),
            results_df['naive2_mape'].mean(),
            results_df['naive2_smape'].mean(),
            results_df['naive2_rmse'].mean()
        ],
        'Naive2 (Median)': [
            results_df['naive2_mae'].median(),
            results_df['naive2_mape'].median(),
            results_df['naive2_smape'].median(),
            results_df['naive2_rmse'].median()
        ]
    }
    summary_df = pd.DataFrame(summary_stats)
    
    # Save results
    os.makedirs("evaluation/tourism", exist_ok=True)
    excel_path = f"evaluation/tourism/{model_name}_evaluation.xlsx"
    
    with pd.ExcelWriter(excel_path) as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        results_df.to_excel(writer, sheet_name='Detailed', index=False)
    
    logger.info(f"Saved evaluation results to {excel_path}")
    
    # Log summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"Number of series evaluated: {len(results_df)}")
    logger.info("\nMean MAPE:")
    logger.info(f"Transformer: {results_df['transformer_mape'].mean():.2f}%")
    logger.info(f"Naive2: {results_df['naive2_mape'].mean():.2f}%")
    logger.info("\nMean SMAPE:")
    logger.info(f"Transformer: {results_df['transformer_smape'].mean():.2f}%")
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