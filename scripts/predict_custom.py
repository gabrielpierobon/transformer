import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import argparse
import sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import random

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import custom loss functions and metrics
from src.models.transformer import gaussian_nll, smape_loss, hybrid_loss

def mae_prob(y_true, y_pred):
    """Custom MAE metric for probabilistic models that only uses mean prediction."""
    mu, _ = tf.split(y_pred, 2, axis=-1)
    return tf.keras.metrics.mean_absolute_error(y_true, mu)

def generate_synthetic_time_series(length=100, seasonality=12, trend=0.01, noise=0.1):
    """
    Generate a synthetic time series with trend, seasonality, and noise.
    
    Args:
        length: Length of the time series
        seasonality: Seasonality period
        trend: Trend coefficient
        noise: Noise level
        
    Returns:
        Synthetic time series as numpy array
    """
    # Time component
    t = np.arange(length)
    
    # Trend component
    trend_component = trend * t
    
    # Seasonal component
    seasonal_component = np.sin(2 * np.pi * t / seasonality)
    
    # Noise component
    noise_component = noise * np.random.randn(length)
    
    # Combine components
    series = trend_component + seasonal_component + noise_component
    
    return series

def preprocess_time_series(series, sequence_length=60):
    """
    Preprocess a time series for model input.
    
    Args:
        series: Input time series
        sequence_length: Length of input sequences
        
    Returns:
        Scaled series and scaler object
    """
    # Reshape for scaler
    series_reshaped = series.reshape(-1, 1)
    
    # Scale the series
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_series = scaler.fit_transform(series_reshaped).flatten()
    
    return scaled_series, scaler

def create_model_input(scaled_series, sequence_length=60):
    """
    Create model input from scaled series.
    
    Args:
        scaled_series: Scaled time series
        sequence_length: Length of input sequences
        
    Returns:
        Model input tensor
    """
    # Use the last sequence_length values as input
    if len(scaled_series) >= sequence_length:
        model_input = scaled_series[-sequence_length:]
    else:
        # If series is shorter than sequence_length, pad with zeros
        model_input = np.zeros(sequence_length)
        model_input[-len(scaled_series):] = scaled_series
    
    # Reshape for model input (batch_size, sequence_length, features)
    model_input = model_input.reshape(1, sequence_length, 1)
    
    return model_input

def recursive_forecast_point(model, initial_series, n_steps, sequence_length=60):
    """
    Generate point forecasts recursively.
    
    Args:
        model: Trained point prediction model
        initial_series: Initial series values (scaled)
        n_steps: Number of steps to forecast
        sequence_length: Length of input sequences
        
    Returns:
        Array of predicted values
    """
    series = initial_series.copy()
    predictions = []
    
    for step in range(n_steps):
        # Create model input
        model_input = create_model_input(series, sequence_length)
        
        # Make prediction
        next_value = model.predict(model_input, verbose=0)[0, 0]
        
        # Store prediction
        predictions.append(next_value)
        
        # Update series with prediction
        series = np.append(series, next_value)
    
    return np.array(predictions)

def recursive_forecast_probabilistic(model, initial_series, n_steps, sequence_length=60, num_samples=1000):
    """
    Generate probabilistic forecasts recursively.
    
    Args:
        model: Trained probabilistic model
        initial_series: Initial series values (scaled)
        n_steps: Number of steps to forecast
        sequence_length: Length of input sequences
        num_samples: Number of samples to generate for each step
        
    Returns:
        Dictionary with mean predictions and samples
    """
    series = initial_series.copy()
    mean_predictions = []
    all_samples = []
    
    for step in range(n_steps):
        # Create model input
        model_input = create_model_input(series, sequence_length)
        
        # Make prediction (mean and log variance)
        prediction = model.predict(model_input, verbose=0)[0]
        mean = prediction[0]
        log_var = prediction[1]
        std = np.exp(0.5 * log_var)
        
        # Store mean prediction
        mean_predictions.append(mean)
        
        # Generate samples from the predicted distribution
        samples = np.random.normal(mean, std, num_samples)
        all_samples.append(samples)
        
        # Update series with mean prediction
        series = np.append(series, mean)
    
    return {
        'mean': np.array(mean_predictions),
        'samples': np.array(all_samples)
    }

def calculate_confidence_intervals(samples, low_bound_conf=25, high_bound_conf=75):
    """
    Calculate confidence intervals from samples.
    
    Args:
        samples: Array of samples for each time step
        low_bound_conf: Lower bound percentile
        high_bound_conf: Upper bound percentile
        
    Returns:
        Lower and upper confidence bounds
    """
    lower_bound = np.percentile(samples, low_bound_conf, axis=1)
    upper_bound = np.percentile(samples, high_bound_conf, axis=1)
    
    return lower_bound, upper_bound

def plot_forecasts(history, point_forecast, proba_forecast, scaler, low_bound_conf=25, high_bound_conf=75):
    """
    Plot history and forecasts.
    
    Args:
        history: Original time series
        point_forecast: Point forecast values (scaled)
        proba_forecast: Probabilistic forecast dictionary (scaled)
        scaler: Scaler used to normalize the data
        low_bound_conf: Lower bound percentile
        high_bound_conf: Upper bound percentile
    """
    # Inverse transform all values
    history_original = scaler.inverse_transform(history.reshape(-1, 1)).flatten()
    
    point_forecast_original = scaler.inverse_transform(
        point_forecast.reshape(-1, 1)).flatten()
    
    proba_mean_original = scaler.inverse_transform(
        proba_forecast['mean'].reshape(-1, 1)).flatten()
    
    # Calculate confidence intervals
    lower_bound, upper_bound = calculate_confidence_intervals(
        proba_forecast['samples'], low_bound_conf, high_bound_conf)
    
    lower_bound_original = scaler.inverse_transform(
        lower_bound.reshape(-1, 1)).flatten()
    
    upper_bound_original = scaler.inverse_transform(
        upper_bound.reshape(-1, 1)).flatten()
    
    # Create time indices
    history_idx = np.arange(len(history_original))
    forecast_idx = np.arange(len(history_original), len(history_original) + len(point_forecast_original))
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot history
    plt.plot(history_idx, history_original, 'b-', label='History')
    
    # Plot point forecast
    plt.plot(forecast_idx, point_forecast_original, 'r-', label='Point Forecast')
    
    # Plot probabilistic forecast
    plt.plot(forecast_idx, proba_mean_original, 'g-', label='Probabilistic Mean')
    plt.fill_between(forecast_idx, lower_bound_original, upper_bound_original, 
                     color='g', alpha=0.2, 
                     label=f'{low_bound_conf}-{high_bound_conf} Percentile')
    
    # Add labels and legend
    plt.axvline(x=len(history_original)-1, color='k', linestyle='--')
    plt.title('Time Series Forecast')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    figures_dir = Path('reports/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / 'custom_forecast.png')
    
    plt.show()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate forecasts for a custom time series')
    
    parser.add_argument(
        '--point-model',
        type=str,
        default='models/final/transformer_1.0_directml_point_M1_M2',
        help='Path to point prediction model'
    )
    
    parser.add_argument(
        '--proba-model',
        type=str,
        default='models/final/transformer_1.0_directml_proba_hybrid_0.8_M1_M2',
        help='Path to probabilistic model'
    )
    
    parser.add_argument(
        '--loss-type',
        type=str,
        default='hybrid',
        choices=['gaussian_nll', 'smape', 'hybrid'],
        help='Loss type used for the probabilistic model'
    )
    
    parser.add_argument(
        '--loss-alpha',
        type=float,
        default=0.8,
        help='Alpha value for hybrid loss'
    )
    
    parser.add_argument(
        '--n-steps',
        type=int,
        default=36,
        help='Number of steps to forecast'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of samples for probabilistic forecast'
    )
    
    parser.add_argument(
        '--low-bound-conf',
        type=int,
        default=25,
        help='Lower bound percentile'
    )
    
    parser.add_argument(
        '--high-bound-conf',
        type=int,
        default=75,
        help='Upper bound percentile'
    )
    
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=60,
        help='Length of input sequences'
    )
    
    parser.add_argument(
        '--series-length',
        type=int,
        default=100,
        help='Length of synthetic time series'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    # Generate synthetic time series
    print("Generating synthetic time series...")
    series = generate_synthetic_time_series(
        length=args.series_length,
        seasonality=12,  # Monthly seasonality
        trend=0.01,
        noise=0.1
    )
    
    # Preprocess time series
    print("Preprocessing time series...")
    scaled_series, scaler = preprocess_time_series(series, args.sequence_length)
    
    # Prepare custom objects dictionary based on loss type
    custom_objects = {
        'mae_prob': mae_prob,
        'gaussian_nll': gaussian_nll,
        'smape_loss': smape_loss
    }
    
    # Add the appropriate loss function based on the specified loss type
    if args.loss_type == 'hybrid':
        hybrid_loss_fn = hybrid_loss(alpha=args.loss_alpha)
        custom_objects['hybrid_loss'] = hybrid_loss_fn
        custom_objects['loss'] = hybrid_loss_fn
        custom_objects['loss_fn'] = hybrid_loss_fn  # Add this for internal function name
    elif args.loss_type == 'gaussian_nll':
        custom_objects['loss'] = gaussian_nll
        custom_objects['loss_fn'] = gaussian_nll  # Add this for internal function name
    elif args.loss_type == 'smape':
        # For SMAPE with probabilistic model
        def smape_prob(y_true, y_pred):
            mu, _ = tf.split(y_pred, 2, axis=-1)
            return smape_loss(y_true, mu)
        custom_objects['loss'] = smape_prob
        custom_objects['smape_prob'] = smape_prob
        custom_objects['loss_fn'] = smape_prob  # Add this for internal function name
    
    # Load models
    print(f"Loading point prediction model from {args.point_model}...")
    point_model = tf.keras.models.load_model(args.point_model)
    
    print(f"Loading probabilistic model from {args.proba_model}...")
    print(f"Using loss type: {args.loss_type}" + (f" with alpha={args.loss_alpha}" if args.loss_type == 'hybrid' else ""))
    proba_model = tf.keras.models.load_model(args.proba_model, custom_objects=custom_objects)
    
    # Generate forecasts
    print(f"Generating point forecast for {args.n_steps} steps...")
    point_forecast = recursive_forecast_point(
        point_model, 
        scaled_series, 
        args.n_steps, 
        args.sequence_length
    )
    
    print(f"Generating probabilistic forecast for {args.n_steps} steps...")
    proba_forecast = recursive_forecast_probabilistic(
        proba_model, 
        scaled_series, 
        args.n_steps, 
        args.sequence_length,
        args.num_samples
    )
    
    # Plot results
    print("Plotting forecasts...")
    plot_forecasts(
        scaled_series, 
        point_forecast, 
        proba_forecast, 
        scaler,
        args.low_bound_conf,
        args.high_bound_conf
    )
    
    print("Done! Forecast plot saved to reports/figures/custom_forecast.png")

if __name__ == "__main__":
    main() 