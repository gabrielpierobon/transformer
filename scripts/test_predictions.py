import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from pathlib import Path
import sys
import argparse

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

def recursive_forecast(model, initial_series, steps, sequence_length, feature_dim):
    """
    Recursively forecast future values using the trained model.
    
    Args:
        model: Trained TensorFlow model
        initial_series: Initial time series values
        steps: Number of steps to forecast
        sequence_length: Length of input sequences
        feature_dim: Number of features (typically 1 for univariate)
    
    Returns:
        Array of predicted values
    """
    series = initial_series.copy()
    predictions = []

    for step in range(steps):
        # Ensure the series is the right shape: (1, sequence_length, feature_dim)
        series_reshaped = series[-sequence_length:].reshape(1, sequence_length, feature_dim)
        
        # Predict the next step
        next_prediction = model.predict(series_reshaped, verbose=0)
        
        # Reshape the prediction and append it to the series
        predictions.append(next_prediction.flatten()[0])
        series = np.append(series, next_prediction.flatten()[0])

    return np.array(predictions)

def plot_with_predictions(X_test, predictions, random_idx, n_steps):
    """
    Plot the actual series and predictions.
    
    Args:
        X_test: Test dataset
        predictions: Array of predicted values
        random_idx: Index of the series to plot
        n_steps: Number of predicted steps
    """
    actual_series = X_test[random_idx].flatten()
    time_steps_actual = range(len(actual_series))
    time_steps_predicted = range(len(actual_series), len(actual_series) + n_steps)

    plt.figure(figsize=(12, 6))

    # Plot the actual series
    plt.plot(time_steps_actual, actual_series, label='Actual Series')

    # Plot the predictions
    plt.scatter(time_steps_predicted, predictions, color='red')
    plt.plot(time_steps_predicted, predictions, color='red', label=f'Predicted Next {n_steps} Values')

    plt.title(f"Time Series (Series Index: {random_idx}) with Extended Predictions")
    plt.xlabel('Time Steps')
    plt.ylabel('Sales (Normalized)')
    plt.legend()
    plt.grid(True)
    
    # Create figures directory if it doesn't exist
    figures_dir = Path('reports/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(figures_dir / f'forecast_series_{random_idx}.png')
    plt.show()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test transformer model predictions')
    parser.add_argument(
        '--start-series',
        type=int,
        required=True,
        help='Starting series index (e.g., 1 for M1)'
    )
    parser.add_argument(
        '--end-series',
        type=int,
        required=True,
        help='Ending series index (e.g., 2 for M2)'
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=36,
        help='Number of steps to forecast'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create series range string
    series_range = f'M{args.start_series}_M{args.end_series}'
    
    # Load the saved model
    model_path = Path(f'models/final/transformer_1.0_directml_{series_range}')
    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Load test data
    data_path = Path(f'data/processed/X_val_{series_range}.npy')
    print(f"Loading validation data from: {data_path}")
    X_val = np.load(data_path)
    print(f"Validation data shape: {X_val.shape}")
    
    # Parameters
    sequence_length = X_val.shape[1]  # Should be 60
    feature_dim = 1
    
    # Generate predictions for all validation series
    n_samples = len(X_val)
    print(f"\nGenerating {args.n_steps}-step forecasts for {n_samples} validation series...")
    
    for idx in range(n_samples):
        print(f"\nGenerating forecast for series index: {idx}")
        
        extended_predictions = recursive_forecast(
            model, 
            X_val[idx].flatten(), 
            args.n_steps, 
            sequence_length, 
            feature_dim
        )
        
        plot_with_predictions(X_val, extended_predictions, idx, args.n_steps)

if __name__ == '__main__':
    main() 