import requests
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import pandas as pd
import argparse
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import function to generate synthetic data
from scripts.predict_custom import generate_synthetic_time_series

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test the forecasting API')
    
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8000',
        help='URL of the forecasting API'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['point', 'probabilistic'],
        default='probabilistic',
        help='Type of model to use for forecasting'
    )
    
    parser.add_argument(
        '--loss-type',
        type=str,
        choices=['hybrid', 'gaussian_nll', 'smape'],
        default='hybrid',
        help='Loss type for probabilistic model'
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
        '--series-length',
        type=int,
        default=100,
        help='Length of synthetic time series'
    )
    
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=60,
        help='Length of input sequences'
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
    
    return parser.parse_args()

def generate_data(length=100, seasonality=12, trend=0.01, noise=0.1):
    """Generate synthetic time series data."""
    return generate_synthetic_time_series(length, seasonality, trend, noise)

def call_forecast_api(api_url, values, model_type='point', n_steps=36, sequence_length=60, 
                     loss_type='hybrid', loss_alpha=0.8, low_bound_conf=25, high_bound_conf=75):
    """Call the forecast API and return the response."""
    endpoint = f"{api_url}/forecast/"
    
    # Prepare request payload
    payload = {
        "values": values.tolist() if isinstance(values, np.ndarray) else values,
        "model_type": model_type,
        "n_steps": n_steps,
        "sequence_length": sequence_length
    }
    
    # Add probabilistic model parameters if needed
    if model_type == "probabilistic":
        payload.update({
            "loss_type": loss_type,
            "loss_alpha": loss_alpha,
            "low_bound_conf": low_bound_conf,
            "high_bound_conf": high_bound_conf,
            "num_samples": 1000
        })
    
    # Make API call
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        sys.exit(1)

def plot_forecast(history, forecast_response):
    """Plot the history and forecast."""
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot history
    history_idx = np.arange(len(history))
    plt.plot(history_idx, history, 'b-', label='History')
    
    # Get forecast data
    forecast = forecast_response['forecast']
    dates = forecast_response['dates']
    
    # Create forecast indices
    forecast_idx = np.arange(len(history), len(history) + len(forecast))
    
    # Plot forecast
    plt.plot(forecast_idx, forecast, 'r-', label='Forecast')
    
    # Plot confidence intervals if available
    if 'lower_bound' in forecast_response and forecast_response['lower_bound'] is not None:
        lower_bound = forecast_response['lower_bound']
        upper_bound = forecast_response['upper_bound']
        plt.fill_between(forecast_idx, lower_bound, upper_bound, 
                         color='r', alpha=0.2, 
                         label='Confidence Interval')
    
    # Add labels and legend
    plt.axvline(x=len(history)-1, color='k', linestyle='--')
    plt.title('Time Series Forecast')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    
    # Add date labels on x-axis for forecast
    if dates:
        # Create a secondary x-axis for dates
        ax2 = plt.gca().twiny()
        ax2.set_xlim(plt.gca().get_xlim())
        
        # Set date ticks at forecast positions
        ax2.set_xticks(forecast_idx)
        
        # Format dates for better display
        formatted_dates = [datetime.strptime(d, "%Y-%m-%d").strftime("%b %Y") for d in dates]
        ax2.set_xticklabels(formatted_dates, rotation=45)
        
        # Move the second x-axis to the bottom
        ax2.spines['top'].set_position(('outward', 0))
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_position(('outward', 36))
        ax2.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_label_position('bottom')
        ax2.set_xlabel('Forecast Dates')
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure
    figures_dir = Path('reports/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / 'api_forecast.png')
    
    plt.show()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Generate synthetic data
    print("Generating synthetic time series...")
    series = generate_data(
        length=args.series_length,
        seasonality=12,
        trend=0.01,
        noise=0.1
    )
    
    print(f"Generated time series with {len(series)} data points")
    
    # Call forecast API
    print(f"Calling forecast API at {args.api_url}...")
    print(f"Model type: {args.model_type}")
    if args.model_type == "probabilistic":
        print(f"Loss type: {args.loss_type}" + (f" with alpha={args.loss_alpha}" if args.loss_type == "hybrid" else ""))
    
    forecast_response = call_forecast_api(
        api_url=args.api_url,
        values=series,
        model_type=args.model_type,
        n_steps=args.n_steps,
        sequence_length=args.sequence_length,
        loss_type=args.loss_type,
        loss_alpha=args.loss_alpha,
        low_bound_conf=args.low_bound_conf,
        high_bound_conf=args.high_bound_conf
    )
    
    # Print forecast summary
    print("\nForecast Summary:")
    print(f"Number of forecast steps: {len(forecast_response['forecast'])}")
    if 'dates' in forecast_response and forecast_response['dates']:
        print(f"Forecast period: {forecast_response['dates'][0]} to {forecast_response['dates'][-1]}")
    
    if 'lower_bound' in forecast_response and forecast_response['lower_bound'] is not None:
        print(f"Confidence interval: {args.low_bound_conf}% - {args.high_bound_conf}%")
    
    # Plot forecast
    print("\nPlotting forecast...")
    plot_forecast(series, forecast_response)
    
    print("\nDone! Forecast plot saved to reports/figures/api_forecast.png")

if __name__ == "__main__":
    main() 