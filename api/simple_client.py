import requests
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def send_time_series_to_api(values, model_type="point", n_steps=36):
    """
    Send a time series to the forecasting API and get results back.
    
    Args:
        values: List or numpy array of time series values
        model_type: 'point' or 'probabilistic'
        n_steps: Number of steps to forecast
        
    Returns:
        Dictionary containing the forecast results
    """
    # API endpoint
    url = "http://localhost:8000/forecast/"
    
    # Convert numpy array to list if needed
    if isinstance(values, np.ndarray):
        values = values.tolist()
    
    # Prepare the request payload
    payload = {
        "values": values,
        "model_type": model_type,
        "n_steps": n_steps,
        "sequence_length": 60  # Default sequence length
    }
    
    # Add probabilistic model parameters if needed
    if model_type == "probabilistic":
        payload.update({
            "loss_type": "hybrid",
            "loss_alpha": 0.8,
            "low_bound_conf": 25,
            "high_bound_conf": 75,
            "num_samples": 1000
        })
    
    # Send the request
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return None

def plot_forecast_results(values, forecast_results):
    """
    Plot the original time series and the forecast results.
    
    Args:
        values: Original time series values
        forecast_results: Dictionary containing forecast results from the API
    """
    plt.figure(figsize=(12, 6))
    
    # Plot original values
    x_hist = np.arange(len(values))
    plt.plot(x_hist, values, 'b-', label='Historical Data')
    
    # Plot forecast
    forecast = forecast_results['forecast']
    x_forecast = np.arange(len(values), len(values) + len(forecast))
    plt.plot(x_forecast, forecast, 'r-', label='Forecast')
    
    # Plot confidence intervals if available
    if 'lower_bound' in forecast_results and forecast_results['lower_bound'] is not None:
        lower_bound = forecast_results['lower_bound']
        upper_bound = forecast_results['upper_bound']
        plt.fill_between(x_forecast, lower_bound, upper_bound, 
                         color='r', alpha=0.2, 
                         label='Confidence Interval')
    
    # Add vertical line at the forecast start
    plt.axvline(x=len(values)-1, color='k', linestyle='--')
    
    # Add labels and legend
    plt.title('Time Series Forecast')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    figures_dir = Path('reports/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / 'simple_api_forecast.png')
    
    plt.show()

def main():
    # Generate a simple synthetic time series (sine wave with noise)
    t = np.linspace(0, 4*np.pi, 100)
    values = np.sin(t) + 0.1 * np.random.randn(len(t))
    
    print("Sending time series to API...")
    
    # Send to API and get forecast
    forecast_results = send_time_series_to_api(
        values=values,
        model_type="point",  # Use "point" for point forecasts
        n_steps=36  # Forecast 36 steps ahead
    )
    
    if forecast_results:
        print("Forecast received successfully!")
        print(f"Number of forecast steps: {len(forecast_results['forecast'])}")
        
        # Print first few forecast values
        print("\nFirst 5 forecast values:")
        for i, val in enumerate(forecast_results['forecast'][:5]):
            print(f"Step {i+1}: {val:.4f}")
        
        # Print confidence intervals if available
        if 'lower_bound' in forecast_results and forecast_results['lower_bound'] is not None:
            print("\nWith confidence intervals (first 5 steps):")
            for i in range(5):
                lower = forecast_results['lower_bound'][i]
                upper = forecast_results['upper_bound'][i]
                forecast = forecast_results['forecast'][i]
                print(f"Step {i+1}: {forecast:.4f} [{lower:.4f}, {upper:.4f}]")
        
        # Plot the results
        plot_forecast_results(values, forecast_results)
        print("\nForecast plot saved to reports/figures/simple_api_forecast.png")
    else:
        print("Failed to get forecast from API.")

if __name__ == "__main__":
    main() 