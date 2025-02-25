# Time Series Forecasting API

This API provides endpoints for generating time series forecasts using transformer models. It supports both point forecasts and probabilistic forecasts with confidence intervals.

## Features

- Generate point forecasts for time series data
- Generate probabilistic forecasts with confidence intervals
- Support for different loss functions (Gaussian NLL, SMAPE, Hybrid)
- Automatic handling of time series of any length (truncation or padding to required sequence length)
- Model caching for improved performance
- Health check endpoint

## Installation

1. Make sure you have all the required dependencies installed:

```bash
pip install fastapi uvicorn tensorflow pandas numpy scikit-learn matplotlib requests
```

2. Make sure you have the trained models in the correct location:
   - Point model: `models/final/transformer_1.0_directml_point_M1_M2`
   - Probabilistic model: `models/final/transformer_1.0_directml_proba_hybrid_0.8_M1_M2`

## Running the API

You can start the API server using one of the provided scripts:

### On Windows:

```bash
.\api\start_api.bat
```

### On Unix/Linux/Mac:

```bash
chmod +x ./api/start_api.sh
./api/start_api.sh
```

### Directly with Python:

```bash
cd api
python forecast_api.py
```

This will start the server at `http://localhost:8000`.

## API Endpoints

### Health Check

```
GET /health
```

Returns the health status of the API.

### Generate Forecast

```
POST /forecast/
```

Generates a forecast for the provided time series data.

#### Request Body

```json
{
  "values": [1.0, 2.0, 3.0, ...],
  "model_type": "point",
  "n_steps": 36,
  "sequence_length": 60,
  "num_samples": 1000,
  "low_bound_conf": 25,
  "high_bound_conf": 75,
  "loss_type": "hybrid",
  "loss_alpha": 0.8
}
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| values | array | Time series values | Required |
| model_type | string | Type of model to use ("point" or "probabilistic") | "point" |
| n_steps | integer | Number of steps to forecast | 36 |
| sequence_length | integer | Length of input sequences | 60 |
| num_samples | integer | Number of samples for probabilistic forecast | 1000 |
| low_bound_conf | integer | Lower bound percentile | 25 |
| high_bound_conf | integer | Upper bound percentile | 75 |
| loss_type | string | Loss type for probabilistic model ("hybrid", "gaussian_nll", or "smape") | "hybrid" |
| loss_alpha | float | Alpha value for hybrid loss | 0.8 |

#### Response

```json
{
  "forecast": [4.0, 5.0, 6.0, ...],
  "lower_bound": [3.5, 4.5, 5.5, ...],
  "upper_bound": [4.5, 5.5, 6.5, ...],
  "dates": ["2023-01-01", "2023-02-01", "2023-03-01", ...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| forecast | array | Forecasted values |
| lower_bound | array | Lower bound of confidence interval (only for probabilistic models) |
| upper_bound | array | Upper bound of confidence interval (only for probabilistic models) |
| dates | array | Dates for the forecast |

## Example Usage

### Using the Simple Client Script

A simple client script is provided to quickly test the API:

```bash
python api/simple_client.py
```

This will:
1. Generate a synthetic sine wave time series with noise
2. Send it to the API to generate a point forecast (no confidence intervals)
3. Print the forecast values
4. Plot the results and save the plot to `reports/figures/simple_api_forecast.png`

To use the probabilistic model instead, you can modify the script to change:
```python
model_type="point"
```
to:
```python
model_type="probabilistic"
```

### Using the Full Client Example

A more comprehensive client script is also provided with command-line options:

```bash
python api/client_example.py --model-type probabilistic --loss-type hybrid --loss-alpha 0.8 --n-steps 36
```

For point forecasts:
```bash
python api/client_example.py --model-type point --n-steps 36
```

This will:
1. Generate synthetic time series data
2. Call the API to generate a forecast
3. Plot the results and save the plot to `reports/figures/api_forecast.png`

### Using cURL

```bash
curl -X POST "http://localhost:8000/forecast/" \
     -H "Content-Type: application/json" \
     -d '{"values": [1.0, 2.0, 3.0, 4.0, 5.0], "model_type": "point", "n_steps": 12}'
```

### Using Python Requests

```python
import requests
import numpy as np

# Generate some time series data
data = np.sin(np.arange(100) * 0.1) + np.random.normal(0, 0.1, 100)

# Prepare request payload
payload = {
    "values": data.tolist(),
    "model_type": "point",  # Use "point" for point forecasts or "probabilistic" for probabilistic forecasts
    "n_steps": 36
}

# Add probabilistic parameters if needed
if payload["model_type"] == "probabilistic":
    payload.update({
        "loss_type": "hybrid",
        "loss_alpha": 0.8,
        "low_bound_conf": 25,
        "high_bound_conf": 75,
        "num_samples": 1000
    })

# Make API call
response = requests.post("http://localhost:8000/forecast/", json=payload)
forecast_data = response.json()

# Access forecast results
forecast = forecast_data["forecast"]
# For probabilistic forecasts only:
if "lower_bound" in forecast_data:
    lower_bound = forecast_data["lower_bound"]
    upper_bound = forecast_data["upper_bound"]
if "dates" in forecast_data:
    dates = forecast_data["dates"]
```

## API Documentation

When the API is running, you can access the interactive API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

These interfaces provide detailed information about the API endpoints, request/response models, and allow you to try out the API directly from your browser.

## Error Handling

The API returns appropriate HTTP status codes and error messages for different types of errors:

- 400 Bad Request: Invalid request parameters (e.g., empty values list)
- 500 Internal Server Error: Server-side errors (e.g., model loading errors)

Error responses include a detail field with a description of the error.

## Testing

The API includes a test suite that can be run with:

```bash
python api/test_api.py
```

This will run a series of tests to verify that the API is functioning correctly, including:
- Health check endpoint
- Point forecasting
- Probabilistic forecasting
- Error handling for invalid requests

## Performance Considerations

- The API uses model caching to avoid reloading models for each request
- For large time series, consider reducing the number of samples for probabilistic forecasts to improve performance
- The API processes requests synchronously, so it may not be suitable for high-throughput applications without additional scaling 