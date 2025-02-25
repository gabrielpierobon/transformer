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
pip install fastapi uvicorn tensorflow pandas numpy scikit-learn matplotlib
```

2. Make sure you have the trained models in the correct location:
   - Point model: `models/final/transformer_1.0_directml_point_M1_M2`
   - Probabilistic models: `models/final/transformer_1.0_directml_proba_[loss_type]_[alpha]_M1_M2`

## Running the API

Start the API server with:

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
  "model_type": "probabilistic",
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

### Using the Client Script

A client script is provided to demonstrate how to use the API:

```bash
cd api
python client_example.py --model-type probabilistic --loss-type hybrid --loss-alpha 0.8 --n-steps 36
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
    "model_type": "probabilistic",
    "n_steps": 36,
    "loss_type": "hybrid",
    "loss_alpha": 0.8
}

# Make API call
response = requests.post("http://localhost:8000/forecast/", json=payload)
forecast_data = response.json()

# Access forecast results
forecast = forecast_data["forecast"]
lower_bound = forecast_data["lower_bound"]
upper_bound = forecast_data["upper_bound"]
dates = forecast_data["dates"]
```

## API Documentation

When the API is running, you can access the interactive API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

These interfaces provide detailed information about the API endpoints, request/response models, and allow you to try out the API directly from your browser.

## Error Handling

The API returns appropriate HTTP status codes and error messages for different types of errors:

- 400 Bad Request: Invalid request parameters
- 500 Internal Server Error: Server-side errors (e.g., model loading errors)

Error responses include a detail field with a description of the error.

## Performance Considerations

- The API uses model caching to avoid reloading models for each request
- For large time series, consider reducing the number of samples for probabilistic forecasts to improve performance
- The API processes requests synchronously, so it may not be suitable for high-throughput applications without additional scaling 