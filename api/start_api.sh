#!/bin/bash

# Start the API server
echo "Starting Time Series Forecasting API..."
# Change to the directory where this script is located
cd "$(dirname "$0")"
python forecast_api.py 