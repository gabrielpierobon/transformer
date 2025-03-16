#!/bin/bash

# Start the API server
echo "Starting Time Series Forecasting API..."
# Change to the directory where this script is located
cd "$(dirname "$0")"
python forecast_api.py 

# Add a pause at the end to keep the terminal window open
echo ""
echo "API execution completed."
read -p "Press Enter to close this window..." 