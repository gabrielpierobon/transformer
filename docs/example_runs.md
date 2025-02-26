# Example Runs

This document provides examples of how to run the transformer models for time series forecasting.

## Air Passengers Demo

The Air Passengers dataset is a classic time series dataset that records monthly totals of international airline passengers from 1949 to 1960. We'll use it to demonstrate the forecasting capabilities of our transformer models.

### Running the Demo

1. **Activate your virtual environment**:
   ```bash
   # If using venv
   source venv/bin/activate  # On Unix/macOS
   .\venv\Scripts\activate  # On Windows
   ```

2. **Run the script with a point model**:
   ```bash
   python scripts/air_passengers_test.py --model_name transformer_1.0_directml_point_M1_M48000_sampled1000
   ```

   This will:
   - Load the Air Passengers dataset
   - Apply trend decomposition
   - Generate forecasts for the next 36 months
   - Display predictions in the terminal
   - Show a plot with:
     - Historical data (blue line)
     - Point forecasts (red line)
     - Confidence intervals (red shaded area)

### Expected Output

The script will show:
```
Time Series to predict:
            unique_id    y
ds                                
1949-01-01  series_1  112
1949-02-01  series_1  118
...

Point Predictions:
                     ds  q_0.5  unique_id
0  1961-01-01  417.3  series_1
1  1961-02-01  435.2  series_1
...
```

A plot will also be displayed showing:
- The historical passenger numbers (1949-1960)
- The forecasted values (next 36 months)
- Uncertainty bounds around the predictions

### Model Types

The script automatically detects the model type from its name:
- If the name contains "point" → Point prediction model
- If the name contains "proba" → Probabilistic model

### Available Models

Current models in the `models/final` directory:
- `transformer_1.0_directml_point_M1_M48000_sampled1000`: Point prediction model
- `transformer_1.0_directml_proba_hybrid_0.8_M1_M2`: Probabilistic model (if available)
- `transformer_1.0_directml_point_M1_M2`: Alternative point model (if available)

### Script Parameters

The script accepts the following parameters:
- `--model_name`: (Required) Name of the model directory in models/final/
- Future parameters may include:
  - Forecast horizon
  - Confidence interval bounds
  - Number of samples for uncertainty estimation
  - Input sequence length 