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

## Synthetic Data Demo

The synthetic data demo uses generated time series with different patterns to test the transformer models. This is useful for testing model behavior with various types of time series patterns.

### Running the Demo

1. **Activate your virtual environment**:
   ```bash
   # If using venv
   source venv/bin/activate  # On Unix/macOS
   .\venv\Scripts\activate  # On Windows
   ```

2. **Run the script with a point model**:
   ```bash
   python scripts/dummy_data_test.py --model_name transformer_1.0_directml_point_M1_M48000_sampled1000
   ```

   Optional parameters:
   - `--input_length`: Length of input time series (default: 60)
   ```bash
   # Example with custom input length
   python scripts/dummy_data_test.py --model_name transformer_1.0_directml_point_M1_M48000_sampled1000 --input_length 48
   ```

   This will:
   - Generate three synthetic time series:
     1. Sine wave with trend
     2. Exponential growth
     3. Damped oscillation
   - Apply trend decomposition to each series
   - Generate forecasts for the next 24 months
   - Display predictions in the terminal
   - Show plots for each series with:
     - Historical data (blue line)
     - Point forecasts (red line)
     - Confidence intervals (red shaded area)

### Expected Output

The script will show:
```
Generated Time Series:
            unique_id      y
ds                                
2019-01-01     id_1   10.5
2019-02-01     id_1   11.2
...

Predictions for id_1:
                     ds  q_0.5  unique_id
0  2024-01-01  15.3      id_1
1  2024-02-01  16.2      id_1
...
```

Multiple plots will be displayed, one for each synthetic series, showing:
- The historical synthetic data
- The forecasted values
- Uncertainty bounds around the predictions

### Generated Series Types

The script generates three different types of time series:
1. **Sine wave with trend**: Combines periodic behavior with linear trend
2. **Exponential growth**: Shows exponential increase over time
3. **Damped oscillation**: Demonstrates decaying periodic behavior

These patterns help test the model's ability to handle different types of time series behaviors. 

## Validation Set Test

The validation set test uses real data from our validation set to evaluate model performance. It compares model predictions with ground truth values and calculates error metrics.

### Running the Demo

1. **Activate your virtual environment**:
   ```bash
   # If using venv
   source venv/bin/activate  # On Unix/macOS
   .\venv\Scripts\activate  # On Windows
   ```

2. **Run the script with a point model**:
   ```bash
   # Test with a random series
   python scripts/validation_set_test.py --model_name transformer_1.0_directml_point_M1_M48000_sampled1000

   # Test with a specific series
   python scripts/validation_set_test.py --model_name transformer_1.0_directml_point_M1_M48000_sampled1000 --series_index 42
   ```

   Parameters:
   - `--model_name`: (Required) Name of the model directory in models/final/
   - `--series_index`: (Optional) Index of the specific series to test. If not provided, a random series is selected.

   This will:
   - Load the validation data from `data/processed`
   - Select a series (random or specified)
   - Generate forecasts
   - Compare with ground truth values
   - Calculate error metrics (MSE, RMSE, MAE, MAPE)
   - Display a plot showing:
     - Historical data (blue line)
     - Model forecasts (red line)
     - Confidence intervals (red shaded area)
     - Ground truth values (green dashed line)

### Expected Output

The script will show:
```
Input Series:
            unique_id      y
ds                                
2019-01-01  validation_series  156.3
2019-02-01  validation_series  162.8
...

Prediction Metrics:
MSE: 245.6789
RMSE: 15.6742
MAE: 12.3456
MAPE: 8.7654
```

A plot will be displayed showing:
- The input time series
- The model's predictions
- The actual ground truth values
- Confidence intervals around predictions

### Data Files

The script uses the following files from `data/processed`:
- `X_val_M1_M48000_sampled1000.npy`: Validation input sequences
- `y_val_M1_M48000_sampled1000.npy`: Validation ground truth values

### Error Metrics

The script calculates several error metrics:
- **MSE** (Mean Squared Error): Average squared difference between predictions and ground truth
- **RMSE** (Root Mean Square Error): Square root of MSE, in the same units as the data
- **MAE** (Mean Absolute Error): Average absolute difference between predictions and ground truth
- **MAPE** (Mean Absolute Percentage Error): Average percentage difference between predictions and ground truth 