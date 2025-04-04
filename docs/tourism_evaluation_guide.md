# Tourism Dataset Evaluation Guide

This guide explains how to evaluate transformer models using the Tourism Monthly dataset as described in the paper "The Tourism Forecasting Competition" by Athanasopoulos et al. (2011).

## Dataset Overview

The Tourism Monthly dataset consists of 366 monthly tourism series. The paper uses these series to evaluate different forecasting methods with a forecast horizon of 24 periods. Key characteristics:

- 366 monthly tourism series
- Forecast horizon of 24 months
- Performance measured primarily using MAPE (Mean Absolute Percentage Error)

## Preparation

### 1. Download the Dataset

First, you need to download the tourism dataset in TSF format:

1. Create a directory for the raw data:
   ```bash
   mkdir -p data/raw
   ```

2. Download the Tourism Monthly dataset in TSF format and place it in the `data/raw` directory. The file should be named `tourism_monthly_dataset.tsf`.

### 2. Convert TSF to CSV Format

The dataset needs to be converted from TSF format to the CSV format that our model can use:

```bash
python scripts/convert_tourism_tsf_to_csv.py
```

This will:
- Parse the TSF file from `data/raw/tourism_monthly_dataset.tsf`
- Convert to a format with columns for unique_id, timestamp, and value
- Split the data into training and test sets
- Save the processed files to `data/processed/tourism_monthly_dataset.csv` and `data/processed/tourism_monthly_test.csv`

## Running the Evaluation

### Basic Evaluation

To evaluate a transformer model on the Tourism Monthly dataset:

```bash
python scripts/evaluate_tourism.py --model-name <MODEL_NAME> --sample-size 366 --forecast-horizon 24
```

Where:
- `<MODEL_NAME>` is the name of your trained model (e.g., `transformer_1.0_directml_point_mse_M1_M48000_sampled2101_full_4epoch`)
- `--sample-size 366` uses all 366 monthly series (as in the paper)
- `--forecast-horizon 24` sets the forecast horizon to 24 months (as in the paper)

### With Log Transformation

For series with increasing variance, you can apply log transformation:

```bash
python scripts/evaluate_tourism.py --model-name <MODEL_NAME> --sample-size 366 --forecast-horizon 24 --log-transform
```

### Benchmark Comparisons

The evaluation script automatically compares your model's performance against the Seasonal Naive (SNaive) benchmark. To also include a Naive2 benchmark in your comparison:

```bash
python scripts/evaluate_tourism.py --model-name <MODEL_NAME> --sample-size 366 --forecast-horizon 24 --include-naive2
```

This will:
- Calculate Naive2 forecasts for each series (using the average of the last 2 observed values)
- Include Naive2 metrics in the summary results
- Show improvement percentages over both SNaive and Naive2

You can combine this with other options:

```bash
python scripts/evaluate_tourism.py --model-name <MODEL_NAME> --sample-size 366 --forecast-horizon 24 --log-transform --include-naive2
```

### Using Batch Files

For convenience, you can use the provided batch file (Windows):

```bash
scripts\run_tourism_evaluation.bat
```

Or with log transformation:

```bash
scripts\run_tourism_evaluation.bat --log-transform
```

With Naive2 benchmark:
```bash
scripts\run_tourism_evaluation.bat --include-naive2
```

### Using Python Script (Linux/macOS)

For Linux/macOS users:

```bash
python scripts/run_tourism_evaluation.py
```

Or with log transformation:

```bash
python scripts/run_tourism_evaluation.py --log-transform
```

With Naive2 benchmark:
```bash
python scripts/run_tourism_evaluation.py --include-naive2
```

## Understanding the Results

The evaluation produces several outputs:

1. **Detailed Results**: CSV file with metrics for each series
2. **Summary Metrics**: Overall performance metrics across all series
3. **Plots**: Visualization of forecasts for each series in `evaluation/tourism/plots/`
4. **Benchmark Comparisons**: Comparison of your model against SNaive and Naive2 (if enabled)

The primary metric used in the paper is MAPE (Mean Absolute Percentage Error). Other metrics included are:

- MAE (Mean Absolute Error)
- SMAPE (Symmetric Mean Absolute Percentage Error)
- RMSE (Root Mean Square Error)
- WMAPE (Weighted Mean Absolute Percentage Error)
- MASE (Mean Absolute Scaled Error)

## Benchmarks from the Paper

For comparison, here are the MAPE values for monthly data from the original paper:

| Method             | Monthly MAPE |
|--------------------|--------------|
| SNaïve             | 22.56        |
| Theta              | 22.11        |
| ForePro            | 19.91        |
| ETS                | 21.15        |
| ARIMA              | 21.13        |
| N-BEATS-G (Theirs) | 19.17        |
| N-BEATS-I (Theirs) | 19.82        |
| N-BEATS-I+G (Theirs)| 19.29      |

A lower MAPE indicates better performance.

## Advanced Options

### Disabling Linear Detrending

If you want to disable linear detrending during inference:

```bash
python scripts/evaluate_tourism.py --model-name <MODEL_NAME> --sample-size 366 --forecast-horizon 24 --log-transform --no-detrend
```

This can be useful when you want to preserve the linear trend in the data.

### Changing Input Length

You can modify the input length used for prediction:

```bash
python scripts/evaluate_tourism.py --model-name <MODEL_NAME> --sample-size 366 --forecast-horizon 24 --input-length 60
```

The default is 60 data points.

## Implementing Naive2 Benchmark

The Naive2 benchmark generates forecasts by averaging the last 2 observed values in a time series. This simple method serves as an additional reference point that is slightly more sophisticated than the basic Naive method.

If you need to implement this benchmark yourself, here's a basic implementation:

```python
def naive2_forecast(series, forecast_horizon):
    """
    Generate Naive2 forecasts (average of last 2 observations)
    
    Args:
        series: Array of historical values
        forecast_horizon: Number of periods to forecast
        
    Returns:
        Array of forecasts
    """
    if len(series) < 2:
        # Fall back to Naive if not enough data
        return np.repeat(series[-1], forecast_horizon)
    
    # Average last 2 observations
    last_value = np.mean(series[-2:])
    
    # Repeat for the forecast horizon
    return np.repeat(last_value, forecast_horizon)
```

## References

Athanasopoulos, G., Hyndman, R. J., Song, H., & Wu, D. C. (2011). The tourism forecasting competition. International Journal of Forecasting, 27(3), 822-844. 