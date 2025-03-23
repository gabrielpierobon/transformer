# Quick Training Guide

This guide provides a sequential workflow for training and evaluating transformer models.

## 1. Creating a Dataset

### Standard Dataset Creation

```bash
# Create a full dataset
python scripts/create_dataset.py --start-series 1 --end-series 48000 --random-seed 42

# Create a sampled dataset (faster)
python scripts/create_dataset.py --start-series 1 --end-series 48000 --sample-size 1000 --random-seed 42

# Create a different sample using a different seed
python scripts/create_dataset.py --start-series 1 --end-series 48000 --sample-size 1000 --random-seed 43
```

The `--random-seed` parameter ensures reproducible sampling and allows you to create different samples by changing the seed value.

### Balanced Dataset Creation (Alternative 1)

This approach processes all series individually, taking an equal number of subsequences from each series to ensure balanced representation:

```bash
# Create a balanced dataset with 375 subsequences from each series (total ~18M subsequences)
python scripts/create_balanced_dataset.py --random-seed 42

# Adjust the number of subsequences per series
python scripts/create_balanced_dataset.py --subsequences-per-series 200 --random-seed 42

# Process only a subset of series
python scripts/create_balanced_dataset.py --start-series 1 --end-series 10000 --random-seed 42
```

The balanced approach ensures equal representation from all series, which can improve model generalization.

### Right-Most Subsequences Dataset (Alternative 2)

This approach focuses on the most recent data points in each time series, creating subsequences starting from the end of each series:

```bash
# Create a dataset with 375 right-most subsequences from each series
python scripts/create_rightmost_dataset.py --random-seed 42

# Adjust the number of subsequences per series
python scripts/create_rightmost_dataset.py --subsequences-per-series 200 --random-seed 42

# Process only a subset of series
python scripts/create_rightmost_dataset.py --start-series 1 --end-series 10000 --random-seed 42
```

The right-most approach prioritizes the most recent patterns in each time series, which can be particularly valuable for forecasting tasks where recent patterns are more relevant than older ones.

## 2. Training a New Model

### Dataset Types

The training script supports three dataset types:

- **standard**: Traditional datasets created with `create_dataset.py`
- **balanced**: Balanced datasets created with `create_balanced_dataset.py`
- **rightmost**: Right-most subsequences datasets created with `create_rightmost_dataset.py`

Use the `--dataset-type` parameter to specify which type of dataset to use.

### Model Architecture Information

During training, detailed information about the model architecture is displayed in the logs:
- Complete model summary showing all layers
- Total parameter count
- Number and percentage of trainable parameters
- Number and percentage of non-trainable parameters

This information is displayed right after the model is built, between clearly marked separator lines:
```
################################################################################
############################## MODEL ARCHITECTURE SUMMARY ##############################
################################################################################
...model layers, shapes, and parameter counts...

################################################################################
############################## MODEL PARAMETERS SUMMARY ##############################
################################################################################
Total parameters: X,XXX,XXX
Trainable parameters: X,XXX,XXX (100.00%)
Non-trainable parameters: 0 (0.00%)
################################################################################
```

This information helps you understand the complexity of your model and verify that it matches your expectations.

### Point Model

```bash
# Standard dataset - SMAPE loss (default)
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --dataset-type standard

# Standard dataset - MSE loss
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --loss-type mse --dataset-type standard

# Balanced dataset - SMAPE loss (default)
python scripts/train.py --sample-size 17979000 --random-seed 42 --batch-size 64 --epochs 50 --dataset-type balanced

# Balanced dataset - MSE loss
python scripts/train.py --sample-size 17979000 --random-seed 42 --batch-size 64 --epochs 50 --loss-type mse --dataset-type balanced

# Right-most dataset - SMAPE loss (default)
python scripts/train.py --sample-size 19177600 --random-seed 42 --batch-size 64 --epochs 50 --dataset-type rightmost

# Right-most dataset - MSE loss
python scripts/train.py --sample-size 19177600 --random-seed 42 --batch-size 64 --epochs 50 --loss-type mse --dataset-type rightmost
```

### Probabilistic Model

```bash
# Standard dataset - Gaussian NLL loss (default)
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --probabilistic --dataset-type standard

# Standard dataset - Hybrid loss
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --probabilistic --loss-type hybrid --loss-alpha 0.8 --dataset-type standard

# Balanced dataset - Gaussian NLL loss (default)
python scripts/train.py --sample-size 17979000 --random-seed 42 --batch-size 64 --epochs 50 --probabilistic --dataset-type balanced

# Balanced dataset - Hybrid loss
python scripts/train.py --sample-size 17979000 --random-seed 42 --batch-size 64 --epochs 50 --probabilistic --loss-type hybrid --loss-alpha 0.8 --dataset-type balanced

# Right-most dataset - Gaussian NLL loss (default)
python scripts/train.py --sample-size 19177600 --random-seed 42 --batch-size 64 --epochs 50 --probabilistic --dataset-type rightmost

# Right-most dataset - Hybrid loss
python scripts/train.py --sample-size 19177600 --random-seed 42 --batch-size 64 --epochs 50 --probabilistic --loss-type hybrid --loss-alpha 0.8 --dataset-type rightmost
```

## 3. Converting Weights to a Full Model

Models are saved in weights-only format (with .index and .data-* files). For evaluation, you need to convert to a full model format.

```bash
# Check model format
python scripts/check_model_format.py transformer_1.0_directml_point_mse_M1_M48000_sampled1000

# Convert to full model
python scripts/fix_model_format.py transformer_1.0_directml_point_mse_M1_M48000_sampled1000
```

## 4. Continuing Training

You can continue training from either weights-only models or full models:

### Dataset Selection for Continued Training

When continuing training, you need to specify which dataset to use:

- For **standard** datasets, the script will try to find a dataset matching the model's series range and sample size. If not found, it will try to find any available standard dataset.
- For **balanced** and **rightmost** datasets, you need to specify the sample size and random seed.

```bash
# Continue training a weights-only model - Standard dataset with SMAPE loss
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --dataset-type standard

# Continue training a weights-only model - Standard dataset with MSE loss
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --loss-type mse --dataset-type standard

# Optimized continuation - Standard dataset with MSE loss
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --loss-type mse --disable-memory-growth --batch-size 64 --aggressive-cleanup --dataset-type standard

# Continue training with a different standard sample - SMAPE loss
python scripts/create_dataset.py --start-series 1 --end-series 48000 --sample-size 2000 --random-seed 43
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --start-series 1 --end-series 48000 --sample-size 2000 --dataset-type standard

# Continue training with a different standard sample - MSE loss
python scripts/create_dataset.py --start-series 1 --end-series 48000 --sample-size 2000 --random-seed 43
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --loss-type mse --start-series 1 --end-series 48000 --sample-size 2000 --dataset-type standard

# Continue training with a balanced dataset - SMAPE loss
python scripts/create_balanced_dataset.py --random-seed 42
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --sample-size 17979000 --random-seed 42 --dataset-type balanced

# Continue training with a balanced dataset - MSE loss
python scripts/create_balanced_dataset.py --random-seed 42
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --loss-type mse --sample-size 17979000 --random-seed 42 --dataset-type balanced

# Continue training with a right-most dataset - SMAPE loss
python scripts/create_rightmost_dataset.py --random-seed 42
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --sample-size 19177600 --random-seed 42 --dataset-type rightmost

# Continue training with a right-most dataset - MSE loss
python scripts/create_rightmost_dataset.py --random-seed 42
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --loss-type mse --sample-size 19177600 --random-seed 42 --dataset-type rightmost
```

## 5. Evaluating on the M4 Test Set

Evaluation requires a full model format (after conversion):

```bash
# Evaluate on a sample of series
python scripts/evaluate_m4_scripts/evaluate_m4.py --model_name transformer_1.0_directml_point_mse_M1_M48000_sampled1000_full --sample_size 400

# Full evaluation
python scripts/evaluate_m4_scripts/evaluate_m4.py --model_name transformer_1.0_directml_point_mse_M1_M48000_sampled1000_full --sample_size 48000
```

## 6. Testing with Air Passengers Dataset

The Air Passengers test script allows you to evaluate your model on a well-known time series dataset and includes backtesting capabilities to compare predictions with actual values.

### Basic Usage

```bash
# Simple forecast without backtesting
python scripts/zero_shot_test/air_passengers_test.py --model_name transformer_1.0_directml_point_mse_M1_M48000_sampled1000_full
```

### Backtesting

You can evaluate how well your model performs on historical data by using the backtesting feature:

```bash
# Test with 12 months of backtesting and 24 months of future forecast
python scripts/zero_shot_test/air_passengers_test.py --model_name transformer_1.0_directml_point_mse_M1_M48000_sampled1000_full --backtest_months 12 --forecast_months 24
```

The backtesting feature:
- Uses the last N months (specified by --backtest_months) to compare predictions with actual values
- Generates M months of future predictions (specified by --forecast_months)
- Shows detailed metrics comparing predictions with actual values:
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)
- Displays a comparison table showing:
  - Actual values
  - Predicted values
  - Absolute error
  - Percentage error
- Plots both historical data and predictions in a single visualization

This helps you evaluate your model's performance before using it for future predictions.

### Log Transformation

You can apply log transformation to the data before forecasting, which is useful for time series with increasing variance (like the Air Passengers dataset). The log transformation helps stabilize the variance and often improves forecast accuracy for such data.

```bash
# Apply log transformation with backtesting
python scripts/zero_shot_test/air_passengers_test.py --model_name transformer_1.0_directml_point_mse_M1_M48000_sampled1000_full --backtest_months 12 --forecast_months 24 --log_transform

# Apply log transformation for simple forecasting
python scripts/zero_shot_test/air_passengers_test.py --model_name transformer_1.0_directml_point_mse_M1_M48000_sampled1000_full --log_transform
```

The script automatically applies the inverse transformation to the predictions, so the results are presented in the original scale. The plots and metrics will reflect the data in its original units, making it easy to interpret the results.

## 7. Testing with Probabilistic Forecasts

The probabilistic forecast script allows you to generate forecasts with confidence intervals using a probabilistic model, providing a range of possible future values rather than just a point forecast.

```bash
# Generate probabilistic forecasts with multiple confidence intervals
python scripts/probabilistic_capabilities_test/air_passengers_proba_forecast.py --model_name transformer_1.0_directml_proba_gaussian_nll_M1_M48000_sampled2103_full --forecast_months 24 --confidence_levels 50 80 95 --log_transform --num_samples 1000

# Test with limited history
python scripts/probabilistic_capabilities_test/air_passengers_proba_forecast.py --model_name transformer_1.0_directml_proba_gaussian_nll_M1_M48000_sampled2103_full --history_length 24 --log_transform

# Customize confidence levels
python scripts/probabilistic_capabilities_test/air_passengers_proba_forecast.py --model_name transformer_1.0_directml_proba_gaussian_nll_M1_M48000_sampled2103_full --confidence_levels 50 90 99
```

The probabilistic forecast script:
- Generates forecasts with multiple confidence intervals (default: 50%, 80%, 95%)
- Visualizes the forecast with shaded confidence intervals
- Creates a detailed forecast table with point predictions and confidence bounds
- Supports log transformation for data with increasing variance
- Allows testing with limited historical data
- Exports predictions to CSV for further analysis

This script is particularly useful for understanding the uncertainty in your forecasts and for making risk-informed decisions based on the range of possible future outcomes.

## 8. Testing Model Performance on Short Time Series

- Tests the model on multiple short time series lengths (default: 3, 6, 9, 12, 18, 24, 48, and 60 months)
- Predicts the same forecast horizon (default: 24 months) for each series length
- Automatically pads shorter series with zeros to meet the model's input requirements (default: 60 time steps)
- Creates a multi-panel visualization comparing predictions across different series lengths
- Shows metrics (MAPE, RMSE) for any predictions that overlap with actual data
- Exports a CSV with detailed metrics for each series length
- Supports log transformation for time series with increasing variance

This script is particularly useful for demonstrating that the transformer model can work effectively even with very short time series, which is traditionally a challenge for many forecasting methods.

### Interpreting the Results

The output visualization shows:
- Each panel represents a different time series length
- Blue line: Input data (historical values used for prediction)
- Gray dots: Zero padding (for series shorter than the model's input length)
- Green line: Actual future values (for comparison)
- Red dashed line: Model predictions
- Vertical dotted line: Boundary between historical and future data
- Gray vertical line: Boundary between padding and actual historical data (if padding was used)
- Metrics in the panel title: MAPE and RMSE for the portion where predictions overlap with actual data

The summary metrics table shows how prediction accuracy changes as the input time series length increases, helping you identify the minimum data requirements for achieving acceptable forecast accuracy with your model.

```bash
# Run short series test with a specific model and range of historical lengths
python scripts/short_series_test/air_passengers_short_series_test.py --model_name transformer_1.0_directml_point_mse_M1_M48000_sampled2103_full_4epochs --short_series_months 3 6 9 12 18 24 48 60 --forecast_months 24 
```

## 9. Testing with Google Trends Data

The Google Trends test scripts allow you to evaluate your model on search trend data, specifically iPhone search volume. This demonstrates the model's performance on real-world search interest data with seasonal patterns.

### Basic Google Trends Forecasting

```bash
# Simple forecast without backtesting
python scripts/short_series_test/google_trends_test.py --model_name your_model_name

# Test with 12 months of backtesting and 12 months of future forecast
python scripts/short_series_test/google_trends_test.py --model_name your_model_name --backtest_months 12 --forecast_months 12

# Apply log transformation (useful if the trend data has increasing variance)
python scripts/short_series_test/google_trends_test.py --model_name your_model_name --backtest_months 12 --log_transform
```

The Google Trends test script:
- Loads iPhone search volume data from Google Trends
- Supports backtesting to evaluate prediction accuracy on historical data
- Provides detailed metrics (MSE, RMSE, MAE, MAPE) for backtested predictions
- Visualizes both historical data and predictions in a single plot
- Creates a comparison table showing actual vs. predicted values
- Supports log transformation for data handling
- Works with both point and probabilistic forecast models

### Testing with Limited Google Trends History

```bash
# Test model performance with different amounts of historical data
python scripts/short_series_test/google_trends_short_series_test.py --model_name your_model_name

# Customize the series lengths to test
python scripts/short_series_test/google_trends_short_series_test.py --model_name your_model_name --short_series_months 3 6 12 24

# Apply log transformation and adjust forecast horizon
python scripts/short_series_test/google_trends_short_series_test.py --model_name your_model_name --log_transform --forecast_months 24

# Complete example with specific model and extended series lengths
python scripts/short_series_test/google_trends_short_series_test.py --model_name transformer_1.0_directml_point_mse_M1_M48000_sampled2103_full_4epochs --short_series_months 3 6 9 12 18 24 48 60 --forecast_months 24
```

The Google Trends short series script:
- Tests the model on multiple short time series lengths (default: 3, 6, 9, 12, 18, 24, and 36 months)
- Creates a multi-panel visualization comparing predictions across different series lengths
- Handles zero padding for series shorter than the model's input requirement
- Shows detailed metrics (MAPE, RMSE) for predictions that overlap with actual data
- Exports the metrics to a CSV file for further analysis
- Visualizes confidence intervals for probabilistic models

These Google Trends test scripts demonstrate the model's ability to forecast search interest patterns and its effectiveness with limited historical data, which is particularly valuable for new or emerging search trends. 

> **Note**: With the recent reorganization of scripts into subdirectories, the Python path imports in these scripts have been updated to use `Path(__file__).resolve().parents[2]` to correctly find the project root directory when executed from their new locations.

## 10. Script Organization Reference

The scripts have been reorganized into logical subdirectories for better project organization:

1. **Root Scripts** - Core functionality scripts remain in the main scripts directory:
   - Dataset creation (`create_dataset.py`, `create_balanced_dataset.py`, `create_rightmost_dataset.py`)
   - Training (`train.py`, `continue_training.py`)
   - Model utilities (`fix_model_format.py`, `check_model_format.py`, `convert_model_format.py`)
   - Performance tools (`optimize_training.py`, `verify_gpu.py`)

2. **Evaluation Scripts** - Located in `scripts/evaluate_m4_scripts/`:
   - `evaluate_m4.py` - Main M4 evaluation script
   - `evaluate_m4_full.py` - Extended M4 evaluation

3. **Zero-Shot Testing** - Located in `scripts/zero_shot_test/`:
   - `air_passengers_test.py` - Tests using the Air Passengers dataset
   - `google_trends_test.py` - Tests using Google Trends data

4. **Short Series Testing** - Located in `scripts/short_series_test/`:
   - `air_passengers_short_series_test.py` - Tests with limited Air Passengers history
   - `google_trends_short_series_test.py` - Tests with limited Google Trends history

5. **Probabilistic Capabilities** - Located in `scripts/probabilistic_capabilities_test/`:
   - `air_passengers_proba_forecast.py` - Probabilistic forecasting with confidence intervals

When running these scripts, be sure to use the correct path for the script location as shown in the examples throughout this guide. 