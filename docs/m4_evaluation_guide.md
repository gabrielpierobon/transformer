# M4 Evaluation Guide

This guide explains how to evaluate transformer models on the M4 competition test data using the `evaluate_m4.py` script.

## Overview

The evaluation process:

1. Loads the M4 training and test data
2. Randomly samples series from the dataset (or uses all series)
3. Takes the last 60 points of each training series
4. Generates forecasts for the next 18 months
5. Compares forecasts with actual values from the test set
6. Calculates performance metrics (MAE, MAPE, SMAPE, etc.)
7. Saves results to Excel files in the `evaluation/` directory

## Prerequisites

Before running the evaluation, ensure you have:

1. Downloaded the M4 competition data files:
   - `Monthly-train.csv`
   - `Monthly-test.csv`
   
   Place these files in the `data/raw/` directory.
   
   > The M4 dataset can be downloaded from the [M4 Competition GitHub repository](https://github.com/Mcompetitions/M4-methods/tree/master/Dataset).

2. Trained a transformer model or have a pre-trained model available in the `models/final/` directory.

## Running the Evaluation

### Basic Usage

To run the evaluation with default settings:

```bash
python scripts/evaluate_m4.py
```

This will:
- Use the default model (`transformer_1.0_directml_point_M1_M48000_sampled2000`)
- Sample 100 random series from the M4 dataset
- Generate forecasts for 18 months
- Calculate metrics and save results

### Command Line Arguments

The script supports several command line arguments:

```bash
python scripts/evaluate_m4.py --model_name MODEL_NAME --sample_size SAMPLE_SIZE --random_seed RANDOM_SEED
```

- `--model_name`: Name of the model directory in `models/final/` (default: `transformer_1.0_directml_point_M1_M48000_sampled2000`)
- `--sample_size`: Number of series to sample for evaluation (default: 100, use a larger number for more comprehensive evaluation)
- `--random_seed`: Random seed for reproducibility (default: 42)

### Examples

Evaluate with a specific model and larger sample size:

```bash
python scripts/evaluate_m4.py --model_name transformer_1.0_directml_proba_hybrid_0.8_M1_M2 --sample_size 1000
```

Evaluate on all available series:

```bash
python scripts/evaluate_m4.py --sample_size 48000
```

Use a different random seed:

```bash
python scripts/evaluate_m4.py --random_seed 123
```

## Configuring Detrending for Evaluation

The evaluation can be run with or without detrending by modifying the `inference_config.yaml` file. This is particularly useful if you want to evaluate models that were trained without detrending.

1. Open the `config/inference_config.yaml` file
2. Set the `apply_detrending` parameter:
   ```yaml
   # Whether to apply detrending during inference
   apply_detrending: false  # Set to false to disable detrending
   ```

## Understanding the Results

### Output Files

The evaluation script generates several output files in the `evaluation/` directory:

1. **Excel file with results**: `{model_name}_results_{sample_size}series_{timestamp}.xlsx`
   - Contains multiple sheets:
     - `Summary`: Overall metrics for each series and the entire dataset
     - `Detailed`: Detailed metrics for each horizon and series
     - `OWA Metrics`: Overall Weighted Average metrics

2. **Metrics distribution plot**: `{model_name}_metrics_distribution_{sample_size}series_{timestamp}.png`
   - Shows the distribution of different metrics across all evaluated series

3. **Benchmark comparison plot**: `{model_name}_benchmark_comparison_{sample_size}series_{timestamp}.png`
   - Shows the percentage of series that beat the Naïve2 benchmark

4. **Individual series plots**: Stored in `evaluation/plots/` directory
   - Each plot shows the historical data, forecast, and ground truth for a single series

### Key Metrics

The evaluation calculates several metrics:

- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error (primary metric used in M4)
- **RMSE**: Root Mean Squared Error
- **WMAPE**: Weighted Mean Absolute Percentage Error
- **MASE**: Mean Absolute Scaled Error

### Overall Weighted Average (OWA)

The OWA is a key metric from the M4 competition that combines:
- Relative SMAPE: The ratio of the model's SMAPE to the Naïve2 benchmark's SMAPE
- Relative MASE: The ratio of the model's MASE to the Naïve2 benchmark's MASE

OWA = (Relative SMAPE + Relative MASE) / 2

An OWA < 1 means the model outperforms the Naïve2 benchmark.

## Interpreting the Results

When analyzing the results, pay attention to:

1. **Overall OWA**: The primary measure of model performance relative to the benchmark
2. **Percentage of series that beat the benchmark**: Indicates consistency of performance
3. **Distribution of metrics**: Shows the spread of performance across different series
4. **Performance by horizon**: Reveals how accuracy changes with forecast distance

## Troubleshooting

### Common Issues

1. **Missing data files**:
   ```
   FileNotFoundError: M4 data files not found
   ```
   Ensure the M4 data files are in the `data/raw/` directory.

2. **Model not found**:
   ```
   ValueError: Model directory not found
   ```
   This error often occurs when trying to evaluate a model saved in the weights-only format. See [Understanding Model Formats and Conversion](model_format_guide.md) for details on how to fix this issue.
   
   Quick fix:
   ```bash
   python scripts/fix_model_format.py your_model_name
   ```
   Then evaluate using:
   ```bash
   python scripts/evaluate_m4.py --model_name your_model_name_full --sample_size 400
   ```

3. **Memory issues with large sample sizes**:
   If evaluating many series, you might encounter memory issues. Try reducing the sample size or running on a machine with more RAM.

4. **NaN values in results**:
   This can happen if the model produces invalid predictions or if there are issues with the metrics calculation. Check the logs for specific error messages.

## Advanced Usage

### Processing Series in Random Order

By default, the script processes series in a random order, which is useful when evaluating a large number of series and you want to monitor progress across a representative sample.

### Progress Tracking

The script provides detailed progress information:
- Percentage of series processed
- Processing speed (series per second)
- Estimated time remaining
- Current series being processed

This information is displayed every 10 series and at the end of processing.

### Customizing Plots

If you want to customize the generated plots, you can modify the `plot_series_forecast` and `plot_summary_metrics` functions in the script.

## Conclusion

The M4 evaluation script provides a comprehensive way to assess the performance of transformer models on time series forecasting tasks. By comparing against the Naïve2 benchmark using the OWA metric, you can determine how well your model performs relative to established baselines. 