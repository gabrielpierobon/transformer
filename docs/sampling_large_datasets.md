# Training with Sampled Time Series Data

This document explains how to use the sampling feature to train transformer models on a subset of the M4 competition dataset. This approach is useful when working with the full dataset (48,000 time series) would be computationally expensive or time-consuming.

## Why Sample Time Series?

The M4 competition dataset contains 48,000 diverse time series, which can be challenging to process and train on due to:

1. **Computational Resources**: Training on all series requires significant memory and processing power
2. **Training Time**: Full dataset training can take days or weeks depending on hardware
3. **Experimentation Speed**: Rapid iteration for architecture and hyperparameter tuning is difficult with the full dataset

By randomly sampling a subset of time series, you can:
- Accelerate the development cycle
- Test model architectures more quickly
- Validate approaches before committing to full dataset training
- Achieve reasonable performance with fewer computational resources

## Sampling Feature Implementation

The sampling feature is integrated into the existing data processing and training pipeline:

1. The `DatasetLoader` class supports random sampling of series
2. The `create_dataset.py` script includes a `--sample-size` parameter
3. The `train.py` script recognizes sampled datasets and maintains the naming convention
4. The `test_predictions.py` script works with models trained on sampled data

## How to Use Sampling

### Step 1: Create a Processed Dataset with Sampling

To create a processed dataset by randomly sampling 1000 series from the M1-M48000 range:

```bash
python scripts/create_dataset.py --start-series 1 --end-series 48000 --sample-size 1000
```

This command:
1. Loads the raw M4 competition data
2. Randomly samples 1000 series from the M1-M48000 range
3. Processes the sampled series into sequences
4. Saves the processed data with the naming convention:
   - `X_train_M1_M48000_sampled1000.npy`
   - `y_train_M1_M48000_sampled1000.npy`
   - `X_val_M1_M48000_sampled1000.npy`
   - `y_val_M1_M48000_sampled1000.npy`

### Step 2: Train the Model on the Sampled Dataset

You can train either a probabilistic model or a point prediction model on the sampled dataset.

#### Training a Probabilistic Model

For a probabilistic model that predicts both mean and variance (uncertainty estimation):

```bash
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 100 --probabilistic --loss-type gaussian_nll
```

This command:
1. Loads the processed data with the sampled1000 suffix
2. Trains a probabilistic transformer model with Gaussian NLL loss
3. Saves checkpoints with the naming convention:
   - `models/checkpoints/model_1.0_directml_proba_gaussian_nll_M1_M48000_sampled1000_{epoch:02d}.h5`
4. Saves the final model with the naming convention:
   - `models/final/transformer_1.0_directml_proba_gaussian_nll_M1_M48000_sampled1000`
5. Saves training history plots with the naming convention:
   - `reports/figures/training_history_1.0_directml_proba_gaussian_nll_M1_M48000_sampled1000.png`

#### Training a Point Prediction Model

For a point prediction model that forecasts exact values (without uncertainty):

```bash
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 100
```

Note the absence of the `--probabilistic` flag. This command:
1. Loads the processed data with the sampled1000 suffix
2. Trains a point prediction transformer model with MSE loss
3. Saves checkpoints with the naming convention:
   - `models/checkpoints/model_1.0_directml_point_M1_M48000_sampled1000_{epoch:02d}.h5`
4. Saves the final model with the naming convention:
   - `models/final/transformer_1.0_directml_point_M1_M48000_sampled1000`
5. Saves training history plots with the naming convention:
   - `reports/figures/training_history_1.0_directml_point_M1_M48000_sampled1000.png`

### Step 3: Test the Model's Predictions

The model is evaluated on the validation set, which consists of 20% of the processed sequences from the sampled time series (as specified by the `validation_split=0.2` parameter in the configuration).

To test the model's predictions:

```bash
python scripts/test_predictions.py --start-series 1 --end-series 48000 --sample-size 1000 --n-steps 36
```

This command:
1. Loads the trained model with the sampled1000 suffix
2. Loads the validation data (`X_val_M1_M48000_sampled1000.npy`)
3. For each validation sequence:
   - Uses the last 60 timesteps as input
   - Recursively generates the next 36 steps of predictions
   - Compares predictions with actual values

#### Evaluation Metrics

For point prediction models:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

For probabilistic models:
- Negative Log Likelihood (NLL)
- Mean Absolute Error (MAE) of the mean predictions
- Calibration metrics for uncertainty estimates

#### Visualization

The script generates plots for each validation sequence showing:
- The actual time series values (blue line)
- The predicted values (red line/points)
- For probabilistic models: uncertainty bands around predictions

The plots are saved in:
```
reports/figures/forecast_series_{index}.png
```

#### Interpreting Results

When evaluating models trained on sampled data, keep in mind:
- Performance is measured only on the validation sequences from the sampled series
- Results may vary from full dataset performance
- Larger sample sizes generally provide more reliable evaluation metrics
- Consider testing the final model on a separate holdout set from the full dataset

## Sampling Strategies

### Random Sampling

The current implementation uses random sampling with a fixed seed (42) for reproducibility. This ensures that:
- Different runs with the same parameters will produce the same samples
- Results can be compared across experiments

### Sample Size Considerations

When choosing a sample size, consider:

1. **Representativeness**: Larger samples better represent the full dataset
2. **Training Time**: Larger samples take longer to train
3. **Memory Requirements**: Larger samples require more memory

Recommended sample sizes:
- **Small experiments (quick iteration)**: 100-500 series
- **Medium experiments (architecture testing)**: 1,000-5,000 series
- **Large experiments (final validation)**: 10,000+ series

## Naming Conventions

The sampling feature uses consistent naming conventions:

1. **Processed Data Files**:
   ```
   {data_type}_{split}_M{start_series}_M{end_series}_sampled{sample_size}.npy
   ```
   Example: `X_train_M1_M48000_sampled1000.npy`

2. **Model Checkpoints**:
   ```
   model_{version}_{model_type}_{loss_type}_M{start_series}_M{end_series}_sampled{sample_size}_{epoch:02d}.h5
   ```
   Example: `model_1.0_directml_proba_gaussian_nll_M1_M48000_sampled1000_05.h5`

3. **Final Models**:
   ```
   transformer_{version}_{model_type}_{loss_type}_M{start_series}_M{end_series}_sampled{sample_size}
   ```
   Example: `transformer_1.0_directml_proba_gaussian_nll_M1_M48000_sampled1000`

4. **Training History Plots**:
   ```
   training_history_{version}_{model_type}_{loss_type}_M{start_series}_M{end_series}_sampled{sample_size}.png
   ```
   Example: `training_history_1.0_directml_proba_gaussian_nll_M1_M48000_sampled1000.png`

## Performance Considerations

### Training Time

Training time scales approximately linearly with the number of series:
- 100 series: ~1 hour (depending on hardware)
- 1,000 series: ~10 hours
- 10,000 series: ~100 hours

### Memory Usage

Memory usage also scales with the number of series:
- 100 series: ~2GB RAM
- 1,000 series: ~8GB RAM
- 10,000 series: ~40GB RAM

### GPU Acceleration

GPU acceleration is highly recommended for training, especially with larger sample sizes. The code uses DirectML for GPU support, which works with both NVIDIA and AMD GPUs.

## Best Practices

1. **Start Small**: Begin with a small sample (100-500 series) for initial experiments
2. **Increase Gradually**: Once you have a working model, increase the sample size
3. **Validate Results**: Compare performance across different sample sizes
4. **Use Consistent Seeds**: Keep the random seed consistent for reproducibility
5. **Monitor Resources**: Watch memory usage and training time as you scale up

## Example Workflow

A typical workflow using sampling might look like:

1. **Initial Exploration**: Train with 100 series to validate basic functionality
   ```bash
   python scripts/create_dataset.py --start-series 1 --end-series 48000 --sample-size 100
   python scripts/train.py --start-series 1 --end-series 48000 --sample-size 100 --epochs 20
   ```

2. **Architecture Tuning**: Train with 1,000 series to refine the model
   ```bash
   python scripts/create_dataset.py --start-series 1 --end-series 48000 --sample-size 1000
   python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --epochs 50
   ```

3. **Final Validation**: Train with 10,000 series for a more robust model
   ```bash
   python scripts/create_dataset.py --start-series 1 --end-series 48000 --sample-size 10000
   python scripts/train.py --start-series 1 --end-series 48000 --sample-size 10000 --epochs 100
   ```

4. **Full Dataset (Optional)**: Train on the entire dataset for production
   ```bash
   python scripts/create_dataset.py --start-series 1 --end-series 48000
   python scripts/train.py --start-series 1 --end-series 48000 --epochs 200
   ```

## Conclusion

The sampling feature provides a flexible way to work with the M4 competition dataset at different scales. By starting with smaller samples and gradually increasing the size, you can develop and refine your models more efficiently before committing to full dataset training.

For any questions or issues, please open an issue in the repository. 