# Slapformer: Transformer for Time Series Forecasting with Limited Historical Data

A deep learning project implementing Slapformer (Short-history Learning with Augmentation Probabilistic Former), a transformer-based approach for monthly time series forecasting that achieves competitive performance with only five years of historical data or less. This model is designed for real-world forecasting scenarios where extended historical data is often unavailable.

## Key Innovations

- 🚀 **Dual Data Augmentation Strategy** that generates both sliding window subsequences and multiple padded versions with varying historical context, multiplying training data to over 20 million examples from just 2,100 time series
- 🕰️ **Variable-Length Input Processing** enabling effective forecasting with as little as 12 months of historical data
- 📊 **Competitive Performance** (sMAPE of 13.38) using only 5 years or less of historical data, approaching sophisticated statistical approaches (13.00)
- 🔄 **Zero-Shot Generalization** to new domains without retraining, with optimal results through domain-specific fine-tuning
- 💻 **Compact Model Architecture** (1.58M parameters) that can be trained on modest hardware while maintaining strong performance

## Model Architecture

- **Transformer-based Design**: Utilizes self-attention mechanisms to capture complex temporal dependencies
- **Input Embedding + Positional Encoding**: Preserves temporal information in the time series
- **Multi-Head Attention (4 heads)**: Allows the model to focus on different aspects of the input sequence
- **Feed-Forward Network**: Processes normalized attention outputs with a dimension of 2048
- **Global Average Pooling**: Reduces the sequence dimension by averaging across time steps

## Project Structure

```
.
├── api/                # API implementation
├── config/             # Configuration files
├── data/               # Data directory
│   ├── raw/            # Raw M4 competition data
│   └── processed/      # Processed numpy arrays
├── docs/               # Documentation
├── logs/               # Training logs
├── models/             # Saved models
│   ├── checkpoints/    # Model checkpoints
│   └── final/          # Final trained models
├── reports/            # Generated analysis reports
│   └── figures/        # Generated graphics
├── scripts/            # Utility scripts
└── src/                # Source code
    ├── data/           # Data processing modules
    ├── models/         # Model implementations
    ├── classes/        # Model wrapper classes
    └── visualization/  # Visualization tools
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd transformer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv_directml
source venv_directml/bin/activate  # Linux/Mac
# OR
.\\venv_directml\\Scripts\\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Verify GPU Setup

```bash
python scripts/verify_gpu.py
```

### 2. Create Dataset

Three dataset types are supported with different augmentation strategies:
```bash
# Standard dataset with subsequence sampling
python scripts/create_dataset.py --start-series 1 --end-series 48000 --sample-size 1000 --random-seed 42

# Balanced dataset with equal representation across series types
python scripts/create_balanced_dataset.py --random-seed 42

# Rightmost dataset focusing on the most recent parts of series
python scripts/create_rightmost_dataset.py --random-seed 42
```

### 3. Train Models

```bash
# Train a point prediction model
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --dataset-type standard

# Train a probabilistic model with uncertainty quantification
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --probabilistic --loss-type gaussian_nll --dataset-type standard
```

## Forecasting Capabilities

### Point Prediction

- Outputs single value predictions
- Uses MSE or sMAPE loss functions
- Best for applications requiring exact value forecasts
- Demonstrated sMAPE of 13.38 across 5,100 diverse time series

### Probabilistic Forecasting

- Outputs both mean and variance (uncertainty) estimates
- Loss function options:
  - `gaussian_nll`: Gaussian Negative Log Likelihood
  - `smape`: Symmetric Mean Absolute Percentage Error
  - `hybrid`: Combination of sMAPE and Gaussian NLL
- Provides confidence intervals at various levels (e.g., 50%, 80%, 95%)
- Enables risk assessment and confidence-based decision making

## Real-World Applications

### Short History Forecasting
The model can generate reasonable forecasts with as little as 12 months of historical data, making it ideal for:
- New products or services with limited history
- Businesses with recent system changes affecting data relevance
- Applications where data collection has only recently begun

### Zero-Shot Prediction
The model can be applied to previously unseen time series without retraining:
- Cross-domain forecasting with minimal preprocessing
- Quick deployment across diverse business units
- Handling of new product lines with limited historical data

### Practical Business Uses
- **Inventory Management**: Setting stock levels based on forecasts and uncertainty
- **Resource Allocation**: Planning capacity based on expected demand
- **Financial Planning**: Creating budgets with confidence intervals
- **Sales Forecasting**: Predicting future sales with quantified uncertainty

## Evaluation

Models are evaluated using:
- Multiple metrics (sMAPE, MAE, MSE)
- Visualization of predictions and uncertainty
- Comparison against benchmark models

Run evaluation on the M4 dataset:
```bash
python scripts/evaluate_m4_scripts/evaluate_m4.py --model_name your_model_name --sample_size 1000
```

> **Note**: The M4 dataset can be downloaded from the [M4 Competition GitHub repository](https://github.com/Mcompetitions/M4-methods/tree/master/Dataset).

Test on the classic Air Passengers dataset:
```bash
python scripts/air_passengers_test.py --model_name your_model_name
```

For probabilistic forecasting with uncertainty:
```bash
python scripts/air_passengers_proba_forecast.py --model_name your_model_name --forecast_months 24 --confidence_levels 50 80 95
```

## Resource Requirements

Memory usage scales with sample size, but models can be trained on modest hardware:
- 100 series: ~2GB RAM
- 1,000 series: ~8GB RAM
- 10,000 series: ~40GB RAM

The compact architecture (1.58M parameters) enables training on consumer-grade GPUs, including:
- NVIDIA GeForce RTX 3050 (4GB VRAM)
- AMD Radeon Graphics GPUs via DirectML

## Documentation

For detailed information about specific features:

- [Quick Commands Guide](docs/QUICK_COMMANDS_GUIDE.md): Comprehensive reference of all command-line options for scripts with practical examples.
- [Sampling Large Datasets](docs/sampling_large_datasets.md): Strategies for efficiently working with large time series datasets through sampling techniques.
- [M4 Evaluation Guide](docs/m4_evaluation_guide.md): Step-by-step instructions for evaluating model performance on the M4 competition dataset.
- [Continue Training Guide](docs/continue_training_guide.md): Methods for resuming or extending training of existing models with new data or parameters.
- [Example Runs](docs/example_runs.md): Curated examples showcasing the model's performance on various time series with visualizations.
- [Model Format Guide](docs/model_format_guide.md): Explanation of model storage formats and conversion utilities for deployment.
- [Training Performance Optimization](docs/training_performance.md): Techniques to optimize training speed and memory usage on different hardware.

## Paper Reference

Please cite the following paper if you use this code:

```
@article{pierobon2023slapformer,
  title={Slapformer: A Transformer for Monthly Time Series Forecasting with Limited Historical Data},
  author={Pierobon, Gabriel},
  year={2023}
}
```

## License

[Your License Here]

## Acknowledgments

- M4 Competition for the dataset
- Microsoft for DirectML support
