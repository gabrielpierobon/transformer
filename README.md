# Transformer Time Series Forecasting

A deep learning project implementing transformer models for time series forecasting, with support for both point predictions and probabilistic forecasting. The project is designed to work with the M4 competition dataset and includes features for efficient data sampling and GPU acceleration via DirectML.

## Features

- 🤖 Transformer-based architecture with multi-head attention
- 📊 Support for both point predictions and probabilistic forecasting
- 🎯 Multiple loss functions: MSE, Gaussian NLL, sMAPE, and hybrid loss
- 🔄 Efficient data sampling for quick experimentation
- 💻 GPU acceleration with DirectML (supports both NVIDIA and AMD GPUs)
- 📈 Comprehensive visualization tools
- 🧪 Validation and testing pipeline

## Project Structure

```
.
├── api/                # API implementation
├── config/             # Configuration files
├── data/              # Data directory
│   ├── raw/           # Raw M4 competition data
│   └── processed/     # Processed numpy arrays
├── docs/              # Documentation
├── logs/              # Training logs
├── models/            # Saved models
│   ├── checkpoints/   # Model checkpoints
│   └── final/         # Final trained models
├── reports/           # Generated analysis reports
│   └── figures/       # Generated graphics
├── scripts/           # Utility scripts
├── src/               # Source code
│   ├── data/          # Data processing modules
│   ├── models/        # Model implementations
│   └── visualization/ # Visualization tools
└── tests/             # Test files
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

For quick experimentation with 1000 series:
```bash
python scripts/create_dataset.py --start-series 1 --end-series 48000 --sample-size 1000
```

For full dataset:
```bash
python scripts/create_dataset.py --start-series 1 --end-series 48000
```

### 3. Train Models

Train a point prediction model:
```bash
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000
```

Train a probabilistic model:
```bash
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --probabilistic --loss-type gaussian_nll
```

## Model Types

### Point Prediction Model

- Outputs single value predictions
- Uses MSE loss by default
- Metrics include MAE
- Best for applications requiring exact value forecasts

### Probabilistic Model

- Outputs both mean and uncertainty estimates
- Three loss function options:
  1. `gaussian_nll`: Gaussian Negative Log Likelihood
  2. `smape`: Symmetric Mean Absolute Percentage Error
  3. `hybrid`: Combination of sMAPE and Gaussian NLL
- Best for applications requiring uncertainty quantification

## Training Options

Key command-line arguments:

```bash
--start-series INT     # Starting series index
--end-series INT      # Ending series index
--sample-size INT     # Number of series to sample
--batch-size INT      # Training batch size
--epochs INT          # Number of training epochs
--sequence-length INT # Input sequence length
--probabilistic       # Enable probabilistic predictions
--loss-type STR      # Loss function type
--loss-alpha FLOAT   # Weight for hybrid loss
```

## Resource Requirements

Memory usage scales with sample size:
- 100 series: ~2GB RAM
- 1,000 series: ~8GB RAM
- 10,000 series: ~40GB RAM

Training time (approximate):
- 100 series: ~1 hour
- 1,000 series: ~10 hours
- 10,000 series: ~100 hours

## Best Practices

1. Start with small samples (100-500 series) for initial experiments
2. Use medium samples (1,000-5,000) for architecture tuning
3. Use large samples (10,000+) for final validation
4. Train on full dataset for production deployment
5. Monitor GPU memory usage during training
6. Use consistent random seeds for reproducibility

## Evaluation

Models are evaluated using:
- Validation set (20% of processed sequences)
- Multiple metrics (MAE, MSE, NLL for probabilistic models)
- Visualization of predictions and uncertainty

Run evaluation:
```bash
python scripts/test_predictions.py --start-series 1 --end-series 48000 --sample-size 1000 --n-steps 36
```

## Documentation

For detailed information about specific features:

- [Sampling Large Datasets](docs/sampling_large_datasets.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License Here]

## Acknowledgments

- M4 Competition for the dataset
- Microsoft for DirectML support
