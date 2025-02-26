# Transformer for Time Series Forecasting

## Introduction

This repository implements a Transformer architecture for time series forecasting that leverages the M4 competition dataset for pretraining. The goal is to create a robust, general-purpose time series model that can be fine-tuned for specific forecasting tasks. By pretraining on M4's diverse collection of 48,000 time series, the model learns fundamental temporal patterns that transfer well to various domains and use cases.

The model introduces several key innovations:

1. **Adaptive Sequence Processing**: Unlike traditional Transformers that require fixed-length sequences, our implementation uses a dynamic masking mechanism that allows the model to adapt to any time series length, making it versatile for different applications.

2. **Dual Prediction Modes**:
   - **Point Predictions**: Direct forecasting of future values using MSE loss, suitable for standard forecasting tasks
   - **Probabilistic Forecasting**: Uncertainty estimation by predicting both mean and variance, critical for risk assessment and decision-making

3. **Time-Aware Attention**: The model incorporates positional encoding specifically designed for time series data, enabling it to learn both short-term and long-term dependencies that are common across different types of time series.

4. **Intelligent Trend Handling**: The model features sophisticated trend detection and handling:
   - Automatic detection of sufficient non-zero data points for reliable trend estimation
   - Support for both STL decomposition and linear regression methods
   - Dynamic trend continuation with dampening for long-term forecasts
   - Zero-padding aware calculations to prevent trend distortion

Key Features:
- Pretrained on diverse time series from M4 competition
- Easy fine-tuning for specific forecasting tasks
- Supports recursive multi-step forecasting
- Provides uncertainty estimates for both probabilistic and point prediction models
- Intelligent trend handling with zero-padding awareness
- GPU-accelerated training and inference
- Scalable from small datasets to large-scale applications

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv_directml
.\venv_directml\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The main dependencies are:
- TensorFlow 2.10.0 with DirectML support
- NumPy 1.24.3
- Pandas 2.1.4
- Matplotlib 3.10.0
- scikit-learn 1.6.1

## Project Structure

```
├── config/
│   └── data_config.yaml     # Configuration for data processing
├── data/
│   ├── raw/                 # Raw M4 competition data
│   └── processed/           # Processed numpy arrays
├── models/
│   ├── checkpoints/         # Model checkpoints during training
│   └── final/              # Final trained models
├── reports/
│   └── figures/            # Generated plots and visualizations
├── scripts/
│   ├── create_dataset.py   # Create processed datasets
│   ├── validate_dataset.py # Validate processed datasets
│   ├── train.py           # Train the transformer model
│   └── test_predictions.py # Test model predictions
└── src/
    ├── data/              # Data processing modules
    ├── models/            # Model architecture
    └── visualization/     # Plotting utilities
```

## Workflow

### 1. Data Preparation

First, create the processed tensors from the raw M4 data:

```bash
python scripts/create_dataset.py --start-series 1 --end-series 50
```

This will:
- Load the raw M4 monthly data
- Process series M1 through M50
- Create sequences with proper padding
- Save the processed arrays in data/processed/

You can validate the created tensors:

```bash
python scripts/validate_dataset.py --start-series 1 --end-series 50
```

This will:
- Load the processed arrays
- Display basic statistics
- Show sample sequences
- Generate validation plots

### 2. Training

Train the transformer model:

```bash
python scripts/train.py --start-series 1 --end-series 50 --batch-size 32 --epochs 50
```

Parameters:
- `--start-series`: First series to include (e.g., 1 for M1)
- `--end-series`: Last series to include (e.g., 50 for M50)
- `--batch-size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--sequence-length`: Length of input sequences (default: 60)
- `--loss-type`: Type of loss function to use (options: 'mse', 'gaussian_nll', 'smape', 'hybrid')

#### Training Models

The script supports both point prediction and probabilistic models:

1. Point Prediction Model (MSE loss):
```bash
python scripts/train.py --start-series 1 --end-series 50 --loss-type mse
```

2. Probabilistic Model with Gaussian NLL:
```bash
python scripts/train.py --start-series 1 --end-series 50 --loss-type gaussian_nll
```

3. Probabilistic Model with sMAPE:
```bash
python scripts/train.py --start-series 1 --end-series 50 --loss-type smape
```

4. Hybrid Loss (combining sMAPE and Gaussian NLL):
```bash
python scripts/train.py --start-series 1 --end-series 50 --loss-type hybrid --loss-alpha 0.8
```

The `loss-alpha` parameter controls the weight between sMAPE (alpha) and Gaussian NLL (1-alpha). Higher alpha values give more weight to sMAPE.

Model files will be saved with descriptive names indicating the type:
- Point predictions: `transformer_1.0_directml_point_M1_M50/`
- Probabilistic with Gaussian NLL: `transformer_1.0_directml_proba_gaussian_nll_M1_M50/`
- Probabilistic with sMAPE: `transformer_1.0_directml_proba_smape_M1_M50/`
- Probabilistic with Hybrid loss: `transformer_1.0_directml_proba_hybrid_0.8_M1_M50/`

The training script will:
- Create a transformer model
- Train using DirectML GPU acceleration
- Save checkpoints in models/checkpoints/
- Save the final model in models/final/
- Generate training plots in reports/figures/

### 3. Testing

Test the trained model's predictions:

```bash
python scripts/test_predictions.py --start-series 1 --end-series 2 --n-steps 36
```

Parameters:
- `--start-series`: First series to test
- `--end-series`: Last series to test
- `--n-steps`: Number of steps to forecast (default: 36)
- `--loss-type`: Type of loss function used in training (default: 'mse')
- `--model-path`: Path to the model (optional, will use default if not specified)

This will:
- Load the trained model
- Generate recursive predictions with uncertainty bounds
- Create plots showing:
  - Original series (excluding zero padding)
  - Point predictions
  - Uncertainty bounds (calculated differently for probabilistic and point prediction models)
- Save plots in reports/figures/

## Model Architecture

The transformer model consists of:
- Input embeddings with positional encoding
- Multi-head self-attention layers
- Feed-forward networks
- Layer normalization
- Global average pooling
- Output layer:
  - Single value for point predictions
  - Mean and variance for probabilistic predictions

Key hyperparameters:
- Sequence length: 60
- Model dimension: 512
- Number of heads: 4
- Feed-forward dimension: 512
- Dropout rate: 0.05

## Trend Handling

The model implements sophisticated trend handling:

1. **Zero-Padding Awareness**:
   - Identifies and excludes zero-padded values from trend calculations
   - Requires minimum number of non-zero points for reliable trend estimation
   - Preserves trend information only in actual data regions

2. **Multiple Detrending Methods**:
   - STL Decomposition: For series with sufficient seasonal patterns
   - Linear Regression: For simpler trend patterns
   - Automatic method selection based on data characteristics

3. **Trend Continuation**:
   - Intelligent trend extrapolation for forecasts
   - Dampening factor to prevent unrealistic long-term growth
   - Separate handling for different trend types

## GPU Acceleration

The model uses DirectML for GPU acceleration, supporting both NVIDIA and AMD GPUs. The training script will automatically detect and use available GPUs.

## Results

The model generates:
- Training history plots (loss and metrics)
- Prediction plots for validation series
- Performance metrics (MSE, MAE)

Results are saved in:
- Model checkpoints: models/checkpoints/
- Final models: models/final/
- Plots: reports/figures/

## License

MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.