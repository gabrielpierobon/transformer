# Quick Training Guide

This guide provides a sequential workflow for training and evaluating transformer models.

## 1. Creating a Dataset

```bash
# Create a full dataset
python scripts/create_dataset.py --start-series 1 --end-series 48000

# Create a sampled dataset (faster)
python scripts/create_dataset.py --start-series 1 --end-series 48000 --sample-size 1000
```

## 2. Training a New Model

### Point Model

```bash
# With SMAPE loss (default)
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50

# With MSE loss
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --loss-type mse
```

### Probabilistic Model

```bash
# With Gaussian NLL loss (default)
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --probabilistic

# With hybrid loss
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --probabilistic --loss-type hybrid --loss-alpha 0.8
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

```bash
# Continue training a weights-only model (with .index file)
python scripts/continue_training.py models/final/transformer_1.0_directml_point_mse_M1_M48000_sampled1000 --epochs 10 --loss-type mse

# Optimized continuation
python scripts/continue_training.py models/final/transformer_1.0_directml_point_mse_M1_M48000_sampled1000 --epochs 10 --loss-type mse --disable-memory-growth --batch-size 64 --aggressive-cleanup
```

## 5. Evaluating on the M4 Test Set

Evaluation requires a full model format (after conversion):

```bash
# Evaluate on a sample of series
python scripts/evaluate_m4.py --model_name transformer_1.0_directml_point_mse_M1_M48000_sampled1000_full --sample_size 400

# Full evaluation
python scripts/evaluate_m4.py --model_name transformer_1.0_directml_point_mse_M1_M48000_sampled1000_full --sample_size 48000
``` 