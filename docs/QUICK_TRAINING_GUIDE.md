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

### Balanced Dataset Creation (Alternative)

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

## 2. Training a New Model

### Point Model

```bash
# Standard dataset - SMAPE loss (default)
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50

# Standard dataset - MSE loss
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --loss-type mse

# Balanced dataset - SMAPE loss (default)
python scripts/train.py --sample-size 17979000 --random-seed 42 --batch-size 64 --epochs 50

# Balanced dataset - MSE loss
python scripts/train.py --sample-size 17979000 --random-seed 42 --batch-size 64 --epochs 50 --loss-type mse
```

### Probabilistic Model

```bash
# Standard dataset - Gaussian NLL loss (default)
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --probabilistic

# Standard dataset - Hybrid loss
python scripts/train.py --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --probabilistic --loss-type hybrid --loss-alpha 0.8

# Balanced dataset - Gaussian NLL loss (default)
python scripts/train.py --sample-size 17979000 --random-seed 42 --batch-size 64 --epochs 50 --probabilistic

# Balanced dataset - Hybrid loss
python scripts/train.py --sample-size 17979000 --random-seed 42 --batch-size 64 --epochs 50 --probabilistic --loss-type hybrid --loss-alpha 0.8
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
# Continue training a weights-only model - Standard dataset with SMAPE loss
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10

# Continue training a weights-only model - Standard dataset with MSE loss
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --loss-type mse

# Optimized continuation - Standard dataset with MSE loss
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --loss-type mse --disable-memory-growth --batch-size 64 --aggressive-cleanup

# Continue training with a different standard sample - SMAPE loss
python scripts/create_dataset.py --start-series 1 --end-series 48000 --sample-size 2000 --random-seed 43
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --start-series 1 --end-series 48000 --sample-size 2000

# Continue training with a different standard sample - MSE loss
python scripts/create_dataset.py --start-series 1 --end-series 48000 --sample-size 2000 --random-seed 43
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --loss-type mse --start-series 1 --end-series 48000 --sample-size 2000

# Continue training with a balanced dataset - SMAPE loss
python scripts/create_balanced_dataset.py --random-seed 42
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --sample-size 17979000 --random-seed 42

# Continue training with a balanced dataset - MSE loss
python scripts/create_balanced_dataset.py --random-seed 42
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --loss-type mse --sample-size 17979000 --random-seed 42
```

## 5. Evaluating on the M4 Test Set

Evaluation requires a full model format (after conversion):

```bash
# Evaluate on a sample of series
python scripts/evaluate_m4.py --model_name transformer_1.0_directml_point_mse_M1_M48000_sampled1000_full --sample_size 400

# Full evaluation
python scripts/evaluate_m4.py --model_name transformer_1.0_directml_point_mse_M1_M48000_sampled1000_full --sample_size 48000
``` 