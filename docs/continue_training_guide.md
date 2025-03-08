# Continuing Training from a Saved Model

This guide explains how to resume training from a previously saved transformer model. This is useful when you want to:

- Train a model for additional epochs
- Continue training that was interrupted
- Fine-tune an existing model on new data
- Adjust learning rate or other parameters for an existing model

## Quick Start

The simplest way to continue training is to use the `continue_training.py` script:

```bash
python scripts/continue_training.py models/final/your_model_name --epochs 10
```

This will continue training the model for 10 additional epochs.

## Understanding Model Types and Loss Functions

Our transformer models can be configured in different ways:

- **Point models**: Generate single-point forecasts (default)
  - Default loss function: SMAPE (Symmetric Mean Absolute Percentage Error)
  
- **Probabilistic models**: Generate probabilistic forecasts with uncertainty
  - Default loss function: Gaussian NLL (Negative Log-Likelihood)

The model name indicates its type:
- `transformer_1.0_directml_point_M1_M48000_sampled3`: A point model
- `transformer_1.0_directml_proba_M1_M48000_sampled3`: A probabilistic model
- `transformer_1.0_directml_point_gaussian_nll_M1_M48000_sampled3`: A point model with non-default Gaussian NLL loss

## Continuing Training with the Same Configuration

The `continue_training.py` script automatically detects the model type and configuration from the model name:

```bash
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled3 --epochs 10
```

This will:
1. Load the model weights
2. Detect the model type (point/probabilistic)
3. Determine the appropriate loss function
4. Continue training for 10 more epochs

## Options for Continuing Training

### Basic Options

- `--epochs`: Number of additional epochs to train (default: 10)
- `--initial-epoch`: Initial epoch to start from (useful for logging, default: 0)
- `--batch-size`: Override the batch size for training
- `--memory-limit`: Set GPU memory limit in MB

Example:
```bash
python scripts/continue_training.py models/final/your_model_name --epochs 20 --batch-size 16
```

### Advanced Options

- `--loss-type`: Override the detected loss function (`gaussian_nll`, `smape`, or `hybrid`)
- `--loss-alpha`: Weight for hybrid loss (if using hybrid loss)
- `--probabilistic`: Force probabilistic model (overrides detection)
- `--original-gaussian-loss`: Specify that an older point model was trained with Gaussian NLL loss

Example with advanced options:
```bash
python scripts/continue_training.py models/final/your_model_name --epochs 10 --loss-type smape
```

## Special Case: Older Point Models

For older point models trained before the default loss function was changed to SMAPE, you need to specify that they used Gaussian NLL loss:

```bash
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled3 --original-gaussian-loss
```

This ensures the model continues training with the same loss function it was originally trained with.

## Using the Underlying train.py Script Directly

You can also use the `train.py` script directly with the `--continue-from` parameter:

```bash
python scripts/train.py --continue-from models/final/your_model_name --epochs 10
```

This gives you access to all the training parameters available in `train.py`.

## Monitoring Training Progress

When continuing training, the script will:

1. Print the detected model configuration
2. Show memory usage after cleanup
3. Display training progress for each epoch
4. Save the final model to the `models/final/` directory

The model name will be preserved, so the continued model will overwrite the original unless you specify a different version with `--version`.

## Handling Memory Issues

If you encounter memory issues during continued training:

1. Reduce the batch size: `--batch-size 4`
2. Set a lower memory limit: `--memory-limit 1024`
3. Clean up other processes using GPU memory

The script includes memory cleanup between epochs to help manage memory usage.

For more advanced memory optimization options and troubleshooting slow training, see the [Training Performance Optimization Guide](training_performance.md).

## Examples

### Continue Training a Point Model

```bash
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled3 --epochs 10
```

### Continue Training a Probabilistic Model

```bash
python scripts/continue_training.py models/final/transformer_1.0_directml_proba_M1_M48000_sampled3 --epochs 10
```

### Continue Training with a Different Loss Function

```bash
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled3 --epochs 10 --loss-type gaussian_nll
```

### Continue Training an Older Model with Gaussian NLL Loss

```bash
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled3 --epochs 10 --original-gaussian-loss
```

### Continue Training with Performance Optimizations

```bash
python scripts/continue_training.py models/final/transformer_1.0_directml_point_M1_M48000_sampled3 --epochs 10 --disable-memory-growth --mixed-precision --batch-size 32
```

## Model Compatibility Between Training and Evaluation

### Understanding Model Saving Formats

Our system supports two different ways of saving models:

1. **Full Model Format (Legacy)**: Saves the entire model architecture and weights as a directory.
   - Used by older versions of the training script
   - Creates a directory structure with model files

2. **Weights-Only Format (Current)**: Saves only the model weights as files.
   - Used by the current version of the training script
   - Creates files instead of directories
   - More flexible for continuing training

For a detailed explanation of model formats and conversion, see the [Model Format Guide](model_format_guide.md).

### Compatibility Issues

When continuing training from a model saved in the full model format, the new model will be saved in the weights-only format. This can cause issues when trying to evaluate the model using the evaluation scripts, which might expect the old format.

We've updated the `ModelLoader` class to handle both formats, but you might encounter errors if you're using older evaluation scripts.

### Troubleshooting Model Loading Errors

If you encounter errors like `Model directory not found` when evaluating a model:

1. Check if the model was saved using the new weights-only format
2. Make sure you're using the latest version of the evaluation scripts
3. Use the `fix_model_format.py` script to convert your model:

```bash
# Convert model to the format needed for evaluation
python scripts/fix_model_format.py your_model_name
```

This will automatically detect your model format and convert it if needed.