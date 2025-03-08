# Understanding Model Formats and Conversion

This guide explains the different model saving formats used in the transformer project, how to convert between them, and how to troubleshoot common issues.

## Model Saving Formats

Our system supports two different ways of saving models:

1. **Full Model Format (Legacy)**:
   - Saves the entire model architecture and weights as a directory
   - Used by older versions of the training script
   - Creates a directory structure with model files
   - Required by older evaluation scripts

2. **Weights-Only Format (Current)**:
   - Saves only the model weights as files (TensorFlow checkpoint format)
   - Used by the current version of the training script
   - Creates files instead of directories (`.index` and `.data-*` files)
   - More flexible for continuing training
   - Requires the updated `ModelLoader` class for evaluation

## Identifying Model Formats

You can identify the format of a saved model by looking at its structure:

- **Full Model Format**: A directory containing model files
  ```
  models/final/transformer_1.0_directml_point_M1_M48000_sampled2000/
  ```

- **Weights-Only Format**: Files with `.index` and `.data-*` extensions
  ```
  models/final/transformer_1.0_directml_point_M1_M48000_sampled2001.index
  models/final/transformer_1.0_directml_point_M1_M48000_sampled2001.data-00000-of-00001
  ```

### Using the Check Model Format Script

We provide a convenient script to check the format of your models:

```bash
# List all models and their formats
python scripts/check_model_format.py

# Check a specific model
python scripts/check_model_format.py transformer_1.0_directml_point_M1_M48000_sampled2001
```

This script will tell you:
- What format your model is in
- Whether it needs conversion for evaluation
- The appropriate commands to use for evaluation or conversion

## Converting Between Formats

### Using the Fix Model Format Script (Recommended)

The easiest way to convert models is to use the `fix_model_format.py` script, which automatically detects the model format and converts it if needed:

```bash
python scripts/fix_model_format.py transformer_1.0_directml_point_M1_M48000_sampled2001
```

This script will:
1. Check if your model is in weights-only format or full model format
2. Convert it to the full model format if needed for evaluation
3. Tell you the exact command to use for evaluation

### Using the Convert Model Format Script (Advanced)

For more control over the conversion process, you can use the `convert_model_format.py` script directly:

```bash
# Convert from weights-only to full model format
python scripts/convert_model_format.py --model-path models/final/your_model_name --to-format full

# Convert from full model to weights-only format
python scripts/convert_model_format.py --model-path models/final/your_model_name --to-format weights
```

## Common Issues and Troubleshooting

### Model Not Found Error

If you encounter an error like `ValueError: Model directory not found` when evaluating a model:

1. Check if the model is in weights-only format (look for `.index` and `.data-*` files)
2. Run the fix model format script:
   ```bash
   python scripts/fix_model_format.py your_model_name
   ```
3. Use the converted model for evaluation:
   ```bash
   python scripts/evaluate_m4.py --model_name your_model_name_full --sample_size 400
   ```

### Conversion Fails

If the conversion process fails:

1. Check that the model files exist in the `models/final/` directory
2. Ensure you're using the correct model name (without file extensions)
3. Look for error messages in the output for specific issues

## Best Practices

1. **For Training**: Use the weights-only format (current default)
2. **For Evaluation**: Use the full model format or the updated `ModelLoader`
3. **For Continuing Training**: Use the weights-only format
4. **For Sharing Models**: Include both formats or instructions for conversion

## When to Convert Models

- **Convert to Full Model Format**: When you need to evaluate a model with older scripts that expect the full model format
- **Convert to Weights-Only Format**: When you need to continue training a model that was saved in full model format

## Conclusion

Understanding the different model formats and how to convert between them will help you avoid common issues when training and evaluating transformer models. The `fix_model_format.py` script provides a simple way to ensure your models are in the correct format for evaluation. 