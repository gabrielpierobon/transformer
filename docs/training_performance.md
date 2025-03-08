# Optimizing Training Performance

This guide explains how to optimize the performance of transformer model training, particularly for continued training sessions that might experience slowdowns.

## Common Performance Issues

When training transformer models, especially when continuing training from a saved model, you might encounter the following issues:

1. **Slower Epochs**: Continued training epochs taking much longer than initial training epochs
2. **Memory Leaks**: Increasing memory usage over time
3. **GPU Memory Fragmentation**: Inefficient use of GPU memory
4. **Out of Memory Errors**: Training crashes due to insufficient memory

## Using the Optimize Training Script

The easiest way to optimize training performance is to use the `optimize_training.py` script:

```bash
python scripts/optimize_training.py --model-path models/final/your_model_name --epochs 10
```

This script applies several optimizations automatically:

- Clears TensorFlow session before training
- Performs garbage collection
- Configures optimal GPU memory settings
- Provides options for mixed precision and memory growth

### Command Line Arguments

```bash
python scripts/optimize_training.py --model-path MODEL_PATH [OPTIONS]
```

- `--model-path`: Path to the model to continue training (required)
- `--batch-size`: Batch size for training (default: 16)
- `--memory-limit`: GPU memory limit in MB (default: 4096)
- `--disable-memory-growth`: Disable memory growth (can help with DirectML issues)
- `--epochs`: Number of epochs to train (default: 10)
- `--mixed-precision`: Enable mixed precision training

## Memory Optimization Options

### Memory Growth

By default, TensorFlow is configured to grow GPU memory allocation as needed. This can sometimes cause fragmentation issues with DirectML. You can disable memory growth with:

```bash
python scripts/continue_training.py models/final/your_model_name --disable-memory-growth
```

### Memory Limit

Setting an appropriate memory limit can prevent out-of-memory errors:

```bash
python scripts/continue_training.py models/final/your_model_name --memory-limit 4096
```

This sets a 4GB limit on GPU memory usage.

### Batch Size

Reducing batch size can significantly decrease memory usage:

```bash
python scripts/continue_training.py models/final/your_model_name --batch-size 8
```

### Mixed Precision

Mixed precision training uses float16 for certain operations, which can speed up training and reduce memory usage on compatible GPUs:

```bash
python scripts/continue_training.py models/final/your_model_name --mixed-precision
```

### Aggressive Memory Cleanup

For systems with severe memory issues, you can enable aggressive cleanup between epochs:

```bash
python scripts/continue_training.py models/final/your_model_name --aggressive-cleanup
```

This forces additional garbage collection and attempts to release memory back to the system.

## Troubleshooting Specific Issues

### Slow Continued Training

If continued training is much slower than initial training:

1. Try disabling memory growth:
   ```bash
   python scripts/continue_training.py models/final/your_model_name --disable-memory-growth
   ```

2. Use a larger batch size if memory allows:
   ```bash
   python scripts/continue_training.py models/final/your_model_name --batch-size 32
   ```

3. Enable mixed precision if your GPU supports it:
   ```bash
   python scripts/continue_training.py models/final/your_model_name --mixed-precision
   ```

### Out of Memory Errors

If you encounter out of memory errors:

1. Reduce batch size:
   ```bash
   python scripts/continue_training.py models/final/your_model_name --batch-size 4
   ```

2. Set a lower memory limit:
   ```bash
   python scripts/continue_training.py models/final/your_model_name --memory-limit 2048
   ```

3. Enable aggressive cleanup:
   ```bash
   python scripts/continue_training.py models/final/your_model_name --aggressive-cleanup
   ```

## Monitoring Performance

You can monitor GPU memory usage during training with tools like:

- Windows Task Manager (Performance tab)
- `nvidia-smi` for NVIDIA GPUs
- TensorBoard memory profiler

## Best Practices

1. **Start with Default Settings**: Begin with default settings and adjust as needed
2. **Experiment with Batch Size**: Find the optimal batch size for your model and hardware
3. **Monitor Memory Usage**: Keep an eye on memory usage during training
4. **Restart for Long Sessions**: For very long training sessions, consider saving and restarting periodically
5. **Use Mixed Precision**: Enable mixed precision on compatible GPUs for faster training

## Conclusion

By applying these optimization techniques, you can significantly improve training performance, especially for continued training sessions. The `optimize_training.py` script provides a convenient way to apply these optimizations automatically. 