import os
import sys
import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import gc  # Add garbage collector

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import get_model
from src.visualization.plot_utils import plot_random_subsequences

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train the transformer model')
    
    # Data parameters
    parser.add_argument(
        '--start-series',
        type=int,
        default=1,
        help='Starting series index (e.g., 1 for M1)'
    )
    parser.add_argument(
        '--end-series',
        type=int,
        default=500,
        help='Ending series index (e.g., 500 for M500)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Number of series used in sampling (if the dataset was sampled)'
    )
    
    # Training parameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,  # Reduced default batch size
        help='Batch size for training'
    )
    parser.add_argument(
        '--memory-limit',
        type=int,
        default=2048,  # 2GB default
        help='GPU memory limit in MB'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs to train'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=60,
        help='Length of input sequences'
    )
    parser.add_argument(
        '--probabilistic',
        action='store_true',
        help='Use probabilistic predictions'
    )
    
    # Model parameters
    parser.add_argument(
        '--version',
        type=str,
        default='1.0_directml',
        help='Model version identifier'
    )
    
    # Loss function parameters
    parser.add_argument(
        '--loss-type',
        type=str,
        choices=['gaussian_nll', 'smape', 'hybrid'],
        default='gaussian_nll',
        help='Type of loss function to use'
    )
    parser.add_argument(
        '--loss-alpha',
        type=float,
        default=0.9,
        help='Weight for sMAPE in hybrid loss (1-alpha for Gaussian NLL)'
    )
    
    return parser.parse_args()

def diagnose_gpu():
    """Diagnose GPU availability and configuration."""
    print("\n=== System and GPU Diagnostics ===")
    
    # System Information
    print("\n1. System Information:")
    import psutil
    total_ram = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB
    available_ram = psutil.virtual_memory().available / (1024 ** 3)  # Convert to GB
    print(f"Total RAM: {total_ram:.2f} GB")
    print(f"Available RAM: {available_ram:.2f} GB")
    print(f"CPU Count: {psutil.cpu_count()} cores")
    
    # Check TensorFlow's built-in GPU detection
    print("\n2. TensorFlow GPU Detection:")
    physical_devices = tf.config.list_physical_devices()
    print("All physical devices:", physical_devices)
    
    physical_gpus = tf.config.list_physical_devices('GPU')
    print("GPU devices:", physical_gpus)
    
    # Check DirectML
    print("\n3. DirectML Information:")
    try:
        import tensorflow_directml as tfdml
        print("TensorFlow DirectML plugin is installed")
        print(f"DirectML plugin version: {tfdml.__version__}")
        
        # Get DirectML adapter info
        try:
            import ctypes
            from tensorflow_directml import _pywrap_directml
            adapter_info = _pywrap_directml.get_adapter_info()
            print("\nDirectML Adapter Information:")
            print(f"Description: {adapter_info['Description']}")
            print(f"Vendor ID: {adapter_info['VendorID']}")
            print(f"Device ID: {adapter_info['DeviceID']}")
            print(f"Driver Version: {adapter_info['DriverVersion']}")
        except Exception as e:
            print(f"Could not get detailed DirectML adapter info: {e}")
            
    except ImportError:
        print("TensorFlow DirectML plugin is not installed")
    
    # Print TensorFlow build information
    print("\n4. TensorFlow Build Information:")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"Built with ROCm: {tf.test.is_built_with_rocm()}")
    
    # Memory information
    print("\n5. Memory Information:")
    try:
        if physical_gpus:
            for gpu in physical_gpus:
                memory_info = tf.config.experimental.get_memory_info(gpu.device_type)
                print(f"\nGPU Memory Info for {gpu.name}:")
                print(f"Total memory: {memory_info['current'] / (1024**3):.2f} GB")
                print(f"Peak memory usage: {memory_info['peak'] / (1024**3):.2f} GB")
    except Exception as e:
        print(f"Could not get GPU memory info: {e}")
    
    # Try to run a simple operation on GPU
    print("\n6. Testing GPU Operation:")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print("Successfully ran test operation on GPU")
            print("Result:", c.numpy())
    except Exception as e:
        print("Failed to run test operation on GPU")
        print("Error:", str(e))
    
    print("\n=== End of System and GPU Diagnostics ===\n")

def configure_gpu():
    """Configure GPU settings for optimal training."""
    # Get memory limit from arguments
    args = parse_args()
    memory_limit_mb = args.memory_limit
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Set memory limit based on argument
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                )
            
            # Print GPU information
            print("\nGPU Information:")
            print(f"Number of GPUs available: {len(gpus)}")
            print(f"GPU devices: {gpus}")
            print(f"GPU memory limit set to: {memory_limit_mb}MB")
            
        except RuntimeError as e:
            print(f"\nError configuring GPU: {e}")
    else:
        print("\nNo GPU devices found. Training will proceed on CPU.")
        print("Warning: Training on CPU will be significantly slower!")

def create_dataset_from_array(X, y, batch_size, is_training=True):
    """Create a TF dataset with memory-efficient loading."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if is_training:
        # Use a smaller shuffle buffer
        buffer_size = min(2000, len(X))
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
    
    # Use smaller prefetch and no caching
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def cleanup_memory():
    """Clean up memory before heavy operations."""
    # Clear TensorFlow session and its memory
    tf.keras.backend.clear_session()
    
    # Clear memory
    gc.collect()
    
    # Print memory status
    import psutil
    process = psutil.Process(os.getpid())
    print(f"\nMemory usage after cleanup: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initial memory cleanup
    cleanup_memory()
    
    # Run GPU diagnostics
    diagnose_gpu()
    
    # Configure GPU
    configure_gpu()
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Create series range string
    series_range = f'M{args.start_series}_M{args.end_series}'
    
    # Add sampling information if provided
    if args.sample_size is not None:
        series_range += f'_sampled{args.sample_size}'
    
    # Add model type and loss type to version
    model_type = 'proba' if args.probabilistic else 'point'
    if args.probabilistic and args.loss_type == 'hybrid':
        model_type += f'_hybrid_{args.loss_alpha}'
    elif args.probabilistic:
        model_type += f'_{args.loss_type}'
    model_version = f'{args.version}_{model_type}_{series_range}'
    
    # Define data paths
    base_directory = Path('data/processed/')
    x_train_path = base_directory / f'X_train_{series_range}.npy'
    x_val_path = base_directory / f'X_val_{series_range}.npy'
    y_train_path = base_directory / f'y_train_{series_range}.npy'
    y_val_path = base_directory / f'y_val_{series_range}.npy'
    
    # Check if processed data exists
    if not all(p.exists() for p in [x_train_path, x_val_path, y_train_path, y_val_path]):
        print(f"Processed data not found for series range {series_range}.")
        print("Please run create_dataset.py first with appropriate arguments.")
        sys.exit(1)
    
    # Load data in chunks to reduce memory usage
    print(f"Loading pre-processed data for series {series_range}...")
    
    # Use memory mapping for data loading
    X_train = np.load(x_train_path, mmap_mode='r')
    X_val = np.load(x_val_path, mmap_mode='r')
    y_train = np.load(y_train_path, mmap_mode='r')
    y_val = np.load(y_val_path, mmap_mode='r')
    
    print("Data loaded successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    
    # Plot random subsequences
    plot_random_subsequences(X_train, n=5, max_length=args.sequence_length)
    
    # Memory cleanup before dataset creation
    cleanup_memory()
    
    # Create datasets with memory-efficient loading
    print("\nCreating TensorFlow datasets...")
    train_dataset = create_dataset_from_array(X_train, y_train, args.batch_size, is_training=True)
    val_dataset = create_dataset_from_array(X_val, y_val, args.batch_size, is_training=False)
    
    # Memory cleanup before model creation
    cleanup_memory()
    
    # Get model with adjusted parameters for DirectML
    print("Building model...")
    model = get_model(
        sequence_length=args.sequence_length,
        probabilistic=args.probabilistic,
        loss_type=args.loss_type,
        loss_alpha=args.loss_alpha
    )
    model.summary()
    
    # Create directories for checkpoints and logs
    checkpoint_dir = Path('models/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Define callbacks with adjusted settings
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / f'model_{model_version}_{{epoch:02d}}.h5'),
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir / f'transformer_{model_version}'),
            histogram_freq=1,
            update_freq='epoch'
        ),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: cleanup_memory()
        )
    ]
    
    # Print training configuration
    print("\nTraining Configuration:")
    print(f"Series Range: {series_range}")
    print(f"Model Type: {model_type}")
    print(f"Loss Type: {args.loss_type}")
    if args.loss_type == 'hybrid':
        print(f"Loss Alpha: {args.loss_alpha}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Model Version: {model_version}")
    
    # Train model with adjusted parameters
    print("\nStarting training...")
    try:
        history = model.fit(
            train_dataset,
            epochs=args.epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        metric_name = 'mae_prob' if args.probabilistic else 'mae'
        plt.plot(history.history[metric_name], label=f'Training {metric_name.upper()}')
        plt.plot(history.history[f'val_{metric_name}'], label=f'Validation {metric_name.upper()}')
        plt.title(f'Model {metric_name.upper()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.upper())
        plt.legend()
        
        # Create figures directory if it doesn't exist
        figures_dir = Path('reports/figures')
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(figures_dir / f'training_history_{model_version}.png')
        plt.close()
        
        # Final evaluation
        print("\nEvaluating model...")
        val_loss, val_mae = model.evaluate(val_dataset, verbose=1)
        print(f"Final Validation Loss: {val_loss:.6f}")
        print(f"Final Validation MAE: {val_mae:.6f}")
        
        # Save final model with model type in name
        final_model_dir = Path('models/final')
        final_model_dir.mkdir(parents=True, exist_ok=True)
        model.save(final_model_dir / f'transformer_{model_version}')
        print(f"\nModel saved to {final_model_dir}/transformer_{model_version}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 