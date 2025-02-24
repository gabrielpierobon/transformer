import os
import sys
import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

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
    
    # Training parameters
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
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
    
    return parser.parse_args()

def diagnose_gpu():
    """Diagnose GPU availability and configuration."""
    print("\n=== GPU Diagnostics ===")
    
    # Check TensorFlow's built-in GPU detection
    print("\n1. TensorFlow GPU Detection:")
    physical_devices = tf.config.list_physical_devices()
    print("All physical devices:", physical_devices)
    
    physical_gpus = tf.config.list_physical_devices('GPU')
    print("GPU devices:", physical_gpus)
    
    # Check DirectML
    print("\n2. DirectML Information:")
    try:
        import tensorflow_directml as tfdml
        print("TensorFlow DirectML plugin is installed")
        print(f"DirectML plugin version: {tfdml.__version__}")
    except ImportError:
        print("TensorFlow DirectML plugin is not installed")
    
    # Print TensorFlow build information
    print("\n3. TensorFlow Build Information:")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Try to run a simple operation on GPU
    print("\n4. Testing GPU Operation:")
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
    
    print("\n=== End of GPU Diagnostics ===\n")

def configure_gpu():
    """Configure GPU settings for optimal training."""
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Print GPU information
            print("\nGPU Information:")
            print(f"Number of GPUs available: {len(gpus)}")
            print(f"GPU devices: {gpus}")
            print("GPU memory growth is enabled")
            
        except RuntimeError as e:
            print(f"\nError configuring GPU: {e}")
    else:
        print("\nNo GPU devices found. Training will proceed on CPU.")
        print("Warning: Training on CPU will be significantly slower!")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Run GPU diagnostics
    diagnose_gpu()
    
    # Configure GPU
    configure_gpu()
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Create series range string
    series_range = f'M{args.start_series}_M{args.end_series}'
    
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
    
    # Load the pre-processed arrays
    print(f"Loading pre-processed data for series {series_range}...")
    X_train = np.load(x_train_path)
    X_val = np.load(x_val_path)
    y_train = np.load(y_train_path)
    y_val = np.load(y_val_path)
    
    print("Data loaded successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")
    
    # Plot random subsequences
    plot_random_subsequences(X_train, n=5, max_length=args.sequence_length)
    
    # Create TensorFlow datasets with optimized settings
    buffer_size = min(10000, len(X_train))  # Prevent excessive memory usage
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.cache().shuffle(buffer_size).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Get model with adjusted parameters for DirectML
    print("Building model...")
    model = get_model(
        sequence_length=args.sequence_length,
        probabilistic=args.probabilistic
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
            filepath=str(checkpoint_dir / f'model_{args.version}_{series_range}_{{epoch:02d}}.h5'),
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
            log_dir=str(log_dir / f'transformer_{args.version}_{series_range}'),
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    
    # Print training configuration
    print("\nTraining Configuration:")
    print(f"Series Range: {series_range}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Probabilistic: {args.probabilistic}")
    print(f"Model Version: {args.version}")
    
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
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        # Create figures directory if it doesn't exist
        figures_dir = Path('reports/figures')
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(figures_dir / f'training_history_{args.version}_{series_range}.png')
        plt.close()
        
        # Final evaluation
        print("\nEvaluating model...")
        val_loss, val_mae = model.evaluate(val_dataset, verbose=1)
        print(f"Final Validation Loss: {val_loss:.6f}")
        print(f"Final Validation MAE: {val_mae:.6f}")
        
        # Save final model
        final_model_dir = Path('models/final')
        final_model_dir.mkdir(parents=True, exist_ok=True)
        model.save(final_model_dir / f'transformer_{args.version}_{series_range}')
        print(f"\nModel saved to {final_model_dir}/transformer_{args.version}_{series_range}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 