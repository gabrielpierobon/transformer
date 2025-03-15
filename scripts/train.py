import os
import sys
import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import gc  # Add garbage collector
import re  # Add regular expressions

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
    parser.add_argument(
        '--random-seed',
        type=int,
        help='Random seed used for dataset creation (needed for balanced datasets)'
    )
    parser.add_argument(
        '--dataset-type',
        type=str,
        choices=['standard', 'balanced', 'rightmost'],
        help='Type of dataset to use (standard, balanced, or rightmost)'
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
    
    # Continue training from saved model
    parser.add_argument(
        '--continue-from',
        type=str,
        help='Path to saved model to continue training from (e.g., models/final/transformer_1.0_directml_point_M1_M48000_sampled3)'
    )
    parser.add_argument(
        '--initial-epoch',
        type=int,
        default=0,
        help='Initial epoch to start from when continuing training (default: 0)'
    )
    parser.add_argument(
        '--original-gaussian-loss',
        action='store_true',
        help='Specify that the original model was trained with gaussian_nll loss (for point models)'
    )
    
    # Loss function parameters
    parser.add_argument(
        '--loss-type',
        type=str,
        choices=['gaussian_nll', 'smape', 'hybrid', 'mse'],
        help='Loss function type (default: gaussian_nll for probabilistic, smape for point)'
    )
    parser.add_argument(
        '--loss-alpha',
        type=float,
        default=0.9,
        help='Weight for sMAPE in hybrid loss (1-alpha for Gaussian NLL)'
    )
    
    # Memory optimization parameters
    parser.add_argument(
        '--disable-memory-growth',
        action='store_true',
        help='Disable memory growth (can help with some DirectML issues)'
    )
    
    parser.add_argument(
        '--mixed-precision',
        action='store_true',
        help='Enable mixed precision training (can speed up training on some GPUs)'
    )
    
    parser.add_argument(
        '--aggressive-cleanup',
        action='store_true',
        help='Perform aggressive memory cleanup between epochs'
    )
    
    args = parser.parse_args()
    
    # Set default loss type based on model type if not specified
    if args.loss_type is None:
        if args.probabilistic:
            args.loss_type = 'gaussian_nll'
        else:
            args.loss_type = 'smape'
    
    return args

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
    """Configure GPU settings."""
    # Get memory limit from arguments
    args = parse_args()
    memory_limit_mb = args.memory_limit
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                # Set memory growth based on argument
                if not args.disable_memory_growth:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError as e:
                        print(f"Warning: {e}")
                else:
                    print("Memory growth disabled as requested")
                
                # Set memory limit based on argument
                try:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                    )
                except RuntimeError as e:
                    print(f"Warning: Could not set memory limit: {e}")
            
            # Print GPU information
            print("\nGPU Information:")
            print(f"Number of GPUs available: {len(gpus)}")
            print(f"GPU devices: {gpus}")
            print(f"GPU memory limit set to: {memory_limit_mb}MB")
            print(f"Memory growth: {'Disabled' if args.disable_memory_growth else 'Enabled'}")
            
        except RuntimeError as e:
            print(f"\nWarning when configuring GPU: {e}")
            print("Training will continue with default GPU configuration.")
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
    # Get arguments
    args = parse_args()
    
    # Clear TensorFlow session and its memory
    tf.keras.backend.clear_session()
    
    # Clear memory
    gc.collect()
    
    # More aggressive cleanup if requested
    if hasattr(args, 'aggressive_cleanup') and args.aggressive_cleanup:
        # Force a second garbage collection pass
        gc.collect()
        
        # Try to release as much memory as possible
        import ctypes
        if hasattr(ctypes, 'windll'):  # Windows
            try:
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
            except Exception as e:
                print(f"Warning: Could not release memory: {e}")
    
    # Print memory status
    import psutil
    process = psutil.Process(os.getpid())
    print(f"\nMemory usage after cleanup: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Configure GPU first, before any TensorFlow operations
    configure_gpu()
    
    # Enable mixed precision if requested
    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("\nEnabled mixed precision training (float16)")
    
    # Initial memory cleanup
    cleanup_memory()
    
    # Run GPU diagnostics
    diagnose_gpu()
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Extract model information from model name or arguments
    if args.continue_from:
        model_path = Path(args.continue_from)
        index_file = Path(f"{args.continue_from}.index")
        
        # Check if model exists in either format
        if not model_path.exists() and not index_file.exists():
            print(f"Error: Model path {args.continue_from} does not exist.")
            
            # List available models
            models_dir = Path('models/final')
            if models_dir.exists():
                print("\nAvailable models:")
                for item in models_dir.iterdir():
                    if item.is_dir():
                        print(f"  - {item.name} (full model)")
                    elif item.name.endswith('.index'):
                        print(f"  - {item.name[:-6]} (weights-only)")
            
            sys.exit(1)
        
        # Extract model name from path
        if model_path.exists():
            model_name = model_path.name
        else:
            # For weights-only models, use the path without the directory
            model_name = model_path.name
        
        print(f"Loading model: {model_name}")
        
        # Extract model information from model name (format: transformer_version_type_seriesrange)
        parts = model_name.split('_')
        
        # Set default model type and loss type
        model_type = "point"  # Default
        loss_type = None
        
        # Extract model type (point/probabilistic) and loss type
        for i, part in enumerate(parts):
            # Check for point/probabilistic model type
            if part == 'point':
                args.probabilistic = False
                model_type = "point"
                
                # Check for loss type after "point"
                if i+1 < len(parts):
                    next_part = parts[i+1]
                    if next_part in ['gaussian_nll', 'smape', 'hybrid', 'mse']:
                        if not args.loss_type:  # Only set if not explicitly provided
                            args.loss_type = next_part
                            loss_type = next_part
                        print(f"Detected loss type from model name: {next_part}")
            
            elif part == 'proba':
                args.probabilistic = True
                model_type = "proba"
                
                # Check for loss type after "proba"
                if i+1 < len(parts):
                    next_part = parts[i+1]
                    if next_part in ['gaussian_nll', 'smape', 'hybrid']:
                        if not args.loss_type:  # Only set if not explicitly provided
                            args.loss_type = next_part
                            loss_type = next_part
                        print(f"Detected loss type from model name: {next_part}")
            
            # Also check if the part itself is a loss type (for models with format transformer_version_point_mse_...)
            elif part in ['gaussian_nll', 'smape', 'hybrid', 'mse']:
                if not args.loss_type and not loss_type:  # Only set if not already set
                    args.loss_type = part
                    loss_type = part
                    print(f"Detected loss type from model name: {part}")
        
        # Extract series range using regular expression
        # Pattern: M\d+_M\d+(_sampled\d+)?
        series_range_match = re.search(r'(M\d+_M\d+(?:_sampled\d+)?)', model_name)
        if series_range_match:
            series_range = series_range_match.group(1)
            print(f"Extracted series range from model name: {series_range}")
        else:
            # If we couldn't extract the series range, use default
            series_range = f'M{args.start_series}_M{args.end_series}'
            if args.sample_size is not None:
                series_range += f'_sampled{args.sample_size}'
            print(f"Using default series range: {series_range}")
        
        # If loss type wasn't found in the model name, use default or command line argument
        if not loss_type:
            if args.loss_type:
                loss_type = args.loss_type
                print(f"Using loss type from command line: {loss_type}")
            else:
                # Set default loss type based on model type
                if args.probabilistic:
                    loss_type = 'gaussian_nll'
                    args.loss_type = loss_type
                    print(f"Using default loss type for probabilistic model: {loss_type}")
                else:
                    # For point models, check if it's an older model
                    if args.original_gaussian_loss:
                        loss_type = 'gaussian_nll'
                        args.loss_type = loss_type
                        print(f"Using gaussian_nll loss for older point model (as specified by --original-gaussian-loss)")
                    else:
                        loss_type = 'smape'
                        args.loss_type = loss_type
                        print(f"Using default loss type for point model: {loss_type}")
    else:
        # New training session
        # Set model type based on arguments
        model_type = "proba" if args.probabilistic else "point"
        
        # Set loss type based on arguments
        if args.loss_type:
            loss_type = args.loss_type
        else:
            # Default loss types
            loss_type = "gaussian_nll" if args.probabilistic else "smape"
        
        # For hybrid loss, include alpha in model name
        loss_name = loss_type
        if loss_type == "hybrid" and args.loss_alpha is not None:
            loss_name = f"{loss_type}_{args.loss_alpha}"
        
        # Set series range for model name
        if args.sample_size:
            series_range = f"M{args.start_series}_M{args.end_series}_sampled{args.sample_size}"
        else:
            series_range = f"M{args.start_series}_M{args.end_series}"
        
        # Construct model name
        model_name = f"transformer_{args.version}_{model_type}_{loss_name}_{series_range}"
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path('models/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Define checkpoint callback
    checkpoint_path = checkpoint_dir / f'model_{model_name}_{{epoch:02d}}.h5'
    
    # Define data paths
    base_directory = Path('data/processed/')
    
    # Check for balanced dataset format first
    balanced_suffix = f"balanced_sampled{args.sample_size}_seed{args.random_seed}" if args.sample_size and hasattr(args, 'random_seed') else None
    
    # Check for right-most dataset format
    rightmost_suffix = f"rightmost_sampled{args.sample_size}_seed{args.random_seed}" if args.sample_size and hasattr(args, 'random_seed') else None
    
    # Try different naming patterns in order of preference
    data_suffixes = []
    
    # If dataset type is specified, prioritize that format
    if hasattr(args, 'dataset_type') and args.dataset_type:
        if args.dataset_type == 'balanced' and balanced_suffix:
            data_suffixes.append(balanced_suffix)
        elif args.dataset_type == 'rightmost' and rightmost_suffix:
            data_suffixes.append(rightmost_suffix)
        elif args.dataset_type == 'standard':
            # For standard datasets, try the exact series range first
            if args.sample_size:
                data_suffixes.append(f"{series_range}")
            
            # Then try with the command-line specified parameters
            if args.start_series and args.end_series and args.sample_size:
                cmd_series_range = f"M{args.start_series}_M{args.end_series}_sampled{args.sample_size}"
                if cmd_series_range != series_range:
                    data_suffixes.append(cmd_series_range)
            
            # Also try without sampling if specified
            if args.start_series and args.end_series:
                data_suffixes.append(f"M{args.start_series}_M{args.end_series}")
    else:
        # 1. Try balanced dataset format if sample_size is provided
        if balanced_suffix:
            data_suffixes.append(balanced_suffix)
        
        # 2. Try right-most dataset format if sample_size is provided
        if rightmost_suffix:
            data_suffixes.append(rightmost_suffix)
        
        # 3. Try series range with sampling
        if args.sample_size:
            data_suffixes.append(f"{series_range}")
        
        # 4. Try just the balanced part without seed
        if args.sample_size:
            data_suffixes.append(f"balanced_sampled{args.sample_size}")
        
        # 5. Try just the right-most part without seed
        if args.sample_size:
            data_suffixes.append(f"rightmost_sampled{args.sample_size}")
    
    # Always try the series range without sampling as a fallback
    series_range_no_sample = f"M{args.start_series}_M{args.end_series}"
    if series_range_no_sample not in data_suffixes:
        data_suffixes.append(series_range_no_sample)
    
    # Try each suffix until we find matching files
    found_data = False
    for suffix in data_suffixes:
        x_train_path = base_directory / f'X_train_{suffix}.npy'
        x_val_path = base_directory / f'X_val_{suffix}.npy'
        y_train_path = base_directory / f'y_train_{suffix}.npy'
        y_val_path = base_directory / f'y_val_{suffix}.npy'
        
        if all(p.exists() for p in [x_train_path, x_val_path, y_train_path, y_val_path]):
            found_data = True
            print(f"Found dataset with suffix: {suffix}")
            break
    
    # If we still haven't found the data and dataset_type is 'standard', try all available standard datasets
    if not found_data and hasattr(args, 'dataset_type') and args.dataset_type == 'standard':
        print("Searching for available standard datasets...")
        
        # List all available datasets
        x_train_files = list(base_directory.glob('X_train_M*.npy'))
        if x_train_files:
            # Sort by sample size (largest first) to prioritize larger datasets
            x_train_files.sort(key=lambda x: x.name, reverse=True)
            
            for file in x_train_files:
                suffix = file.name[8:-4]  # Remove 'X_train_' and '.npy'
                
                # Skip balanced and rightmost datasets
                if 'balanced_' in suffix or 'rightmost_' in suffix:
                    continue
                
                x_val_path = base_directory / f'X_val_{suffix}.npy'
                y_train_path = base_directory / f'y_train_{suffix}.npy'
                y_val_path = base_directory / f'y_val_{suffix}.npy'
                
                if all(p.exists() for p in [file, x_val_path, y_train_path, y_val_path]):
                    x_train_path = file
                    found_data = True
                    print(f"Found standard dataset with suffix: {suffix}")
                    break
    
    # If we still haven't found the data, try a direct file search
    if not found_data:
        print("Dataset not found with standard naming patterns. Searching for available datasets...")
        
        # List all available datasets
        x_train_files = list(base_directory.glob('X_train_*.npy'))
        if x_train_files:
            print("\nAvailable datasets:")
            for file in x_train_files:
                suffix = file.name[8:-4]  # Remove 'X_train_' and '.npy'
                print(f"  - {suffix}")
            
            # Ask user to specify dataset
            print("\nPlease run the script again with the appropriate dataset parameters.")
            sys.exit(1)
        else:
            print("No datasets found in data/processed/ directory.")
            print("Please run create_dataset.py, create_balanced_dataset.py, or create_rightmost_dataset.py first.")
            sys.exit(1)
    
    # Check if processed data exists
    if not all(p.exists() for p in [x_train_path, x_val_path, y_train_path, y_val_path]):
        print(f"Processed data not found for series range {series_range}.")
        print("Please run create_dataset.py or create_balanced_dataset.py first with appropriate arguments.")
        sys.exit(1)
    
    # Load data in chunks to reduce memory usage
    print(f"Loading pre-processed data...")
    
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
    print(f"Model configuration: probabilistic={args.probabilistic}, loss_type={args.loss_type}")
    if args.probabilistic:
        if args.loss_type == 'gaussian_nll':
            print("Using default Gaussian NLL loss for probabilistic model")
        elif args.loss_type == 'hybrid':
            print(f"Using hybrid loss with alpha={args.loss_alpha}")
        else:
            print(f"Using {args.loss_type} loss for probabilistic model")
    else:
        if args.loss_type == 'smape':
            print("Using default SMAPE loss for point model")
        else:
            print(f"Using {args.loss_type} loss for point model")
    
    # Set loss function based on arguments
    if args.loss_type == 'gaussian_nll' or (args.probabilistic and args.loss_type is None):
        loss_type = 'gaussian_nll'
        loss = gaussian_nll
    elif args.loss_type == 'smape' or (not args.probabilistic and args.loss_type is None):
        loss_type = 'smape'
        loss = smape_loss
    elif args.loss_type == 'mse':
        loss_type = 'mse'
        loss = 'mse'  # TensorFlow's built-in MSE loss
    elif args.loss_type == 'hybrid':
        loss_type = 'hybrid'
        # Use custom hybrid loss with alpha parameter
        alpha = args.loss_alpha if args.loss_alpha is not None else 0.5
        loss = lambda y_true, y_pred: hybrid_loss(y_true, y_pred, alpha)
    else:
        # Default to smape for point models, gaussian_nll for probabilistic
        if args.probabilistic:
            loss_type = 'gaussian_nll'
            loss = gaussian_nll
        else:
            loss_type = 'smape'
            loss = smape_loss
    
    model = get_model(
        sequence_length=args.sequence_length,
        probabilistic=args.probabilistic,
        loss_type=loss_type,
        loss_alpha=args.loss_alpha
    )
    model.summary()
    
    # Load model weights if continuing training
    if args.continue_from:
        try:
            # Check if we're loading from a weights-only model (with .index file)
            if Path(f"{args.continue_from}.index").exists():
                print(f"Loading weights from weights-only model: {args.continue_from}")
                model.load_weights(args.continue_from)
            # Check if we're loading from a full model (directory)
            elif Path(args.continue_from).is_dir():
                print(f"Loading weights from full model: {args.continue_from}")
                model.load_weights(f"{args.continue_from}/variables/variables")
            else:
                print(f"Error: Could not determine model format for {args.continue_from}")
                sys.exit(1)
                
            print("Model weights loaded successfully")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            sys.exit(1)
    
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
            filepath=str(checkpoint_path),
            save_weights_only=True,
            save_best_only=False,
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
            log_dir=str(log_dir / model_name),
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
    print(f"Loss Type: {loss_type}")
    if loss_type == 'hybrid':
        print(f"Loss Alpha: {args.loss_alpha}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Model Name: {model_name}")
    
    # Train model with adjusted parameters
    print("\nStarting training...")
    try:
        history = model.fit(
            train_dataset,
            epochs=args.epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1,
            initial_epoch=args.initial_epoch
        )
        
        # Save training history plot
        figures_dir = Path('reports/figures')
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        try:
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
            plt.plot(history.history[metric_name], label='Training MAE')
            plt.plot(history.history[f'val_{metric_name}'], label='Validation MAE')
            plt.title('Model MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(figures_dir / f'training_history_{model_name}.png')
            plt.close()
            
            # Print final metrics
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            final_mae = history.history[metric_name][-1]
            val_mae = history.history[f'val_{metric_name}'][-1]
            
            print(f"\nFinal Training Loss: {final_loss:.6f}")
            print(f"Final Validation Loss: {final_val_loss:.6f}")
            print(f"Final Training MAE: {final_mae:.6f}")
            print(f"Final Validation MAE: {val_mae:.6f}")
            
            # Save model
            final_model_dir = Path('models/final')
            final_model_dir.mkdir(parents=True, exist_ok=True)
            final_model_path = final_model_dir / model_name
            
            try:
                # Save weights only for easier continued training
                model.save_weights(str(final_model_path))
                print(f"\nModel weights saved to {final_model_path}")
            except Exception as e:
                print(f"\nError saving model weights: {e}")
            
            print("\nTraining completed successfully!")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            
            # Save the model even if interrupted
            try:
                final_model_dir = Path('models/final')
                final_model_dir.mkdir(parents=True, exist_ok=True)
                final_model_path = final_model_dir / f'{model_name}_interrupted'
                model.save_weights(str(final_model_path))
                print(f"\nInterrupted model saved to: {final_model_path}")
            except Exception as e:
                print(f"Error saving interrupted model: {e}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        
    finally:
        # Final cleanup
        cleanup_memory()
        print("\nTraining session ended.")

if __name__ == "__main__":
    main() 