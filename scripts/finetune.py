#!/usr/bin/env python
# scripts/finetune.py

import sys
import os
from pathlib import Path
import logging
import argparse
import yaml
import numpy as np
import tensorflow as tf
import time
import gc
import json
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import psutil
import re
import subprocess
import traceback

# Add the project root directory to the Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.data.config import DataConfig
from src.data.dataset_loader import DatasetLoader
from src.models.transformer import get_model, gaussian_nll, smape_loss, hybrid_loss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def diagnose_gpu():
    """Diagnose GPU availability and configuration."""
    logger.info("\n=== System and GPU Diagnostics ===")
    
    # System Information
    logger.info("\n1. System Information:")
    total_ram = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB
    available_ram = psutil.virtual_memory().available / (1024 ** 3)  # Convert to GB
    logger.info(f"Total RAM: {total_ram:.2f} GB")
    logger.info(f"Available RAM: {available_ram:.2f} GB")
    logger.info(f"CPU Count: {psutil.cpu_count()} cores")
    
    # Check TensorFlow's built-in GPU detection
    logger.info("\n2. TensorFlow GPU Detection:")
    physical_devices = tf.config.list_physical_devices()
    logger.info(f"All physical devices: {physical_devices}")
    
    physical_gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"GPU devices: {physical_gpus}")
    
    # Check DirectML
    logger.info("\n3. DirectML Information:")
    try:
        import tensorflow_directml as tfdml
        logger.info("TensorFlow DirectML plugin is installed")
        logger.info(f"DirectML plugin version: {tfdml.__version__}")
        
        # Get DirectML adapter info
        try:
            import ctypes
            from tensorflow_directml import _pywrap_directml
            adapter_info = _pywrap_directml.get_adapter_info()
            logger.info("\nDirectML Adapter Information:")
            logger.info(f"Description: {adapter_info['Description']}")
            logger.info(f"Vendor ID: {adapter_info['VendorID']}")
            logger.info(f"Device ID: {adapter_info['DeviceID']}")
            logger.info(f"Driver Version: {adapter_info['DriverVersion']}")
        except Exception as e:
            logger.info(f"Could not get detailed DirectML adapter info: {e}")
            
    except ImportError:
        logger.info("TensorFlow DirectML plugin is not installed")
    
    # Print TensorFlow build information
    logger.info("\n4. TensorFlow Build Information:")
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    logger.info(f"Built with ROCm: {tf.test.is_built_with_rocm()}")

def configure_gpu(args):
    """Configure GPU settings."""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                if not args.disable_memory_growth:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError as e:
                        logger.warning(f"Could not set memory growth: {e}")
                
                # Set memory limit if specified
                memory_limit_mb = args.memory_limit
                try:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                    )
                except RuntimeError as e:
                    logger.warning(f"Could not set memory limit: {e}")
            
            # Print GPU information
            logger.info("\nGPU Information:")
            logger.info(f"Number of GPUs available: {len(gpus)}")
            logger.info(f"GPU devices: {gpus}")
            logger.info(f"GPU memory limit set to: {memory_limit_mb}MB")
            logger.info(f"Memory growth: {'Disabled' if args.disable_memory_growth else 'Enabled'}")
            
        except RuntimeError as e:
            logger.warning(f"\nWarning when configuring GPU: {e}")
            logger.warning("Training will continue with default GPU configuration.")
    else:
        logger.warning("\nNo GPU devices found. Training will proceed on CPU.")
        logger.warning("Warning: Training on CPU will be significantly slower!")

def load_config(config_path: str) -> DataConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return DataConfig(
        train_data_path=config['train_data_path'],
        test_data_path=config['test_data_path'],
        processed_data_path=config['processed_data_path'],
        extension=config['extension'],
        batch_size=config['batch_size'],
        validation_split=config['validation_split'],
        detrend_data=config.get('detrending', {}).get('enabled', False),
        min_points_stl=config.get('detrending', {}).get('min_points_stl', 12),
        min_points_linear=config.get('detrending', {}).get('min_points_linear', 5),
        force_linear_detrend=config.get('detrending', {}).get('force_linear', False),
        dtype_precision=config.get('dtype', {}).get('precision', 'float16')
    )

def load_pretrained_model(model_name: str, trainable_layers: Optional[List[str]] = None) -> tf.keras.Model:
    """Load a pre-trained model and set specified layers as trainable."""
    # Check if model path exists in both formats (full model or weights-only)
    model_path = Path(f'models/final/{model_name}')
    weights_path = model_path / 'model_weights'
    saved_model_path = model_path
    
    if not model_path.exists():
        logger.error(f"Error: Model path {model_path} does not exist.")
        
        # List available models
        models_dir = Path('models/final')
        if models_dir.exists():
            logger.info("\nAvailable models:")
            for item in models_dir.iterdir():
                if item.is_dir():
                    logger.info(f"  - {item.name} (full model)")
                elif item.name.endswith('.index'):
                    logger.info(f"  - {item.name[:-6]} (weights-only)")
        
        raise FileNotFoundError(f"Model {model_name} not found")

    # Extract model information from name
    model_type, loss_type = extract_model_info_from_name(model_name)
    logger.info(f"Detected model type: {model_type}")
    logger.info(f"Detected loss type: {loss_type}")

    # Try loading as SavedModel first
    try:
        logger.info(f"Attempting to load SavedModel from {saved_model_path}")
        model = tf.keras.models.load_model(
            str(saved_model_path),
            custom_objects={
                'gaussian_nll': gaussian_nll,
                'smape_loss': smape_loss,
                'hybrid_loss': hybrid_loss
            }
        )
        logger.info("Successfully loaded SavedModel format")
    except Exception as e:
        logger.warning(f"Could not load SavedModel format: {e}")
        
        # Try loading weights-only format
        if weights_path.with_suffix('.index').exists():
            logger.info(f"Loading weights from {weights_path}")
            
            # Load model configuration
            config_path = model_path / 'config.json'
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found at {config_path}")

            with open(config_path, 'r') as f:
                config = json.load(f)

            # Create a new model instance with the same configuration
            model = get_model(
                sequence_length=config.get('sequence_length', 60),
                probabilistic=config.get('probabilistic', False),
                loss_type=config.get('loss_type', 'mse'),
                loss_alpha=config.get('loss_alpha', 0.9)
            )
            
            # Load weights
            model.load_weights(str(weights_path))
            logger.info("Successfully loaded weights-only format")
        else:
            raise FileNotFoundError(f"Neither SavedModel nor weights found at {model_path}")

    # Set all layers to non-trainable first
    for layer in model.layers:
        layer.trainable = False

    # Make specified layers trainable
    if trainable_layers:
        for layer_name in trainable_layers:
            layer = None
            try:
                layer = model.get_layer(layer_name)
            except ValueError:
                logger.warning(f"Layer '{layer_name}' not found in model")
                continue

            if layer:
                logger.info(f"Setting layer '{layer_name}' as trainable")
                layer.trainable = True

    return model

def extract_model_info_from_name(model_name: str) -> Tuple[str, str]:
    """Extract model type and loss type from model name."""
    parts = model_name.split('_')
    
    model_type = "point"  # Default
    loss_type = None
    
    for i, part in enumerate(parts):
        # Check for point/probabilistic model type
        if part == 'point':
            model_type = "point"
            if i+1 < len(parts) and parts[i+1] in ['gaussian_nll', 'smape', 'hybrid', 'mse']:
                loss_type = parts[i+1]
        elif part == 'proba':
            model_type = "proba"
            if i+1 < len(parts) and parts[i+1] in ['gaussian_nll', 'smape', 'hybrid']:
                loss_type = parts[i+1]
        # Also check if the part itself is a loss type
        elif part in ['gaussian_nll', 'smape', 'hybrid', 'mse']:
            loss_type = part
    
    # Set default loss type if none found
    if not loss_type:
        loss_type = 'gaussian_nll' if model_type == 'proba' else 'mse'
    
    return model_type, loss_type

def extract_series_info_from_model_name(model_name: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Extract series information from model name."""
    series_range_match = re.search(r'M(\d+)_M(\d+)(?:_sampled(\d+))?', model_name)
    if series_range_match:
        start_series, end_series, sample_size = series_range_match.groups()
        return int(start_series), int(end_series), int(sample_size) if sample_size else None
    return None, None, None

def cleanup_memory(args=None):
    """Clean up memory before heavy operations."""
    # Clear TensorFlow session and its memory
    tf.keras.backend.clear_session()
    
    # Clear memory
    gc.collect()
    
    # More aggressive cleanup if requested
    if args and hasattr(args, 'aggressive_cleanup') and args.aggressive_cleanup:
        # Force a second garbage collection pass
        gc.collect()
        
        # Try to release as much memory as possible
        import ctypes
        if hasattr(ctypes, 'windll'):  # Windows
            try:
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
            except Exception as e:
                logger.warning(f"Could not release memory: {e}")
    
    # Print memory status
    process = psutil.Process(os.getpid())
    logger.info(f"Memory usage after cleanup: {process.memory_info().rss / 1024 / 1024:.2f} MB")

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

def prepare_dataset(config: DataConfig, dataset_suffix: str) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Prepare dataset for finetuning.
    
    Args:
        config: Data processing configuration
        dataset_suffix: Suffix for the dataset files
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Create output directory path
    output_dir = Path(config.processed_data_path)
    
    # Define file paths
    x_train_path = output_dir / f"X_train{dataset_suffix}.npy"
    y_train_path = output_dir / f"y_train{dataset_suffix}.npy"
    x_val_path = output_dir / f"X_val{dataset_suffix}.npy"
    y_val_path = output_dir / f"y_val{dataset_suffix}.npy"
    
    # Check if all files exist
    if not all(p.exists() for p in [x_train_path, y_train_path, x_val_path, y_val_path]):
        raise FileNotFoundError(f"Some dataset files not found with suffix: {dataset_suffix}")
    
    # Load data with memory mapping for efficiency
    logger.info("Loading pre-processed dataset files...")
    
    # Use memory mapping for data loading
    X_train = np.load(x_train_path, mmap_mode='r')
    X_val = np.load(x_val_path, mmap_mode='r')
    y_train = np.load(y_train_path, mmap_mode='r')
    y_val = np.load(y_val_path, mmap_mode='r')
    
    logger.info(f"Loaded dataset - X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"Loaded dataset - X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    # Create TensorFlow datasets
    logger.info("Creating TensorFlow datasets...")
    batch_size = config.batch_size
    train_dataset = create_dataset_from_array(X_train, y_train, batch_size, is_training=True)
    val_dataset = create_dataset_from_array(X_val, y_val, batch_size, is_training=False)
    
    logger.info("Datasets created successfully")
    
    return train_dataset, val_dataset

def finetune_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    model_name: str,
    epochs: int = 10,
    learning_rate: float = 0.0001,
    patience: int = 5,
    callbacks: list = None,
    args: Optional[argparse.Namespace] = None
) -> tf.keras.Model:
    """
    Finetune the model on new data.
    
    Args:
        model: TensorFlow model instance with pre-trained model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        model_name: Name to use for the finetuned model
        epochs: Number of epochs to train
        learning_rate: Learning rate for finetuning
        patience: Patience for early stopping
        callbacks: Additional callbacks
        args: Command line arguments
        
    Returns:
        The finetuned model
    """
    logger.info(f"Finetuning model with learning rate: {learning_rate}")
    
    # Set up optimizer with reduced learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Configure loss function
    if args.loss_type:
        if args.loss_type == 'gaussian_nll':
            loss = tf.keras.losses.GaussianNLL()
        elif args.loss_type == 'smape':
            def smape_loss(y_true, y_pred):
                epsilon = tf.keras.backend.epsilon()
                summ = tf.abs(y_true) + tf.abs(y_pred) + epsilon
                smape = 2.0 * tf.abs(y_pred - y_true) / summ
                return 100.0 * tf.reduce_mean(smape)
            loss = smape_loss
        elif args.loss_type == 'hybrid':
            if args.loss_alpha is None:
                args.loss_alpha = 0.5
            def hybrid_loss(y_true, y_pred):
                mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
                smape = smape_loss(y_true, y_pred)
                return args.loss_alpha * smape + (1 - args.loss_alpha) * mse
            loss = hybrid_loss
        else:
            loss = 'mse'
    else:
        # Try to infer loss from model name
        model_name = Path(args.pretrained_model).stem.lower()
        if 'gaussian' in model_name or 'nll' in model_name:
            loss = tf.keras.losses.GaussianNLL()
        elif 'smape' in model_name:
            def smape_loss(y_true, y_pred):
                epsilon = tf.keras.backend.epsilon()
                summ = tf.abs(y_true) + tf.abs(y_pred) + epsilon
                smape = 2.0 * tf.abs(y_pred - y_true) / summ
                return 100.0 * tf.reduce_mean(smape)
            loss = smape_loss
        else:
            # Default to MSE
            loss = 'mse'
    
    # Set up metrics
    metrics = ['mse']  # Always track MSE
    if loss != 'mse':
        # Add custom sMAPE metric
        def smape_metric(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            summ = tf.abs(y_true) + tf.abs(y_pred) + epsilon
            smape = 2.0 * tf.abs(y_pred - y_true) / summ
            return 100.0 * tf.reduce_mean(smape)
        metrics.append(smape_metric)
    
    # Compile model with new optimizer, loss and metrics
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    # Print model summary
    model.summary()
    
    # Set up callbacks
    if callbacks is None:
        callbacks = []
    
    # Add early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Add model checkpoint callback
    checkpoint_dir = Path(ROOT_DIR) / "models" / "checkpoints" / f"{model_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / "checkpoint-{epoch:02d}-{val_loss:.4f}"),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Add TensorBoard callback
    log_dir = Path(ROOT_DIR) / "logs" / "finetuning" / f"{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    callbacks.append(tensorboard_callback)
    
    # Add memory cleanup callback
    memory_cleanup_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: cleanup_memory(args)
    )
    callbacks.append(memory_cleanup_callback)
    
    # Finetune the model
    logger.info("Starting finetuning...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the finetuned model
    output_dir = Path(ROOT_DIR) / "models" / "final" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model.save_weights(str(output_dir / "model_weights"))
    
    # Save full model
    model.save(str(output_dir))
    
    # Check model structure to determine properties
    is_probabilistic = len(model.outputs[0].shape) > 1 and model.outputs[0].shape[-1] > 1
    
    # Try to determine input and output sequence lengths from model
    input_seq_len = model.input_shape[1] if hasattr(model, 'input_shape') else None
    output_seq_len = 1  # Default for most models
    
    # Save model configuration
    config = {
        "embedding_dim": 128,  # Default values, might not match actual model
        "num_heads": 4,
        "ff_dim": 2048,
        "dropout_rate": 0.1,
        "input_seq_len": input_seq_len,
        "output_seq_len": output_seq_len,
        "probabilistic": is_probabilistic,
        "loss_type": "gaussian_nll" if is_probabilistic else "mse",
        "finetuned_from": model_name.replace("_finetuned", "").replace("_tourism", ""),
        "finetuning_date": datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        "learning_rate": learning_rate,
        "epochs": epochs
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    # Save training history
    with open(output_dir / "history.json", 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        history_dict = {}
        for key, value in history.history.items():
            history_dict[key] = [float(v) for v in value]
        json.dump(history_dict, f, indent=4)
    
    # Print final metrics
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    logger.info(f"Final Training Loss: {final_loss:.6f}")
    logger.info(f"Final Validation Loss: {final_val_loss:.6f}")
    logger.info(f"Finetuned model saved to {output_dir}")
    
    return model

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Finetune a pre-trained transformer model')
    
    # Required arguments
    parser.add_argument(
        '--pretrained-model',
        type=str,
        required=True,
        help='Path to the pre-trained model to finetune'
    )
    
    # Dataset parameters
    parser.add_argument(
        '--dataset-suffix',
        type=str,
        default='_tourism',
        help='Suffix of the dataset files (default: _tourism)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs to finetune (default: 10)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate for finetuning (default: 0.0001)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Number of epochs to wait for improvement before early stopping (default: 5)'
    )
    
    # Model parameters
    parser.add_argument(
        '--freeze-layers',
        type=str,
        choices=['none', 'embeddings', 'attention', 'all_but_final'],
        default='none',
        help='Which layers to freeze during finetuning'
    )
    
    # Loss function parameters
    parser.add_argument(
        '--loss-type',
        type=str,
        choices=['gaussian_nll', 'smape', 'hybrid', 'mse'],
        help='Loss function to use (default: same as pretrained model)'
    )
    
    parser.add_argument(
        '--loss-alpha',
        type=float,
        help='Weight for sMAPE in hybrid loss (1-alpha for Gaussian NLL)'
    )
    
    # Memory optimization parameters
    parser.add_argument(
        '--memory-limit',
        type=int,
        help='GPU memory limit in MB'
    )
    
    parser.add_argument(
        '--disable-memory-growth',
        action='store_true',
        help='Disable memory growth (can help with some DirectML issues)'
    )
    
    parser.add_argument(
        '--mixed-precision',
        action='store_true',
        help='Enable mixed precision training'
    )
    
    parser.add_argument(
        '--aggressive-cleanup',
        action='store_true',
        help='Perform aggressive memory cleanup between epochs'
    )
    
    return parser.parse_args()

def load_tourism_dataset(dataset_suffix: str):
    """Load the processed tourism dataset."""
    data_dir = Path(ROOT_DIR) / "data" / "processed"
    
    try:
        X_train = np.load(data_dir / f"X_train{dataset_suffix}.npy")
        y_train = np.load(data_dir / f"y_train{dataset_suffix}.npy")
        X_val = np.load(data_dir / f"X_val{dataset_suffix}.npy")
        y_val = np.load(data_dir / f"y_val{dataset_suffix}.npy")
        
        logger.info(f"Loaded tourism dataset:")
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"X_val shape: {X_val.shape}")
        logger.info(f"y_val shape: {y_val.shape}")
        
        return X_train, y_train, X_val, y_val
    
    except Exception as e:
        logger.error(f"Error loading tourism dataset: {str(e)}")
        logger.error("Please run create_tourism_dataset.py first")
        sys.exit(1)

def freeze_model_layers(model, freeze_option: str):
    """Freeze specified layers of the model."""
    if freeze_option == 'none':
        return
    
    logger.info(f"Freezing layers: {freeze_option}")
    
    for layer in model.layers:
        if freeze_option == 'all_but_final':
            if 'output' not in layer.name.lower():
                layer.trainable = False
        elif freeze_option == 'embeddings':
            if 'embedding' in layer.name.lower():
                layer.trainable = False
        elif freeze_option == 'attention':
            if 'attention' in layer.name.lower():
                layer.trainable = False

def main():
    """Main function to finetune the model."""
    args = parse_args()
    
    # Load tourism dataset
    X_train, y_train, X_val, y_val = load_tourism_dataset(args.dataset_suffix)
    
    # Set up GPU memory growth if needed
    if not args.disable_memory_growth:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    
    # Set memory limit if specified
    if args.memory_limit:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.memory_limit)]
        )
    
    # Enable mixed precision if requested
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Load pre-trained model
    try:
        model = tf.keras.models.load_model(args.pretrained_model)
        logger.info(f"Loaded pre-trained model from {args.pretrained_model}")
    except Exception as e:
        logger.error(f"Error loading pre-trained model: {str(e)}")
        sys.exit(1)
    
    # Freeze layers as specified
    freeze_model_layers(model, args.freeze_layers)
    
    # Set up optimizer with new learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    # Configure loss function
    if args.loss_type:
        if args.loss_type == 'gaussian_nll':
            loss = tf.keras.losses.GaussianNLL()
        elif args.loss_type == 'smape':
            def smape_loss(y_true, y_pred):
                epsilon = tf.keras.backend.epsilon()
                summ = tf.abs(y_true) + tf.abs(y_pred) + epsilon
                smape = 2.0 * tf.abs(y_pred - y_true) / summ
                return 100.0 * tf.reduce_mean(smape)
            loss = smape_loss
        elif args.loss_type == 'hybrid':
            if args.loss_alpha is None:
                args.loss_alpha = 0.5
            def hybrid_loss(y_true, y_pred):
                mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
                smape = smape_loss(y_true, y_pred)
                return args.loss_alpha * smape + (1 - args.loss_alpha) * mse
            loss = hybrid_loss
        else:
            loss = 'mse'
    else:
        # Try to infer loss from model name
        model_name = Path(args.pretrained_model).stem.lower()
        if 'gaussian' in model_name or 'nll' in model_name:
            loss = tf.keras.losses.GaussianNLL()
        elif 'smape' in model_name:
            def smape_loss(y_true, y_pred):
                epsilon = tf.keras.backend.epsilon()
                summ = tf.abs(y_true) + tf.abs(y_pred) + epsilon
                smape = 2.0 * tf.abs(y_pred - y_true) / summ
                return 100.0 * tf.reduce_mean(smape)
            loss = smape_loss
        else:
            # Default to MSE
            loss = 'mse'
    
    # Set up metrics
    metrics = ['mse']  # Always track MSE
    if loss != 'mse':
        # Add custom sMAPE metric
        def smape_metric(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            summ = tf.abs(y_true) + tf.abs(y_pred) + epsilon
            smape = 2.0 * tf.abs(y_pred - y_true) / summ
            return 100.0 * tf.reduce_mean(smape)
        metrics.append(smape_metric)
    
    # Compile model with new optimizer, loss and metrics
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    # Print model summary
    model.summary()
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=args.patience // 2,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'logs/finetuning/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]

    try:
        history = model.fit(
            X_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model weights instead of the full model
        output_dir = Path('models/finetuned')
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = output_dir / f"finetuned_tourism_{Path(args.pretrained_model).stem}_weights"
        model.save_weights(weights_path)
        logger.info(f"Finetuning completed successfully!")
        logger.info(f"Model weights saved to: {weights_path}")
        
    except Exception as e:
        logger.error(f"Error during finetuning: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    finally:
        # Cleanup if requested
        if args.aggressive_cleanup:
            import gc
            gc.collect()
            tf.keras.backend.clear_session()
            
        # Log final model information
        try:
            logger.info("\nFinal model summary:")
            model.summary(print_fn=logger.info)
            trainable_count = sum(layer.count_params() for layer in model.trainable_weights)
            non_trainable_count = sum(layer.count_params() for layer in model.non_trainable_weights)
            total_count = trainable_count + non_trainable_count
            logger.info(f"\nTotal parameters: {total_count:,}")
            logger.info(f"Trainable parameters: {trainable_count:,} ({trainable_count/total_count*100:.2f}%)")
            logger.info(f"Non-trainable parameters: {non_trainable_count:,} ({non_trainable_count/total_count*100:.2f}%)")
        except Exception as e:
            logger.warning(f"Could not print final model information: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
        logger.error(f"Full error traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1) 