import os
import sys
import tensorflow as tf
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_loader import DatasetLoader
from src.data.config import DataConfig
from src.models.transformer import get_model

def main():
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Load configuration
    config = DataConfig()
    
    # Initialize data loader
    data_loader = DatasetLoader(config)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, y_train, X_val, y_val = data_loader.load_data(verbose=True)
    
    # Create TensorFlow datasets
    buffer_size = 10000
    batch_size = 256
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.cache().shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Get model
    print("Building model...")
    model = get_model()
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/transformer_checkpoint.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_dataset,
        epochs=100,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('models/transformer_final.h5')
    print("Training completed. Model saved.")

if __name__ == "__main__":
    main() 