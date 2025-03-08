#!/usr/bin/env python
"""
Script to convert between different model saving formats.

This script can convert models between the full model format (directory)
and the weights-only format (files).
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import tensorflow as tf

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import get_model, gaussian_nll

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert between model formats')
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the model to convert'
    )
    
    parser.add_argument(
        '--to-format',
        type=str,
        choices=['full', 'weights'],
        required=True,
        help='Target format: "full" for full model, "weights" for weights-only'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        help='Output path for the converted model (default: original path with suffix)'
    )
    
    return parser.parse_args()

def extract_model_params(model_path):
    """Extract model parameters from the model name."""
    model_name = os.path.basename(model_path)
    
    # Determine if model is probabilistic
    is_probabilistic = "proba" in model_name.lower()
    
    # Determine loss type
    loss_type = "gaussian_nll"  # Default for probabilistic models
    if not is_probabilistic:
        if "gaussian_nll" in model_name:
            loss_type = "gaussian_nll"
        elif "smape" in model_name:
            loss_type = "smape"
        elif "hybrid" in model_name:
            loss_type = "hybrid"
        else:
            # For older point models, default to smape unless specified
            loss_type = "smape"
    
    return {
        "is_probabilistic": is_probabilistic,
        "loss_type": loss_type
    }

def convert_to_full_model(model_path, output_path):
    """Convert a weights-only model to a full model."""
    logger.info(f"Converting weights-only model at {model_path} to full model")
    
    # Extract model parameters
    params = extract_model_params(model_path)
    
    # Create model with the same architecture
    model = get_model(
        sequence_length=60,  # Default value, will be overridden by weights
        probabilistic=params["is_probabilistic"],
        loss_type=params["loss_type"]
    )
    
    # Compile the model
    custom_objects = {}
    if params["loss_type"] == "gaussian_nll" or params["is_probabilistic"]:
        custom_objects["loss"] = gaussian_nll
        custom_objects["gaussian_nll"] = gaussian_nll
    
    model.compile()
    
    # Load weights
    try:
        # Check if we need to add .index extension (TensorFlow checkpoint format)
        if not os.path.exists(model_path) and os.path.exists(model_path + '.index'):
            logger.info(f"Detected TensorFlow checkpoint format")
            model_path_to_load = model_path
        else:
            model_path_to_load = model_path
            
        model.load_weights(model_path_to_load)
        logger.info(f"Successfully loaded weights from {model_path}")
    except Exception as e:
        logger.error(f"Error loading weights: {str(e)}")
        raise ValueError(f"Failed to load weights: {str(e)}")
    
    # Save as full model
    try:
        model.save(output_path, save_format='tf')
        logger.info(f"Successfully saved full model to {output_path}")
    except Exception as e:
        logger.error(f"Error saving full model: {str(e)}")
        raise ValueError(f"Failed to save full model: {str(e)}")

def convert_to_weights_only(model_path, output_path):
    """Convert a full model to a weights-only model."""
    logger.info(f"Converting full model at {model_path} to weights-only model")
    
    # Load the full model
    custom_objects = {}
    if "proba" in model_path.lower():
        custom_objects["loss"] = gaussian_nll
        custom_objects["gaussian_nll"] = gaussian_nll
    
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects
        )
        logger.info(f"Successfully loaded full model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading full model: {str(e)}")
        raise ValueError(f"Failed to load full model: {str(e)}")
    
    # Save weights only
    try:
        model.save_weights(output_path)
        logger.info(f"Successfully saved weights to {output_path}")
    except Exception as e:
        logger.error(f"Error saving weights: {str(e)}")
        raise ValueError(f"Failed to save weights: {str(e)}")

def main():
    """Main function to convert model formats."""
    args = parse_args()
    
    # Normalize the path to handle Windows backslashes
    model_path_str = args.model_path.replace('\\', '/')
    
    # Check if model path exists - try both with and without backslashes
    model_path = Path(model_path_str)
    index_file = Path(f"{model_path_str}.index")
    
    # Check if we're dealing with a TensorFlow checkpoint format
    is_tf_checkpoint = index_file.exists()
    
    if not model_path.exists() and not is_tf_checkpoint:
        # Try with backslashes on Windows
        model_path = Path(args.model_path)
        index_file = Path(f"{args.model_path}.index")
        is_tf_checkpoint = index_file.exists()
        
        if not model_path.exists() and not is_tf_checkpoint:
            logger.error(f"Model path {args.model_path} does not exist")
            
            # Print available models for user reference
            try:
                models_dir = Path('models/final')
                if models_dir.exists():
                    logger.info("Available models in models/final directory:")
                    for item in models_dir.iterdir():
                        logger.info(f"  - {item.name}")
            except Exception:
                pass
                
            sys.exit(1)
    
    # Determine output path if not provided
    if args.output_path:
        output_path = args.output_path
    else:
        if args.to_format == 'full':
            output_path = str(model_path) + '_full'
        else:
            output_path = str(model_path) + '_weights'
    
    # Convert model
    try:
        if args.to_format == 'full':
            # Check if input is a file (weights-only) or a TensorFlow checkpoint
            if os.path.isfile(model_path) or is_tf_checkpoint:
                # Use the base path (without .index) for TensorFlow checkpoints
                model_path_str = str(model_path)
                convert_to_full_model(model_path_str, output_path)
            else:
                logger.error(f"Input model {args.model_path} is already in full model format or not found")
                sys.exit(1)
        else:  # weights
            # Check if input is a directory (full model)
            if model_path.is_dir():
                convert_to_weights_only(str(model_path), output_path)
            else:
                logger.error(f"Input model {args.model_path} is already in weights-only format or not found")
                sys.exit(1)
        
        logger.info(f"Model conversion completed successfully")
    
    except Exception as e:
        logger.error(f"Error converting model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 