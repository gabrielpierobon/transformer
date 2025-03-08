#!/usr/bin/env python
"""
Script to check the format of a saved model.

This script helps users identify whether a model is saved in the full model format
(directory) or the weights-only format (files).
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Check model format')
    
    parser.add_argument(
        'model_name',
        type=str,
        nargs='?',
        help='Name of the model to check (e.g., transformer_1.0_directml_point_M1_M48000_sampled2001)'
    )
    
    return parser.parse_args()

def main():
    """Main function to check model format."""
    args = parse_args()
    
    # If no model name provided, list all models
    if args.model_name is None:
        logger.info("No model name provided. Listing all models in models/final directory:")
        list_all_models()
        return
    
    # Construct model paths
    model_name = args.model_name
    models_dir = Path('models/final')
    model_path = models_dir / model_name
    index_file = models_dir / f"{model_name}.index"
    
    # Check what format the model is in
    is_directory = model_path.is_dir()
    has_index_file = index_file.exists()
    
    if is_directory:
        logger.info(f"Model {model_name} is in full model format (directory)")
        logger.info("This format is compatible with evaluation scripts.")
        logger.info(f"You can evaluate using:")
        logger.info(f"python scripts/evaluate_m4.py --model_name {model_name} --sample_size 400")
    elif has_index_file:
        logger.info(f"Model {model_name} is in weights-only format (checkpoint files)")
        logger.info("This format is optimized for continuing training but may need conversion for evaluation.")
        logger.info("To convert for evaluation, run:")
        logger.info(f"python scripts/fix_model_format.py {model_name}")
    else:
        logger.error(f"Model {model_name} not found in models/final directory")
        list_all_models()
        sys.exit(1)

def list_all_models():
    """List all models in the models/final directory."""
    models_dir = Path('models/final')
    
    if not models_dir.exists():
        logger.error(f"Directory {models_dir} does not exist")
        return
    
    # Track unique model names (without extensions)
    model_names = set()
    full_models = []
    weights_models = []
    
    for item in models_dir.iterdir():
        if item.is_dir():
            full_models.append(item.name)
            model_names.add(item.name)
        elif item.name.endswith('.index'):
            # Extract model name from .index file
            base_name = item.name[:-6]  # Remove .index
            weights_models.append(base_name)
            model_names.add(base_name)
    
    if full_models:
        logger.info("\nFull model format (directories):")
        for name in sorted(full_models):
            logger.info(f"  - {name}")
    
    if weights_models:
        logger.info("\nWeights-only format (checkpoint files):")
        for name in sorted(weights_models):
            logger.info(f"  - {name}")
    
    if not model_names:
        logger.info("No models found in models/final directory")
    else:
        logger.info("\nTo check a specific model, run:")
        logger.info("python scripts/check_model_format.py MODEL_NAME")

if __name__ == "__main__":
    main() 