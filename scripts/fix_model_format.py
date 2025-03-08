#!/usr/bin/env python
"""
Script to fix model format issues.

This is a convenience wrapper around convert_model_format.py that automatically
detects the model format and converts it to the format needed for evaluation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import subprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fix model format issues')
    
    parser.add_argument(
        'model_name',
        type=str,
        help='Name of the model to fix (e.g., transformer_1.0_directml_point_M1_M48000_sampled2001)'
    )
    
    return parser.parse_args()

def main():
    """Main function to fix model format."""
    args = parse_args()
    
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
        logger.info("No conversion needed for evaluation")
        logger.info(f"You can evaluate using:")
        logger.info(f"python scripts/evaluate_m4.py --model_name {model_name} --sample_size 400")
    elif has_index_file:
        logger.info(f"Model {model_name} is in weights-only format (checkpoint files)")
        logger.info("Converting to full model format for evaluation...")
        
        # Build command to convert model
        cmd = [
            sys.executable,
            "scripts/convert_model_format.py",
            "--model-path", f"models/final/{model_name}",
            "--to-format", "full"
        ]
        
        # Execute command
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Conversion successful. You can now evaluate using:")
            logger.info(f"python scripts/evaluate_m4.py --model_name {model_name}_full --sample_size 400")
        except subprocess.CalledProcessError as e:
            logger.error(f"Conversion failed with exit code {e.returncode}")
            if e.stdout:
                logger.error(f"Output: {e.stdout}")
            if e.stderr:
                logger.error(f"Error: {e.stderr}")
            sys.exit(e.returncode)
    else:
        logger.error(f"Model {model_name} not found in models/final directory")
        
        # List available models
        logger.info("Available models in models/final directory:")
        
        # Track unique model names (without extensions)
        model_names = set()
        
        for item in models_dir.iterdir():
            if item.is_dir():
                logger.info(f"  - {item.name} (full model format)")
                model_names.add(item.name)
            elif item.name.endswith('.index'):
                # Extract model name from .index file
                base_name = item.name[:-6]  # Remove .index
                logger.info(f"  - {base_name} (weights-only format)")
                model_names.add(base_name)
            else:
                # Just list other files for reference
                logger.info(f"  - {item.name}")
        
        # Suggest closest model name if possible
        if model_names:
            logger.info("\nDid you mean one of these models?")
            for name in model_names:
                logger.info(f"  python scripts/fix_model_format.py {name}")
        
        sys.exit(1)

if __name__ == "__main__":
    main() 