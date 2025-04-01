#!/usr/bin/env python
"""
Script to run the tourism dataset evaluation process.

This script:
1. Converts the tourism TSF file to CSV format
2. Runs the evaluation on the transformer model
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command):
    """Run a shell command and log output"""
    logger.info(f"Running command: {command}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True
    )
    
    # Stream output
    for line in process.stdout:
        print(line.strip())
    
    # Get return code
    process.wait()
    return process.returncode

def main():
    """Run the tourism dataset evaluation process"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run tourism dataset evaluation')
    parser.add_argument('--log-transform', action='store_true',
                        help='Apply log transformation to the data before forecasting')
    args = parser.parse_args()
    
    logger.info("Starting tourism dataset evaluation")
    
    # Create directories if they don't exist
    for directory in ["data/processed", "evaluation/tourism", "evaluation/tourism/plots"]:
        os.makedirs(directory, exist_ok=True)
    
    # Convert TSF file to CSV format
    logger.info("Converting TSF file to CSV format")
    conversion_cmd = "python scripts/convert_tourism_tsf_to_csv.py"
    if run_command(conversion_cmd) != 0:
        logger.error("Failed to convert tourism dataset to CSV format")
        return 1
    
    # Check if conversion was successful
    if not Path("data/processed/tourism_monthly_dataset.csv").exists():
        logger.error("Tourism dataset CSV file not found")
        return 1
    
    # Use the correct model for evaluation
    model_name = "transformer_1.0_directml_point_mse_M1_M48000_sampled2101_full_4epoch"
    
    # Run evaluation with all 366 monthly series as per the paper
    logger.info("Running evaluation with all 366 monthly series (as per the paper)")
    
    # Add log transform parameter if requested
    log_transform_param = "--log-transform" if args.log_transform else ""
    eval_cmd = f"python scripts/evaluate_tourism.py --model-name {model_name} --sample-size 366 --forecast-horizon 24 {log_transform_param}"
    
    if run_command(eval_cmd) != 0:
        logger.error("Evaluation failed")
        return 1
    
    logger.info("Tourism dataset evaluation complete")
    logger.info("Results have been saved to evaluation/tourism/ directory")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 