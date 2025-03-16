#!/bin/bash

# Make script exit on error
set -e

# Function to display help message
show_help() {
    echo "Usage: ./run_in_docker.sh [COMMAND] [ARGUMENTS]"
    echo ""
    echo "Commands:"
    echo "  build                Build the Docker image"
    echo "  shell                Start a shell in the Docker container"
    echo "  create-dataset       Create a dataset with custom arguments"
    echo "  create-dataset-docker Create a dataset optimized for Docker (handles large datasets)"
    echo "  create-balanced      Create a balanced dataset with custom arguments"
    echo "  create-rightmost     Create a right-most dataset with custom arguments"
    echo "  train                Train a model with custom arguments"
    echo "  check-model MODEL    Check model format"
    echo "  convert-model MODEL  Convert a model to full format"
    echo "  continue-training MODEL  Continue training a model with custom arguments"
    echo "  evaluate             Evaluate a model on M4 test set with custom arguments"
    echo "  help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_in_docker.sh build"
    echo "  ./run_in_docker.sh create-dataset --start-series 1 --end-series 48000 --sample-size 1000 --random-seed 42"
    echo "  ./run_in_docker.sh create-dataset-docker --start-series 1 --end-series 48000 --sample-size 2500 --max-batch-size 1000 --random-seed 42"
    echo "  ./run_in_docker.sh train --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --dataset-type standard"
    echo "  ./run_in_docker.sh train --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --probabilistic --dataset-type standard"
    echo "  ./run_in_docker.sh check-model transformer_1.0_directml_point_mse_M1_M48000_sampled1000"
    echo "  ./run_in_docker.sh convert-model transformer_1.0_directml_point_mse_M1_M48000_sampled1000"
    echo "  ./run_in_docker.sh continue-training models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --dataset-type standard"
    echo "  ./run_in_docker.sh evaluate --model_name transformer_1.0_directml_point_mse_M1_M48000_sampled1000_full --sample_size 400"
}

# Check if command is provided
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# Get the command (first argument)
COMMAND="$1"
shift  # Remove the first argument (command) from $@

# Process commands
case "$COMMAND" in
    build)
        echo "Building Docker image..."
        docker-compose build
        ;;
    shell)
        echo "Starting shell in Docker container..."
        docker-compose run --rm transformer bash
        ;;
    create-dataset)
        echo "Creating dataset with custom arguments..."
        docker-compose run --rm transformer python scripts/create_dataset.py "$@"
        ;;
    create-dataset-docker)
        echo "Creating dataset with Docker-optimized script (for large datasets)..."
        docker-compose run --rm transformer python scripts/create_dataset_docker.py "$@"
        ;;
    create-balanced)
        echo "Creating balanced dataset with custom arguments..."
        docker-compose run --rm transformer python scripts/create_balanced_dataset.py "$@"
        ;;
    create-rightmost)
        echo "Creating right-most dataset with custom arguments..."
        docker-compose run --rm transformer python scripts/create_rightmost_dataset.py "$@"
        ;;
    train)
        echo "Training model with custom arguments..."
        docker-compose run --rm transformer python scripts/train.py "$@"
        ;;
    check-model)
        if [ $# -eq 0 ]; then
            echo "Error: Model name is required"
            echo "Usage: ./run_in_docker.sh check-model MODEL_NAME"
            exit 1
        fi
        echo "Checking model format..."
        docker-compose run --rm transformer python scripts/check_model_format.py "$@"
        ;;
    convert-model)
        if [ $# -eq 0 ]; then
            echo "Error: Model name is required"
            echo "Usage: ./run_in_docker.sh convert-model MODEL_NAME"
            exit 1
        fi
        echo "Converting model to full format..."
        docker-compose run --rm transformer python scripts/fix_model_format.py "$@"
        ;;
    continue-training)
        if [ $# -eq 0 ]; then
            echo "Error: Model path is required"
            echo "Usage: ./run_in_docker.sh continue-training MODEL_PATH [ARGUMENTS]"
            exit 1
        fi
        echo "Continuing training with custom arguments..."
        docker-compose run --rm transformer python scripts/continue_training.py "$@"
        ;;
    evaluate)
        echo "Evaluating model with custom arguments..."
        docker-compose run --rm transformer python scripts/evaluate_m4.py "$@"
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

# Add a pause at the end to keep the terminal window open
echo ""
echo "Command execution completed."
read -p "Press Enter to close this window..." 