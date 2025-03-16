# PowerShell script to run transformer project in Docker

# Function to display help message
function Show-Help {
    Write-Host "Usage: .\run_in_docker.ps1 [COMMAND] [ARGUMENTS]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  build                Build the Docker image"
    Write-Host "  shell                Start a shell in the Docker container"
    Write-Host "  create-dataset       Create a dataset with custom arguments"
    Write-Host "  create-dataset-docker Create a dataset optimized for Docker (handles large datasets)"
    Write-Host "  create-balanced      Create a balanced dataset with custom arguments"
    Write-Host "  create-rightmost     Create a right-most dataset with custom arguments"
    Write-Host "  train                Train a model with custom arguments"
    Write-Host "  check-model MODEL    Check model format"
    Write-Host "  convert-model MODEL  Convert a model to full format"
    Write-Host "  continue-training MODEL  Continue training a model with custom arguments"
    Write-Host "  evaluate             Evaluate a model on M4 test set with custom arguments"
    Write-Host "  help                 Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run_in_docker.ps1 build"
    Write-Host "  .\run_in_docker.ps1 create-dataset --start-series 1 --end-series 48000 --sample-size 1000 --random-seed 42"
    Write-Host "  .\run_in_docker.ps1 create-dataset-docker --start-series 1 --end-series 48000 --sample-size 2500 --max-batch-size 1000 --random-seed 42"
    Write-Host "  .\run_in_docker.ps1 train --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --dataset-type standard"
    Write-Host "  .\run_in_docker.ps1 train --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --probabilistic --dataset-type standard"
    Write-Host "  .\run_in_docker.ps1 check-model transformer_1.0_directml_point_mse_M1_M48000_sampled1000"
    Write-Host "  .\run_in_docker.ps1 convert-model transformer_1.0_directml_point_mse_M1_M48000_sampled1000"
    Write-Host "  .\run_in_docker.ps1 continue-training models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --dataset-type standard"
    Write-Host "  .\run_in_docker.ps1 evaluate --model_name transformer_1.0_directml_point_mse_M1_M48000_sampled1000_full --sample_size 400"
}

# Check if command is provided
if ($args.Count -eq 0) {
    Show-Help
    exit 1
}

# Get the command (first argument)
$command = $args[0]

# Process commands
switch ($command) {
    "build" {
        Write-Host "Building Docker image..."
        docker-compose build
    }
    "shell" {
        Write-Host "Starting shell in Docker container..."
        docker-compose run --rm transformer bash
    }
    "create-dataset" {
        Write-Host "Creating dataset with custom arguments..."
        # Pass each argument individually
        $argArray = @("scripts/create_dataset.py")
        for ($i = 1; $i -lt $args.Count; $i++) {
            $argArray += $args[$i]
        }
        docker-compose run --rm transformer python $argArray
    }
    "create-dataset-docker" {
        Write-Host "Creating dataset with Docker-optimized script (for large datasets)..."
        # Pass each argument individually
        $argArray = @("scripts/create_dataset_docker.py")
        for ($i = 1; $i -lt $args.Count; $i++) {
            $argArray += $args[$i]
        }
        docker-compose run --rm transformer python $argArray
    }
    "create-balanced" {
        Write-Host "Creating balanced dataset with custom arguments..."
        $scriptArgs = $args[1..($args.Count-1)] -join " "
        docker-compose run --rm transformer python scripts/create_balanced_dataset.py $scriptArgs
    }
    "create-rightmost" {
        Write-Host "Creating right-most dataset with custom arguments..."
        $scriptArgs = $args[1..($args.Count-1)] -join " "
        docker-compose run --rm transformer python scripts/create_rightmost_dataset.py $scriptArgs
    }
    "train" {
        Write-Host "Training model with custom arguments..."
        $scriptArgs = $args[1..($args.Count-1)] -join " "
        docker-compose run --rm transformer python scripts/train.py $scriptArgs
    }
    "check-model" {
        if ($args.Count -lt 2) {
            Write-Host "Error: Model name is required"
            Write-Host "Usage: .\run_in_docker.ps1 check-model MODEL_NAME"
            exit 1
        }
        Write-Host "Checking model format..."
        $scriptArgs = $args[1..($args.Count-1)] -join " "
        docker-compose run --rm transformer python scripts/check_model_format.py $scriptArgs
    }
    "convert-model" {
        if ($args.Count -lt 2) {
            Write-Host "Error: Model name is required"
            Write-Host "Usage: .\run_in_docker.ps1 convert-model MODEL_NAME"
            exit 1
        }
        Write-Host "Converting model $($args[1]) to full format..."
        $scriptArgs = $args[1..($args.Count-1)] -join " "
        docker-compose run --rm transformer python scripts/fix_model_format.py $scriptArgs
    }
    "continue-training" {
        if ($args.Count -lt 2) {
            Write-Host "Error: Model path is required"
            Write-Host "Usage: .\run_in_docker.ps1 continue-training MODEL_PATH [ARGUMENTS]"
            exit 1
        }
        Write-Host "Continuing training with custom arguments..."
        $scriptArgs = $args[1..($args.Count-1)] -join " "
        docker-compose run --rm transformer python scripts/continue_training.py $scriptArgs
    }
    "evaluate" {
        Write-Host "Evaluating model with custom arguments..."
        $scriptArgs = $args[1..($args.Count-1)] -join " "
        docker-compose run --rm transformer python scripts/evaluate_m4.py $scriptArgs
    }
    "help" {
        Show-Help
    }
    default {
        Write-Host "Unknown command: $command"
        Show-Help
        exit 1
    }
} 

# Add a pause at the end to keep the window open
Write-Host ""
Write-Host "Command execution completed."
Read-Host -Prompt "Press Enter to close this window" 