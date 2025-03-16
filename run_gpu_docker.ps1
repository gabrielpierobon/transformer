# PowerShell script to run Docker with GPU passthrough for DirectML

# Function to display help message
function Show-Help {
    Write-Host "Usage: .\run_gpu_docker.ps1 [COMMAND] [ARGUMENTS]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  build                Build the Docker image"
    Write-Host "  train                Train a model with custom arguments"
    Write-Host "  create-dataset       Create a dataset with custom arguments"
    Write-Host "  create-dataset-docker Create a dataset optimized for Docker (handles large datasets)"
    Write-Host "  shell                Start a shell in the Docker container"
    Write-Host "  help                 Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run_gpu_docker.ps1 build"
    Write-Host "  .\run_gpu_docker.ps1 train --start-series 1 --end-series 48000 --sample-size 2100 --batch-size 128 --epochs 3 --loss-type mse --dataset-type standard"
    Write-Host "  .\run_gpu_docker.ps1 create-dataset-docker --start-series 1 --end-series 48000 --sample-size 2500 --max-batch-size 1000 --random-seed 42"
}

# Check if command is provided
if ($args.Count -eq 0) {
    Show-Help
    exit 1
}

# Get the command (first argument)
$command = $args[0]

# Get the correct image name from docker-compose project
$projectName = (Get-Item -Path ".\").Name.ToLower()
$imageName = "${projectName}-transformer"

# Process commands
switch ($command) {
    "build" {
        Write-Host "Building Docker image with DirectML support..."
        docker-compose build
        
        # Verify the image was built
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error: Failed to build Docker image."
            exit 1
        }
        
        Write-Host "Docker image built successfully: $imageName"
    }
    "shell" {
        Write-Host "Starting shell in Docker container with DirectML support..."
        
        # Check if image exists, build if not
        $imageExists = docker image inspect $imageName 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Image not found. Building first..."
            docker-compose build
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Error: Failed to build Docker image."
                exit 1
            }
        }
        
        # Run Docker with GPU passthrough
        docker run --rm -it `
            --device=/dev/dxg `
            --gpus=all `
            -v ${PWD}/data/raw:/app/data/raw `
            -v ${PWD}/data/processed:/app/data/processed `
            -v ${PWD}/models:/app/models `
            -v ${PWD}/logs:/app/logs `
            -v ${PWD}/reports:/app/reports `
            -e DIRECTML_GPU_ENABLE=1 `
            -e TF_FORCE_GPU_ALLOW_GROWTH=true `
            -e DML_VISIBLE_DEVICES=0 `
            -e TF_DIRECTML_KERNEL_CACHE=/app/directml_cache `
            $imageName bash
    }
    "train" {
        Write-Host "Training model with custom arguments in Docker with DirectML support..."
        
        # Check if image exists, build if not
        $imageExists = docker image inspect $imageName 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Image not found. Building first..."
            docker-compose build
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Error: Failed to build Docker image."
                exit 1
            }
        }
        
        # Build the docker run command with all arguments
        $dockerCmd = "docker run --rm --device=/dev/dxg --gpus=all " + `
            "-v ${PWD}/data/raw:/app/data/raw " + `
            "-v ${PWD}/data/processed:/app/data/processed " + `
            "-v ${PWD}/models:/app/models " + `
            "-v ${PWD}/logs:/app/logs " + `
            "-v ${PWD}/reports:/app/reports " + `
            "-e DIRECTML_GPU_ENABLE=1 " + `
            "-e TF_FORCE_GPU_ALLOW_GROWTH=true " + `
            "-e DML_VISIBLE_DEVICES=0 " + `
            "-e TF_DIRECTML_KERNEL_CACHE=/app/directml_cache " + `
            "$imageName python scripts/train.py"
        
        # Add each argument individually
        for ($i = 1; $i -lt $args.Count; $i++) {
            $dockerCmd += " " + $args[$i]
        }
        
        # Execute the command
        Write-Host "Executing: $dockerCmd"
        Invoke-Expression $dockerCmd
    }
    "create-dataset" {
        Write-Host "Creating dataset with custom arguments in Docker with DirectML support..."
        
        # Check if image exists, build if not
        $imageExists = docker image inspect $imageName 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Image not found. Building first..."
            docker-compose build
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Error: Failed to build Docker image."
                exit 1
            }
        }
        
        # Build the docker run command with all arguments
        $dockerCmd = "docker run --rm --device=/dev/dxg --gpus=all " + `
            "-v ${PWD}/data/raw:/app/data/raw " + `
            "-v ${PWD}/data/processed:/app/data/processed " + `
            "-v ${PWD}/models:/app/models " + `
            "-v ${PWD}/logs:/app/logs " + `
            "-v ${PWD}/reports:/app/reports " + `
            "-e DIRECTML_GPU_ENABLE=1 " + `
            "-e TF_FORCE_GPU_ALLOW_GROWTH=true " + `
            "-e DML_VISIBLE_DEVICES=0 " + `
            "-e TF_DIRECTML_KERNEL_CACHE=/app/directml_cache " + `
            "$imageName python scripts/create_dataset.py"
        
        # Add each argument individually
        for ($i = 1; $i -lt $args.Count; $i++) {
            $dockerCmd += " " + $args[$i]
        }
        
        # Execute the command
        Write-Host "Executing: $dockerCmd"
        Invoke-Expression $dockerCmd
    }
    "create-dataset-docker" {
        Write-Host "Creating dataset with Docker-optimized script (for large datasets) with DirectML support..."
        
        # Check if image exists, build if not
        $imageExists = docker image inspect $imageName 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Image not found. Building first..."
            docker-compose build
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Error: Failed to build Docker image."
                exit 1
            }
        }
        
        # Build the docker run command with all arguments
        $dockerCmd = "docker run --rm --device=/dev/dxg --gpus=all " + `
            "-v ${PWD}/data/raw:/app/data/raw " + `
            "-v ${PWD}/data/processed:/app/data/processed " + `
            "-v ${PWD}/models:/app/models " + `
            "-v ${PWD}/logs:/app/logs " + `
            "-v ${PWD}/reports:/app/reports " + `
            "-e DIRECTML_GPU_ENABLE=1 " + `
            "-e TF_FORCE_GPU_ALLOW_GROWTH=true " + `
            "-e DML_VISIBLE_DEVICES=0 " + `
            "-e TF_DIRECTML_KERNEL_CACHE=/app/directml_cache " + `
            "$imageName python scripts/create_dataset_docker.py"
        
        # Add each argument individually
        for ($i = 1; $i -lt $args.Count; $i++) {
            $dockerCmd += " " + $args[$i]
        }
        
        # Execute the command
        Write-Host "Executing: $dockerCmd"
        Invoke-Expression $dockerCmd
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