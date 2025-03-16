# Running Transformer Time Series Forecasting in Docker

This guide explains how to run the transformer time series forecasting project in a Docker container, following the steps in the QUICK_TRAINING_GUIDE.md.

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/) installed on your system
- [Docker Compose](https://docs.docker.com/compose/install/) installed on your system (usually comes with Docker Desktop)
- M4 competition dataset files already in your local `data/raw` directory:
  - `Monthly-train.csv`
  - `Monthly-test.csv`

## Getting Started

1. Make sure you have the M4 competition dataset files in your local `data/raw` directory.

2. Build the Docker image:

   ```bash
   # For Linux/macOS
   ./run_in_docker.sh build
   
   # For Windows PowerShell
   .\run_in_docker.ps1 build
   ```

## Running the Quick Training Guide Steps

The `run_in_docker.sh` (Linux/macOS) or `run_in_docker.ps1` (Windows) script allows you to run **exactly the same commands** from the QUICK_TRAINING_GUIDE.md in the Docker container. The only difference is that you need to prefix the commands with the Docker run script and use the appropriate command name.

### Command Mapping

The Docker scripts use slightly different command names than the Python scripts:

| Docker Command     | Python Script                  |
|--------------------|-------------------------------|
| create-dataset     | scripts/create_dataset.py     |
| create-dataset-docker | scripts/create_dataset_docker.py |
| create-balanced    | scripts/create_balanced_dataset.py |
| create-rightmost   | scripts/create_rightmost_dataset.py |
| train              | scripts/train.py              |
| check-model        | scripts/check_model_format.py |
| convert-model      | scripts/fix_model_format.py   |
| continue-training  | scripts/continue_training.py  |
| evaluate           | scripts/evaluate_m4.py        |

### Examples with Exact Same Arguments

#### 1. Creating a Dataset

```bash
# Standard dataset with exact same arguments from QUICK_TRAINING_GUIDE.md
./run_in_docker.sh create-dataset --start-series 1 --end-series 48000 --random-seed 42
./run_in_docker.sh create-dataset --start-series 1 --end-series 48000 --sample-size 1000 --random-seed 42

# Docker-optimized dataset creation for large datasets (handles memory limitations)
./run_in_docker.sh create-dataset-docker --start-series 1 --end-series 48000 --sample-size 2500 --max-batch-size 1000 --random-seed 42

# Balanced dataset with exact same arguments
./run_in_docker.sh create-balanced --random-seed 42
./run_in_docker.sh create-balanced --subsequences-per-series 200 --random-seed 42
./run_in_docker.sh create-balanced --start-series 1 --end-series 10000 --random-seed 42

# Right-most dataset with exact same arguments
./run_in_docker.sh create-rightmost --random-seed 42
./run_in_docker.sh create-rightmost --subsequences-per-series 200 --random-seed 42
./run_in_docker.sh create-rightmost --start-series 1 --end-series 10000 --random-seed 42
```

#### 2. Training a New Model

```bash
# Point model with SMAPE loss (default) - exact same arguments
./run_in_docker.sh train --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --dataset-type standard

# Point model with MSE loss - exact same arguments
./run_in_docker.sh train --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --loss-type mse --dataset-type standard

# Probabilistic model with Gaussian NLL loss (default) - exact same arguments
./run_in_docker.sh train --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --probabilistic --dataset-type standard

# Probabilistic model with Hybrid loss - exact same arguments
./run_in_docker.sh train --start-series 1 --end-series 48000 --sample-size 1000 --batch-size 64 --epochs 50 --probabilistic --loss-type hybrid --loss-alpha 0.8 --dataset-type standard
```

#### 3. Converting Weights to a Full Model

Models are saved in weights-only format (with .index and .data-* files). For evaluation, you need to convert to a full model format.

```bash
# Check model format - exact same arguments as in QUICK_TRAINING_GUIDE.md
./run_in_docker.sh check-model transformer_1.0_directml_point_mse_M1_M48000_sampled1000

# Convert to full model - exact same arguments
./run_in_docker.sh convert-model transformer_1.0_directml_point_mse_M1_M48000_sampled1000
```

#### 4. Continuing Training

You can continue training from either weights-only models or full models. Here are examples with exact same arguments as in QUICK_TRAINING_GUIDE.md:

```bash
# Continue training a weights-only model - Standard dataset with SMAPE loss
./run_in_docker.sh continue-training models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --dataset-type standard

# Continue training a weights-only model - Standard dataset with MSE loss
./run_in_docker.sh continue-training models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --loss-type mse --dataset-type standard

# Optimized continuation - Standard dataset with MSE loss
./run_in_docker.sh continue-training models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --loss-type mse --disable-memory-growth --batch-size 64 --aggressive-cleanup --dataset-type standard
```

For sequential commands in the QUICK_TRAINING_GUIDE.md, you can run them one after another:

```bash
# Continue training with a different standard sample - SMAPE loss
./run_in_docker.sh create-dataset --start-series 1 --end-series 48000 --sample-size 2000 --random-seed 43
./run_in_docker.sh continue-training models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --start-series 1 --end-series 48000 --sample-size 2000 --dataset-type standard

# Continue training with a balanced dataset - SMAPE loss
./run_in_docker.sh create-balanced --random-seed 42
./run_in_docker.sh continue-training models/final/transformer_1.0_directml_point_M1_M48000_sampled1000 --epochs 10 --sample-size 17979000 --random-seed 42 --dataset-type balanced
```

#### 5. Evaluating on the M4 Test Set

Evaluation requires a full model format (after conversion):

```bash
# Evaluate on a sample of series - exact same arguments
./run_in_docker.sh evaluate --model_name transformer_1.0_directml_point_mse_M1_M48000_sampled1000_full --sample_size 400

# Full evaluation - exact same arguments
./run_in_docker.sh evaluate --model_name transformer_1.0_directml_point_mse_M1_M48000_sampled1000_full --sample_size 48000
```

## Running a Shell in the Container

If you need to run custom commands or explore the container, you can start a shell:

```bash
# For Linux/macOS
./run_in_docker.sh shell

# For Windows PowerShell
.\run_in_docker.ps1 shell
```

## Data Persistence

The Docker setup uses a combination of host mounts and named volumes:

- **Host Mount**:
  - `./data/raw:/app/data/raw` - Mounts your local raw data directory into the container

- **Named Volumes** (persist between container runs):
  - `transformer_processed:/app/data/processed` - Processed datasets
  - `transformer_models:/app/models` - Trained models
  - `transformer_logs:/app/logs` - Training logs
  - `transformer_reports:/app/reports` - Evaluation reports

This setup ensures that:
1. Your existing raw data is used by the container
2. All generated data (processed datasets, models, logs, reports) is stored in Docker volumes
3. The generated data persists between container runs but is separate from your local filesystem

## Troubleshooting

### GPU Access

The Docker container uses TensorFlow with DirectML for GPU acceleration. If you encounter GPU-related issues:

1. Make sure your GPU drivers are up to date
2. Check that DirectML is properly installed in the container
3. Try running with a smaller batch size or sample size

### Memory Issues

If you encounter memory issues during dataset creation or training:

1. For dataset creation with large sample sizes (>2000):
   - Use the Docker-optimized dataset creation script: `create-dataset-docker`
   - Adjust the `--max-batch-size` parameter (lower values use less memory)
   - Example: `./run_in_docker.sh create-dataset-docker --start-series 1 --end-series 48000 --sample-size 2500 --max-batch-size 1000 --random-seed 42`

2. For training:
   - Reduce the batch size (e.g., `--batch-size 32` or `--batch-size 16`)
   - Use a smaller sample size (e.g., `--sample-size 500` instead of `--sample-size 1000`)
   - Use float16 precision instead of float32 (check `config/data_config.yaml`)

### Container Access Issues

If you have issues accessing the container:

1. Make sure Docker is running
2. Check that you have permission to run Docker commands
3. Try running Docker commands with sudo (Linux/macOS) 