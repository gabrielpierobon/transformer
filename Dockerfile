FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install TensorFlow DirectML plugin explicitly
RUN pip install --no-cache-dir tensorflow-directml-plugin==0.4.0.dev230202

# Create necessary directories with proper permissions
RUN mkdir -p data/raw data/processed \
    models/checkpoints models/final \
    logs reports/figures \
    && chmod -R 777 data models logs reports

# Create DirectML cache directory with proper permissions
RUN mkdir -p /app/directml_cache && chmod -R 777 /app/directml_cache

# Set environment variables for DirectML
ENV TF_DIRECTML_KERNEL_CACHE=1
ENV TF_DIRECTML_KERNEL_PATH=/app/directml_cache
ENV DIRECTML_GPU_ENABLE=1
ENV DML_VISIBLE_DEVICES=0
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Copy the rest of the application
COPY . .

# Default command
CMD ["bash"] 