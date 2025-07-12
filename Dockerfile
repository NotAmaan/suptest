# Use official PyTorch image with CUDA 12.8 for RTX 5090 compatibility
# Note: Using PyTorch 2.5.1 which has better CUDA 12.8 support
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04


# Set Python unbuffered for better logging
ENV PYTHONUNBUFFERED=1

# Set mode and workspace directory
ARG MODE_TO_RUN="pod"
ENV MODE_TO_RUN=$MODE_TO_RUN
ENV WORKSPACE_DIR=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR $WORKSPACE_DIR

# Copy requirements first for better caching
COPY requirements.txt .

# Update pip and install essential packages
RUN pip install --upgrade pip

# Install xformers with CUDA 12.8 support
RUN pip install -U xformers --index-url https://download.pytorch.org/whl/cu128

# Install RunPod SDK (needed for serverless mode)
RUN pip install runpod>=1.7.0

# Install remaining requirements
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="${WORKSPACE_DIR}:${PYTHONPATH:-}"

# Update model paths to use /workspace mount
ENV MODEL_BASE_PATH="/workspace/models"

# Verify CUDA installation
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Use start script for dual-mode support
CMD ["./start.sh"]