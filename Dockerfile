# Use official PyTorch image with CUDA 12.8 for RTX 5090 compatibility
# Note: Using PyTorch 2.5.1 which has better CUDA 12.8 support
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set Python unbuffered for better logging
ENV PYTHONUNBUFFERED=1

# Set mode and workspace directory
ARG MODE_TO_RUN="pod"
ENV MODE_TO_RUN=$MODE_TO_RUN
WORKDIR /app

# Install system dependencies in a single layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements for caching
COPY requirements.txt .

# Install core and specialized Python packages
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -U xformers --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir runpod>=1.7.0

# Install Flash Attention 2 from source and cleanup
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    pip install --no-cache-dir . && \
    cd .. && \
    rm -rf flash-attention

# Install the remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Update model paths to use /workspace mount
ENV MODEL_BASE_PATH="/workspace/models"

# Verify CUDA installation
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Use start script for dual-mode support
CMD ["./start.sh"]