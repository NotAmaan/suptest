# =========================================================================
# Stage 1: The Builder
# Use the exact RunPod development image you need for CUDA 12.8
# =========================================================================
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 AS builder

WORKDIR /app

# Install build-time system dependencies (just git is needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment to hold all our python packages.
# This makes it very easy and clean to copy them to the next stage.
RUN python3 -m venv /app/venv
# Activate the venv for all subsequent RUN commands in this stage
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip and install wheels
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -U xformers --index-url https://download.pytorch.org/whl/cu128

# Build Flash Attention from source. This is the step that needs the -devel image.
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
# We clone, build, and then remove the source to save space
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    pip install --no-cache-dir . && \
    cd .. && \
    rm -rf flash-attention

# Finally, install the rest of your application's requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# =========================================================================
# Stage 2: The Final Production Image
# THIS IS THE KEY CORRECTION FOR YOUR REQUIREMENT
# =========================================================================
# Use the official, minimal NVIDIA runtime image for CUDA 12.8.1
# This image is guaranteed to exist and is the correct partner for your builder.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.11 and other essential runtime dependencies.
# The base NVIDIA image is minimal and does not include Python.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire Python virtual environment with all installed packages from the builder stage
COPY --from=builder /app/venv /app/venv

# Copy your application's source code
COPY . .

# Activate the virtual environment by setting the PATH. This makes `python` and `pip`
# from our venv the default for the container.
ENV PATH="/app/venv/bin:$PATH"

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Make your start script executable
RUN chmod +x start.sh

# Set the command to run your application
CMD ["./start.sh"]