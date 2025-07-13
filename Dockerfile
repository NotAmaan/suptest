# =========================================================================
# Stage 1: The Builder
# Use the large development image which has the CUDA compiler (nvcc)
# =========================================================================
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 AS builder

WORKDIR /app

# Install build-time system dependencies (just git is needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment to hold all our python packages.
# This makes it very easy to copy them to the next stage.
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
# Use the smaller 'runtime' image which does NOT have the bulky compiler
# =========================================================================
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install only the essential RUNTIME system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire virtual environment with all installed packages from the builder stage
COPY --from=builder /app/venv /app/venv

# Copy your application's source code
COPY . .

# Activate the virtual environment by setting the PATH
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