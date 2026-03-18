FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG RUNTIME=nvidia

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# HuggingFace cache for model persistence across rebuilds
ENV HF_HOME=/app/hf_cache
# Enable HF fast downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1
# CUDA performance: allow TF32 for faster matmul on Ampere+
ENV NVIDIA_TF32_OVERRIDE=1
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink for python3 to be python for convenience
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set up working directory
WORKDIR /app

# Install PyTorch with CUDA 12.8 first (largest layer, best cache hit)
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    torch==2.7.0+cu128 \
    torchvision==0.22.0+cu128 \
    torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Copy and install remaining requirements (without torch, already installed)
COPY requirements-nvidia.txt .
RUN pip3 install --no-cache-dir -r requirements-nvidia.txt

# Copy the rest of the application code
COPY . .

# Create required directories
RUN mkdir -p model_cache reference_audio outputs voices logs hf_cache

# Expose the port the application will run on
EXPOSE 8004

# Command to run the application
CMD ["python3", "server.py"]
