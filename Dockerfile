# Use CUDA 12.1 devel image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch compatible with CUDA 12.1
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth and dependencies
RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install other libraries
# 1. numpy<2 prevents the FAISS error
# 2. We install the rest of the stack
RUN pip install --no-cache-dir \
    "numpy<2" \
    transformers \
    sentence-transformers \
    faiss-gpu \
    protobuf \
    scipy \
    accelerate \
    bitsandbytes \
    gradio

# --- THE CRITICAL FIX ---
# torchao causes version conflicts with stable PyTorch.
# We don't need it for BitsAndBytes quantization, so we remove it to prevent the crash.
RUN pip uninstall -y torchao

# Default command
CMD ["python", "ultimate_chatbot_final.py"]