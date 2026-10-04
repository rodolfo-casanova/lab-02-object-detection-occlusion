FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch (much lighter than the CUDA build)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
