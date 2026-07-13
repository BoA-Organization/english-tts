FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    MODEL_ID=k2-fsa/OmniVoice \
    SPEAKER_AUDIO=/app/speaker.mp3 \
    SPEAKER_TEXT_FILE=/app/speaker.txt

WORKDIR /app

# Audio libraries, Git, and health-check utility
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade \
    pip \
    setuptools \
    wheel

# The base image already contains CUDA-enabled torch 2.8.0.
# Install only the matching torchaudio package without replacing torch.
RUN python -m pip install \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128 \
    --no-deps

COPY requirements.txt /app/requirements.txt

RUN python -m pip install -r /app/requirements.txt

# Verify required packages during the image build
RUN python -c "\
import torch; \
import torchaudio; \
import omnivoice; \
print('PyTorch:', torch.__version__); \
print('CUDA build:', torch.version.cuda); \
print('TorchAudio:', torchaudio.__version__); \
print('OmniVoice import successful')"

COPY app.py /app/app.py
COPY speaker.mp3 /app/speaker.mp3
COPY speaker.txt /app/speaker.txt

RUN mkdir -p \
    /models/huggingface \
    /models/torch

EXPOSE 9005

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=300s \
    --retries=5 \
    CMD curl --fail http://localhost:9005/health || exit 1

CMD ["python","-m","uvicorn","app:app","--host","0.0.0.0","--port","9005","--workers","1"]
 