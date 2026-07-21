FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    MODEL_ID=k2-fsa/OmniVoice \
    SPEAKER_AUDIO=/app/speaker.wav \
    SPEAKER_TEXT_FILE=/app/speaker.txt

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
        torch \
        torchaudio \
        --index-url https://download.pytorch.org/whl/cu130

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
COPY speaker.wav /app/speaker.wav
COPY speaker.txt /app/speaker.txt

RUN mkdir -p /models/huggingface && \
    python -c "import torch; import torchaudio; import omnivoice; print('Torch:', torch.__version__); print('TorchAudio:', torchaudio.__version__); print('CUDA build:', torch.version.cuda); print('Architectures:', torch.cuda.get_arch_list()); print('OmniVoice import successful')"

EXPOSE 9005

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9005/health')" || exit 1

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9005", "--workers", "1"]
