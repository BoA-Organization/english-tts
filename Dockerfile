FROM nvcr.io/nvidia/pytorch:25.08-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    TRANSFORMERS_CACHE=/models/huggingface \
    MODEL_ID=k2-fsa/OmniVoice \
    SPEAKER_AUDIO=/app/speaker.mp3 \
    SPEAKER_TEXT_FILE=/app/speaker.txt

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 curl git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    torch==2.8.0+cu128 \
    torchaudio==2.8.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

RUN python -c "import torch, torchaudio, omnivoice; print(torch.__version__); print(torchaudio.__version__)"

COPY app.py /app/app.py
COPY speaker.mp3 /app/speaker.mp3
COPY speaker.txt /app/speaker.txt

RUN mkdir -p /models/huggingface /models/torch

EXPOSE 9005

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl --fail http://localhost:9005/health || exit 1

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9005", "--workers", "1"]
