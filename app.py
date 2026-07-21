import io
import os
import threading
import time
from contextlib import asynccontextmanager

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from omnivoice import OmniVoice
from pydantic import BaseModel, Field


MODEL_ID = os.getenv("MODEL_ID", "k2-fsa/OmniVoice")
SPEAKER_AUDIO = os.getenv("SPEAKER_AUDIO", "/app/speaker.mp3")
SPEAKER_TEXT_FILE = os.getenv("SPEAKER_TEXT_FILE", "/app/speaker.txt")
SAMPLE_RATE = 24_000

model = None
speaker_text = None
speaker_audio = None
generation_lock = threading.Lock()


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, speaker_text, speaker_audio

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was not detected. Start the container with --gpus all."
        )

    if not os.path.isfile(SPEAKER_AUDIO):
        raise RuntimeError(
            f"Speaker audio was not found: {SPEAKER_AUDIO}"
        )

    if not os.path.isfile(SPEAKER_TEXT_FILE):
        raise RuntimeError(
            f"Speaker text was not found: {SPEAKER_TEXT_FILE}"
        )

    with open(SPEAKER_TEXT_FILE, "r", encoding="utf-8") as file:
        speaker_text = file.read().strip()

    if not speaker_text:
        raise RuntimeError("speaker.txt is empty.")

    print(f"Loading speaker audio from {SPEAKER_AUDIO}...")
    speaker_audio, _ = sf.read(SPEAKER_AUDIO)
    print(f"Speaker audio loaded: shape={speaker_audio.shape}")

    print(f"Loading {MODEL_ID} on GPU...")

    model = OmniVoice.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        torch_dtype=torch.float16,
    )

    # Ensure model stays on GPU
    model.eval()
    
    print(f"OmniVoice loaded on {torch.cuda.get_device_name(0)}")
    
    # Warmup: Run a test generation to compile kernels and cache operations
    print("Warming up model with test generation...")
    try:
        with torch.inference_mode():
            _ = model.generate(
                text="Hello world",
                ref_audio=speaker_audio,
                ref_text=speaker_text,
            )
        print("Model warmup completed successfully")
    except Exception as e:
        print(f"Warning: Model warmup failed: {e}")

    yield

    model = None
    speaker_audio = None
    torch.cuda.empty_cache()


app = FastAPI(
    title="OmniVoice TTS API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    gpu_available = torch.cuda.is_available()
    healthy = model is not None and gpu_available

    return {
        "status": "healthy" if healthy else "unhealthy",
        "model_loaded": model is not None,
        "gpu_available": gpu_available,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if gpu_available
            else None
        ),
        "speaker_audio": os.path.basename(SPEAKER_AUDIO),
    }


@app.post("/tts/en")
def generate_speech(request: TTSRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The TTS model is not loaded.",
        )

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    started_at = time.perf_counter()

    try:
        with generation_lock:
            with torch.inference_mode():
                audio_list = model.generate(
                    text=text,
                    ref_audio=speaker_audio,
                    ref_text=speaker_text,
                )

        if not audio_list:
            raise RuntimeError("The model returned no audio.")

        audio = audio_list[0]

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().float().cpu().numpy()

        if audio.ndim > 1:
            audio = audio.squeeze()

        wav_buffer = io.BytesIO()

        sf.write(
            wav_buffer,
            audio,
            SAMPLE_RATE,
            format="WAV",
        )

        wav_buffer.seek(0)

        elapsed = time.perf_counter() - started_at

        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="speech.wav"',
                "X-Generation-Time-Seconds": f"{elapsed:.3f}",
                "X-Sample-Rate": str(SAMPLE_RATE),
            },
        )

    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()

        raise HTTPException(
            status_code=503,
            detail="GPU memory is exhausted.",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Speech generation failed: {exc}",
        ) from exc
