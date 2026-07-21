import io
import os
import threading
import time
import traceback
from contextlib import asynccontextmanager

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from omnivoice import OmniVoice
from pydantic import BaseModel, Field
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

    logger.info("Starting application lifespan...")

    if not torch.cuda.is_available():
        logger.error("CUDA GPU not detected")
        raise RuntimeError(
            "CUDA GPU was not detected. Start the container with --gpus all."
        )

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    if not os.path.isfile(SPEAKER_AUDIO):
        logger.error(f"Speaker audio not found: {SPEAKER_AUDIO}")
        raise RuntimeError(
            f"Speaker audio was not found: {SPEAKER_AUDIO}"
        )

    if not os.path.isfile(SPEAKER_TEXT_FILE):
        logger.error(f"Speaker text file not found: {SPEAKER_TEXT_FILE}")
        raise RuntimeError(
            f"Speaker text was not found: {SPEAKER_TEXT_FILE}"
        )

    with open(SPEAKER_TEXT_FILE, "r", encoding="utf-8") as file:
        speaker_text = file.read().strip()

    if not speaker_text:
        logger.error("speaker.txt is empty")
        raise RuntimeError("speaker.txt is empty.")

    logger.info(f"Speaker text loaded: {speaker_text[:50]}...")

    logger.info(f"Loading speaker audio from {SPEAKER_AUDIO}...")
    try:
        speaker_audio, sample_rate = sf.read(SPEAKER_AUDIO)
        logger.info(f"Speaker audio loaded: shape={speaker_audio.shape}, sample_rate={sample_rate}, dtype={speaker_audio.dtype}")
    except Exception as e:
        logger.error(f"Failed to load speaker audio: {e}")
        logger.error(traceback.format_exc())
        raise

    logger.info(f"Loading {MODEL_ID} on GPU...")

    try:
        model = OmniVoice.from_pretrained(
            MODEL_ID,
            device_map="cuda:0",
            torch_dtype=torch.float16,
        )

        # Ensure model stays on GPU
        model.eval()
        
        logger.info(f"OmniVoice loaded on {torch.cuda.get_device_name(0)}")
        logger.info(f"Model device: {next(model.parameters()).device}")
        
        # Warmup: Run a test generation to compile kernels and cache operations
        logger.info("Warming up model with test generation...")
        try:
            with torch.inference_mode():
                warmup_result = model.generate(
                    text="Hello world",
                    ref_audio=speaker_audio,
                    ref_text=speaker_text,
                )
            logger.info(f"Model warmup completed successfully. Result type: {type(warmup_result)}")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
            logger.warning(traceback.format_exc())
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        raise

    yield

    logger.info("Shutting down application...")
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
    logger.info(f"Received TTS request: text length={len(request.text)}")
    
    if model is None:
        logger.error("Model is not loaded")
        raise HTTPException(
            status_code=503,
            detail="The TTS model is not loaded.",
        )

    text = request.text.strip()

    if not text:
        logger.warning("Empty text received after stripping")
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    logger.debug(f"Processing text: {text[:100]}...")
    started_at = time.perf_counter()

    try:
        logger.info("Acquiring generation lock...")
        with generation_lock:
            logger.info("Lock acquired, starting generation...")
            logger.debug(f"Speaker audio shape: {speaker_audio.shape}, dtype: {speaker_audio.dtype}")
            logger.debug(f"Speaker text: {speaker_text[:50]}...")
            
            with torch.inference_mode():
                logger.info("Calling model.generate()...")
                audio_list = model.generate(
                    text=text,
                    ref_audio=speaker_audio,
                    ref_text=speaker_text,
                )
                logger.info(f"Generation completed. Result type: {type(audio_list)}, length: {len(audio_list) if audio_list else 0}")

        if not audio_list:
            logger.error("Model returned empty audio list")
            raise RuntimeError("The model returned no audio.")

        audio = audio_list[0]
        logger.debug(f"Audio type: {type(audio)}, shape: {audio.shape if hasattr(audio, 'shape') else 'N/A'}")

        if isinstance(audio, torch.Tensor):
            logger.debug("Converting tensor to numpy...")
            audio = audio.detach().float().cpu().numpy()
            logger.debug(f"Converted audio shape: {audio.shape}, dtype: {audio.dtype}")

        if audio.ndim > 1:
            logger.debug(f"Squeezing audio from {audio.ndim}D to 1D")
            audio = audio.squeeze()

        logger.info(f"Final audio shape: {audio.shape}, dtype: {audio.dtype}")
        logger.info("Writing WAV to buffer...")
        
        wav_buffer = io.BytesIO()

        sf.write(
            wav_buffer,
            audio,
            SAMPLE_RATE,
            format="WAV",
        )

        wav_buffer.seek(0)

        elapsed = time.perf_counter() - started_at
        logger.info(f"TTS generation successful in {elapsed:.3f}s")

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
        logger.error("GPU out of memory")
        logger.error(traceback.format_exc())
        torch.cuda.empty_cache()

        raise HTTPException(
            status_code=503,
            detail="GPU memory is exhausted.",
        ) from exc

    except Exception as exc:
        logger.error(f"Speech generation failed: {exc}")
        logger.error(f"Exception type: {type(exc).__name__}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail=f"Speech generation failed: {str(exc)}",
        ) from exc
