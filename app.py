# main.py
# Urban Sound Narrative API — Railway-optimized (CPU-only, low-RAM safe)
# - Whisper (base) is lazy-loaded and aggressively unloaded after each use
# - PANNs AudioTagging is a singleton kept resident
# - Pipeline is serialized via an asyncio.Semaphore to avoid RAM spikes
# - SSE streaming for frontend progress
# - Hard caps on upload size and duration
# - External services: Groq (LLM), ElevenLabs (TTS)

import os
import io
import gc
import re
import uuid
import json
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager

import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# External services
import whisper
from panns_inference import AudioTagging, labels
from groq import Groq
from pydub import AudioSegment
import requests

# -----------------------------
# Settings & Globals
# -----------------------------
torch.set_num_threads(1)  # keep CPU use predictable on small containers
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))                 # hard request limit
MAX_AUDIO_DURATION_SEC = int(os.getenv("MAX_AUDIO_DURATION_SEC", "180"))  # 3 minutes
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))      # serialize heavy ops

WORK_DIR = Path(os.getenv("WORK_DIR", "/app"))
AUDIO_DIR = WORK_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"  # Railway CPU container
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

MODELS = {
    "panns": None,   # singleton kept resident
    "whisper": None  # loaded per call and then freed
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("urban-narration")


# -----------------------------
# FastAPI App & Lifespan
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load PANNs once (singleton)
    try:
        log.info("Loading PANNs AudioTagging singleton (CPU)...")
        MODELS["panns"] = AudioTagging(checkpoint_path=None, device=DEVICE)
        log.info("PANNs ready.")
    except Exception:
        log.exception("Failed to load PANNs model")
        raise

    yield

    # Shutdown cleanup
    MODELS["panns"] = None
    gc.collect()
    log.info("Shutdown complete: models cleared.")

app = FastAPI(
    title="Urban Sound Narrative API",
    description="Transforms urban sounds + speech into cinematic narration",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# -----------------------------
# DTOs
# -----------------------------
class PipelineResult(BaseModel):
    narration: str
    audio_url: str
    detected_sounds: List[str]
    transcript: str


# -----------------------------
# Utilities
# -----------------------------
def _safe_name(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]', '_', name)

def _ensure_duration_ok(filepath: Path):
    try:
        info = torchaudio.info(str(filepath))
        seconds = info.num_frames / max(1, info.sample_rate)
        if seconds > MAX_AUDIO_DURATION_SEC:
            raise HTTPException(
                status_code=413,
                detail=f"Audio too long ({seconds:.1f}s). Max is {MAX_AUDIO_DURATION_SEC}s."
            )
    except Exception as e:
        # If torchaudio backend can’t parse (rare), allow small files; otherwise reject
        log.warning(f"Duration check warning: {e}")

def _resample_mono_32k(src_path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(src_path))
    if wav.ndim == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 32000:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=32000)(wav)
    # Return CPU tensor, shape (1, T)
    return wav

def _sse(msg: dict) -> str:
    return json.dumps(msg) + "\n"


# -----------------------------
# Whisper (lazy, unload after use)
# -----------------------------
async def _transcribe_with_whisper(src_path: Path) -> str:
    model = None
    try:
        log.info("Lazy-loading Whisper base (CPU)...")
        model = whisper.load_model("base", device=DEVICE)
        result = model.transcribe(str(src_path))
        text = (result.get("text") or "").strip() or "(no speech detected)"
        log.info(f"Transcription length: {len(text)} chars")
        return text
    except Exception:
        log.exception("Whisper transcription failed")
        return "(transcription failed)"
    finally:
        # Aggressively free memory
        try:
            del model  # type: ignore
        except:
            pass
        gc.collect()
        log.info("Whisper unloaded.")


# -----------------------------
# PANNs (singleton)
# -----------------------------
def _detect_sounds_panns(waveform_mono_32k: torch.Tensor) -> List[str]:
    clipwise, _ = MODELS["panns"].inference(waveform_mono_32k)
    clipwise = clipwise.squeeze()
    top_idx = clipwise.argsort()[-5:][::-1]
    sounds = [labels[int(i)] for i in top_idx]
    log.info(f"Detected sounds: {sounds}")
    return sounds


# -----------------------------
# Groq LLM
# -----------------------------
def _build_llm_prompt(sound_ctx: str, transcript: str) -> str:
    return (
        "You are a master storyteller crafting a cinematic urban narrative. "
        f"Detected sounds: {sound_ctx}. "
        "Write a vivid, emotionally evocative 2–3 sentence scene that feels alive with sensory detail. "
        f"Seamlessly integrate this spoken phrase: '{transcript}'. "
        "Use [bracketed_emotion] only for dialogue tone (e.g., [thoughtfully]). "
        "Avoid literal sound effect words or onomatopoeia."
    )

async def _narrate_groq(sound_types: List[str], transcript: str) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing")
    client = Groq(api_key=GROQ_API_KEY)
    prompt = _build_llm_prompt(", ".join(sound_types), transcript)
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=150,
        top_p=0.9,
        messages=[{"role": "user", "content": prompt}],
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="LLM returned empty narration")
    log.info(f"Narration length: {len(text)} chars")
    return text


# -----------------------------
# ElevenLabs TTS
# -----------------------------
async def _tts_elevenlabs(narration: str, out_name: str) -> Path:
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY missing")

    # Voice: replace with your voice id if needed
    url = "https://api.elevenlabs.io/v1/text-to-speech/cgSgspJ2msm6clMCkdW9"
    payload = {
        "text": narration,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.75,
            "style": 0.8,
            "use_speaker_boost": True
        },
        "output_format": "mp3_44100_128"
    }
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }

    r = requests.post(url, headers=headers, json=payload, timeout=45)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"ElevenLabs error: {r.text}")

    tmp_path = AUDIO_DIR / f"tmp_{out_name}"
    with open(tmp_path, "wb") as f:
        f.write(r.content)

    # Gentle polish (requires ffmpeg in container)
    audio = AudioSegment.from_mp3(tmp_path)
    audio = audio + 6
    audio = audio.normalize()
    audio = audio.fade_in(80).fade_out(160)

    out_path = AUDIO_DIR / out_name
    audio.export(out_path, format="mp3", bitrate="192k", parameters=["-q:a", "0"])
    tmp_path.unlink(missing_ok=True)

    log.info(f"TTS ready: {out_path}")
    return out_path


# -----------------------------
# API Endpoints
# -----------------------------
@app.post("/process-audio-stream")
async def process_audio_stream(file: UploadFile = File(...)):
    # Simple extension check
    if not (file.filename.endswith(".mp3") or file.filename.endswith(".wav")):
        raise HTTPException(status_code=400, detail="Only .mp3 or .wav allowed")

    # Persist to /tmp (Railway supports this)
    uid = str(uuid.uuid4())
    safe_in = _safe_name(file.filename)
    in_path = Path(tempfile.gettempdir()) / f"upload_{uid}_{safe_in}"
    out_name = f"narration_{uid}.mp3"

    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_MB} MB.")

    with open(in_path, "wb") as f:
        f.write(content)

    async def _run():
        try:
            yield _sse({"stage": "loading", "message": "Analyzing audio...", "progress": 10})
            _ensure_duration_ok(in_path)

            async with SEMAPHORE:
                # Transcribe (lazy-load + unload Whisper)
                yield _sse({"stage": "transcribing", "message": "Transcribing speech...", "progress": 20})
                transcript = await _transcribe_with_whisper(in_path)
                yield _sse({"stage": "transcribe_complete", "message": "Speech transcribed.", "progress": 35})

                # Prepare waveform (mono 32k)
                yield _sse({"stage": "extracting", "message": "Preparing waveform...", "progress": 45})
                wf = _resample_mono_32k(in_path)

                # Tag sounds (PANNs singleton)
                yield _sse({"stage": "identifying", "message": "Identifying urban soundscapes...", "progress": 55})
                sounds = _detect_sounds_panns(wf)
                yield _sse({"stage": "sounds_detected", "sounds": sounds, "message": "Sounds detected.", "progress": 65})

                # LLM narration (remote)
                yield _sse({"stage": "ai_processing", "message": "Crafting narrative...", "progress": 75})
                narration = await _narrate_groq(sounds, transcript)

                # TTS (remote) + polish
                yield _sse({"stage": "voice_generation", "message": "Generating expressive narration...", "progress": 90})
                out_path = await _tts_elevenlabs(narration, out_name)

            audio_url = f"/audio/{out_name}"
            yield _sse({
                "stage": "complete",
                "message": "Done.",
                "progress": 100,
                "narration": narration,
                "audio_url": audio_url,
                "detected_sounds": sounds,
                "transcript": transcript
            })
        except HTTPException as he:
            yield _sse({"stage": "error", "message": he.detail, "progress": 0})
        except Exception as e:
            log.exception("Pipeline failed")
            yield _sse({"stage": "error", "message": str(e), "progress": 0})
        finally:
            try:
                in_path.unlink(missing_ok=True)
            except:
                pass
            gc.collect()

    return StreamingResponse(_run(), media_type="text/event-stream")


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    fp = AUDIO_DIR / _safe_name(filename)
    if not fp.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(
        str(fp),
        media_type="audio/mpeg",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
            "Access-Control-Expose-Headers": "*"
        }
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "panns_loaded": MODELS["panns"] is not None,
        "device": DEVICE,
        "limits": {
            "max_upload_mb": MAX_UPLOAD_MB,
            "max_duration_sec": MAX_AUDIO_DURATION_SEC,
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS
        }
    }


@app.post("/warmup")
async def warmup():
    # Touch PANNs quickly to ensure weights are resident
    dummy = torch.zeros(1, 32000)  # 1 second of 32k mono
    _ = _detect_sounds_panns(dummy)
    return {"status": "warmed"}


@app.get("/")
async def root():
    return {
        "name": "Urban Sound Narrative API",
        "version": "3.0.0",
        "message": "Whisper(base) lazy + PANNs singleton; SSE pipeline; Railway-optimized",
    }


# Optional: local dev entrypoint (ignored by Railway which uses Procfile)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
