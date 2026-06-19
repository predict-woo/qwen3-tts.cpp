"""OpenAI-compatible TTS server for qwen3-tts.cpp.

Serves POST /v1/audio/speech following the OpenAI TTS API spec.
Uses the qwen3-tts C shared library via ctypes.
"""

from __future__ import annotations

import array
import asyncio
import io
import json
import os
import struct
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from icl_cache import IclPromptCache, build_icl_prompt_cache_key
from qwen3_tts_binding import QwenTTS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = os.environ.get("QWEN3TTS_MODEL_DIR", str(Path(__file__).parent.parent / "models"))
VOICES_DIR = os.environ.get("QWEN3TTS_VOICES_DIR", str(Path(__file__).parent.parent / "voices"))
N_THREADS = int(os.environ.get("QWEN3TTS_THREADS", "4"))
ICL_CACHE_SIZE = int(os.environ.get("QWEN3TTS_ICL_CACHE_SIZE", "8"))
DEFAULT_LANGUAGE_ID = 2050

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

tts_engine: QwenTTS | None = None
voice_embeddings: dict[str, list[float]] = {}
icl_prompt_cache = IclPromptCache(ICL_CACHE_SIZE)
_synthesis_lock = threading.Lock()  # C API is not thread-safe


def load_voices():
    """Scan voices/ directory and cache speaker embeddings."""
    voices_path = Path(VOICES_DIR)
    if not voices_path.is_dir():
        return
    for f in voices_path.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            if "data" in data and isinstance(data["data"], list):
                voice_embeddings[f.stem] = data["data"]
                print(f"  Loaded voice: {f.stem} ({len(data['data'])} dims)")
        except Exception as e:
            print(f"  Warning: failed to load voice {f.name}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load TTS engine and voices at startup."""
    global tts_engine
    print(f"Loading qwen3-tts models from: {MODEL_DIR}")
    tts_engine = QwenTTS(MODEL_DIR, n_threads=N_THREADS)
    print(f"Model loaded: type={tts_engine.model_type} size={tts_engine.model_size}")
    print(f"Scanning voices from: {VOICES_DIR}")
    load_voices()
    presets = tts_engine.list_speakers()
    print(f"ICL prompt cache size: {icl_prompt_cache.max_entries}")
    print(f"Ready. {len(voice_embeddings)} JSON voice(s), {len(presets)} model preset(s).")
    yield
    icl_prompt_cache.clear()
    if tts_engine:
        tts_engine.close()
        tts_engine = None


def _resolve_voice(voice: str):
    """Resolve a voice name to ('json', name) | ('preset', name) | ('default', None).

    Precedence: exact match in local JSON voices dir, then model presets.
    """
    if voice == "default":
        return ("default", None)
    if voice in voice_embeddings:
        return ("json", voice)
    if tts_engine is not None:
        for p in tts_engine.list_speakers():
            if p["name"] == voice:
                return ("preset", voice)
    return (None, None)


app = FastAPI(title="qwen3-tts server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# WAV encoding
# ---------------------------------------------------------------------------

def pcm_float32_to_wav(samples: list[float], sample_rate: int) -> bytes:
    """Pack PCM float32 samples into a WAV file (IEEE float format)."""
    n_samples = len(samples)
    n_channels = 1
    bits_per_sample = 32
    byte_rate = sample_rate * n_channels * (bits_per_sample // 8)
    block_align = n_channels * (bits_per_sample // 8)
    data_size = n_samples * (bits_per_sample // 8)

    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))  # file size - 8
    buf.write(b"WAVE")
    # fmt chunk (format tag 3 = IEEE float)
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 3))   # format: IEEE float
    buf.write(struct.pack("<H", n_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits_per_sample))
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    arr = array.array('f', samples)
    if sys.byteorder != 'little':
        arr.byteswap()
    buf.write(arr.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    model: str = "qwen3-tts"
    input: str
    voice: str = "default"
    response_format: str = "wav"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

    # ICL voice cloning: supply a path to a reference WAV on the server's
    # filesystem together with its transcript. When both are set, the request
    # takes the Mimi codec path; `voice` is ignored in that case.
    reference_audio_path: str | None = None
    reference_text: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail={"error": {"message": "input is required", "type": "invalid_request_error"}})

    if request.response_format != "wav":
        raise HTTPException(status_code=400, detail={"error": {"message": f"Unsupported format '{request.response_format}'. Only 'wav' is supported.", "type": "invalid_request_error"}})

    # Map speed to temperature: speed=1.0 → temp=0.9, speed=2.0 → temp=0.45
    temperature = min(2.0, max(0.1, 0.9 / request.speed))

    # ICL path: both reference_audio_path and reference_text provided.
    icl_ref_audio = (request.reference_audio_path or "").strip()
    icl_ref_text  = (request.reference_text or "").strip()
    if icl_ref_audio and icl_ref_text:
        if not os.path.isfile(icl_ref_audio):
            raise HTTPException(status_code=400, detail={"error": {"message": f"reference_audio_path does not exist: {icl_ref_audio}", "type": "invalid_request_error"}})
        try:
            def _do_icl():
                with _synthesis_lock:
                    cache_key = build_icl_prompt_cache_key(
                        tts_engine.icl_model_identities,
                        icl_ref_audio,
                        icl_ref_text,
                        DEFAULT_LANGUAGE_ID,
                    )
                    prompt = icl_prompt_cache.get(cache_key)
                    cache_enabled = icl_prompt_cache.max_entries > 0
                    if prompt is not None:
                        print(f"ICL prompt cache hit: {icl_ref_audio}")
                    else:
                        print(f"ICL prompt cache miss: {icl_ref_audio}")
                        prompt = tts_engine.prepare_icl_prompt(
                            icl_ref_audio, icl_ref_text,
                            temperature=temperature,
                            language_id=DEFAULT_LANGUAGE_ID,
                        )
                        if cache_enabled:
                            icl_prompt_cache.put(cache_key, prompt)
                    try:
                        return tts_engine.synthesize_with_icl_prompt(
                            request.input, prompt,
                            temperature=temperature,
                            language_id=DEFAULT_LANGUAGE_ID,
                        )
                    finally:
                        if not cache_enabled:
                            prompt.close()
            loop = asyncio.get_event_loop()
            samples, sample_rate = await loop.run_in_executor(None, _do_icl)
            wav_data = pcm_float32_to_wav(samples, sample_rate)
            return Response(content=wav_data, media_type="audio/wav")
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail={"error": {"message": str(e), "type": "server_error"}})
    elif icl_ref_audio or icl_ref_text:
        raise HTTPException(status_code=400, detail={"error": {"message": "reference_audio_path and reference_text must be provided together for ICL mode", "type": "invalid_request_error"}})

    voice = request.voice.lower()

    # Resolve voice: 'default' | local JSON voice | model preset
    kind, resolved = _resolve_voice(voice)
    if kind is None:
        preset_names = [p["name"] for p in (tts_engine.list_speakers() if tts_engine else [])]
        available = sorted(set(["default"] + list(voice_embeddings.keys()) + preset_names))
        raise HTTPException(status_code=400, detail={"error": {"message": f"Unknown voice '{request.voice}'. Available: {', '.join(available)}", "type": "invalid_request_error"}})

    try:
        def _do_synthesis():
            with _synthesis_lock:
                if kind == "default":
                    return tts_engine.synthesize(
                        request.input, temperature=temperature,
                    )
                if kind == "json":
                    embedding = voice_embeddings[resolved]
                    return tts_engine.synthesize_with_embedding(
                        request.input, embedding, temperature=temperature,
                    )
                # kind == "preset"
                return tts_engine.synthesize_with_preset(
                    request.input, resolved, temperature=temperature,
                )

        loop = asyncio.get_event_loop()
        samples, sample_rate = await loop.run_in_executor(None, _do_synthesis)

        wav_data = pcm_float32_to_wav(samples, sample_rate)
        return Response(content=wav_data, media_type="audio/wav")

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail={"error": {"message": str(e), "type": "server_error"}})


@app.get("/v1/audio/voices")
async def list_voices():
    """List available voice names (local JSONs + model presets)."""
    presets = tts_engine.list_speakers() if tts_engine else []
    json_voices = [{"name": n, "source": "json"} for n in sorted(voice_embeddings.keys())]
    preset_voices = [
        {"name": p["name"], "source": "preset", "dialect": p.get("dialect", "")}
        for p in presets
    ]
    return {
        "voices": ["default"] + sorted(voice_embeddings.keys()) + [p["name"] for p in presets],
        "detail": [{"name": "default", "source": "default"}] + json_voices + preset_voices,
        "model_type": tts_engine.model_type if tts_engine else "",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": tts_engine is not None}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("QWEN3TTS_HOST", "0.0.0.0")
    port = int(os.environ.get("QWEN3TTS_PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port)
