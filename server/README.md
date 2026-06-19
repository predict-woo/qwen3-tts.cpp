# qwen3-tts Server

OpenAI-compatible TTS HTTP server backed by the qwen3-tts.cpp shared library.

## Prerequisites

1. Build qwen3-tts.cpp with the shared library (done by default — works on macOS/Linux/Windows, CPU-only or any GPU backend; see main README for per-platform build instructions):
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j$(nproc)
   ```
   This produces `build/libqwen3tts.{so,dylib,dll}`. On Windows, copy the GGML DLLs (`ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, and any backend DLL) into the same directory as `libqwen3tts.dll` before launching the server, or point `QWEN3TTS_LIB_PATH` at an absolute path whose directory holds them.

2. Download or convert GGUF models into `models/` (see main README).

3. Install Python dependencies:
   ```bash
   cd server
   pip install -r requirements.txt
   ```

## Quick Start

```bash
cd server
python main.py
```

The server starts on `http://0.0.0.0:8000`.

## API

### POST /v1/audio/speech

OpenAI-compatible text-to-speech endpoint.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from qwen3-tts!", "model": "qwen3-tts"}' \
  -o output.wav
```

Request body:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `"qwen3-tts"` | Accepted but ignored (single model) |
| `input` | string | (required) | Text to synthesize |
| `voice` | string | `"default"` | Voice name (see below) |
| `response_format` | string | `"wav"` | Only `"wav"` supported |
| `speed` | float | `1.0` | Speech speed (0.25-4.0, maps to temperature) |

### GET /v1/audio/voices

List available voice names.

```bash
curl http://localhost:8000/v1/audio/voices
```

### GET /health

Health check.

## Voice Setup

Voices are speaker embeddings stored as JSON files in the `voices/` directory.

### Extract a voice from reference audio:

```bash
# Using the CLI
./build/qwen3-tts-cli -m models -r reference.wav \
  --dump-speaker-embedding voices/myvoice.json \
  -t "extraction" -o /dev/null
```

Then use it in API requests:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello!", "voice": "myvoice"}' \
  -o output.wav
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN3TTS_MODEL_DIR` | `../models` | Path to GGUF model directory |
| `QWEN3TTS_VOICES_DIR` | `../voices` | Path to speaker embedding JSON files |
| `QWEN3TTS_LIB_PATH` | (auto-detect) | Explicit path to `libqwen3tts.{so,dylib}` |
| `QWEN3TTS_THREADS` | `4` | Number of compute threads |
| `QWEN3TTS_ICL_CACHE_SIZE` | `8` | Maximum prepared ICL prompt handles to keep for repeated reference-audio requests. Set `0` to disable. |
| `QWEN3TTS_HOST` | `0.0.0.0` | Server bind address |
| `QWEN3TTS_PORT` | `8000` | Server port |

## Runtime Caches and Concurrency

The server keeps two voice-related caches:

- JSON speaker embeddings from `QWEN3TTS_VOICES_DIR` are loaded at startup and reused for `voice` requests backed by `.json` embedding files.
- ICL requests that provide `reference_audio_path` and `reference_text` use a bounded prepared-prompt cache controlled by `QWEN3TTS_ICL_CACHE_SIZE`. Entries store the native prepared prompt handle: reference audio speaker embedding, reference Mimi codec tokens, reference transcript, and related prompt metadata. They do not cache generated speech.

ICL cache keys include role-specific model file identities for the TTS model, speaker encoder, codec encoder, and tokenizer/vocoder decoder; the reference audio path, size, and mtime; a hash of `reference_text`; and the language id. Some roles may point at the same GGUF file today, but they are tracked separately so future split-model layouts invalidate correctly.

All synthesis paths still run under one process-local synthesis lock because the C API is not request-thread-safe. The ICL cache reduces repeated reference preparation latency; it does not enable parallel synthesis.

## OpenAI SDK Compatibility

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.audio.speech.create(
    model="qwen3-tts",
    voice="default",
    input="Hello from the OpenAI SDK!",
)
response.stream_to_file("output.wav")
```
