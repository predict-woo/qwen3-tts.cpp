# qwen3-tts.cpp

![PyTorch vs qwen3-tts.cpp benchmark](./docs/benchmark_pytorch_vs_cpp.png)

**Benchmark Snapshot (PyTorch vs qwen3-tts.cpp):** Basic 3.19x faster, Clone 4.07x faster. Peak RSS delta: Basic +19.0%, Clone +7.7%.

C++ inference for [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) using the [GGML](https://github.com/ggml-org/ggml) tensor library.

Runs the full TTS pipeline in pure C++17, including text tokenization, speaker encoding, transformer code generation, and vocoder decoding, without Python or PyTorch at inference time.

## Features

- Full text-to-speech pipeline in C++17 with GGML backend
- **0.6B and 1.7B model support** (Base, CustomVoice, VoiceDesign)
- Voice cloning from reference audio (ECAPA-TDNN x-vector extraction)
- **Speaker embedding save/load** for fast repeated cloning (skip encoder)
- **Voice steering instructions** for controlling speech style and emotion
- **Unicode/multilingual tokenizer** (Chinese, Japanese, Korean, German, etc.)
- Greedy and sampled decoding (temperature, top-k, repetition penalty, seed)
- GGUF model format with **built-in C++ quantizer** (F16, Q8_0, Q4_K, etc.)
- Runtime backend selection: Metal (macOS), Vulkan (Linux/Windows), CUDA (NVIDIA)
- Cross-platform: macOS, Linux, Windows (MSVC)
- C API for FFI integration (Python, Rust, Nim, etc.)
- Deterministic reference tests comparing C++ output against Python

## Prerequisites

- **Compiler:** GCC 9+ or Clang 10+ (Linux/macOS), or MSVC 2019+ (Windows)
- **CMake:** 3.14+
- **GGML:** Built from source (vendored as git submodule)
- **GPU backends (optional):**
  - Metal — macOS only, built into Xcode Command Line Tools
  - [Vulkan SDK](https://vulkan.lunarg.com/) — Linux and Windows (AMD, NVIDIA, Intel)
  - [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) — NVIDIA GPUs
- **Python 3.10+** with [uv](https://github.com/astral-sh/uv) — model conversion only; not needed if using [pre-built GGUF files](#pre-built-models-skip-python)

## Quickstart (macOS)

Run these commands from a fresh clone. For **Linux** or **Windows**, see the [Build](#build) section below.

```bash
git clone https://github.com/predict-woo/qwen3-tts.cpp.git
cd qwen3-tts.cpp
git submodule update --init --recursive

# 1) Build GGML with Metal
cmake -S ggml -B ggml/build -DGGML_METAL=ON
cmake --build ggml/build -j4

# 2) Build qwen3-tts.cpp
cmake -S . -B build
cmake --build build -j4

# 3) Create a uv Python environment for setup/conversion tools
uv venv .venv
source .venv/bin/activate

# 4) Install Python dependencies
uv pip install --upgrade pip
uv pip install huggingface_hub gguf torch safetensors numpy tqdm coremltools

# Optional if model access requires auth:
# huggingface-cli login

# 5) Download and generate all runtime model artifacts
python scripts/setup_pipeline_models.py

# 6) Basic synthesis example
./build/qwen3-tts-cli \
  -m models \
  -t "Hello from qwen3-tts.cpp running on macOS with CoreML by default." \
  -o examples/readme_example_basic.wav

# 7) Voice-clone example using sample audio in this repo
./build/qwen3-tts-cli \
  -m models \
  -r examples/readme_clone_input.wav \
  -t "This is a voice cloning example generated from the sample audio file in this directory." \
  -o examples/readme_example_clone.wav
```

Expected model artifacts after step 5:

- `models/qwen3-tts-0.6b-f16.gguf`
- `models/qwen3-tts-tokenizer-f16.gguf`
- `models/coreml/code_predictor.mlpackage` (on macOS)

Expected audio outputs after steps 6-7:

- `examples/readme_example_basic.wav`
- `examples/readme_example_clone.wav`

Included voice-clone input/output pair (so users can compare directly):

- Input reference audio: `examples/readme_clone_input.wav`
- Generated output audio: `examples/readme_example_clone.wav`

Audio preview (inline):

<audio controls src="./examples/readme_clone_input.wav"></audio>
<br/>
<audio controls src="./examples/readme_example_clone.wav"></audio>

If your Markdown renderer does not show inline controls, use direct links:

- [Play input reference WAV](./examples/readme_clone_input.wav)
- [Play generated output WAV](./examples/readme_example_clone.wav)

## Build

### Clone and initialize

```bash
git clone https://github.com/predict-woo/qwen3-tts.cpp.git
cd qwen3-tts.cpp
git submodule update --init --recursive
```

> **Note:** The top-level CMake expects GGML in `./ggml` with libraries under `./ggml/build/src`.

### Build matrix

Pick the row that matches your platform and GPU. CPU backend is always compiled in — GPU flags just add an additional backend that `QWEN3_TTS_BACKEND=auto` (the default) will prefer when available.

| Platform | Mode | GGML flags (step 1) | Notes |
|---|---|---|---|
| macOS (Apple Silicon / Intel) | Metal + CPU (default) | *(none — Metal and Apple Accelerate BLAS are default-on)* | CoreML code predictor also enabled by default; see below. |
| macOS | CPU only | `-DGGML_METAL=OFF -DGGML_BLAS=OFF` + top-level `-DQWEN3_TTS_COREML=OFF` | Useful for bisecting GPU issues or on machines without a usable GPU. |
| Linux | CPU only | *(none)* | Opt into BLAS with `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS` (or similar) if installed. |
| Linux | Vulkan + CPU | `-DGGML_VULKAN=ON` | Works on AMD, NVIDIA, Intel; install the [Vulkan SDK](https://vulkan.lunarg.com/) first. |
| Linux | CUDA + CPU | `-DGGML_CUDA=ON` | NVIDIA only; requires the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). |
| Windows (MSVC) | CPU only | *(none)* | Open a Developer Command Prompt for VS 2019/2022. |
| Windows | Vulkan + CPU | `-DGGML_VULKAN=ON` | Copy `ggml-vulkan.dll` next to `qwen3-tts-cli.exe`. |
| Windows | CUDA + CPU | `-DGGML_CUDA=ON` | Copy `ggml-cuda.dll` next to `qwen3-tts-cli.exe`. |

"GPU + CPU" is the normal operating mode for any GPU build — the GGML scheduler runs most work on GPU with automatic CPU fallback for unsupported ops. See [Backend Selection](#backend-selection) for how to force one or the other at runtime.

### macOS (Metal + CoreML, default)

```bash
# Step 1: Build GGML. Metal and Accelerate BLAS are on by default on Apple;
# passing -DGGML_METAL=ON is optional and just makes the intent explicit.
cmake -S ggml -B ggml/build -DCMAKE_BUILD_TYPE=Release
cmake --build ggml/build -j$(sysctl -n hw.ncpu)

# Step 2: Build qwen3-tts.cpp. QWEN3_TTS_COREML is ON by default on APPLE.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
```

CoreML acceleration for the code predictor stage is enabled by default when `models/coreml/code_predictor.mlpackage` exists. Generate that with `python scripts/setup_pipeline_models.py --coreml on`. Turn CoreML off at runtime with `QWEN3_TTS_USE_COREML=0` or at compile time with `-DQWEN3_TTS_COREML=OFF`.

### macOS (CPU only)

```bash
# Disable Metal, Accelerate BLAS, and the CoreML code predictor bridge.
cmake -S ggml -B ggml/build -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=OFF -DGGML_BLAS=OFF
cmake --build ggml/build -j$(sysctl -n hw.ncpu)

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQWEN3_TTS_COREML=OFF
cmake --build build -j$(sysctl -n hw.ncpu)
```

You can also keep a Metal build and force CPU at runtime with `QWEN3_TTS_BACKEND=cpu` — no rebuild required.

### Linux (CPU only)

```bash
cmake -S ggml -B ggml/build -DCMAKE_BUILD_TYPE=Release
cmake --build ggml/build -j$(nproc)

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Linux (Vulkan + CPU)

Install the [Vulkan SDK](https://vulkan.lunarg.com/) first, then:

```bash
cmake -S ggml -B ggml/build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ggml/build -j$(nproc)

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Linux (CUDA + CPU)

Requires the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), then:

```bash
cmake -S ggml -B ggml/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ggml/build -j$(nproc)

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

At runtime, set `QWEN3_TTS_BACKEND=cuda` to force the CUDA backend, or leave it at the default `auto` to let the scheduler pick it when available (see [Backend Selection](#backend-selection)).

### Windows (MSVC)

Open a **Developer Command Prompt for VS 2022** (or 2019), then pick one:

```cmd
:: CPU only (default)
cmake -S ggml -B ggml\build -DCMAKE_BUILD_TYPE=Release
cmake --build ggml\build --config Release

:: Vulkan + CPU (install Vulkan SDK from https://vulkan.lunarg.com/)
:: cmake -S ggml -B ggml\build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release

:: CUDA + CPU (install CUDA Toolkit)
:: cmake -S ggml -B ggml\build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release

:: Then build qwen3-tts.cpp against the chosen GGML build.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**DLL placement:** Copy these DLLs next to `qwen3-tts-cli.exe` (and `libqwen3tts.dll` if you use the shared lib):

- Always: `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`
- Vulkan build: also `ggml-vulkan.dll`
- CUDA build: also `ggml-cuda.dll` plus CUDA runtime DLLs from your toolkit install

These are produced under `ggml\build\bin\Release\` (or `ggml\build\src\` depending on generator).

## Model Setup (Recommended)

Use the one-shot setup script:

```bash
source .venv/bin/activate
python scripts/setup_pipeline_models.py
```

Useful flags:

- `--force` re-downloads and re-generates all artifacts.
- `--coreml auto|on|off` controls CoreML export behavior.
- `--skip-download` skips HF download and uses existing local model dirs.

### Pre-built Models (Skip Python)

Community-hosted GGUF files are available on HuggingFace. Download directly — no Python required:

```bash
mkdir -p models
curl -L -o models/qwen3-tts-0.6b-f16.gguf \
  "https://huggingface.co/koboldcpp/tts/resolve/main/qwen3-tts-0.6b-f16.gguf"
curl -L -o models/qwen3-tts-tokenizer-f16.gguf \
  "https://huggingface.co/koboldcpp/tts/resolve/main/qwen3-tts-tokenizer-f16.gguf"
```

These are F16 (full-precision) models. Q8\_0 quantized variants are also available and will be auto-detected if named `qwen3-tts-0.6b-q8_0.gguf`.

## Manual Model Conversion (Advanced)

Convert HuggingFace models to GGUF format:

```bash
# Download the model
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --local-dir models/Qwen3-TTS-12Hz-0.6B-Base

# Convert TTS model (transformer + speaker encoder + tokenizer)
python scripts/convert_tts_to_gguf.py \
    models/Qwen3-TTS-12Hz-0.6B-Base \
    models/qwen3-tts-0.6b-f16.gguf

# Convert vocoder (audio decoder)
python scripts/convert_tokenizer_to_gguf.py \
    models/Qwen3-TTS-12Hz-0.6B-Base \
    models/qwen3-tts-tokenizer-f16.gguf
```

Place both `.gguf` files in a `models/` directory.

## Usage

```bash
# Basic synthesis
./build/qwen3-tts-cli -m models -t "Hello, world!" -o hello.wav

# Voice cloning from reference audio
./build/qwen3-tts-cli -m models -t "Hello! How are you?" -r reference.wav -o cloned.wav

# Greedy decoding with max length
./build/qwen3-tts-cli -m models -t "Hello!" -r ref.wav -o out.wav \
    --temperature 0 --max-tokens 2048
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model <dir>` | Model directory containing GGUF files | (required) |
| `--tts-model <file>` | TTS model filename (overrides auto-detection) | (auto) |
| `--tokenizer-model <file>` | Tokenizer/vocoder filename | (auto) |
| `-t, --text <text>` | Text to synthesize | (required) |
| `-o, --output <file>` | Output WAV file path | `output.wav` |
| `-r, --reference <file>` | Reference audio for voice cloning | (none) |
| `--speaker-embedding <file>` | Use precomputed speaker embedding (.json/.bin) | (none) |
| `--dump-speaker-embedding <file>` | Save extracted embedding from `--reference` | (none) |
| `--speaker <name>` | Use a named preset voice (CustomVoice / VoiceDesign models) | (none) |
| `--list-speakers` | Print the preset voices baked into the loaded model and exit | — |
| `-i, --instruction, --instruct <text>` | Voice steering instruction (e.g. "Speak happily") | (none) |
| `--temperature <val>` | Sampling temperature (0 = greedy) | 0.9 |
| `--top-k <n>` | Top-k sampling (0 = disabled) | 50 |
| `--top-p <val>` | Top-p (nucleus) sampling cutoff | 1.0 |
| `--max-tokens <n>` | Maximum audio frames (codec tokens) to generate | 2048 |
| `--repetition-penalty <val>` | Repetition penalty on codebook-0 token generation | 1.05 |
| `--seed <n>` | RNG seed for reproducible output | (random) |
| `--no-f32-acc` | Disable f32 matmul accumulation (faster, less precise) | (off) |
| `-l, --language <lang>` | Language: en, ru, zh, ja, ko, de, fr, es | en |
| `-j, --threads <n>` | Number of compute threads | 4 |

### Preset Voices (CustomVoice / VoiceDesign)

The `Qwen3-TTS-12Hz-1.7B-CustomVoice` and `-VoiceDesign` repos ship a set of built-in preset voices (e.g. `serena`, `ryan`, `dylan`) that are reserved codec token IDs. No reference audio is required — just pick a preset by name.

```bash
# Convert the CustomVoice checkpoint (once)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --local-dir models/hf/qwen3-tts-1.7b-customvoice
python scripts/convert_tts_to_gguf.py \
    --input models/hf/qwen3-tts-1.7b-customvoice \
    --output models/qwen3-tts-1.7b-customvoice-f16.gguf --type f16

# List preset voices baked into the model
./build/qwen3-tts-cli -m models \
  --tts-model qwen3-tts-1.7b-customvoice-f16.gguf --list-speakers

# Synthesize with a named preset
./build/qwen3-tts-cli -m models \
  --tts-model qwen3-tts-1.7b-customvoice-f16.gguf \
  --speaker serena -t "Hello from the preset voice!" -o out.wav
```

Base variants (`-Base` on HF) carry no preset list — use voice cloning via `-r reference.wav` instead. The Python server's `/v1/audio/voices` endpoint lists both local JSON embeddings (from `voices/`) and model-level presets; `/v1/audio/speech` accepts either kind in the `voice` field.

### Speaker Embedding Workflow

Precompute a speaker embedding once (saves ~20s per synthesis):

```bash
# Extract and save embedding from reference audio
./build/qwen3-tts-cli -m models -r reference.wav --dump-speaker-embedding speaker.json \
  -t "Initial extraction." -o /dev/null

# Reuse embedding for fast voice-cloned synthesis (no encoder needed)
./build/qwen3-tts-cli -m models --speaker-embedding speaker.json \
  -t "Fast voice cloning with cached embedding." -o output.wav
```

### Voice Steering Instructions

Control voice characteristics with the `--instruction` flag (works best with 1.7B VoiceDesign models):

```bash
./build/qwen3-tts-cli -m models --instruction "Speak in a cheerful tone" \
  -t "Good morning, everyone!" -o cheerful.wav
```

### Quantization

Quantize models from F16 to smaller formats without Python:

```bash
./build/qwen3-tts-quantize --input models/qwen3-tts-0.6b-f16.gguf \
  --output models/qwen3-tts-0.6b-q8_0.gguf --type q8_0
```

Supported types: `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q4_k`, `q5_k`, `q6_k`. The CLI auto-detects Q8_0 models when present.

### Backend Selection

At runtime, each component (tokenizer, encoder, transformer, decoder) independently selects a backend and logs it — for example, `TTSTransformer backend: MTL0` or `AudioTokenizerDecoder backend: Vulkan0`.

`QWEN3_TTS_BACKEND` controls the selection:

| Value | Effect |
|---|---|
| `auto` (default) | Scheduler picks the best available device in the order `IGPU` → `GPU` → `ACCEL` → `CPU`. Metal, Vulkan, and CUDA all surface through `auto` when the corresponding GGML flag was enabled at build time. |
| `cpu` | Forces CPU-only execution regardless of what was built in. Useful for reproducibility and for bisecting GPU issues without rebuilding. |
| `cuda` | Forces the CUDA backend. Only valid when GGML was built with `-DGGML_CUDA=ON`; otherwise falls back to CPU with a warning. |

There is no separate `metal` or `vulkan` value — those are reached via `auto`. If you need to disable them, either rebuild without their GGML flag or set `QWEN3_TTS_BACKEND=cpu`.

"Hybrid GPU + CPU" is the default behavior of any GPU build: the GGML scheduler keeps most work on the GPU and automatically falls back to CPU for ops the GPU backend does not support.

### Runtime environment variables

| Variable | Default | Purpose |
|---|---|---|
| `QWEN3_TTS_BACKEND` | `auto` | Runtime backend override. See [Backend Selection](#backend-selection). |
| `QWEN3_TTS_DEVICE` | `0` | CUDA device index when `QWEN3_TTS_BACKEND=cuda`. Ignored otherwise. |
| `QWEN3_TTS_DECODER_GPU_MAX_FRAMES` | `34` | Max frames per CUDA vocoder chunk. Lower it if the GPU OOMs during decode. |
| `QWEN3_TTS_DECODER_GPU_CONTEXT_FRAMES` | `12` | Left-context frames per CUDA vocoder chunk. |
| `QWEN3_TTS_LOW_MEM` | unset | Set to `1` to enable low-memory pipeline mode (loads/unloads components in sequence instead of holding everything resident). |
| `QWEN3_TTS_USE_COREML` | `1` on macOS when model exists | Set to `0` to disable the CoreML code-predictor bridge without rebuilding. |
| `QWEN3_TTS_COREML_MODEL` | auto-detected | Absolute path override for a custom `.mlpackage` location (macOS only). |

## C API

The shared library `libqwen3tts.{so,dylib,dll}` is built automatically by CMake (the `qwen3tts_shared` target). It provides a C-linkage API for FFI integration from Python, Rust, Nim, Go, etc.

### Structs

```c
typedef struct Qwen3TtsParams {
    int32_t max_audio_tokens;    // default: 2048
    float   temperature;         // default: 0.9, 0=greedy
    float   top_p;               // default: 1.0
    int32_t top_k;               // default: 50, 0=disabled
    int32_t n_threads;           // default: 4
    float   repetition_penalty;  // default: 1.05
    int32_t language_id;         // 2050=en, 2058=ja, 2055=zh, etc.
} Qwen3TtsParams;

typedef struct Qwen3TtsAudio {
    const float* samples;  // PCM float32 mono, 24kHz
    int32_t n_samples;
    int32_t sample_rate;   // always 24000
} Qwen3TtsAudio;
```

### Functions

| Function | Description |
|---|---|
| `qwen3_tts_default_params(params)` | Fill `Qwen3TtsParams` with defaults |
| `qwen3_tts_create(model_dir, n_threads)` | Load models, return opaque handle (NULL on failure) |
| `qwen3_tts_is_loaded(tts)` | Check if models are loaded |
| `qwen3_tts_synthesize(tts, text, params)` | Text to audio |
| `qwen3_tts_synthesize_with_voice_file(tts, text, wav_path, params)` | Voice clone from WAV |
| `qwen3_tts_synthesize_with_voice_samples(tts, text, samples, n, params)` | Voice clone from float32 |
| `qwen3_tts_extract_embedding_file(tts, wav_path, buf, max)` | Extract speaker embedding |
| `qwen3_tts_synthesize_with_embedding(tts, text, emb, size, params)` | Synthesize with cached embedding |
| `qwen3_tts_sample_rate(tts)` | Returns 24000 |
| `qwen3_tts_free_audio(audio)` | Free `Qwen3TtsAudio*` (required) |
| `qwen3_tts_destroy(tts)` | Destroy engine |
| `qwen3_tts_get_error(tts)` | Get last error string |

### C Usage Example

```c
#include "qwen3tts_c_api.h"

Qwen3Tts *tts = qwen3_tts_create("./models", 4);
if (!tts) { fprintf(stderr, "load failed\n"); return 1; }

Qwen3TtsParams params;
qwen3_tts_default_params(&params);

Qwen3TtsAudio *audio = qwen3_tts_synthesize(tts, "Hello, world!", &params);
if (audio) {
    // audio->samples is PCM float32, audio->n_samples samples at 24kHz
    // ... write to WAV file ...
    qwen3_tts_free_audio(audio);
}

qwen3_tts_destroy(tts);
```

### Python ctypes

A full Python binding is provided in `server/qwen3_tts_binding.py`. See the [Server README](server/README.md) for an OpenAI-compatible HTTP API.

### Memory Management

- Every `Qwen3TtsAudio*` returned by synthesis functions must be freed with `qwen3_tts_free_audio()`
- The engine handle must be destroyed with `qwen3_tts_destroy()`
- The `qwen3_tts_get_error()` string is owned by the engine and valid until the next API call

## Architecture

```
Text ──► [Tokenizer] ──► token IDs
                              │
Reference Audio ──► [Speaker Encoder] ──► speaker embedding
                              │
token IDs + speaker embedding ──► [TTS Transformer] ──► speech codes (N frames x 16 codebooks)
                              │
speech codes ──► [Vocoder] ──► audio waveform (24kHz)
```

### Source Files

| Module | Files | Description |
|--------|-------|-------------|
| `src/common/` | `gguf_loader`, `coreml_code_predictor`, `speaker_embedding_io` | Shared infrastructure |
| `src/tokenizer/` | `text_tokenizer`, `tokenizer_unicode` | BPE tokenizer with Unicode/GPT-2 regex |
| `src/transformer/` | `tts_transformer` | Qwen2 talker + code predictor (0.6B and 1.7B) |
| `src/encoder/` | `audio_tokenizer_encoder` | ECAPA-TDNN x-vector speaker encoder |
| `src/decoder/` | `audio_tokenizer_decoder` | WavTokenizer vocoder (codes to 24kHz audio) |
| `src/pipeline/` | `qwen3_tts`, `qwen3tts_c_api` | Pipeline orchestration + C API |
| `src/` | `main.cpp`, `qwen3_tts_quantize.cpp` | CLI entry point + quantization tool |

### TTS Transformer Details

The transformer generates speech codes in two stages per frame:

1. **Talker** (28 layers, 16 heads, 1024 hidden) produces a hidden state and codebook-0 logits.
2. **Code Predictor** (5 layers) autoregressively generates codebooks 1-15 from that hidden state.

The prefill embedding mirrors the Python pipeline exactly:
- Positions 0-2: text-projected role tokens (`<|im_start|>`, `assistant`, `\n`)
- Positions 3-6: TTS pad + codec embeddings (think tokens, language ID)
- Position 7: TTS pad + speaker embedding
- Position 8: TTS BOS + codec pad embedding
- Position 9+: text-projected text tokens + codec BOS/embeddings

## Testing

```bash
# End-to-end user smoketest (macOS): build → download → synthesize on
# Metal + CPU × F16 + Q8_0 × basic/clone/instruction + Python server
bash scripts/run_user_smoketest.sh            # see scripts/USER_SMOKETEST.md

# Run full test suite (component + E2E reference comparison)
bash scripts/run_all_tests.sh

# Individual component tests
./build/test_tokenizer --model models/qwen3-tts-0.6b-f16.gguf
./build/test_encoder --tokenizer models/qwen3-tts-0.6b-f16.gguf \
    --audio clone.wav --reference reference/ref_audio_embedding.bin
./build/test_transformer --model models/qwen3-tts-0.6b-f16.gguf \
    --ref-dir reference/
./build/test_decoder --tokenizer models/qwen3-tts-tokenizer-f16.gguf \
    --codes reference/speech_codes.bin --reference reference/decoded_audio.bin

# End-to-end Python vs C++ comparison
uv run python scripts/compare_e2e.py

# Generate deterministic reference data from Python
uv run python scripts/generate_deterministic_reference.py
```

### Test Results (F16 model)

- Prefill logits: cosine similarity = 0.99999994 with Python reference
- Codebook 0 match rate: 81% (frame-level exact match)
- Codebooks 1-4: ~84% match rate
- Audio output is perceptually equivalent; low waveform correlation is expected due to autoregressive divergence from F16 precision

## Profiling

Build with compile-time timing instrumentation (zero overhead when disabled):

```bash
cmake .. -DQWEN3_TTS_TIMING=ON
make -j4
```

Example output (92 frames, 7.3s audio):

```
=== Detailed Generation Timing (92 frames) ===

  Prefill:
      Compute:           175.9 ms

  Talker forward_step:
      Graph build:        21.8 ms   (0.2 ms/frame)
      Graph alloc:        34.1 ms   (0.4 ms/frame)
      Compute:          7717.4 ms   (83.9 ms/frame)

  Code predictor:
      Init/KV/embed:       7.7 ms   (0.1 ms/frame)
      Prefill (2tok):   1393.2 ms   (15.1 ms/frame)
      Steps (14):      19531.7 ms   (212.3 ms/frame)
      Compute:         20702.6 ms   (225.0 ms/frame)

  Total generate:      28915.0 ms   (3.2 frames/s)
```

The code predictor (15 sequential forward passes per frame) accounts for ~71% of generation time.

## Troubleshooting

### Linux: shared library link error (`recompile with -fPIC`)

If you see `relocation R_X86_64_32S ... recompile with -fPIC` when building the shared library, your CMakeLists.txt is missing `set(CMAKE_POSITION_INDEPENDENT_CODE ON)`. This is fixed in current builds.

### Vulkan: segfault when loading vocoder

Older versions had a bug where `normalize_codebooks()` wrote directly to GPU-mapped memory, causing a segfault on Vulkan backends. This is fixed in current builds. If you hit this on a fork, see [issue #20](https://github.com/predict-woo/qwen3-tts.cpp/issues/20) for the fix.

### Windows: missing DLL errors at runtime

Copy all required DLLs next to `qwen3-tts-cli.exe`:
- `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`
- Backend DLL: `ggml-vulkan.dll` or `ggml-cuda.dll` (if using GPU)

These are built under `ggml\build\bin\Release\` (or `ggml\build\src\` depending on generator).

### CUDA: slower than expected

Ensure you are using the CUDA backend at runtime:

```bash
QWEN3_TTS_BACKEND=cuda QWEN3_TTS_DEVICE=0 ./build/qwen3-tts-cli -m models -t "test" -o out.wav
```

Tune chunked vocoder decode with `QWEN3_TTS_DECODER_GPU_MAX_FRAMES` (default 34) and `QWEN3_TTS_DECODER_GPU_CONTEXT_FRAMES` (default 12) if you see OOM or slow decode on your GPU.

### Vocoder runs on CPU despite GPU backend shown

If logs show `AudioTokenizerDecoder backend: Vulkan0` but vocoder decode is still slow (~17s), this is the backend/buffer mismatch bug. Fixed in current builds — update to the latest version.

## Acknowledgments

- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) by Alibaba Qwen team
- [GGML](https://github.com/ggml-org/ggml) tensor library by Georgi Gerganov
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) vocoder architecture
- [predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) — original upstream project
- [gonwan/qwen3-tts.cpp](https://github.com/gonwan/qwen3-tts.cpp) — 1.7B model support, speaker embedding save/load, explicit model selection, seed parameter, Windows `_fseeki64` fix, `--no-f32-acc` flag
- [Danmoreng/qwen3-tts.cpp](https://github.com/Danmoreng/qwen3-tts.cpp) — Unicode/UTF-8 GPT-2 regex tokenizer (derived from [llama.cpp](https://github.com/ggml-org/llama.cpp)), voice steering instructions, C++ quantizer tool, decoder graph caching, `ggml_pad_reflect_1d` optimization
- [SiaoZeng](https://github.com/SiaoZeng) — Vulkan vocoder backend-safe `normalize_codebooks` fix ([issue #20](https://github.com/predict-woo/qwen3-tts.cpp/issues/20))
- [kevinzhow/clawd20130](https://github.com/kevinzhow) — CUDA runtime backend override and chunked GPU vocoder decode ([PR #11](https://github.com/predict-woo/qwen3-tts.cpp/pull/11))
- [DeryabinIvan](https://github.com/DeryabinIvan) — Windows build fixes ([PR #2](https://github.com/predict-woo/qwen3-tts.cpp/pull/2))
- [TheTastefulToastie](https://github.com/TheTastefulToastie) — cleaner CMake approach for MSVC `_USE_MATH_DEFINES` / `NOMINMAX`
