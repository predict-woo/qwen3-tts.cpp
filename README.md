# qwen3-tts.cpp

![PyTorch vs qwen3-tts.cpp benchmark](./docs/benchmark_pytorch_vs_cpp.png)

**Benchmark Snapshot (PyTorch vs qwen3-tts.cpp):** Basic 3.19x faster, Clone 4.07x faster. Peak RSS delta: Basic +19.0%, Clone +7.7%.

C++ inference for [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) using the [GGML](https://github.com/ggml-org/ggml) tensor library.

Runs the full TTS pipeline in pure C++17, including text tokenization, speaker encoding, transformer code generation, and vocoder decoding, without Python or PyTorch at inference time.

## Features

- Full text-to-speech pipeline in C++17 with GGML backend
- Voice cloning from reference audio (ECAPA-TDNN x-vector extraction)
- Greedy and sampled decoding (temperature, top-k, repetition penalty)
- GGUF model format (F16 and Q8_0 quantization)
- Runtime backend selection with GPU/Metal preference and CPU fallback
- Deterministic reference tests comparing C++ output against Python
- Compile-time timing instrumentation with zero overhead in normal builds

## Prerequisites

- C++17 compiler (GCC 9+ or Clang 10+)
- CMake 3.14+
- [GGML](https://github.com/ggml-org/ggml) built from source
- Python 3.10+ with [uv](https://github.com/astral-sh/uv) (model conversion only)

## Quickstart (macOS, copy/paste)

Run these commands from a fresh clone:

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

```bash
git clone https://github.com/predict-woo/qwen3-tts.cpp.git
cd qwen3-tts.cpp
git submodule update --init --recursive

# Build GGML (vendored in ./ggml)
cmake -S ggml -B ggml/build -DGGML_METAL=ON
cmake --build ggml/build -j4

# Build qwen3-tts.cpp
cmake -S . -B build
cmake --build build -j4
```

> **Note:** The top-level CMake currently expects GGML in `./ggml` with libraries under `./ggml/build/src`.
> For NVIDIA CUDA, build GGML with `-DGGML_CUDA=ON` and run with `QWEN3_TTS_BACKEND=cuda`.

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
| `-t, --text <text>` | Text to synthesize | (required) |
| `-o, --output <file>` | Output WAV file path | `output.wav` |
| `-r, --reference <file>` | Reference audio for voice cloning | (none) |
| `--temperature <val>` | Sampling temperature (0 = greedy) | 0.9 |
| `--top-k <n>` | Top-k sampling (0 = disabled) | 50 |
| `--top-p <val>` | Top-p sampling | 1.0 |
| `--max-tokens <n>` | Maximum audio frames to generate | 4096 |
| `--repetition-penalty <val>` | Repetition penalty on codebook-0 token generation | 1.05 |
| `-j, --threads <n>` | Number of compute threads | 4 |

`--top-p` is currently parsed by the CLI but not yet wired into transformer sampling.

On macOS, CoreML code predictor is enabled by default when `models/coreml/code_predictor.mlpackage` exists.
Set `QWEN3_TTS_USE_COREML=0` to disable it. Low-memory mode is opt-in via `QWEN3_TTS_LOW_MEM=1`.

### Backend Selection

At runtime, each component logs its selected backend (for example, `TTSTransformer backend: MTL0` or `BLAS`).

- Preferred order: `GPU` -> `IGPU` -> `ACCEL` -> `CPU`
- Encoder and transformer can run on Metal/other accelerators with CPU fallback in the scheduler
- Decoder now follows the same backend preference and will use Metal when available
- `QWEN3_TTS_BACKEND` overrides runtime selection: `auto` (default), `cuda`, or `cpu`
- `QWEN3_TTS_DEVICE` selects CUDA device index when `QWEN3_TTS_BACKEND=cuda` (default device is index 0)
- `QWEN3_TTS_DECODER_GPU_MAX_FRAMES` controls max frames per CUDA vocoder chunk (default: `34`)
- `QWEN3_TTS_DECODER_GPU_CONTEXT_FRAMES` controls left-context frames per CUDA vocoder chunk (default: `12`)

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

| File | Component | Description |
|------|-----------|-------------|
| `text_tokenizer.{h,cpp}` | Tokenizer | BPE text tokenizer from GGUF |
| `audio_tokenizer_encoder.{h,cpp}` | Speaker Encoder | ECAPA-TDNN x-vector extractor |
| `tts_transformer.{h,cpp}` | TTS Transformer | 28-layer Qwen2 talker + 5-layer code predictor |
| `audio_tokenizer_decoder.{h,cpp}` | Vocoder | WavTokenizer decoder (codes to waveform) |
| `qwen3_tts.{h,cpp}` | Pipeline | Full pipeline orchestration |
| `main.cpp` | CLI | Command-line interface |
| `gguf_loader.{h,cpp}` | GGUF | Model loading utilities |

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
# Run full test suite
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

## Acknowledgments

- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) by Alibaba Qwen team
- [GGML](https://github.com/ggml-org/ggml) tensor library by Georgi Gerganov
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) vocoder architecture
