# AGENTS.md

Coding conventions and architecture guide for AI agents working on this codebase.

## Project Overview

`qwen3-tts.cpp` is a pure C++17 implementation of the Qwen3-TTS text-to-speech pipeline using GGML. It converts text to speech through four stages: tokenization, speaker encoding, transformer code generation, and vocoder decoding.

## Repository Structure

```
qwen3-tts.cpp/
  src/                          # C++ source files
    main.cpp                    # CLI entry point
    qwen3_tts.{h,cpp}          # Full pipeline orchestration
    tts_transformer.{h,cpp}    # TTS transformer (talker + code predictor)
    text_tokenizer.{h,cpp}     # BPE text tokenizer
    audio_tokenizer_encoder.{h,cpp}  # ECAPA-TDNN speaker encoder
    audio_tokenizer_decoder.{h,cpp}  # WavTokenizer vocoder
    gguf_loader.{h,cpp}        # GGUF model loading
  tests/                        # Component tests
    test_codebook.cpp
    test_vq_only.cpp
    test_tokenizer.cpp
    test_encoder.cpp
    test_transformer.cpp        # Deterministic reference comparison
    test_decoder.cpp
  scripts/                      # Python utilities
    convert_tts_to_gguf.py      # HuggingFace -> GGUF converter (TTS model)
    convert_tokenizer_to_gguf.py # HuggingFace -> GGUF converter (vocoder)
    generate_deterministic_reference.py  # Generate Python reference data
    compare_e2e.py              # End-to-end Python vs C++ comparison
    run_all_tests.sh            # Test runner
  reference/                    # Reference data (*.bin gitignored, *.json tracked)
  models/                       # GGUF models (gitignored)
  CMakeLists.txt
```

## Build System

- **CMake 3.14+** with C++17
- GGML is vendored under `./ggml` and linked from `./ggml/build/src`
- Build GGML first: `cmake -S ggml -B ggml/build -DGGML_METAL=ON && cmake --build ggml/build -j4`
- Build project: `cmake -S . -B build && cmake --build build -j4`
- Timing build: `cmake -S . -B build -DQWEN3_TTS_TIMING=ON && cmake --build build -j4`
- GGML headers are in `./ggml/include`

## Coding Conventions

### C++ Style

- C++17 standard, no exceptions, no RTTI
- Use `fprintf(stderr, ...)` for logging, not `std::cerr`
- Error handling: methods return `bool`, error details stored in `error_msg_` member
- Memory: GGML contexts own tensor memory; use `ggml_free()` for cleanup
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for macros
- Header guards: `#pragma once`
- All public types in `qwen3_tts` namespace

### GGML Patterns

Every forward pass follows this pattern:

```cpp
// 1. Build computation graph
struct ggml_cgraph * gf = build_xxx_graph(...);

// 2. Allocate graph memory
ggml_backend_sched_alloc_graph(state_.sched, gf);

// 3. Set input tensors
struct ggml_tensor * inp = ggml_graph_get_tensor(gf, "input_name");
ggml_backend_tensor_set(inp, data, 0, size);

// 4. Compute
ggml_backend_sched_graph_compute(state_.sched, gf);

// 5. Get output tensors
struct ggml_tensor * out = ggml_graph_get_tensor(gf, "output_name");
ggml_backend_tensor_get(out, output_data, 0, size);

// 6. Reset scheduler
ggml_backend_sched_reset(state_.sched);
```

Important: the `ggml_cast` to F32 before `ggml_mul_mat` is needed only for F16 weight tensors
(specifically `ffn_down` in both talker and code predictor layers). Gate it on the stored type:
`ggml_is_quantized(layer.ffn_down->type) ? layer.ffn_down : ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32)`.
Passing a quantized weight through the cast is not a no-op: the vendored ggml-cpu type traits derive
`vec_dot_type` from src0's type, so an F32 src0 selects the f32 dot kernel and the whole weight tensor
is dequantized on every call (measured on RK3588, Q8_0: 192.4 -> 62.5 ms per code-predictor request,
3.09x, from gating alone; on f16 the gated form is bit-identical to the unguarded cast).

Backend initialization and scheduling notes:

- Use `init_preferred_backend()` (`src/gguf_loader.cpp`) to select backend in order: `IGPU -> GPU -> ACCEL -> CPU`
- If the selected runtime backend is not CPU, add a CPU backend as scheduler fallback (`backend_cpu`) when calling `ggml_backend_sched_new(...)`
- Decoder follows the same backend preference; load decoder weights with `GGML_BACKEND_DEVICE_TYPE_IGPU` preference for Metal-first execution

### Model Architecture

The TTS transformer has two sub-models:

1. **Talker** — 28-layer Qwen2 transformer (1024 hidden, 16 heads, 8 KV heads, 128 head_dim)
   - Input: prefill embedding or step embedding (float32, [1, 1024])
   - Output: hidden states + codec logits via `codec_head`

2. **Code Predictor** — 5-layer transformer (same attention config)
   - Input: talker hidden state + codebook-0 embedding (2-token prefill)
   - Output: 15 codebook predictions (autoregressive, one per step)
   - Has its own separate KV cache (max 16 tokens)

### Prefill Embedding Structure (10 positions for single-word input)

```
Pos 0:   text_projection(<|im_start|>)
Pos 1:   text_projection(assistant)
Pos 2:   text_projection(\n)
Pos 3:   tts_pad + codec_embd(think_id)
Pos 4:   tts_pad + codec_embd(think_bos_id)
Pos 5:   tts_pad + codec_embd(language_id)
Pos 6:   tts_pad + codec_embd(think_eos_id)
Pos 7:   tts_pad + speaker_embedding
Pos 8:   tts_bos + codec_embd(pad_id)
Pos 9+:  text_projection(text_token[i]) + codec_embd(bos_id or pad_id)
```

This structure must mirror the Python pipeline exactly.

### Special Token IDs

```
tts_bos = 151672, tts_eos = 151673, tts_pad = 151671
codec_bos = 2149, codec_eos = 2150, codec_pad = 2148
codec_think = 2154, codec_think_bos = 2156, codec_think_eos = 2157
english_language_id = 2050
```

### Key Files to Understand

- `tts_transformer.cpp` — The core file (~2300 lines). Contains `generate()`, `forward_prefill()`, `forward_step()`, `predict_codes_autoregressive()`, and all graph builders.
- `qwen3_tts.cpp` — Pipeline orchestration. Calls tokenizer, encoder, transformer, decoder in sequence.
- `CMakeLists.txt` — Build configuration. Each component is a separate static library.

## Testing

### Reference Data

Deterministic reference data is generated by `scripts/generate_deterministic_reference.py` using float32 Python inference with greedy decoding. Files go in `reference/det_*.bin` (gitignored) and `reference/det_*.json` (tracked).

### Test Strategy

- `test_transformer` loads reference data and compares C++ output at each stage
- Pass criteria: prefill logits cosine > 0.99; speech codes partially match (F16 precision causes divergence)
- E2E comparison (`compare_e2e.py`): checks both pipelines produce valid non-silent audio with similar duration; waveform correlation is informational only

### Running Tests

```bash
bash scripts/run_all_tests.sh           # Full suite
./build/test_transformer --ref-dir reference/  # Transformer only
```

## Git Conventions

- Conventional commits: `feat(scope):`, `fix(scope):`, `docs:`
- Scopes: `transformer`, `vocoder`, `test`, `timing`
- One logical change per commit
- Do not commit model files (*.gguf) or reference binaries (*.bin)

## Known Limitations

- F16 model weights cause autoregressive divergence vs Python's float32 — speech codes differ but audio is perceptually equivalent
- M-RoPE uses 1D positions (equivalent for single-batch, may differ for batched inference)
- `--top-p` is parsed in CLI params but currently not used in transformer sampling
- Top-level CMake expects vendored GGML at `./ggml`

## Performance Profile

The code predictor is the primary bottleneck (~71% of generation time) because it runs 15 sequential forward passes per frame (1 prefill + 14 autoregressive steps). The talker accounts for ~27%. Graph build/alloc and data I/O are negligible (<1%).
