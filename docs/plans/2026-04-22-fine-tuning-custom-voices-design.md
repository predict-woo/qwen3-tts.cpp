# Fine-Tuning Custom Voices Guide Design

## Goal

Add end-user documentation that explains the best supported path for creating custom fine-tuned voices for `qwen3-tts.cpp`.

The guide should standardize on the upstream `QwenLM/Qwen3-TTS` fine-tuning workflow for `Qwen3-TTS-12Hz-1.7B-Base`, then show how to convert the resulting checkpoint to GGUF and validate it in `qwen3-tts.cpp`.

## Audience

This documentation is for end users who need a practical recipe from dataset preparation through fine-tuning, GGUF conversion, and inference.

It is not aimed at repository maintainers only, and it is not meant to describe how to build a native training stack inside `qwen3-tts.cpp`.

## Supported Scope

### Supported

- Single-speaker fine-tuning
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` as the training base model
- Linux with NVIDIA CUDA as the supported training environment
- macOS / Apple Silicon for GGUF conversion and local inference validation
- Explicit `--tts-model <file>` loading when running fine-tuned checkpoints in `qwen3-tts.cpp`

### Out of Scope

- `0.6B` fine-tuning
- Multi-speaker fine-tuning
- Native training implementation in `qwen3-tts.cpp`
- Claiming full server or C API support for arbitrary fine-tuned checkpoint names

## Why This Path

`qwen3-tts.cpp` is an inference and model-conversion project, not a training project. The upstream `QwenLM/Qwen3-TTS` repository now ships an official fine-tuning workflow under `finetuning/`, including `prepare_data.py`, `dataset.py`, and `sft_12hz.py`.

The runtime in this repository already supports consuming converted CustomVoice-style GGUF checkpoints when users load them explicitly, but the default server and C API paths still assume stock model filenames. The guide should therefore standardize on upstream training plus local conversion and CLI validation, rather than implying that this repository owns the training workflow end to end.

## Documentation Deliverables

### 1. New guide

Create a new document at:

- `docs/fine-tuning-custom-voices.md`

This guide will be a runbook rather than a high-level overview. It should include:

- Goal and support matrix
- Prerequisites and environment expectations
- Dataset requirements for single-speaker fine-tuning
- Linux/CUDA setup for training
- macOS limitations and what users can still do on Apple Silicon
- checkpoint evaluation and selection before conversion
- Upstream fine-tuning steps
- GGUF conversion steps using `scripts/convert_tts_to_gguf.py`
- Validation steps in `qwen3-tts.cpp`, including explicit speaker validation
- Known limitations and troubleshooting notes

### 2. README integration

Add a short subsection to `README.md` near the existing CustomVoice / VoiceDesign material that:

- points readers to `docs/fine-tuning-custom-voices.md`
- tells users to load fine-tuned checkpoints with `--tts-model <file>`
- does not overstate current server-side support

## Guide Structure

The guide should use the following structure:

1. Overview
2. What is supported
3. Training environment
4. Dataset format and recommendations
5. Upstream data preparation
6. Upstream fine-tuning
7. Checkpoint evaluation and selection
8. Exporting a fine-tuned checkpoint to GGUF
9. Running the fine-tuned model in `qwen3-tts.cpp`
10. Validation checklist
11. Known limitations
12. Troubleshooting

## Key Technical Constraints To Document

### Training support

The guide should recommend `1.7B-Base` only. As of February 10, 2026, the upstream public issue tracker still showed an open `0.6B` fine-tuning dimension mismatch issue, so `0.6B` should remain out of scope for the supported path.

### Model loading in this repo

Fine-tuned checkpoints should be loaded explicitly with `--tts-model`, because auto-detection currently defaults to stock `qwen3-tts-0.6b-{q8_0,f16}.gguf` naming.

### Checkpoint selection

The guide should not imply that the latest epoch is automatically the best output. Before GGUF conversion, users should evaluate multiple upstream checkpoints and pick the one with the best audio quality and speaking-rate stability.

### Speaker encoder expectations

The guide should warn that a fine-tuned checkpoint may work for preset-speaker synthesis while not supporting the legacy speaker-encoder cloning path in the same way as the stock base checkpoint. Users should validate the exact inference path they intend to use.

### macOS expectations

The guide should describe macOS and Apple Silicon as best-effort for preparation tasks unless explicitly validated. The only behaviors this repo can describe confidently in this pass are GGUF conversion and local inference validation on macOS.

### Server and C API limitations

The guide should not claim that the Python server or C API can automatically discover arbitrary fine-tuned model names. For now, the cleanly supported validation path is the CLI with explicit model selection.

## Validation Requirements

Before the documentation work is considered complete:

- The new guide should be checked for consistency with the upstream fine-tuning README and scripts
- The README link should land near the existing voice-related sections
- The commands shown for `qwen3-tts.cpp` should reflect current local behavior, especially explicit `--tts-model` loading
- The guide should require `--list-speakers` after conversion and confirm that the trained `speaker_name` appears in the preset list
- The guide should require a synthesis sanity check using `--speaker <name>` after `--list-speakers` succeeds
- The guide should distinguish supported behavior from best-effort or known-limited behavior
- The guide should tell users to choose a specific upstream checkpoint before conversion rather than assuming the final epoch is correct

## Non-Goals

- No code changes to training logic
- No server feature additions
- No model selection API redesign
- No wrapper scripts around the upstream training repo in this pass

## Risks

- Upstream fine-tuning scripts may continue to evolve, so commands and assumptions may drift over time
- Upstream checkpoint quality may drift across epochs, so conversion of the final checkpoint is not always the right default
- Some users will expect macOS local training; the guide must be explicit that Linux/CUDA is the supported training path
- Users may assume “fine-tuned custom voices” means the same runtime behavior as the stock CustomVoice releases; the guide should call out where that assumption is unsafe

## Recommendation

Proceed with a documentation-only implementation that standardizes:

- upstream `Qwen3-TTS-12Hz-1.7B-Base` fine-tuning
- upstream checkpoint selection before export
- local GGUF conversion in `qwen3-tts.cpp`
- CLI-based validation with explicit model selection, `--list-speakers`, and `--speaker <name>`

This gives users the best currently supportable path without overstating what `qwen3-tts.cpp` owns today.
