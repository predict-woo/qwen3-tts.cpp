# Fine-Tuning Custom Voices Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an end-user guide for creating custom fine-tuned voices with upstream Qwen3-TTS training, GGUF conversion, and `qwen3-tts.cpp` validation.

**Architecture:** This is a docs-only change. Add a new operational runbook under `docs/`, then add a short README entry point near the existing CustomVoice section so users can discover the guide and load fine-tuned checkpoints correctly.

**Tech Stack:** Markdown docs, GitHub README, upstream Qwen3-TTS fine-tuning workflow, local `qwen3-tts.cpp` CLI behavior.

---

### Task 1: Add the end-user training guide

**Files:**
- Create: `docs/fine-tuning-custom-voices.md`
- Reference: `docs/plans/2026-04-22-fine-tuning-custom-voices-design.md`
- Reference: `scripts/convert_tts_to_gguf.py`
- Reference: `src/main.cpp`

- [ ] **Step 1: Draft the guide structure from the approved design**

Write sections for support matrix, environment setup, dataset format, upstream preparation, fine-tuning, checkpoint selection, GGUF conversion, local validation, and troubleshooting.

- [ ] **Step 2: Fill in upstream workflow details from official sources**

Use the current upstream `QwenLM/Qwen3-TTS` fine-tuning README plus `prepare_data.py` and `sft_12hz.py` so the guide matches the official data format and commands.

- [ ] **Step 3: Add repo-specific GGUF conversion guidance**

Document how to convert the chosen fine-tuned checkpoint with `scripts/convert_tts_to_gguf.py`, where to place the output file, and why users should avoid relying on default model auto-detection.

- [ ] **Step 4: Add explicit validation steps**

Require:
- `--list-speakers`
- confirmation that the trained `speaker_name` appears
- a synthesis sanity check with `--speaker <name>`

- [ ] **Step 5: Add limitations and troubleshooting**

Call out:
- Linux/CUDA as the supported training path
- macOS as best-effort for conversion and local inference validation
- `0.6B` out of scope
- checkpoint quality drift across epochs
- server/C API model-selection limitations

### Task 2: Add README discoverability

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a short subsection near CustomVoice / VoiceDesign**

Link to `docs/fine-tuning-custom-voices.md` and explain that fine-tuned checkpoints should be loaded with `--tts-model <file>`.

- [ ] **Step 2: Keep claims aligned with actual repo behavior**

Do not imply that the server or C API automatically support arbitrary fine-tuned checkpoint names.

### Task 3: Verify the docs

**Files:**
- Verify: `docs/fine-tuning-custom-voices.md`
- Verify: `README.md`

- [ ] **Step 1: Re-read the new guide against upstream commands**

Check that JSONL fields, `prepare_data.py`, and `sft_12hz.py` arguments match current upstream behavior.

- [ ] **Step 2: Re-read the local validation steps against `qwen3-tts.cpp`**

Check that the CLI validation flow reflects the current repo:
- `--tts-model <file>`
- `--list-speakers`
- `--speaker <name>`

- [ ] **Step 3: Check the final diff**

Run:

```bash
git diff -- docs/fine-tuning-custom-voices.md README.md docs/plans/2026-04-22-fine-tuning-custom-voices.md
```

Expected: only the new guide, the README link, and this plan file are changed.
