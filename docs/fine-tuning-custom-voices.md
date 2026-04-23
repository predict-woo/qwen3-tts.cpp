# Fine-Tuning Custom Voices for `qwen3-tts.cpp`

This guide documents the best currently supportable path for creating a reusable custom voice for `qwen3-tts.cpp`.

The supported workflow is:

1. Fine-tune `Qwen/Qwen3-TTS-12Hz-1.7B-Base` with the official upstream training scripts from [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning)
2. Pick the best checkpoint by listening to multiple epochs
3. Convert the chosen checkpoint to GGUF with [`scripts/convert_tts_to_gguf.py`](../scripts/convert_tts_to_gguf.py)
4. Validate the resulting GGUF in `qwen3-tts.cpp` with `--list-speakers` and `--speaker <name>`

This document was verified against upstream `QwenLM/Qwen3-TTS` fine-tuning files on April 22, 2026:

- [upstream finetuning README](https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/finetuning/README.md)
- [upstream `prepare_data.py`](https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/finetuning/prepare_data.py)
- [upstream `sft_12hz.py`](https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/finetuning/sft_12hz.py)
- [upstream `dataset.py`](https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/finetuning/dataset.py)

## Support Matrix

### Supported

- Single-speaker fine-tuning
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- Linux + NVIDIA CUDA for training
- GGUF conversion in `qwen3-tts.cpp`
- Local inference validation in `qwen3-tts.cpp`

### Best-effort only

- macOS / Apple Silicon for helper tasks such as file preparation, GGUF conversion, and local inference checks

### Not supported in this guide

- `0.6B` fine-tuning
- Multi-speaker fine-tuning
- Native training inside `qwen3-tts.cpp`
- Assuming the Python server or C API will automatically pick up arbitrary fine-tuned model filenames

## Important Constraints

### 1. Train on Linux/CUDA

The upstream fine-tuning scripts are CUDA-first. The official examples use `--device cuda:0`, and the model-loading path in `sft_12hz.py` is written around GPU inference/training.

Treat Linux + NVIDIA CUDA as the supported path for actual fine-tuning.

### 2. Use `1.7B-Base`, not `0.6B`

The upstream repo advertises fine-tuning for both `1.7B` and `0.6B`, but as of April 22, 2026 there is still an open upstream issue for `0.6B` dimension mismatch during fine-tuning:

- [Issue #198](https://github.com/QwenLM/Qwen3-TTS/issues/198)

For a stable workflow, use `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.

### 3. The fine-tuned model behaves like a single preset CustomVoice model

The current upstream `sft_12hz.py` script:

- rewrites `config.json` so `tts_model_type` becomes `custom_voice`
- writes your `speaker_name` into `talker_config.spk_id`
- assigns that speaker to token id `3000`
- copies the learned target speaker embedding into codec embedding row `3000`
- drops `speaker_encoder.*` weights when saving the fine-tuned checkpoint

That means the safest way to use the resulting model in `qwen3-tts.cpp` is as a preset speaker model:

- validate it with `--list-speakers`
- synthesize with `--speaker <name>`

Do not assume the fine-tuned checkpoint will behave like the stock Base checkpoint for all reference-audio cloning paths.

### 4. Do not rely on default model auto-detection

`qwen3-tts.cpp` auto-detection still defaults to stock `qwen3-tts-0.6b-{q8_0,f16}.gguf` filenames. For a fine-tuned model, always load it explicitly with:

```bash
--tts-model your-model-name.gguf
```

## Environment Setup

### Linux / NVIDIA CUDA training environment

Start from a clean Python virtual environment on a CUDA-capable Linux machine.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install qwen-tts accelerate tensorboard soundfile librosa safetensors
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS/finetuning
```

Notes:

- `qwen-tts` is the required upstream package.
- `accelerate` is imported by `sft_12hz.py`.
- `tensorboard` is a practical dependency because the upstream script initializes `Accelerator(..., log_with="tensorboard")`.
- If your environment already pins these packages differently, isolate the fine-tuning workflow in its own virtual environment.

### macOS / Apple Silicon

Do not treat Apple Silicon as the primary training environment for this workflow.

What macOS is useful for in this guide:

- preparing JSONL manifests
- organizing checkpoints
- converting a selected checkpoint to GGUF
- running local `qwen3-tts.cpp` CLI validation

What this guide does **not** claim:

- that upstream `prepare_data.py` has been validated on `cpu` or `mps`
- that local fine-tuning on macOS is a supported path

## Dataset Format

The upstream fine-tuning README expects a JSONL file with one JSON object per line.

Each line must contain:

- `audio`: path to the target training WAV
- `text`: transcript for that WAV
- `ref_audio`: path to the reference speaker WAV

Example:

```jsonl
{"audio":"./data/utt0001.wav","text":"Actually, I have realized that I pay close attention to other people's emotions.","ref_audio":"./data/ref.wav"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/ref.wav"}
```

Upstream specifically recommends using the **same** `ref_audio` for every training sample because that usually improves speaker consistency and stability.

Practical guidance:

- Keep every clip to a single speaker
- Make sure transcripts match the audio exactly
- Use a clean reference clip for `ref_audio`
- Keep the training data language and the transcripts aligned
- Store paths in a way that will still resolve from the `finetuning/` working directory

## Prepare Data

The upstream preprocessing step adds `audio_codes` to each JSONL record.

Run:

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

Input:

- `train_raw.jsonl` with `audio`, `text`, and `ref_audio`

Output:

- `train_with_codes.jsonl` with an added `audio_codes` field

The official `prepare_data.py` processes `audio` through the `Qwen3TTSTokenizer` and writes `audio_codes` back into the JSONL output.

## Fine-Tune the Base Model

The official training entry point is `sft_12hz.py`.

Minimal command:

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name my_voice
```

Important notes:

- `speaker_name` is the name that should later appear in `qwen3-tts.cpp --list-speakers`
- the upstream README uses `batch_size 32`, but that is only realistic on large GPUs
- if you hit memory pressure, reduce `--batch_size` first

Checkpoints are written as:

- `output/checkpoint-epoch-0`
- `output/checkpoint-epoch-1`
- `output/checkpoint-epoch-2`
- ...

## Evaluate and Select a Checkpoint Before Export

Do **not** assume the final epoch is automatically the best checkpoint.

There is an open upstream report showing speaking-rate drift across epochs during Base fine-tuning:

- [Issue #179](https://github.com/QwenLM/Qwen3-TTS/issues/179)

Before you convert anything to GGUF:

1. Pick a fixed evaluation sentence or short paragraph
2. Run the same inference against multiple checkpoint directories
3. Listen for:
   - speaker similarity
   - noise or collapse
   - speaking-rate drift
   - pronunciation stability
4. Select the best checkpoint by listening, not by epoch number alone

The upstream README’s quick inference pattern is:

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="She said she would be here by noon.",
    speaker="my_voice",
)
sf.write("output.wav", wavs[0], sr)
```

If you want to compare many checkpoints quickly, adapt that snippet into a loop and render one WAV per checkpoint.

## Convert the Chosen Checkpoint to GGUF

After you choose the best checkpoint, switch back to your `qwen3-tts.cpp` checkout and convert it.

Example:

```bash
cd /path/to/qwen3-tts.cpp
python scripts/convert_tts_to_gguf.py \
  --input /path/to/Qwen3-TTS/finetuning/output/checkpoint-epoch-4 \
  --output models/qwen3-tts-1.7b-my-voice-f16.gguf \
  --type f16
```

Recommended naming:

- `models/qwen3-tts-1.7b-my-voice-f16.gguf`

The vocoder/tokenizer GGUF is **not** re-trained by this workflow, so keep using the standard tokenizer/vocoder model:

- `models/qwen3-tts-tokenizer-f16.gguf`

If you have not prepared the standard runtime models yet, use the repo setup flow first:

```bash
python scripts/setup_pipeline_models.py
```

That gives you:

- `models/qwen3-tts-0.6b-f16.gguf`
- `models/qwen3-tts-tokenizer-f16.gguf`
- optional CoreML artifacts on macOS

For a fine-tuned workflow, you replace only the TTS GGUF you point at with `--tts-model`.

## Validate the Fine-Tuned Model in `qwen3-tts.cpp`

This is the minimum validation sequence that proves you produced a reusable custom voice.

### 1. Confirm the speaker metadata is present

```bash
./build/qwen3-tts-cli \
  -m models \
  --tts-model qwen3-tts-1.7b-my-voice-f16.gguf \
  --list-speakers
```

Expected result:

- the model loads successfully
- your `speaker_name` appears in the preset speaker list

If your name does **not** appear, stop there. The conversion did not preserve the speaker metadata you need for preset-speaker inference.

### 2. Run a real synthesis test with the trained speaker

```bash
./build/qwen3-tts-cli \
  -m models \
  --tts-model qwen3-tts-1.7b-my-voice-f16.gguf \
  --speaker my_voice \
  -t "This is a validation sentence for my fine-tuned voice." \
  -o out-my-voice.wav
```

Expected result:

- synthesis completes without error
- the output sounds like the intended speaker
- the speaking rate and pronunciation are acceptable

### 3. Keep using explicit model selection

For fine-tuned checkpoints, keep `--tts-model qwen3-tts-1.7b-my-voice-f16.gguf` in your commands. Do not rely on default model discovery.

## Recommended Runtime Layout

Example `models/` directory:

```text
models/
  qwen3-tts-tokenizer-f16.gguf
  qwen3-tts-0.6b-f16.gguf
  qwen3-tts-1.7b-my-voice-f16.gguf
```

The tokenizer/vocoder GGUF remains shared. Only the TTS model file changes.

## Known Limitations

- This guide only covers single-speaker fine-tuning.
- It does not cover `0.6B` fine-tuning.
- It does not claim that the Python server can select arbitrary fine-tuned TTS model files by name.
- It does not claim that the C API can load arbitrary fine-tuned TTS model names without code changes.
- It does not guarantee that reference-audio cloning through the fine-tuned checkpoint will match the stock Base model behavior.
- Upstream fine-tuning quality can vary across epochs, so checkpoint selection matters.

## Troubleshooting

### `speaker_name` does not appear in `--list-speakers`

Possible causes:

- you converted the wrong checkpoint directory
- the checkpoint was saved before `config.json` was rewritten with `tts_model_type=custom_voice` and `spk_id`
- conversion was run against a directory that did not contain the modified `config.json`

Re-check the chosen checkpoint directory before conversion.

### The model loads, but `--speaker my_voice` fails

Possible causes:

- the `speaker_name` in training does not match the name you are passing
- the GGUF speaker metadata does not match the checkpoint config

Run `--list-speakers` first and use the exact name shown there.

### The voice sounds worse in later epochs

This is a known risk in the upstream workflow. Compare multiple checkpoints and convert the one that sounds best instead of always taking the latest epoch.

Relevant upstream report:

- [Issue #179](https://github.com/QwenLM/Qwen3-TTS/issues/179)

### You tried to fine-tune `0.6B`

Switch to `1.7B-Base`. There is still an open upstream `0.6B` fine-tuning issue:

- [Issue #198](https://github.com/QwenLM/Qwen3-TTS/issues/198)

### You want to use the fine-tuned model through the server

Validate the model with the CLI first. The current server path in `qwen3-tts.cpp` still assumes stock model naming and does not expose clean per-request TTS model selection.

## References

- [QwenLM/Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS)
- [official finetuning README](https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/finetuning/README.md)
- [official `prepare_data.py`](https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/finetuning/prepare_data.py)
- [official `sft_12hz.py`](https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/finetuning/sft_12hz.py)
- [Issue #179: speaking-rate drift across epochs](https://github.com/QwenLM/Qwen3-TTS/issues/179)
- [Issue #198: `0.6B` fine-tuning mismatch](https://github.com/QwenLM/Qwen3-TTS/issues/198)
