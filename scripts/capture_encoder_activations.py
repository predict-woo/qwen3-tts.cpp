#!/usr/bin/env python3
"""Capture per-stage encoder activations for C++ parity debugging.

Hooks every MimiEncoder layer plus the outer transformer/downsample/quantizer
stages and saves each activation to reference/enc_acts/ as .npy.

Usage: python scripts/capture_encoder_activations.py [audio.f32]
"""
import sys
from pathlib import Path

import numpy as np
import torch

MODEL = "/home/willwade/spikes/qwen3-tts.cpp/models/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer"
AUDIO = sys.argv[1] if len(sys.argv) > 1 else "/tmp/will_ref.f32"
OUT = Path("reference/enc_acts")


def main():
    from qwen_tts import Qwen3TTSTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    tok = Qwen3TTSTokenizer.from_pretrained(MODEL, device_map="cpu")
    enc = tok.model.encoder

    acts = {}

    def mk_hook(name):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            if hasattr(t, "last_hidden_state"):
                t = t.last_hidden_state
            acts[name] = t.detach().float().cpu().numpy()
        return hook

    handles = []
    mimi = enc.encoder
    for i, layer in enumerate(mimi.layers):
        handles.append(layer.register_forward_hook(mk_hook(f"layer_{i:02d}")))
    for i, layer in enumerate(enc.encoder_transformer.layers):
        handles.append(layer.register_forward_hook(mk_hook(f"tfm_{i:02d}")))
    handles.append(enc.encoder_transformer.register_forward_hook(mk_hook("transformer_out")))
    handles.append(enc.downsample.register_forward_hook(mk_hook("downsample_out")))
    handles.append(enc.quantizer.register_forward_hook(mk_hook("quantizer_out")))

    def mk_pre_hook(name):
        def pre_hook(_m, args):
            t = args[0]
            acts[name] = t.detach().float().cpu().numpy()
        return pre_hook

    handles.append(enc.downsample.register_forward_pre_hook(mk_pre_hook("downsample_in")))

    audio = np.fromfile(AUDIO, dtype=np.float32)
    with torch.no_grad():
        result = tok.encode(audio, sr=24000)
    codes = np.asarray(result.audio_codes)[0]
    np.save(OUT / "codes.npy", codes)

    for h in handles:
        h.remove()

    for name, arr in acts.items():
        np.save(OUT / f"{name}.npy", arr)
        print(f"{name}: shape={arr.shape}")


if __name__ == "__main__":
    main()
