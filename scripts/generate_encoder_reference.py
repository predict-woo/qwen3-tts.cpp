#!/usr/bin/env python3
"""Generate deterministic reference data for the codec encoder.

Creates a 1-second sine wave, runs it through the Python Mimi encoder,
and saves the codec tokens + intermediate activations for C++ comparison.

Usage: python scripts/generate_encoder_reference.py <speech_tokenizer_dir>
Output: reference/enc_reference.json + reference/enc_*.bin

<speech_tokenizer_dir> is the path to the model's speech_tokenizer
directory (e.g. .../Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer).
"""
import json
import sys
import numpy as np

def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    tokenizer_dir = sys.argv[1]

    # Try loading the tokenizer
    try:
        from qwen_tts import Qwen3TTSTokenizer
    except ImportError:
        print("ERROR: qwen_tts package not available")
        print("Install with: pip install qwen-tts")
        sys.exit(1)

    print(f"Loading tokenizer from {tokenizer_dir}...")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_dir,
        device_map="cpu",
    )

    # Generate 1 second of 440Hz sine wave (same as C++ test)
    n_samples = 24000
    t = np.arange(n_samples) / 24000.0
    audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    print(f"Encoding {n_samples} samples...")
    # Encode using the tokenizer's encode method
    result = tokenizer.encode(audio, sr=24000)

    # The result should be codec tokens
    if isinstance(result, dict) and "codes" in result:
        codes = result["codes"]
    elif hasattr(result, "codes"):
        codes = result.codes
    else:
        codes = result  # assume it's the token array directly

    codes = np.asarray(codes)
    print(f"Codec tokens shape: {codes.shape}")
    print(f"First 20 tokens (cb0): {codes[0, :20] if codes.ndim > 1 else codes[:20]}")

    # Save reference
    np.save("reference/enc_codes.npy", codes)
    with open("reference/enc_reference.json", "w") as f:
        json.dump({
            "n_samples": n_samples,
            "sample_rate": 24000,
            "frequency": 440.0,
            "codes_shape": list(codes.shape),
            "first_10_cb0": (codes[0, :10] if codes.ndim > 1 else codes[:10]).tolist(),
        }, f, indent=2)
    print("Saved reference/enc_codes.npy and reference/enc_reference.json")

if __name__ == "__main__":
    main()
