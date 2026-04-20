qwen3-tts.cpp - precompiled build
==================================

This archive contains compiled binaries for qwen3-tts.cpp. No toolchain
required - just download a model and run the CLI.


What's in here
--------------

  qwen3-tts-cli         The text-to-speech CLI (main binary)
  qwen3-tts-quantize    Standalone GGUF quantizer (optional)
  examples/             Reference WAV files for voice cloning demos
  scripts/              download_model.sh (Linux/macOS) and .bat (Windows)
  LICENSE               Project license

Any *.dll / *.so / *.dylib files in this directory are runtime dependencies
(GGML backend libraries, CUDA runtime, etc.). Leave them next to the binary.


Quick start
-----------

1. Download the model (first run only, ~1.2 GB):

     Linux/macOS:   ./scripts/download_model.sh
     Windows:       scripts\download_model.bat

   This downloads two GGUF files into ./models:
     - qwen3-tts-0.6b-q8-0.gguf           (TTS model, ~986 MB)
     - qwen3-tts-tokenizer-0.6b-q8-0.gguf (tokenizer + vocoder, ~250 MB)

2. Synthesize speech:

     Linux/macOS:
       ./qwen3-tts-cli \
         -m ./models \
         --tts-model qwen3-tts-0.6b-q8-0.gguf \
         --tokenizer-model qwen3-tts-tokenizer-0.6b-q8-0.gguf \
         -p "Hello world" -o out.wav

     Windows:
       qwen3-tts-cli.exe ^
         -m .\models ^
         --tts-model qwen3-tts-0.6b-q8-0.gguf ^
         --tokenizer-model qwen3-tts-tokenizer-0.6b-q8-0.gguf ^
         -p "Hello world" -o out.wav

   The --tts-model and --tokenizer-model flags are required because the
   downloaded filenames do not match the CLI's auto-detection pattern.

3. See all options:

     ./qwen3-tts-cli --help


Voice cloning
-------------

Pass a reference WAV with -r:

     ./qwen3-tts-cli \
       -m ./models \
       --tts-model qwen3-tts-0.6b-q8-0.gguf \
       --tokenizer-model qwen3-tts-tokenizer-0.6b-q8-0.gguf \
       -p "Hello in my voice" \
       -r examples/readme_clone_input.wav \
       -o cloned.wav


Platform notes
--------------

macOS: The binaries are unsigned. On first run, Gatekeeper will block them
with "cannot be opened because the developer cannot be verified." Clear the
quarantine attribute once, from the unzipped directory:

     xattr -dr com.apple.quarantine .

Linux: The GGML backend libraries are shipped next to the binary. If the
dynamic loader can't find them, run with:

     LD_LIBRARY_PATH=. ./qwen3-tts-cli ...

Windows: SmartScreen may warn about unsigned executables. Click "More info"
then "Run anyway." All required DLLs are shipped in this directory.

CUDA builds: Require a recent NVIDIA driver (550+). The cudart / cublas DLLs
are bundled but the driver itself is not.

Vulkan builds: Require a GPU with a working Vulkan driver. Check with:
     Linux:    vulkaninfo
     Windows:  vulkaninfo.exe  (install the Vulkan SDK or use GPU driver ICD)


Source code, issues, docs
-------------------------

     https://github.com/rmusser01/qwen3-tts.cpp
