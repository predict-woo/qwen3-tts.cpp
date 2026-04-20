#!/usr/bin/env bash
#
# download_model.sh — fetch the Qwen3-TTS 0.6B Q8_0 GGUF files from HuggingFace.
#
# Usage: ./download_model.sh [dest_dir]
#   dest_dir defaults to ./models
#
# Downloads two files into dest_dir (both are required by the CLI):
#   - qwen3-tts-0.6b-q8-0.gguf            (~986 MB, TTS model weights)
#   - qwen3-tts-tokenizer-0.6b-q8-0.gguf  (~250 MB, tokenizer + vocoder)
#
# Filenames are kept verbatim from HuggingFace; pass them explicitly to the
# CLI via --tts-model / --tokenizer-model (auto-detection uses different
# filenames).

set -euo pipefail

DEST_DIR="${1:-./models}"

BASE_URL="https://huggingface.co/OpenVoiceOS/qwen3-tts-0.6b-q8-0/resolve/main"
FILES=(
    "qwen3-tts-0.6b-q8-0.gguf"
    "qwen3-tts-tokenizer-0.6b-q8-0.gguf"
)

# Pick a downloader up front so we fail fast if neither is available.
if command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl"
elif command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget"
else
    echo "error: neither 'curl' nor 'wget' is available on PATH" >&2
    echo "       install one of them and re-run this script" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"

download_one() {
    local url="$1"
    local out="$2"

    if [ -e "$out" ]; then
        echo "[skip] $out already exists"
        return 0
    fi

    echo "[download] $url"
    echo "       --> $out"
    case "$DOWNLOADER" in
        curl)
            # -L follow redirects, -f fail on HTTP errors, -# progress bar,
            # -o output path. Write to a .part file and rename on success so
            # an interrupted download doesn't look like a completed one.
            curl -L -f -# -o "$out.part" "$url"
            mv "$out.part" "$out"
            ;;
        wget)
            wget -O "$out.part" "$url"
            mv "$out.part" "$out"
            ;;
    esac
    echo "[ok] $out"
}

for f in "${FILES[@]}"; do
    download_one "$BASE_URL/$f" "$DEST_DIR/$f"
done

echo
echo "done. files in $DEST_DIR:"
for f in "${FILES[@]}"; do
    echo "  $DEST_DIR/$f"
done
