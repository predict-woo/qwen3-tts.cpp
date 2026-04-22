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
# Parallel arrays: filename, expected sha256 of the raw GGUF file.
FILES=(
    "qwen3-tts-0.6b-q8-0.gguf"
    "qwen3-tts-tokenizer-0.6b-q8-0.gguf"
)
SHA256=(
    "11486718b76d5d7fa1d1c6b03c3afac03f7736c53c3acc3d2903db6925f9ca71"
    "f03d490a06d038212d99464ae9a6c2fc3327e34c8a4b840d9e8e6ca67bccaf72"
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

# Pick a sha256 command that exists on the platform.
if command -v sha256sum >/dev/null 2>&1; then
    SHA256_CMD="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
    SHA256_CMD="shasum -a 256"
else
    SHA256_CMD=""
fi

verify_sha256() {
    local file="$1"
    local expected="$2"
    if [ -z "$SHA256_CMD" ]; then
        echo "[warn] no sha256 tool found; skipping checksum verification for $file" >&2
        return 0
    fi
    local actual
    actual=$($SHA256_CMD "$file" | awk '{print $1}')
    if [ "$actual" != "$expected" ]; then
        echo "[error] checksum mismatch for $file" >&2
        echo "        expected: $expected" >&2
        echo "        actual:   $actual" >&2
        return 1
    fi
    echo "[verified] sha256 ok ($file)"
    return 0
}

download_one() {
    local url="$1"
    local out="$2"
    local expected_sha="$3"

    if [ -e "$out" ]; then
        echo "[skip] $out already exists"
        verify_sha256 "$out" "$expected_sha" || {
            echo "       re-download by deleting and re-running, or ignore if you trust the source" >&2
        }
        return 0
    fi

    echo "[download] $url"
    echo "       --> $out"
    # Remove any leftover .part from a previous failed run, and ensure
    # a mid-download failure doesn't leave a .part behind that a user
    # might later mistake for a resumable state.
    rm -f "$out.part"
    trap 'rm -f "$out.part"' ERR
    case "$DOWNLOADER" in
        curl)
            # -L follow redirects, -f fail on HTTP errors, -# progress bar,
            # -o output path. Write to a .part file and rename on success so
            # an interrupted download doesn't look like a completed one.
            curl -L -f -# -o "$out.part" "$url"
            ;;
        wget)
            wget --tries=3 --timeout=30 --show-progress -O "$out.part" "$url"
            ;;
    esac
    mv "$out.part" "$out"
    trap - ERR
    echo "[ok] $out"

    # Verify integrity; bail if corrupt so "already exists" doesn't mask a bad download.
    if ! verify_sha256 "$out" "$expected_sha"; then
        rm -f "$out"
        return 1
    fi
}

for i in "${!FILES[@]}"; do
    f="${FILES[$i]}"
    download_one "$BASE_URL/$f" "$DEST_DIR/$f" "${SHA256[$i]}"
done

echo
echo "done. files in $DEST_DIR:"
for f in "${FILES[@]}"; do
    echo "  $DEST_DIR/$f"
done
