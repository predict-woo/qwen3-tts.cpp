# CI Release Builds Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Provide precompiled `.zip` downloads for Windows/Linux/macOS across CPU/Vulkan/CUDA/Metal backends so end users can synthesize without a toolchain. Nightly artifacts on `main`, versioned Releases on tags.

**Architecture:** Single workflow `.github/workflows/release.yml` with a 7-entry build matrix. Each matrix job builds GGML with the appropriate backend, builds the project, packages binaries + runtime deps + download scripts into a `.zip`, and either uploads as a 14-day artifact (nightly) or attaches to a GitHub Release (tag).

**Tech Stack:** GitHub Actions, CMake, MSVC/gcc/Xcode, LunarG Vulkan SDK, `Jimver/cuda-toolkit`, `softprops/action-gh-release`, `actions/upload-artifact@v4`.

**Design doc:** `docs/plans/2026-04-19-ci-release-builds-design.md` (committed at 74041a1).

**Critical context for the implementer:**
- `ggml/` is a submodule — checkouts MUST use `submodules: recursive`.
- GGML is built **separately before** the main project, with backend flags on the GGML build (not the main build). See `AGENTS.md:53-54`. Main `CMakeLists.txt` expects artifacts at `ggml/build/src/`.
- CLI binary name: `qwen3-tts-cli` (+ `.exe` on Windows). Quantizer: `qwen3-tts-quantize`.
- HuggingFace GGUF URL for `download_model.sh/.bat` is **not yet chosen** — Task 1 picks it. The memory confirms 0.6B Q8_0 is the intended default but no public mirror URL has been validated.

---

## Task 1: Create `download_model.sh`

**Files:**
- Create: `scripts/download_model.sh`

**Step 1: Pick the HuggingFace GGUF URL**

Before writing the script, run:

```bash
curl -sLI "https://huggingface.co/rmusser01/qwen3-tts-0.6B-GGUF/resolve/main/qwen3-tts-0.6B-q8_0.gguf" | head -1
```

If HTTP 200 or 302, use that URL. Otherwise try these candidates in order and pick the first that returns a redirect/200:
- `https://huggingface.co/gonwan/qwen3-tts-0.6B-GGUF/resolve/main/qwen3-tts-0.6B-q8_0.gguf`
- `https://huggingface.co/Danmoreng/qwen3-tts-0.6B-GGUF/resolve/main/qwen3-tts-0.6B-q8_0.gguf`

If none work, STOP and ask the user: "No public GGUF mirror found. Should I (a) use the converter script to build the GGUF in CI and publish it via a separate release workflow, (b) point users to `scripts/convert_tts_to_gguf.py`, or (c) use a specific URL you provide?" Do not proceed with a guessed URL.

Record the chosen URL and filename. The remainder of this plan refers to them as `<GGUF_URL>` and `<GGUF_FILENAME>`.

**Step 2: Write `scripts/download_model.sh`**

```bash
#!/usr/bin/env bash
# Download the Qwen3-TTS 0.6B Q8_0 GGUF model.
# Usage: ./download_model.sh [dest_dir]  (defaults to ./models)
set -euo pipefail

URL="<GGUF_URL>"
FILENAME="<GGUF_FILENAME>"
DEST_DIR="${1:-./models}"

mkdir -p "$DEST_DIR"
OUT="$DEST_DIR/$FILENAME"

if [ -f "$OUT" ]; then
    echo "Model already exists at $OUT"
    exit 0
fi

echo "Downloading $FILENAME to $DEST_DIR ..."
if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar -o "$OUT" "$URL"
elif command -v wget >/dev/null 2>&1; then
    wget --show-progress -O "$OUT" "$URL"
else
    echo "Error: neither curl nor wget found" >&2
    exit 1
fi

echo "Done. Model saved to $OUT"
```

Mark executable: `chmod +x scripts/download_model.sh`.

**Step 3: Test it against the real URL (HEAD-only to avoid full download)**

Run:

```bash
curl -sLI "<GGUF_URL>" | head -1
```

Expected: `HTTP/2 200` or `HTTP/2 302` (redirect to CDN). If not, revisit Step 1.

Then dry-run the script's error path:

```bash
PATH=/usr/bin:/bin bash scripts/download_model.sh /tmp/qwen3tts-dltest
```

Expected: begins downloading, progress bar appears. Cancel after a few MB to confirm it's reaching the server; then delete `/tmp/qwen3tts-dltest`.

**Step 4: Commit**

```bash
git add scripts/download_model.sh
git commit -m "feat(scripts): add download_model.sh for users to fetch Qwen3-TTS GGUF"
```

---

## Task 2: Create `download_model.bat`

**Files:**
- Create: `scripts/download_model.bat`

**Step 1: Write the file**

Use PowerShell via `cmd`'s invoker since it's the most reliable Windows download path. Use the same `<GGUF_URL>` and `<GGUF_FILENAME>` from Task 1.

```batch
@echo off
REM Download the Qwen3-TTS 0.6B Q8_0 GGUF model.
REM Usage: download_model.bat [dest_dir]  (defaults to .\models)
setlocal

set "URL=<GGUF_URL>"
set "FILENAME=<GGUF_FILENAME>"
set "DEST_DIR=%~1"
if "%DEST_DIR%"=="" set "DEST_DIR=.\models"

if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"
set "OUT=%DEST_DIR%\%FILENAME%"

if exist "%OUT%" (
    echo Model already exists at %OUT%
    exit /b 0
)

echo Downloading %FILENAME% to %DEST_DIR% ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ProgressPreference='Continue'; Invoke-WebRequest -Uri '%URL%' -OutFile '%OUT%'"

if errorlevel 1 (
    echo Download failed.
    exit /b 1
)

echo Done. Model saved to %OUT%
endlocal
```

**Step 2: Validate syntax locally (no real execution)**

Run:

```bash
# On macOS/Linux, sanity-check that the file has CRLF line endings (Windows expects them for .bat)
file scripts/download_model.bat
```

Expected: `ASCII text` or `ASCII text, with CRLF line terminators`. If plain ASCII, convert:

```bash
unix2dos scripts/download_model.bat   # or: sed -i '' 's/$/\r/' scripts/download_model.bat on macOS
```

Real execution validation happens in Task 6 when the Windows CI job runs it.

**Step 3: Commit**

```bash
git add scripts/download_model.bat
git commit -m "feat(scripts): add download_model.bat for Windows users"
```

---

## Task 3: Create packaging README template

**Files:**
- Create: `.github/packaging/README.txt`

**Step 1: Write the file**

Plain text (not Markdown) so Windows Notepad renders it cleanly on double-click.

```
qwen3-tts.cpp — precompiled build
==================================

Quick start
-----------
1. Download a model:
     Linux/macOS:  ./scripts/download_model.sh
     Windows:      scripts\download_model.bat

2. Synthesize speech:
     ./qwen3-tts-cli -m ./models -p "Hello world" -o out.wav

3. For more options:
     ./qwen3-tts-cli --help

Voice cloning
-------------
     ./qwen3-tts-cli -m ./models -p "Hello" -r examples/readme_clone_input.wav -o out.wav

Quantize your own model
-----------------------
     ./qwen3-tts-quantize input.gguf output.gguf q8_0

Source code and docs
--------------------
     https://github.com/rmusser01/qwen3-tts.cpp
```

**Step 2: Commit**

```bash
git add .github/packaging/README.txt
git commit -m "chore(packaging): add README template for release archives"
```

---

## Task 4: Scaffold workflow with Linux CPU job

This is the biggest single task — it establishes the workflow skeleton end-to-end with one working job, so the remaining tasks just add matrix entries.

**Files:**
- Create: `.github/workflows/release.yml`

**Step 1: Create workflow directory**

```bash
mkdir -p .github/workflows
```

**Step 2: Write the initial workflow**

```yaml
name: Release builds

on:
  push:
    branches: [main]
    tags: ['v*']
  workflow_dispatch:

jobs:
  compute-version:
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.v.outputs.version }}
      is_tag: ${{ steps.v.outputs.is_tag }}
    steps:
      - id: v
        run: |
          if [[ "$GITHUB_REF" == refs/tags/v* ]]; then
            VERSION="${GITHUB_REF#refs/tags/v}"
            echo "is_tag=true" >> "$GITHUB_OUTPUT"
          else
            VERSION="nightly-$(date -u +%Y%m%d)-${GITHUB_SHA::7}"
            echo "is_tag=false" >> "$GITHUB_OUTPUT"
          fi
          echo "version=$VERSION" >> "$GITHUB_OUTPUT"
          echo "Resolved version: $VERSION"

  build:
    needs: compute-version
    strategy:
      fail-fast: false
      matrix:
        include:
          - { os: ubuntu-latest, backend: cpu,    name: linux-cpu }
    runs-on: ${{ matrix.os }}
    env:
      VERSION: ${{ needs.compute-version.outputs.version }}
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - name: Install build tools (Linux)
        if: runner.os == 'Linux'
        run: sudo apt-get update && sudo apt-get install -y cmake build-essential

      - name: Configure ggml (cpu)
        if: matrix.backend == 'cpu'
        run: cmake -S ggml -B ggml/build -DCMAKE_BUILD_TYPE=Release

      - name: Build ggml
        run: cmake --build ggml/build --config Release -j

      - name: Configure project
        run: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQWEN3_TTS_COREML=OFF

      - name: Build project
        run: cmake --build build --config Release -j --target qwen3-tts-cli qwen3-tts-quantize

      - name: Package
        shell: bash
        run: |
          PKG="qwen3-tts-${VERSION}-${{ matrix.name }}"
          mkdir -p "dist/$PKG/examples" "dist/$PKG/scripts"
          # Binaries
          cp build/qwen3-tts-cli      "dist/$PKG/"
          cp build/qwen3-tts-quantize "dist/$PKG/"
          # GGML runtime libs (Linux: .so next to binary + LD_LIBRARY_PATH workaround in README)
          find ggml/build -maxdepth 4 -name 'libggml*.so*' -exec cp {} "dist/$PKG/" \;
          # Examples + scripts + docs
          cp examples/*.wav               "dist/$PKG/examples/"
          cp scripts/download_model.sh   "dist/$PKG/scripts/"
          cp scripts/download_model.bat  "dist/$PKG/scripts/"
          cp .github/packaging/README.txt "dist/$PKG/"
          cp LICENSE                      "dist/$PKG/"
          chmod +x "dist/$PKG/scripts/download_model.sh"
          # Zip
          cd dist && zip -r "${PKG}.zip" "$PKG"
          ls -la "${PKG}.zip"

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: qwen3-tts-${{ env.VERSION }}-${{ matrix.name }}
          path: dist/qwen3-tts-${{ env.VERSION }}-${{ matrix.name }}.zip
          retention-days: 14
          if-no-files-found: error
```

**Step 3: Push and validate**

```bash
git add .github/workflows/release.yml
git commit -m "ci: add release workflow scaffold with Linux CPU build"
git push
```

Go to the Actions tab, watch the `Release builds` run. Expected: all steps green, `linux-cpu` zip appears as an artifact with a `qwen3-tts-nightly-YYYYMMDD-<sha>-linux-cpu.zip` inside.

**Step 4: Smoke-test the artifact**

Download the zip, unpack locally on a Linux machine (or a Linux VM/container):

```bash
unzip qwen3-tts-*-linux-cpu.zip
cd qwen3-tts-*-linux-cpu
LD_LIBRARY_PATH=. ./qwen3-tts-cli --help
```

Expected: help text prints. If ldd reports missing GGML shared libs, revisit the `find` glob in Step 2 to catch any missing `libggml-*.so`.

If the smoke test fails, fix the packaging step and re-push before moving on — later matrix entries inherit this logic.

**Step 5: Commit any packaging fixes** (if needed) with descriptive message.

---

## Task 5: Add Linux Vulkan + CUDA jobs

**Files:**
- Modify: `.github/workflows/release.yml`

**Step 1: Add Vulkan entry to the matrix**

In the `matrix.include` list, add:

```yaml
- { os: ubuntu-latest, backend: vulkan, name: linux-vulkan }
- { os: ubuntu-latest, backend: cuda,   name: linux-cuda }
```

**Step 2: Add Vulkan SDK install step**

Insert after the "Install build tools (Linux)" step:

```yaml
      - name: Install Vulkan SDK (Linux)
        if: runner.os == 'Linux' && matrix.backend == 'vulkan'
        run: |
          wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo gpg --dearmor -o /usr/share/keyrings/lunarg.gpg
          sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
          sudo apt-get update
          sudo apt-get install -y vulkan-sdk glslc
```

Note: the `-jammy` suffix must match `ubuntu-latest`'s Ubuntu codename at the time of execution. If `ubuntu-latest` has moved past jammy, update to the current codename (`noble`, etc.) before committing — check `cat /etc/os-release` in any Linux step output.

**Step 3: Add CUDA toolkit step**

```yaml
      - name: Install CUDA Toolkit
        if: matrix.backend == 'cuda'
        uses: Jimver/cuda-toolkit@v0.2.19
        with:
          cuda: '12.5.0'
          method: 'network'
          sub-packages: '["nvcc", "cudart", "cublas", "cublas_dev", "thrust"]'
```

**Step 4: Wire backend-specific GGML configure**

Replace the single "Configure ggml (cpu)" step with:

```yaml
      - name: Configure ggml
        shell: bash
        run: |
          FLAGS="-DCMAKE_BUILD_TYPE=Release"
          case "${{ matrix.backend }}" in
            cpu)    : ;;
            vulkan) FLAGS="$FLAGS -DGGML_VULKAN=ON" ;;
            cuda)   FLAGS="$FLAGS -DGGML_CUDA=ON" ;;
            metal)  FLAGS="$FLAGS -DGGML_METAL=ON" ;;
          esac
          cmake -S ggml -B ggml/build $FLAGS
```

**Step 5: Extend the runtime-lib copy glob**

In the Package step, extend the `find` line so it picks up Vulkan and CUDA shared objects produced by ggml:

```bash
find ggml/build -maxdepth 4 \( -name 'libggml*.so*' -o -name 'libggml-vulkan*.so*' -o -name 'libggml-cuda*.so*' \) -exec cp {} "dist/$PKG/" \;
```

(For CUDA you also need cudart/cublas. Since CUDA runtime libs live outside the build tree, add:)

```yaml
      - name: Bundle CUDA runtime libs (Linux)
        if: runner.os == 'Linux' && matrix.backend == 'cuda'
        shell: bash
        run: |
          PKG="qwen3-tts-${VERSION}-${{ matrix.name }}"
          for lib in libcudart.so libcublas.so libcublasLt.so; do
            p=$(ldconfig -p | grep -m1 "^\s*${lib}" | awk '{print $NF}') || true
            if [ -n "$p" ]; then cp -L "$p"* "dist/$PKG/" 2>/dev/null || true; fi
          done
```

Place this BEFORE the zip step (inside the Package step or as a new step right after it and before `cd dist && zip`).

**Step 6: Push and validate**

```bash
git add .github/workflows/release.yml
git commit -m "ci: add Linux Vulkan and CUDA build jobs"
git push
```

Expected: three green matrix entries (`linux-cpu`, `linux-vulkan`, `linux-cuda`), three artifacts uploaded.

**Step 7: Smoke-test each**

For each artifact, download and run `LD_LIBRARY_PATH=. ./qwen3-tts-cli --help` on a suitable machine (Vulkan: any Linux box with a GPU or Mesa software rasterizer; CUDA: any machine with NVIDIA driver ≥ 550). Verify the binary loads and prints help. The CLI should not crash even without a model — `--help` just prints usage.

If either Vulkan or CUDA fails to produce a `.so` to copy, check the GGML build log for that backend's output directory and adjust the glob.

---

## Task 6: Add Windows jobs (CPU, Vulkan, CUDA)

**Files:**
- Modify: `.github/workflows/release.yml`

**Step 1: Add matrix entries**

```yaml
- { os: windows-latest, backend: cpu,    name: windows-cpu }
- { os: windows-latest, backend: vulkan, name: windows-vulkan }
- { os: windows-latest, backend: cuda,   name: windows-cuda }
```

**Step 2: Add Windows-specific Vulkan SDK install**

```yaml
      - name: Install Vulkan SDK (Windows)
        if: runner.os == 'Windows' && matrix.backend == 'vulkan'
        uses: humbletim/install-vulkan-sdk@v1.1.1
        with:
          version: 1.3.290.0
          cache: true
```

**Step 3: CUDA install already conditional on `matrix.backend == 'cuda'`** — it works on Windows too without changes.

**Step 4: Replace unified Configure ggml step**

The existing bash script works on Windows too (GitHub Actions bash is available via `shell: bash`). Keep `shell: bash`.

**Step 5: Add Windows packaging variant**

Zip must use `.exe` binary names and collect `.dll`s instead of `.so`s. Replace the single Package step with a two-variant form using OS conditionals:

```yaml
      - name: Package (Linux/macOS)
        if: runner.os != 'Windows'
        shell: bash
        run: |
          # ... existing Linux packaging logic unchanged ...

      - name: Package (Windows)
        if: runner.os == 'Windows'
        shell: bash
        run: |
          PKG="qwen3-tts-${VERSION}-${{ matrix.name }}"
          mkdir -p "dist/$PKG/examples" "dist/$PKG/scripts"
          cp build/Release/qwen3-tts-cli.exe      "dist/$PKG/"
          cp build/Release/qwen3-tts-quantize.exe "dist/$PKG/"
          # GGML runtime DLLs
          find ggml/build -maxdepth 5 -name 'ggml*.dll' -exec cp {} "dist/$PKG/" \;
          # CUDA runtime DLLs if needed
          if [ "${{ matrix.backend }}" = "cuda" ]; then
            for pat in cudart64_*.dll cublas64_*.dll cublasLt64_*.dll; do
              find "$CUDA_PATH/bin" -name "$pat" -exec cp {} "dist/$PKG/" \; 2>/dev/null || true
            done
          fi
          # Vulkan loader DLL travels with the SDK — usually auto-located via system. If users have no GPU driver with a Vulkan ICD, the app will fail at runtime; that's expected and documented.
          cp examples/*.wav               "dist/$PKG/examples/"
          cp scripts/download_model.sh   "dist/$PKG/scripts/"
          cp scripts/download_model.bat  "dist/$PKG/scripts/"
          cp .github/packaging/README.txt "dist/$PKG/"
          cp LICENSE                      "dist/$PKG/"
          # Use PowerShell zip (works reliably on windows-latest)
          powershell -Command "Compress-Archive -Path 'dist/$PKG/*' -DestinationPath 'dist/${PKG}.zip'"
          ls -la "dist/${PKG}.zip"
```

**Step 6: Push and validate**

```bash
git add .github/workflows/release.yml
git commit -m "ci: add Windows CPU, Vulkan, and CUDA build jobs"
git push
```

Expected: 6 green matrix entries, 6 artifacts.

**Step 7: Smoke-test a Windows artifact**

Download `qwen3-tts-*-windows-cpu.zip`. On a Windows machine (VM if needed):

```cmd
qwen3-tts-cli.exe --help
scripts\download_model.bat .\models
```

Expected: help prints, download script begins fetching. Cancel after a few MB.

If missing DLLs cause a load failure on launch, open the exe in Dependencies (or `dumpbin /dependents`) and extend the copy glob.

---

## Task 7: Add macOS Metal job

**Files:**
- Modify: `.github/workflows/release.yml`

**Step 1: Add matrix entry**

```yaml
- { os: macos-14, backend: metal, name: macos-arm64-metal }
```

**Step 2: Project configure on macOS should keep CoreML ON**

The current `Configure project` step hardcodes `-DQWEN3_TTS_COREML=OFF`. That's correct for Linux/Windows. Adjust to enable it only on macOS:

```yaml
      - name: Configure project
        shell: bash
        run: |
          FLAGS="-DCMAKE_BUILD_TYPE=Release"
          if [ "${{ runner.os }}" != "macOS" ]; then
            FLAGS="$FLAGS -DQWEN3_TTS_COREML=OFF"
          fi
          cmake -S . -B build $FLAGS
```

**Step 3: Add macOS packaging branch**

The existing "Package (Linux/macOS)" step likely already works on macOS, but GGML on macOS produces `.dylib` not `.so`. Extend the find glob:

```bash
find ggml/build -maxdepth 4 \( -name 'libggml*.so*' -o -name 'libggml*.dylib' \) -exec cp {} "dist/$PKG/" \;
```

And add an install_name_tool fixup so the binary finds the `.dylib`s next to it:

```bash
if [ "${{ runner.os }}" = "macOS" ]; then
    for dylib in "dist/$PKG/"*.dylib; do
        install_name_tool -id "@rpath/$(basename "$dylib")" "$dylib" || true
    done
    for bin in qwen3-tts-cli qwen3-tts-quantize; do
        install_name_tool -add_rpath "@executable_path" "dist/$PKG/$bin" || true
    done
fi
```

**Step 4: Push and validate**

```bash
git add .github/workflows/release.yml
git commit -m "ci: add macOS arm64 Metal build job"
git push
```

Expected: 7 green matrix entries, 7 artifacts.

**Step 5: Smoke-test**

On the current macOS dev machine:

```bash
unzip qwen3-tts-*-macos-arm64-metal.zip
cd qwen3-tts-*-macos-arm64-metal
./qwen3-tts-cli --help
```

macOS will quarantine unsigned binaries (Gatekeeper). Expect a "cannot be opened because the developer cannot be verified" dialog on first run. Document this in the README.txt with `xattr -dr com.apple.quarantine <dir>` as the workaround — add a short note to `.github/packaging/README.txt` in this task.

---

## Task 8: Add tag-triggered release publish job

**Files:**
- Modify: `.github/workflows/release.yml`

**Step 1: Add a `release` job gated on tag trigger**

Append to the workflow:

```yaml
  release:
    needs: [compute-version, build]
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          path: all-artifacts
          pattern: 'qwen3-tts-*'
          merge-multiple: true

      - name: List artifacts
        run: ls -la all-artifacts/

      - uses: softprops/action-gh-release@v2
        with:
          files: all-artifacts/*.zip
          generate_release_notes: true
          fail_on_unmatched_files: true
```

**Step 2: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: publish tag builds to GitHub Releases"
git push
```

Tag pushes will now produce a Release. Nightly pushes remain artifact-only.

---

## Task 9: End-to-end tag build validation

**Step 1: Create a throwaway test tag**

```bash
git tag v0.0.0-citest
git push origin v0.0.0-citest
```

**Step 2: Watch Actions tab**

Expected: all 7 `build` jobs green, then `release` job green, a new Release `v0.0.0-citest` appears on the repo with 7 `.zip`s attached.

**Step 3: Clean up**

Delete the Release on GitHub (repo → Releases → the test release → Delete). Then delete the tag:

```bash
git push origin :refs/tags/v0.0.0-citest
git tag -d v0.0.0-citest
```

**Step 4: Announce completion**

If all 7 artifacts were present on the test Release and the clean-up succeeded, the CI is ready for real releases. Use superpowers:requesting-code-review to get a review of the workflow file before merging to `main`.

---

## Out of scope (tracked for later)

- Code signing on Windows/macOS (current builds will trigger SmartScreen / Gatekeeper warnings).
- PR-triggered build validation.
- Cross-run caching of ggml build / CUDA toolkit install.
- Linux ARM64, macOS x86_64, Windows ARM64.
- Bundling the Python server with the release.
