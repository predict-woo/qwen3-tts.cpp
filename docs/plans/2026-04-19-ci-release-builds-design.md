# CI release builds — design

## Goal

Provide precompiled downloads so end users can grab a `.zip`, run one script to fetch a model, and synthesize — no toolchain required. Cover Windows, Linux, and macOS across CPU and GPU backends.

## Workflow structure

Single workflow at `.github/workflows/release.yml` with two triggers:

- `push: tags: ['v*']` — builds all variants, publishes a GitHub Release with the `.zip`s attached.
- `push: branches: [main]` — builds all variants, uploads as workflow artifacts with 14-day retention (nightly channel; downloaded from the Actions tab).

Both triggers share the same build matrix and packaging jobs. Only the final publish step differs. Checkout uses `submodules: recursive` since `ggml/` is a submodule.

## Build matrix

Seven parallel jobs, `fail-fast: false`:

| OS | Backend | Runner | Tooling |
|---|---|---|---|
| Windows | CPU | `windows-latest` | MSVC (default) |
| Windows | Vulkan | `windows-latest` | LunarG Vulkan SDK action |
| Windows | CUDA | `windows-latest` | `Jimver/cuda-toolkit`, CUDA 12.x |
| Linux | CPU | `ubuntu-latest` | gcc (default) |
| Linux | Vulkan | `ubuntu-latest` | `apt install vulkan-sdk` |
| Linux | CUDA | `ubuntu-latest` | `Jimver/cuda-toolkit`, CUDA 12.x |
| macOS | Metal (arm64) | `macos-14` | Xcode (default) |

Build command: `cmake -B build -DCMAKE_BUILD_TYPE=Release -D<backend flags>` then `cmake --build build --config Release -j`.

Not in matrix: Linux macOS x86_64 (legacy), Windows ARM64 (no demand yet). Add later if requested.

## Packaging

Each job produces one archive with this layout:

```
qwen3-tts-<version>-<os>-<backend>/
├── qwen3-tts-cli(.exe)
├── qwen3-tts-quantize(.exe)
├── <runtime libraries>         # Vulkan loader, cudart/cublas, etc.
├── examples/                    # wav samples copied from repo
├── scripts/
│   ├── download_model.sh
│   └── download_model.bat        # both fetch Qwen3-TTS 0.6B Q8_0 GGUF
├── README.txt                    # short: run download_model, then cli --help
└── LICENSE
```

Then zipped as `qwen3-tts-<version>-<os>-<backend>.zip`.

Version string:
- Tag build: tag without `v` prefix — `v1.2.3` → `qwen3-tts-1.2.3-windows-cuda.zip`.
- Nightly: `nightly-YYYYMMDD-<short-sha>` — e.g. `qwen3-tts-nightly-20260419-abc1234-linux-vulkan.zip`.

`download_model.sh` and `download_model.bat` are new files to add under `scripts/`. They fetch the Qwen3-TTS 0.6B Q8_0 GGUF from HuggingFace using `curl` / `Invoke-WebRequest`. Exact HF URL selected at implementation time.

## Publish

**Tag builds:** After all 7 jobs succeed, a `release` job uses `softprops/action-gh-release` to create/update the Release for the tag and attach all 7 `.zip`s. Release notes auto-generated from commits since the last tag; editable by hand afterwards.

**Nightly builds:** Each job uploads its `.zip` via `actions/upload-artifact` with 14-day retention. No Release is created.

## Failure behavior

`fail-fast: false` on the matrix so one broken backend doesn't block the others. On a tag push, if any matrix job fails the `release` job is skipped — no partial Release gets published. Recovery: fix, re-push the tag (or `workflow_dispatch` re-run).

## CI cost

Rough per-run estimate — Windows/Linux CUDA ~15 min (toolkit install dominates), others ~5–10 min. Seven parallel jobs, wall clock ~15 min. Well within the free tier for a public repo.

## Out of scope

Deferred and tracked for later if demand appears:

- Windows code signing (SmartScreen will warn on unsigned `.exe`).
- PR build validation.
- Build caching between runs.
- Linux ARM64 / macOS x86_64 / Windows ARM64 variants.
- Bundling the Python server (requires separate Python runtime; `server/README.md` stays the documented path).
