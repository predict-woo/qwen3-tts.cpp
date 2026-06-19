# Cross-Platform Qwen3-TTS Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add benchmark visibility and low-risk cross-platform speedups for `qwen3-tts.cpp`, then gate deeper GGML scheduler and server-cache optimizations on measured wins.

**Architecture:** Keep public synthesis semantics unchanged. Add structured benchmark output at the CLI/result boundary, fix backend thread propagation through existing runtime boundaries, pilot GGML scheduler reservation behind an opt-in flag, and add explicit ICL prompt caching through a small core/C API/server path.

**Tech Stack:** C++17, GGML scheduler/backends, CMake/CTest, Python `ctypes` server binding, FastAPI server.

---

## Scope Notes

The design covers four related but independently shippable slices. Implement them in order and commit after each task. If Task 1 shows graph build/allocation is not material on the target backend, skip Task 3 and record that result in the docs instead of forcing a scheduler abstraction.

The current repo has unrelated dirty worktree changes. Start implementation in a dedicated worktree from the current committed design head and do not revert unrelated edits in the original checkout.

## File Structure

- `src/common/benchmark_json.h` / `src/common/benchmark_json.cpp`: small JSON writer for benchmark records; no dependency on a third-party JSON library.
- `tests/test_benchmark_json.cpp`: unit tests for JSON escaping and computed timing fields.
- `src/transformer/tts_transformer.h` / `src/transformer/tts_transformer.cpp`: expose last detailed generation timing under `QWEN3_TTS_TIMING`; optionally hold scheduler reservation state.
- `src/pipeline/qwen3_tts.h` / `src/pipeline/qwen3_tts.cpp`: carry detailed timing into `tts_result`, add thread setter, and add prepared ICL prompt APIs.
- `src/common/backend_threads.h` / `src/common/backend_threads.cpp`: backend-thread helper for CPU and backend registry thread hooks.
- `src/common/gguf_loader.cpp`: apply default backend thread count when shared preferred backends are created.
- `src/pipeline/qwen3tts_c_api.h` / `src/pipeline/qwen3tts_c_api.cpp`: expose effective thread behavior and opaque prepared ICL prompt handles.
- `server/qwen3_tts_binding.py`: bind prepared ICL prompt APIs and pass thread settings consistently.
- `server/icl_cache.py`: deterministic ICL prompt cache key and bounded cache.
- `server/main.py`: use `server/icl_cache.py` for repeated ICL requests.
- `tests/test_backend_threads.cpp`: focused helper test for CPU backend thread application.
- `server/tests/test_icl_cache.py`: pure-Python tests for cache key, hit/miss, and invalidation behavior.
- `CMakeLists.txt`: add the new common sources and test targets.
- `README.md` and `server/README.md`: document benchmark JSON, thread behavior, graph reservation flag, and ICL cache scope.

## Task 0: Isolate Implementation Worktree

**Files:**
- No repo file changes.

- [ ] **Step 1: Create a dedicated worktree**

Run:

```bash
git worktree add ../qwen3-tts.cpp-perf-cross-platform HEAD -b perf/cross-platform-qwen3-tts
```

Expected: new worktree created from the current committed spec head.

- [ ] **Step 2: Check the implementation worktree status**

Run:

```bash
git -C ../qwen3-tts.cpp-perf-cross-platform status --short
```

Expected: no unrelated modified files. If dirty, stop and inspect before editing.

- [ ] **Step 3: Configure a timing build**

Run:

```bash
cmake -S . -B build-perf -DQWEN3_TTS_TIMING=ON
cmake --build build-perf -j
```

Expected: build completes. If model-dependent tests are not runnable locally, continue with non-model unit tests and document the blocker.

## Task 1: Add Structured Benchmark Output

**Files:**
- Create: `src/common/benchmark_json.h`
- Create: `src/common/benchmark_json.cpp`
- Create: `tests/test_benchmark_json.cpp`
- Modify: `CMakeLists.txt`
- Modify: `src/transformer/tts_transformer.h`
- Modify: `src/transformer/tts_transformer.cpp`
- Modify: `src/pipeline/qwen3_tts.h`
- Modify: `src/pipeline/qwen3_tts.cpp`
- Modify: `src/main.cpp`

- [ ] **Step 1: Write failing JSON helper tests**

Add `tests/test_benchmark_json.cpp` with tests like:

```cpp
#include "common/benchmark_json.h"

#include <cassert>
#include <string>

int main() {
    qwen3_tts::benchmark_record r;
    r.mode = "default";
    r.backend = "cpu";
    r.text = "hello \"tts\"";
    r.audio_seconds = 2.0;
    r.total_ms = 1000;
    r.generate_ms = 700;

    const std::string json = qwen3_tts::benchmark_record_to_json(r);
    assert(json.find("\"mode\":\"default\"") != std::string::npos);
    assert(json.find("hello \\\"tts\\\"") != std::string::npos);
    assert(json.find("\"speed_x_realtime\":2") != std::string::npos);
    assert(json.find("\"wall_rtf\":0.5") != std::string::npos);
    return 0;
}
```

- [ ] **Step 2: Register the failing test**

Add `src/common/benchmark_json.cpp` to `COMMON_SOURCES`, add a `test_benchmark_json` executable, and add a `benchmark_json_test` CTest entry.

Run:

```bash
cmake --build build-perf -j
ctest --test-dir build-perf -R benchmark_json_test --output-on-failure
```

Expected: build or test fails because `benchmark_json` does not exist yet.

- [ ] **Step 3: Implement the JSON helper**

Create `benchmark_record` with fields for mode, backend, device, thread count, model type, model size, quantization/model names, text, audio seconds, total/tokenize/encode/generate/decode ms, memory snapshots, and optional detailed timing values.

Keep it dependency-free:

```cpp
namespace qwen3_tts {
struct benchmark_record {
    std::string mode;
    std::string backend;
    std::string device;
    int32_t thread_count = 0;
    std::string model_type;
    std::string model_size;
    std::string text;
    double audio_seconds = 0.0;
    int64_t tokenize_ms = 0;
    int64_t encode_ms = 0;
    int64_t generate_ms = 0;
    int64_t decode_ms = 0;
    int64_t total_ms = 0;
};

std::string benchmark_record_to_json(const benchmark_record & r);
bool write_benchmark_record_json(const std::string & path, const benchmark_record & r, std::string & error);
}
```

Computed fields:

- `speed_x_realtime = audio_seconds / (total_ms / 1000.0)`
- `wall_rtf = (total_ms / 1000.0) / audio_seconds`

- [ ] **Step 4: Expose detailed transformer timing to results**

Under `QWEN3_TTS_TIMING`, add a copy of the last `tts_timing` to `TTSTransformer` and `tts_result`.

Sketch:

```cpp
#ifdef QWEN3_TTS_TIMING
const tts_timing * TTSTransformer::last_timing() const {
    return has_last_timing_ ? &last_timing_ : nullptr;
}
#endif
```

At the end of `TTSTransformer::generate`, store the local timing struct before clearing `timing_`. In `Qwen3TTS::synthesize_internal`, copy it to `result` after `transformer_.generate(...)` succeeds.

- [ ] **Step 5: Add CLI flags**

Add to `src/main.cpp`:

- `--benchmark-json <file>`
- `--quiet-progress` or reuse an existing setting to disable progress output for benchmark runs

After synthesis succeeds, populate a `benchmark_record` and write it if `--benchmark-json` is set. Include the mode selected by CLI path:

- `default`
- `speaker_embedding`
- `preset`
- `xvector_voice_clone`
- `icl_voice_clone`

- [ ] **Step 6: Verify helper tests pass**

Run:

```bash
cmake --build build-perf -j
ctest --test-dir build-perf -R benchmark_json_test --output-on-failure
```

Expected: `benchmark_json_test` passes.

- [ ] **Step 7: Smoke benchmark JSON with a local model if available**

Run:

```bash
./build-perf/qwen3-tts-cli -m models \
  -t "Benchmark smoke test." \
  --seed 1234 --max-tokens 32 \
  --benchmark-json /tmp/qwen3-tts-bench.json \
  -o /tmp/qwen3-tts-bench.wav
```

Expected: JSON file exists and includes timing, audio duration, `speed_x_realtime`, and `wall_rtf`. If local models are missing, record the blocker and rely on unit tests.

- [ ] **Step 8: Commit**

```bash
git add CMakeLists.txt src/common/benchmark_json.* tests/test_benchmark_json.cpp \
  src/transformer/tts_transformer.* src/pipeline/qwen3_tts.* src/main.cpp
git commit -m "feat: add structured benchmark output"
```

## Task 2: Wire Effective Backend Thread Controls

**Files:**
- Create: `src/common/backend_threads.h`
- Create: `src/common/backend_threads.cpp`
- Create: `tests/test_backend_threads.cpp`
- Modify: `CMakeLists.txt`
- Modify: `src/common/gguf_loader.h`
- Modify: `src/common/gguf_loader.cpp`
- Modify: `src/pipeline/qwen3_tts.h`
- Modify: `src/pipeline/qwen3_tts.cpp`
- Modify: `src/pipeline/qwen3tts_c_api.cpp`
- Modify: `server/qwen3_tts_binding.py`

- [ ] **Step 1: Write failing backend-thread helper test**

Add `tests/test_backend_threads.cpp`:

```cpp
#include "common/backend_threads.h"
#include "ggml-backend.h"

#include <cassert>

int main() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(cpu != nullptr);
    assert(qwen3_tts::apply_backend_n_threads(cpu, 1));
    assert(qwen3_tts::apply_backend_n_threads(cpu, 2));
    assert(!qwen3_tts::apply_backend_n_threads(nullptr, 2));
    assert(!qwen3_tts::apply_backend_n_threads(cpu, 0));
    ggml_backend_free(cpu);
    return 0;
}
```

Register `test_backend_threads` in `CMakeLists.txt`.

Run:

```bash
cmake --build build-perf -j
ctest --test-dir build-perf -R backend_threads_test --output-on-failure
```

Expected: fails because helper does not exist.

- [ ] **Step 2: Implement `backend_threads` helper**

Use GGML backend registry support when available, and include CPU fallback support:

```cpp
bool apply_backend_n_threads(ggml_backend_t backend, int32_t n_threads);
void set_default_backend_n_threads(int32_t n_threads);
int32_t get_default_backend_n_threads();
```

Implementation should:

- reject `nullptr` and `n_threads <= 0`
- look up `ggml_backend_set_n_threads` through the backend registry
- fall back to `ggml_backend_cpu_set_n_threads` for CPU backends
- return `false` for backends with no thread hook, without treating that as fatal

- [ ] **Step 3: Apply default threads at backend creation**

In `init_preferred_backend(...)`, after backend creation and before storing the shared backend, call `apply_backend_n_threads(backend, get_default_backend_n_threads())`.

Also apply thread count to CPU fallback backends created directly in model components, especially `TTSTransformer::load_model`.

- [ ] **Step 4: Add runtime setter on `Qwen3TTS`**

Add:

```cpp
void Qwen3TTS::set_n_threads(int32_t n_threads);
```

It should update the default backend thread count and apply it to loaded component backends where accessible. Keep unsupported GPU backends non-fatal.

- [ ] **Step 5: Wire C API constructor and per-call params**

In `qwen3_tts_create_with_models(...)`, replace `(void)n_threads` with:

```cpp
if (n_threads > 0) {
    tts->engine.set_n_threads(n_threads);
}
```

Before synthesis calls, apply `cpp_params.n_threads` through `engine.set_n_threads(...)` so CLI and server per-request params are effective.

- [ ] **Step 6: Verify tests**

Run:

```bash
cmake --build build-perf -j
ctest --test-dir build-perf -R "backend_threads_test|benchmark_json_test|sampling_test" --output-on-failure
```

Expected: listed tests pass.

- [ ] **Step 7: Smoke CPU benchmark thread variation if models are available**

Run:

```bash
QWEN3_TTS_BACKEND=cpu ./build-perf/qwen3-tts-cli -m models \
  -t "Thread benchmark." --seed 1234 --max-tokens 32 -j 1 \
  --benchmark-json /tmp/qwen3-tts-j1.json -o /tmp/qwen3-tts-j1.wav

QWEN3_TTS_BACKEND=cpu ./build-perf/qwen3-tts-cli -m models \
  -t "Thread benchmark." --seed 1234 --max-tokens 32 -j 4 \
  --benchmark-json /tmp/qwen3-tts-j4.json -o /tmp/qwen3-tts-j4.wav
```

Expected: both runs complete. The JSON records should show `thread_count` as 1 and 4 respectively. Record observed `generate_ms`, `total_ms`, and whether the CPU backend thread hook was applied. Do not require a fixed speedup threshold, but if timing does not change or the hook cannot be applied, document that result explicitly instead of claiming a speedup.

- [ ] **Step 8: Commit**

```bash
git add CMakeLists.txt src/common/backend_threads.* src/common/gguf_loader.* \
  src/pipeline/qwen3_tts.* src/pipeline/qwen3tts_c_api.cpp \
  server/qwen3_tts_binding.py tests/test_backend_threads.cpp
git commit -m "fix: apply qwen3 tts thread settings"
```

## Task 3: Pilot GGML Scheduler Reservation for Code Predictor

**Files:**
- Modify: `src/transformer/tts_transformer.h`
- Modify: `src/transformer/tts_transformer.cpp`
- Modify: `src/common/benchmark_json.h`
- Modify: `src/common/benchmark_json.cpp`
- Modify: `README.md`

- [ ] **Step 1: Add reservation metrics behind timing**

Extend `tts_timing` with:

```cpp
double t_code_pred_reserve_ms = 0;
int32_t n_code_pred_reserve_attempts = 0;
int32_t n_code_pred_reserve_successes = 0;
```

Add the same fields to `benchmark_record`.

- [ ] **Step 2: Add opt-in reservation flag**

Add an internal helper:

```cpp
bool TTSTransformer::graph_reservation_enabled() const;
```

Read `QWEN3_TTS_GRAPH_RESERVE=1`. Default must be off for the first implementation pass.

- [ ] **Step 3: Implement the smallest code-predictor reservation pilot**

Add state for whether the code-predictor reservation has been attempted for the currently loaded model/backend.

Pilot logic:

1. On first `predict_codes_autoregressive(...)`, build a representative prefill graph.
2. Call `ggml_backend_sched_reserve(state_.sched, prefill_graph)`.
3. Build a representative maximum step graph using `n_past = 15` and `generation_step = 14`.
4. Call `ggml_backend_sched_reserve(state_.sched, step_graph)`.
5. Do not cache raw `ggml_cgraph *` beyond the call.
6. If either reservation fails, disable reservation for the session and continue normal inference.

- [ ] **Step 4: Preserve normal compute behavior**

Keep the existing per-step `build_code_pred_*`, `ggml_backend_sched_alloc_graph`, tensor set/get, compute, sample, and reset sequence. This task is only about avoiding repeated backend buffer planning if GGML can use the reservation.

- [ ] **Step 5: Verify compile and core tests**

Run:

```bash
cmake --build build-perf -j
ctest --test-dir build-perf -R "transformer_test|sampling_test|benchmark_json_test" --output-on-failure
```

Expected: tests pass or model-dependent tests report known missing-model blockers. No regression in model-free tests.

- [ ] **Step 6: Benchmark with and without reservation if models are available**

Run:

```bash
./build-perf/qwen3-tts-cli -m models \
  -t "Graph reservation benchmark." --seed 1234 --max-tokens 64 \
  --benchmark-json /tmp/qwen3-tts-no-reserve.json -o /tmp/qwen3-tts-no-reserve.wav

QWEN3_TTS_GRAPH_RESERVE=1 ./build-perf/qwen3-tts-cli -m models \
  -t "Graph reservation benchmark." --seed 1234 --max-tokens 64 \
  --benchmark-json /tmp/qwen3-tts-reserve.json -o /tmp/qwen3-tts-reserve.wav
```

Expected: both runs complete. Compare code-predictor graph allocation time and total generation time. If reservation does not help or causes backend issues, leave it documented as experimental/off and do not enable by default.

- [ ] **Step 7: Verify deterministic output parity**

Run the same prompt, backend, seed, and max-token settings with reservation disabled and enabled:

```bash
./build-perf/qwen3-tts-cli -m models \
  -t "Graph reservation parity." --seed 1234 --max-tokens 64 \
  --benchmark-json /tmp/qwen3-tts-parity-off.json -o /tmp/qwen3-tts-parity-off.wav

QWEN3_TTS_GRAPH_RESERVE=1 ./build-perf/qwen3-tts-cli -m models \
  -t "Graph reservation parity." --seed 1234 --max-tokens 64 \
  --benchmark-json /tmp/qwen3-tts-parity-on.json -o /tmp/qwen3-tts-parity-on.wav

shasum -a 256 /tmp/qwen3-tts-parity-off.wav /tmp/qwen3-tts-parity-on.wav
```

Expected: exact generated-code parity if an accessible code fixture exists. If only WAV output is available, the WAV hashes should match for this reservation-only change. If local models are unavailable, document the model-fixture blocker and do not claim semantic parity.

- [ ] **Step 8: Commit**

```bash
git add src/transformer/tts_transformer.* src/common/benchmark_json.* README.md
git commit -m "perf: pilot code predictor scheduler reservation"
```

## Task 4: Add Prepared ICL Prompt Cache for Server Reuse

**Files:**
- Create: `server/icl_cache.py`
- Create: `server/tests/test_icl_cache.py`
- Modify: `src/pipeline/qwen3_tts.h`
- Modify: `src/pipeline/qwen3_tts.cpp`
- Modify: `src/pipeline/qwen3tts_c_api.h`
- Modify: `src/pipeline/qwen3tts_c_api.cpp`
- Modify: `server/qwen3_tts_binding.py`
- Modify: `server/main.py`
- Modify: `server/README.md`
- Modify: `CMakeLists.txt` if adding Python cache tests to CTest

- [ ] **Step 1: Add pure Python cache-key tests**

Create `server/tests/test_icl_cache.py` using only the standard library. Test:

- same path/text/model metadata gives same key
- changed model identity changes key
- changed reference text changes key
- changed file mtime/size changes key
- LRU evicts oldest entry when max size is exceeded
- evicted values are explicitly closed/freed

Run:

```bash
python3 server/tests/test_icl_cache.py
```

Expected: fails because `server/icl_cache.py` does not exist.

- [ ] **Step 2: Implement `server/icl_cache.py`**

Provide:

```python
@dataclass(frozen=True)
class IclPromptCacheKey:
    tts_model_path: str
    tts_model_size: int
    tts_model_mtime_ns: int
    speaker_encoder_model_path: str
    speaker_encoder_model_size: int
    speaker_encoder_model_mtime_ns: int
    codec_encoder_model_path: str
    codec_encoder_model_size: int
    codec_encoder_model_mtime_ns: int
    tokenizer_decoder_model_path: str
    tokenizer_decoder_model_size: int
    tokenizer_decoder_model_mtime_ns: int
    reference_path: str
    reference_size: int
    reference_mtime_ns: int
    reference_text_hash: str
    language_id: int

class IclPromptCache:
    def __init__(self, max_entries: int): ...
    def get(self, key): ...
    def put(self, key, value): ...
    def clear(self): ...
```

Use `OrderedDict` for LRU behavior. Include every model component that can affect prepared ICL output. In the current runtime the TTS model path may also contain speaker-encoder tensors, and the tokenizer/vocoder model path may also contain the Mimi codec encoder, but the cache key should name those roles explicitly even when their resolved paths are identical. Cache values must own prompt handles deterministically: when an entry is evicted or the cache is cleared, call `close()` on the cached value so the underlying `qwen3_tts_free_icl_prompt(...)` runs. Do not rely only on Python finalizers.

- [ ] **Step 3: Add C++ prepared prompt model**

In `src/pipeline/qwen3_tts.h`, add:

```cpp
struct icl_prompt {
    std::vector<float> speaker_embedding;
    std::vector<int32_t> ref_codes;
    int32_t n_ref_frames = 0;
    std::string ref_text;
};

bool prepare_icl_prompt(const std::string & reference_audio,
                        const std::string & reference_text,
                        const tts_params & params,
                        icl_prompt & out);

tts_result synthesize_with_icl_prompt(const std::string & text,
                                      const icl_prompt & prompt,
                                      const tts_params & params = tts_params());
```

Move the existing reference load/resample, speaker-encoder, and codec-encoder work from `synthesize_with_voice(...)` into `prepare_icl_prompt(...)`. Keep `synthesize_with_voice(...)` behavior unchanged by having it prepare and immediately use a prompt.

- [ ] **Step 4: Expose opaque C API prompt handles**

Add to `qwen3tts_c_api.h`:

```c
typedef struct Qwen3TtsIclPrompt Qwen3TtsIclPrompt;

Qwen3TtsIclPrompt* qwen3_tts_prepare_icl_prompt_file(
    Qwen3Tts* tts,
    const char* reference_audio_path,
    const char* reference_text,
    const Qwen3TtsParams* params);

Qwen3TtsAudio* qwen3_tts_synthesize_with_icl_prompt(
    Qwen3Tts* tts,
    const char* text,
    const Qwen3TtsIclPrompt* prompt,
    const Qwen3TtsParams* params);

void qwen3_tts_free_icl_prompt(Qwen3TtsIclPrompt* prompt);
```

Implement ownership with a heap-allocated wrapper around `qwen3_tts::icl_prompt`.

- [ ] **Step 5: Bind prompt handles in Python**

In `server/qwen3_tts_binding.py`, add:

- `prepare_icl_prompt(reference_audio_path, reference_text, ...)`
- `synthesize_with_icl_prompt(text, prompt_handle, ...)`
- prompt handle finalizer that calls `qwen3_tts_free_icl_prompt`
- resolved model identity fields for cache keys: TTS model, speaker-encoder model, codec-encoder model, and tokenizer/vocoder decoder model path/size/mtime. These role identities may point to the same underlying GGUF file, but the binding should expose them separately so future split-model layouts do not produce stale cache hits.

Keep raw pointer handling private to the binding.

- [ ] **Step 6: Use cache in server ICL route**

In `server/main.py`, add:

- `QWEN3TTS_ICL_CACHE_SIZE` env var, default `8`
- global `icl_prompt_cache`
- cache key from `server/icl_cache.py`

Inside the existing `_synthesis_lock`, get or prepare the prompt, then call `synthesize_with_icl_prompt(...)`.

Do not add parallel request execution. The cache reduces repeated preprocessing only.

On FastAPI lifespan shutdown, clear the ICL cache before destroying `tts_engine` so cached C prompt handles are freed while the library and engine are still valid.

- [ ] **Step 7: Verify tests**

Run:

```bash
python3 server/tests/test_icl_cache.py
cmake --build build-perf -j
ctest --test-dir build-perf -R "c_api_customvoice_test|benchmark_json_test|backend_threads_test" --output-on-failure
```

Expected: Python cache tests pass. C/C++ tests pass or model-dependent tests report known blockers.

- [ ] **Step 8: Smoke repeated ICL if models and reference audio are available**

Run two identical `/v1/audio/speech` ICL requests and log whether the second request reports an ICL cache hit. If local server dependencies or models are unavailable, document the blocker.

- [ ] **Step 9: Commit**

```bash
git add src/pipeline/qwen3_tts.* src/pipeline/qwen3tts_c_api.* \
  server/qwen3_tts_binding.py server/main.py server/icl_cache.py \
  server/tests/test_icl_cache.py server/README.md CMakeLists.txt
git commit -m "perf: cache prepared icl prompts in server"
```

## Task 5: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `server/README.md`
- Modify: `docs/superpowers/specs/2026-05-04-cross-platform-qwen3-tts-performance-design.md` only if implementation findings change the design assumptions.

- [ ] **Step 1: Document benchmark usage**

Add a short README section with:

```bash
cmake -S . -B build-perf -DQWEN3_TTS_TIMING=ON
cmake --build build-perf -j
./build-perf/qwen3-tts-cli -m models -t "Benchmark." \
  --seed 1234 --max-tokens 64 \
  --benchmark-json bench.json -o bench.wav
```

Explain `speed_x_realtime` and `wall_rtf`.

- [ ] **Step 2: Document thread behavior**

Clarify that `-j/--threads`, C API `n_threads`, and `QWEN3TTS_THREADS` affect CPU-capable backends when GGML exposes a thread hook. GPU-only backends may ignore it.

- [ ] **Step 3: Document scheduler reservation**

If Task 3 remains opt-in, document:

```bash
QWEN3_TTS_GRAPH_RESERVE=1 ./build-perf/qwen3-tts-cli ...
```

Clearly mark it experimental unless measurements justify enabling by default.

- [ ] **Step 4: Document ICL cache**

In `server/README.md`, document:

- `QWEN3TTS_ICL_CACHE_SIZE`
- what is cached
- that JSON speaker embeddings were already cached separately
- that the server still uses the synthesis lock

- [ ] **Step 5: Full verification sweep**

Run:

```bash
git diff --check
cmake --build build-perf -j
ctest --test-dir build-perf --output-on-failure
python3 server/tests/test_icl_cache.py
```

Expected:

- no whitespace errors
- build succeeds
- model-free tests pass
- model-dependent failures are either fixed or explicitly documented as missing local model/reference fixtures

- [ ] **Step 6: Final benchmark capture if models are available**

Capture before/after JSON records from the same command, same seed, same backend, and same `--max-tokens`. Save the numbers in the final implementation summary; do not commit generated WAV/JSON files unless the user asks.

- [ ] **Step 7: Commit**

```bash
git add README.md server/README.md docs/superpowers/specs/2026-05-04-cross-platform-qwen3-tts-performance-design.md
git commit -m "docs: document qwen3 tts performance controls"
```

## Execution Notes

- Prefer Task 1 and Task 2 as the first implementation PR. They are low-risk and make the rest measurable.
- Treat Task 3 as conditional. If graph allocation is not visible in benchmark output, skip it.
- Treat Task 4 as a separate PR if Task 1/2 are already useful; it expands API surface and deserves focused review.
- Do not enable CUDA-only graph capture in this plan.
- Do not change sampling behavior to make graph fusion easier.
