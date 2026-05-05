# Cross-Platform Qwen3-TTS Performance Design

## Goal

Adapt the useful performance lessons from `andimarafioti/faster-qwen3-tts` into `qwen3-tts.cpp` without making the first optimization pass NVIDIA-only.

The first implementation track should improve or validate speedups that can help GGML CPU, Metal, Vulkan, CUDA, and other scheduler-backed backends. CUDA graph capture can remain a later backend-specific track after the cross-platform hot paths are measured and cleaned up.

## Context

`faster-qwen3-tts` gets its major speedups from CUDA graph capture, static KV cache buffers, pre-allocated tensors, and fixed-shape padded attention. Its README reports large RTF and TTFA improvements on CUDA systems and explains that Qwen3-TTS spends substantial time launching many small kernels per decode step.

That exact implementation depends on PyTorch CUDA graph APIs and NVIDIA CUDA behavior. It is not directly portable to a GGML C++ runtime, and it does not help macOS Metal, Vulkan, or CPU users.

The reusable lesson is still relevant: avoid dynamic work inside the per-frame decode loop. In this repository, the local transformer path currently rebuilds, allocates, computes, and resets GGML graphs in hot paths, including the talker step and the code predictor prefill plus 14 autoregressive predictor steps per audio frame.

## Target

Optimize for cross-platform local inference first:

- CPU users should benefit from correctly wired thread settings and lower scheduler overhead.
- Metal, Vulkan, CUDA, and other GGML backends should benefit from less graph lifecycle churn where graph shape allows it.
- Server users should benefit from reuse of speaker and reference prompt work across repeated requests.

## Non-Goals

- Do not start by porting PyTorch `torch.cuda.CUDAGraph`.
- Do not make CUDA the only fast path in the first pass.
- Do not redesign model weights, GGUF conversion, or public inference semantics.
- Do not accept quality regressions or token-shape changes without parity tests.
- Do not treat streaming as a throughput fix by itself. Streaming can reduce time-to-first-audio, but total RTF still depends on the same decode hot path.

## Proposed Phases

### Phase 1: Baseline and Profiling

Create a repeatable benchmark path before changing the runtime. The benchmark should record:

- total synthesis wall time
- generated audio duration and RTF
- TTFA where streaming is available
- talker graph build, allocation, data transfer, and compute time
- code predictor graph build, allocation, data transfer, and compute time
- backend, quantization, thread count, model size, prompt length, and max token settings

The existing timing counters in `TTSTimingStats` already expose much of this information. The implementation plan should first determine whether those counters are sufficient or whether a small benchmark wrapper is needed.

Success criteria:

- One documented CPU benchmark command.
- One documented accelerator benchmark command when a backend is available.
- Timing output that can show whether graph lifecycle overhead is material.

### Phase 2: Runtime Knob Cleanup

Fix low-risk runtime controls before attempting deeper graph changes.

The C API exposes `n_threads`, and the CLI/server pass thread values, but the current create path ignores the constructor argument and comments that threads are set per call. The implementation should trace whether per-call thread count is actually applied to the GGML CPU backend. If not, wire it through deliberately.

The server should expose the same effective knobs as the CLI where practical:

- backend selection
- thread count
- max audio tokens
- quantization/model selection guidance
- timing/profiling mode when compiled with timing enabled

Success criteria:

- CPU thread count has a verified effect or is clearly documented as unsupported.
- CLI and server behavior are consistent enough that benchmark results are comparable.
- Documentation describes practical speed knobs without overstating platform-specific features.

### Phase 3: GGML Graph Lifecycle Optimization

Use the benchmark data to reduce repeated scheduler work in the hottest safe path.

Start with the code predictor. It runs once per generated frame and includes one prefill graph followed by 14 step graphs. This makes it the best candidate for reusable graph reservations or a small cache of stable graph variants.

The implementation should investigate these options in order:

1. Reserve scheduler memory with a max-shape graph where GGML supports it.
2. Reuse graph allocation for stable-shape predictor graphs.
3. Keep input tensors in dedicated contexts where GGML allows static allocation.
4. Only then consider larger structural changes such as unrolling the predictor sequence into fewer graph executions.

Apply the same pattern to the talker step only if profiling shows graph build or allocation time is meaningful there.

Success criteria:

- The graph optimization is guarded by tests or benchmarks that detect regressions.
- Generated output remains compatible with the existing sampling and token generation behavior.
- Backends that cannot benefit still run correctly.

### Phase 4: Server Reuse

Add server-side reuse for work that does not need to be repeated on every request.

Useful reuse candidates:

- speaker embeddings
- reference audio codec tokens
- reference text tokenization
- prompt structures for repeated voice clone or CustomVoice requests

This phase should not change the core transformer math. It should reduce repeated setup latency in server mode and make repeated requests more comparable to the warm-cache behavior described by `faster-qwen3-tts`.

Success criteria:

- Repeated requests for the same voice avoid redundant prompt/reference preprocessing.
- Cache keys include all inputs that affect output correctness.
- Cache size and invalidation behavior are explicit.

### Phase 5: Static-Shape Decode Feasibility

After phases 1-4, decide whether to attempt the deeper adaptation of the faster repo's static-cache idea.

This would mean building a fixed-shape decode path with static KV/cache buffers and explicit masks so that graph structure remains stable while logical sequence length changes. It is the closest cross-platform analog to the faster repo's static cache plus CUDA graph capture design, but it is also the riskiest part.

This phase should start as a feasibility branch, not as an immediate production change.

Success criteria:

- Clear proof that dynamic graph shape is a limiting factor after easier fixes.
- Prefix-parity or perceptual-quality tests for generated codec tokens/audio.
- Backend-specific fallbacks for unsupported cases.

## Architecture

The design should keep optimization logic close to existing runtime boundaries:

- `TTSTransformer` owns graph construction, KV cache state, timing counters, and scheduler interaction.
- The C API and server own user-facing params and should not know graph-cache internals.
- Benchmark/documentation helpers should consume public CLI/server interfaces where possible.

If a reusable graph cache is introduced, it should be a small internal helper with explicit ownership:

- keyed by graph role, step index, shape, backend-relevant dimensions, and model config
- invalidated when context size, backend, or model changes
- hidden from public API consumers

## Testing

The implementation plan should include focused tests at three levels:

- Unit or integration tests for thread-param propagation and cache key behavior.
- Existing transformer tests to catch generation regressions.
- Benchmark comparisons before and after graph lifecycle changes.

Performance tests should report numbers but should not be brittle pass/fail checks unless a stable local fixture already exists.

## Risks

- GGML graph tensors are not always reusable in the same way across backends.
- `n_past` changes graph shape in the current talker path, which may limit reuse.
- Static-shape attention masking could change numerical behavior even if the algorithm is equivalent.
- Server caches can return wrong results if cache keys omit reference audio, reference text, language, speaker, or model identity.
- Existing worktree changes are present, so implementation should keep future edits narrowly staged.

## Recommendation

Proceed with phases 1-4 as the first implementation plan:

1. make performance measurement reproducible
2. fix easy runtime knobs
3. optimize GGML graph lifecycle where profiling proves overhead
4. add safe server-side reuse for repeated prompt/reference work

Treat full static-shape decode as a later feasibility project. It is likely the closest conceptual match to `faster-qwen3-tts`, but it should be justified by measurements after the lower-risk cross-platform work is complete.
