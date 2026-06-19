#include "common/backend_threads.h"
#include "pipeline/qwen3_tts.h"
#include "ggml-backend.h"

#include <cassert>

int main() {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(cpu != nullptr);

    assert(qwen3_tts::apply_backend_n_threads(cpu, 1));
    assert(qwen3_tts::apply_backend_n_threads(cpu, 2));
    assert(!qwen3_tts::apply_backend_n_threads(nullptr, 2));
    assert(!qwen3_tts::apply_backend_n_threads(cpu, 0));
    assert(!qwen3_tts::apply_backend_n_threads(cpu, -1));

    qwen3_tts::set_default_backend_n_threads(3);
    assert(qwen3_tts::get_default_backend_n_threads() == 3);
    qwen3_tts::set_default_backend_n_threads(0);
    assert(qwen3_tts::get_default_backend_n_threads() == 3);

    ggml_backend_free(cpu);

    qwen3_tts::Qwen3TTS engine;
    engine.set_n_threads(8);
    assert(qwen3_tts::get_default_backend_n_threads() == 8);

    qwen3_tts::tts_params default_params;
    assert(default_params.n_threads == 0);
    auto unloaded_result = engine.synthesize_with_voice("thread regression", nullptr, 0, default_params);
    assert(!unloaded_result.success);
    assert(qwen3_tts::get_default_backend_n_threads() == 8);

    qwen3_tts::tts_params override_params;
    override_params.n_threads = 6;
    unloaded_result = engine.synthesize_with_voice("thread override", nullptr, 0, override_params);
    assert(!unloaded_result.success);
    assert(qwen3_tts::get_default_backend_n_threads() == 6);

    return 0;
}
