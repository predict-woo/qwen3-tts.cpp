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
    assert(!qwen3_tts::apply_backend_n_threads(cpu, -1));

    qwen3_tts::set_default_backend_n_threads(3);
    assert(qwen3_tts::get_default_backend_n_threads() == 3);
    qwen3_tts::set_default_backend_n_threads(0);
    assert(qwen3_tts::get_default_backend_n_threads() == 3);

    ggml_backend_free(cpu);
    return 0;
}
