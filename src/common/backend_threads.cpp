#include "common/backend_threads.h"

#include "ggml-cpu.h"

#include <atomic>

namespace qwen3_tts {

namespace {
std::atomic<int32_t> g_default_backend_n_threads{4};
}

bool apply_backend_n_threads(ggml_backend_t backend, int32_t n_threads) {
    if (!backend || n_threads <= 0) {
        return false;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) {
        auto set_n_threads =
            reinterpret_cast<ggml_backend_set_n_threads_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        if (set_n_threads) {
            set_n_threads(backend, n_threads);
            return true;
        }
    }

    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
        return true;
    }

    return false;
}

void set_default_backend_n_threads(int32_t n_threads) {
    if (n_threads > 0) {
        g_default_backend_n_threads.store(n_threads, std::memory_order_relaxed);
    }
}

int32_t get_default_backend_n_threads() {
    return g_default_backend_n_threads.load(std::memory_order_relaxed);
}

} // namespace qwen3_tts
