#pragma once

#include "ggml-backend.h"

#include <cstdint>

namespace qwen3_tts {

bool apply_backend_n_threads(ggml_backend_t backend, int32_t n_threads);
void set_default_backend_n_threads(int32_t n_threads);
int32_t get_default_backend_n_threads();

} // namespace qwen3_tts
