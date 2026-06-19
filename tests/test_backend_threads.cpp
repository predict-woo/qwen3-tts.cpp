#include "common/backend_threads.h"
#include "pipeline/qwen3_tts.h"
#include "ggml-backend.h"

#include <cassert>
#include <string>

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

    qwen3_tts::Qwen3TTS engine_a;
    qwen3_tts::Qwen3TTS engine_b;
    engine_a.set_n_threads(8);
    engine_b.set_n_threads(5);
    assert(qwen3_tts::get_default_backend_n_threads() == 5);

    qwen3_tts::tts_params no_override_params;
    assert(no_override_params.n_threads == 0);
    unloaded_result = engine_a.synthesize_with_voice("engine a default", nullptr, 0, no_override_params);
    assert(!unloaded_result.success);
    assert(qwen3_tts::get_default_backend_n_threads() == 8);

    unloaded_result = engine_b.synthesize_with_voice("engine b default", nullptr, 0, no_override_params);
    assert(!unloaded_result.success);
    assert(qwen3_tts::get_default_backend_n_threads() == 5);

    qwen3_tts::Qwen3TTS prompt_engine;
    qwen3_tts::icl_prompt wrong_prompt;
    wrong_prompt.expected_hidden_size = 2048;
    wrong_prompt.expected_n_codebooks = 16;
    wrong_prompt.speaker_embedding.assign((size_t) wrong_prompt.expected_hidden_size, 0.0f);
    auto prompt_result = prompt_engine.synthesize_with_icl_prompt(
        "prompt mismatch", wrong_prompt, no_override_params);
    assert(!prompt_result.success);
    assert(prompt_result.error_msg.find("hidden_size mismatch") != std::string::npos);

    qwen3_tts::icl_prompt wrong_codes_prompt;
    wrong_codes_prompt.expected_hidden_size = 1024;
    wrong_codes_prompt.expected_n_codebooks = 8;
    wrong_codes_prompt.speaker_embedding.assign((size_t) wrong_codes_prompt.expected_hidden_size, 0.0f);
    wrong_codes_prompt.ref_text = "reference";
    wrong_codes_prompt.n_ref_frames = 1;
    wrong_codes_prompt.ref_codes.assign((size_t) wrong_codes_prompt.expected_n_codebooks, 0);
    prompt_result = prompt_engine.synthesize_with_icl_prompt(
        "codebook mismatch", wrong_codes_prompt, no_override_params);
    assert(!prompt_result.success);
    assert(prompt_result.error_msg.find("n_codebooks mismatch") != std::string::npos);

    return 0;
}
