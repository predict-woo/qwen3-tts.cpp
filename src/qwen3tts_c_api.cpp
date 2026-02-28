/* qwen3tts_c_api.cpp — C API wrapper for qwen3-tts.cpp */

#include "qwen3tts_c_api.h"
#include "qwen3_tts.h"

#include <cstdlib>
#include <cstring>

struct Qwen3Tts {
    qwen3_tts::Qwen3TTS engine;
    std::string last_error;
};

/* Internal helpers (C++ linkage, not exported) */

static qwen3_tts::tts_params convert_params(const Qwen3TtsParams* params) {
    qwen3_tts::tts_params p;
    p.max_audio_tokens = params->max_audio_tokens;
    p.temperature = params->temperature;
    p.top_p = params->top_p;
    p.top_k = params->top_k;
    p.n_threads = params->n_threads;
    p.repetition_penalty = params->repetition_penalty;
    p.language_id = params->language_id;
    p.print_progress = false;
    p.print_timing = false;
    return p;
}

static Qwen3TtsAudio* make_audio_result(const qwen3_tts::tts_result& result) {
    auto* audio = static_cast<Qwen3TtsAudio*>(
        std::malloc(sizeof(Qwen3TtsAudio)));
    if (!audio) return nullptr;

    auto n = static_cast<int32_t>(result.audio.size());
    auto* buf = static_cast<float*>(std::malloc(n * sizeof(float)));
    if (!buf) {
        std::free(audio);
        return nullptr;
    }
    std::memcpy(buf, result.audio.data(), n * sizeof(float));

    audio->samples = buf;
    audio->n_samples = n;
    audio->sample_rate = result.sample_rate;
    return audio;
}

extern "C" {

void qwen3_tts_default_params(Qwen3TtsParams* params) {
    if (!params) return;
    params->max_audio_tokens = 4096;
    params->temperature = 0.9f;
    params->top_p = 1.0f;
    params->top_k = 50;
    params->n_threads = 4;
    params->repetition_penalty = 1.05f;
    params->language_id = 2050;  /* English */
}

Qwen3Tts* qwen3_tts_create(const char* model_dir, int32_t n_threads) {
    if (!model_dir) return nullptr;

    auto* tts = new (std::nothrow) Qwen3Tts();
    if (!tts) return nullptr;

    if (!tts->engine.load_models(model_dir)) {
        tts->last_error = tts->engine.get_error();
        delete tts;
        return nullptr;
    }
    (void)n_threads;  /* threads are per-synthesis via params */
    return tts;
}

int qwen3_tts_is_loaded(const Qwen3Tts* tts) {
    return (tts && tts->engine.is_loaded()) ? 1 : 0;
}

Qwen3TtsAudio* qwen3_tts_synthesize(
    Qwen3Tts* tts,
    const char* text,
    const Qwen3TtsParams* params)
{
    if (!tts || !text || !params) return nullptr;

    auto p = convert_params(params);
    auto result = tts->engine.synthesize(std::string(text), p);
    if (!result.success) {
        tts->last_error = result.error_msg;
        return nullptr;
    }
    return make_audio_result(result);
}

Qwen3TtsAudio* qwen3_tts_synthesize_with_voice_file(
    Qwen3Tts* tts,
    const char* text,
    const char* reference_audio_path,
    const Qwen3TtsParams* params)
{
    if (!tts || !text || !reference_audio_path || !params) return nullptr;

    auto p = convert_params(params);
    auto result = tts->engine.synthesize_with_voice(
        std::string(text), std::string(reference_audio_path), p);
    if (!result.success) {
        tts->last_error = result.error_msg;
        return nullptr;
    }
    return make_audio_result(result);
}

Qwen3TtsAudio* qwen3_tts_synthesize_with_voice_samples(
    Qwen3Tts* tts,
    const char* text,
    const float* ref_samples,
    int32_t n_ref_samples,
    const Qwen3TtsParams* params)
{
    if (!tts || !text || !ref_samples || n_ref_samples <= 0 || !params)
        return nullptr;

    auto p = convert_params(params);
    auto result = tts->engine.synthesize_with_voice(
        std::string(text), ref_samples, n_ref_samples, p);
    if (!result.success) {
        tts->last_error = result.error_msg;
        return nullptr;
    }
    return make_audio_result(result);
}

int32_t qwen3_tts_sample_rate(const Qwen3Tts* tts) {
    (void)tts;
    return 24000;
}

void qwen3_tts_free_audio(Qwen3TtsAudio* audio) {
    if (!audio) return;
    std::free(const_cast<float*>(audio->samples));
    std::free(audio);
}

void qwen3_tts_destroy(Qwen3Tts* tts) {
    delete tts;
}

const char* qwen3_tts_get_error(const Qwen3Tts* tts) {
    if (!tts) return "";
    return tts->last_error.c_str();
}

} /* extern "C" */
