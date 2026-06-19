/* qwen3tts_c_api.h — C API wrapper for qwen3-tts.cpp (Nim FFI) */
#ifndef QWEN3TTS_C_API_H
#define QWEN3TTS_C_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct Qwen3Tts Qwen3Tts;
typedef struct Qwen3TtsIclPrompt Qwen3TtsIclPrompt;

/* Generation parameters */
typedef struct Qwen3TtsParams {
    int32_t max_audio_tokens;    /* default: 2048 */
    float   temperature;         /* default: 0.9, 0=greedy */
    float   top_p;               /* default: 1.0 */
    int32_t top_k;               /* default: 50, 0=disabled */
    int32_t n_threads;           /* default: 0 (use handle default) */
    float   repetition_penalty;  /* default: 1.05 */
    int32_t language_id;         /* 2050=en, 2058=ja, 2055=zh, etc. */
} Qwen3TtsParams;

/* Generated audio result */
typedef struct Qwen3TtsAudio {
    const float* samples;  /* PCM float32 mono */
    int32_t n_samples;
    int32_t sample_rate;   /* always 24000 */
} Qwen3TtsAudio;

/* Fill params with defaults */
void qwen3_tts_default_params(Qwen3TtsParams* params);

/* Create TTS engine and load models from directory.
 * model_dir must contain qwen3-tts-0.6b-f16.gguf and
 * qwen3-tts-tokenizer-f16.gguf.
 * Returns NULL on failure. */
Qwen3Tts* qwen3_tts_create(const char* model_dir, int32_t n_threads);

/* Check if models are loaded */
int qwen3_tts_is_loaded(const Qwen3Tts* tts);

/* Synthesize text to audio. Returns NULL on failure.
 * Caller must free with qwen3_tts_free_audio(). */
Qwen3TtsAudio* qwen3_tts_synthesize(
    Qwen3Tts* tts,
    const char* text,
    const Qwen3TtsParams* params);

/* Get sample rate (always 24000) */
int32_t qwen3_tts_sample_rate(const Qwen3Tts* tts);

/* Free generated audio */
void qwen3_tts_free_audio(Qwen3TtsAudio* audio);

/* Destroy TTS engine */
void qwen3_tts_destroy(Qwen3Tts* tts);

/* Synthesize with voice cloning from WAV file.
 * reference_audio_path: path to reference WAV (24kHz mono recommended).
 * Returns NULL on failure. Caller must free with qwen3_tts_free_audio(). */
Qwen3TtsAudio* qwen3_tts_synthesize_with_voice_file(
    Qwen3Tts* tts,
    const char* text,
    const char* reference_audio_path,
    const Qwen3TtsParams* params);

/* Synthesize with voice cloning from raw samples.
 * ref_samples: 24kHz mono float32 normalized to [-1, 1].
 * Returns NULL on failure. Caller must free with qwen3_tts_free_audio(). */
Qwen3TtsAudio* qwen3_tts_synthesize_with_voice_samples(
    Qwen3Tts* tts,
    const char* text,
    const float* ref_samples,
    int32_t n_ref_samples,
    const Qwen3TtsParams* params);

/* Extract speaker embedding from WAV file (for caching).
 * embedding_out: caller-allocated buffer for the embedding.
 * max_size: size of embedding_out in floats.
 * Returns the actual embedding size (typically 1024), or -1 on failure. */
int32_t qwen3_tts_extract_embedding_file(
    Qwen3Tts* tts,
    const char* reference_audio_path,
    float* embedding_out,
    int32_t max_size);

/* Synthesize with pre-computed speaker embedding (skips encoder).
 * embedding: speaker embedding from qwen3_tts_extract_embedding_file().
 * embedding_size: must match the size returned by extract.
 * Returns NULL on failure. Caller must free with qwen3_tts_free_audio(). */
Qwen3TtsAudio* qwen3_tts_synthesize_with_embedding(
    Qwen3Tts* tts,
    const char* text,
    const float* embedding,
    int32_t embedding_size,
    const Qwen3TtsParams* params);

/* Synthesize with in-context-learning (ICL) voice cloning.
 * Encodes the reference audio through the Mimi codec and threads the
 * resulting codes plus the reference transcript into the talker prefill —
 * Qwen's intended cloning mode for Base variants.
 *
 * reference_audio_path: path to reference WAV (24kHz mono recommended).
 * reference_text:       transcript of the reference audio (same language).
 * Returns NULL on failure. Caller must free with qwen3_tts_free_audio(). */
Qwen3TtsAudio* qwen3_tts_synthesize_icl_file(
    Qwen3Tts* tts,
    const char* text,
    const char* reference_audio_path,
    const char* reference_text,
    const Qwen3TtsParams* params);

/* Prepare reusable in-context-learning prompt state from a reference WAV and
 * transcript. Caller must free the returned handle with
 * qwen3_tts_free_icl_prompt(). */
Qwen3TtsIclPrompt* qwen3_tts_prepare_icl_prompt_file(
    Qwen3Tts* tts,
    const char* reference_audio_path,
    const char* reference_text,
    const Qwen3TtsParams* params);

/* Synthesize using a prompt returned by qwen3_tts_prepare_icl_prompt_file().
 * Returns NULL on failure. Caller must free with qwen3_tts_free_audio(). */
Qwen3TtsAudio* qwen3_tts_synthesize_with_icl_prompt(
    Qwen3Tts* tts,
    const char* text,
    const Qwen3TtsIclPrompt* prompt,
    const Qwen3TtsParams* params);

/* Free a prepared ICL prompt handle. */
void qwen3_tts_free_icl_prompt(Qwen3TtsIclPrompt* prompt);

/* Get last error message (or empty string) */
const char* qwen3_tts_get_error(const Qwen3Tts* tts);

/* --- Model metadata + preset voices ----------------------------------- */

/* Returns the model variant: "base", "custom_voice", or "voice_design".
 * Pointer is owned by the engine and valid until qwen3_tts_destroy(). */
const char* qwen3_tts_model_type(const Qwen3Tts* tts);

/* Returns the model size tag: "0b6", "1b7", etc. Empty on older GGUFs. */
const char* qwen3_tts_model_size(const Qwen3Tts* tts);

/* Resolved model paths used by each role that can affect prepared ICL output.
 * Roles may currently point to the same GGUF path but are exposed separately
 * for stable cache keys across future split-model layouts. */
const char* qwen3_tts_tts_model_path(const Qwen3Tts* tts);
const char* qwen3_tts_speaker_encoder_model_path(const Qwen3Tts* tts);
const char* qwen3_tts_codec_encoder_model_path(const Qwen3Tts* tts);
const char* qwen3_tts_tokenizer_decoder_model_path(const Qwen3Tts* tts);

/* Returns 1 if the model ships an ECAPA-TDNN speaker encoder, else 0. */
int qwen3_tts_has_speaker_encoder(const Qwen3Tts* tts);

/* Number of preset voices in the loaded model (0 for Base variants). */
int32_t qwen3_tts_speaker_count(const Qwen3Tts* tts);

/* Name of the i-th preset voice, or NULL if i is out of range.
 * Pointer is owned by the engine. */
const char* qwen3_tts_speaker_name(const Qwen3Tts* tts, int32_t i);

/* Dialect tag of the i-th preset voice (e.g. "sichuan_dialect").
 * Empty string if the preset is not a dialect, NULL if i is out of range. */
const char* qwen3_tts_speaker_dialect(const Qwen3Tts* tts, int32_t i);

/* Write the preset voice's speaker embedding into embedding_out. Returns the
 * embedding size (hidden_size) on success, -1 if the name is unknown or
 * max_size is insufficient. */
int32_t qwen3_tts_get_speaker_embedding(
    Qwen3Tts* tts,
    const char* speaker_name,
    float* embedding_out,
    int32_t max_size);

#ifdef __cplusplus
}
#endif

#endif /* QWEN3TTS_C_API_H */
