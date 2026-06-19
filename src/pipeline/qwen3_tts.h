#pragma once

#include "tokenizer/text_tokenizer.h"
#include "transformer/tts_transformer.h"
#include "encoder/audio_tokenizer_encoder.h"
#include "encoder/audio_codec_encoder.h"
#include "decoder/audio_tokenizer_decoder.h"

#include <string>
#include <vector>
#include <functional>
#include <cstdint>

namespace qwen3_tts {

// TTS generation parameters
struct tts_params {
    // Maximum number of audio tokens to generate
    int32_t max_audio_tokens = 2048;
    
    // Temperature for sampling (0 = greedy)
    float temperature = 0.9f;
    
    // Top-p sampling
    float top_p = 1.0f;
    
    // Top-k sampling (0 = disabled)
    int32_t top_k = 50;
    
    // Number of threads. 0 means use the engine's configured default.
    int32_t n_threads = 0;
    
    // Print progress during generation
    bool print_progress = false;
    
    // Print timing information
    bool print_timing = true;
    
    // Repetition penalty for CB0 token generation (HuggingFace style)
    float repetition_penalty = 1.05f;

    // Language ID for codec (2050=en, 2069=ru, 2055=zh, 2058=ja, 2064=ko, 2053=de, 2061=fr, 2054=es)
    int32_t language_id = 2050;

    // RNG seed for reproducible output (-1 = random)
    int32_t seed = -1;

    // Force f32 accumulation in matmul (default: true for quality)
    bool f32_acc = true;

    // Voice steering instruction (e.g. "Speak happily", "Use a deep voice")
    std::string instruction;

    // Reference transcript for ICL voice cloning. When non-empty alongside a
    // reference audio source (`-r` / `synthesize_with_voice`), the pipeline
    // encodes the reference audio with the Mimi codec and threads both the
    // codes and the transcript into the talker prefill (Qwen's intended
    // cloning mode for Base variants). If empty, voice cloning falls back to
    // x-vector-only conditioning.
    std::string ref_text;
};

// TTS generation result
struct tts_result {
    // Generated audio samples (24kHz, mono)
    std::vector<float> audio;
    
    // Sample rate
    int32_t sample_rate = 24000;
    
    // Success flag
    bool success = false;
    
    // Error message if failed
    std::string error_msg;
    
    // Timing info (in milliseconds)
    int64_t t_load_ms = 0;
    int64_t t_tokenize_ms = 0;
    int64_t t_encode_ms = 0;
    int64_t t_generate_ms = 0;
    int64_t t_decode_ms = 0;
    int64_t t_total_ms = 0;

    // Process memory snapshots (bytes)
    uint64_t mem_rss_start_bytes = 0;
    uint64_t mem_rss_end_bytes = 0;
    uint64_t mem_rss_peak_bytes = 0;
    uint64_t mem_phys_start_bytes = 0;
    uint64_t mem_phys_end_bytes = 0;
    uint64_t mem_phys_peak_bytes = 0;

#ifdef QWEN3_TTS_TIMING
    bool has_detailed_timing = false;
    tts_timing detailed_timing = {};
#endif
    
};

// Progress callback type
using tts_progress_callback_t = std::function<void(int tokens_generated, int max_tokens)>;

// Main TTS class that orchestrates the full pipeline
class Qwen3TTS {
public:
    Qwen3TTS();
    ~Qwen3TTS();
    
    // Load all models from directory
    // model_dir should contain: transformer.gguf, tokenizer.gguf, vocoder.gguf
    // tts_model/tokenizer_model override auto-detection when non-empty
    bool load_models(const std::string & model_dir,
                     const std::string & tts_model = std::string(),
                     const std::string & tokenizer_model = std::string());
    
    // Generate speech from text
    // text: input text to synthesize
    // params: generation parameters
    tts_result synthesize(const std::string & text,
                          const tts_params & params = tts_params());
    
    // Generate speech with voice cloning
    // text: input text to synthesize
    // reference_audio: path to reference audio file (WAV, 24kHz)
    // params: generation parameters
    tts_result synthesize_with_voice(const std::string & text,
                                      const std::string & reference_audio,
                                      const tts_params & params = tts_params());
    
    // Generate speech with voice cloning from samples
    // text: input text to synthesize
    // ref_samples: reference audio samples (24kHz, mono, normalized to [-1, 1])
    // n_ref_samples: number of reference samples
    // params: generation parameters
    tts_result synthesize_with_voice(const std::string & text,
                                      const float * ref_samples, int32_t n_ref_samples,
                                      const tts_params & params = tts_params());
    
    // Extract speaker embedding from raw audio samples (for caching)
    // ref_samples: 24kHz mono float32 normalized to [-1, 1]
    // embedding: output vector (resized to hidden_size, typically 1024)
    // Returns true on success
    bool extract_speaker_embedding(const float * ref_samples, int32_t n_ref_samples,
                                   std::vector<float> & embedding,
                                   const tts_params & params = tts_params());

    // Synthesize with pre-computed speaker embedding (skips encoder)
    // embedding: speaker embedding from extract_speaker_embedding()
    // embedding_size: must match hidden_size (typically 1024)
    tts_result synthesize_with_embedding(const std::string & text,
                                          const float * embedding, int32_t embedding_size,
                                          const tts_params & params = tts_params());

    // Set progress callback
    void set_progress_callback(tts_progress_callback_t callback);

    // Set default backend thread count and apply it to loaded backends where supported.
    void set_n_threads(int32_t n_threads);

    // Model metadata
    const std::string & get_model_type() const;  // "base" | "custom_voice" | "voice_design"
    const std::string & get_model_size() const;  // e.g. "0b6" | "1b7" (empty on older GGUFs)
    bool has_speaker_encoder() const;            // true if ECAPA-TDNN x-vector path works

    // Preset voice table (CustomVoice / VoiceDesign). Empty for Base and for
    // GGUFs converted before preset-metadata support.
    const std::vector<std::string> & get_speaker_names() const;
    const std::vector<int32_t>     & get_speaker_ids() const;
    const std::vector<std::string> & get_speaker_dialects() const;

    // Look up a preset voice by name. Returns -1 if not found.
    int32_t get_speaker_id(const std::string & name) const;

    // Resolve a preset voice to the speaker embedding (codec_embd row at the
    // preset's token ID). Returns false if the name is unknown or the
    // underlying tensor is missing. On success, `out` is resized to
    // hidden_size and filled with float32 values.
    bool get_speaker_embedding(const std::string & name, std::vector<float> & out);

    // Get error message
    const std::string & get_error() const { return error_msg_; }

    const std::string & get_tts_model_path() const { return tts_model_path_; }
    const std::string & get_decoder_model_path() const { return decoder_model_path_; }

    // Check if models are loaded
    bool is_loaded() const { return models_loaded_; }
    
private:
    int32_t effective_n_threads(const tts_params & params) const;

    tts_result synthesize_internal(const std::string & text,
                                   const float * speaker_embedding,
                                   const tts_params & params,
                                   tts_result & result,
                                   const int32_t * ref_codes = nullptr,
                                   int32_t n_ref_frames = 0);

    TextTokenizer tokenizer_;
    TTSTransformer transformer_;
    AudioTokenizerEncoder audio_encoder_;
    AudioCodecEncoder codec_encoder_;
    AudioTokenizerDecoder audio_decoder_;
    
    bool models_loaded_ = false;
    bool encoder_loaded_ = false;
    bool codec_encoder_loaded_ = false;
    bool transformer_loaded_ = false;
    bool decoder_loaded_ = false;
    bool low_mem_mode_ = false;
    int32_t n_threads_ = 4;
    std::string error_msg_;
    std::string tts_model_path_;
    std::string decoder_model_path_;
    tts_progress_callback_t progress_callback_;
};

// Utility: Load audio file (WAV format)
bool load_audio_file(const std::string & path, std::vector<float> & samples, 
                     int & sample_rate);

// Utility: Save audio file (WAV format)
bool save_audio_file(const std::string & path, const std::vector<float> & samples,
                     int sample_rate);

} // namespace qwen3_tts
