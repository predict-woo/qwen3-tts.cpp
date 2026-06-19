#include "pipeline/qwen3_tts.h"
#include "common/gguf_loader.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <cmath>
#include <fstream>
#include <cstdint>
#include <cstdlib>

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace qwen3_tts {

static int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct process_memory_snapshot {
    uint64_t rss_bytes = 0;
    uint64_t phys_footprint_bytes = 0;
};

static bool get_process_memory_snapshot(process_memory_snapshot & out) {
#ifdef __APPLE__
    mach_task_basic_info_data_t basic_info = {};
    mach_msg_type_number_t basic_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&basic_info), &basic_count) != KERN_SUCCESS) {
        return false;
    }
    out.rss_bytes = (uint64_t) basic_info.resident_size;

    task_vm_info_data_t vm_info = {};
    mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&vm_info), &vm_count) == KERN_SUCCESS) {
        out.phys_footprint_bytes = (uint64_t) vm_info.phys_footprint;
    } else {
        out.phys_footprint_bytes = out.rss_bytes;
    }
    return true;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return false;
    }
    out.rss_bytes = (uint64_t)pmc.WorkingSetSize;
    out.phys_footprint_bytes = out.rss_bytes;
    return true;
#else
    struct rusage usage = {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return false;
    }
    out.rss_bytes = (uint64_t) usage.ru_maxrss * 1024ULL;
    out.phys_footprint_bytes = out.rss_bytes;
    return true;
#endif
}

static std::string format_bytes(uint64_t bytes) {
    static const char * units[] = { "B", "KB", "MB", "GB", "TB" };
    double val = (double) bytes;
    int unit = 0;
    while (val >= 1024.0 && unit < 4) {
        val /= 1024.0;
        ++unit;
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f %s", val, units[unit]);
    return std::string(buf);
}

static void log_memory_usage(const char * label) {
    process_memory_snapshot mem;
    if (!get_process_memory_snapshot(mem)) {
        fprintf(stderr, "  [mem] %-24s unavailable\n", label);
        return;
    }
    fprintf(stderr, "  [mem] %-24s rss=%s  phys=%s\n",
            label, format_bytes(mem.rss_bytes).c_str(),
            format_bytes(mem.phys_footprint_bytes).c_str());
}

static void resample_linear(const float * input, int input_len, int input_rate,
                            std::vector<float> & output, int output_rate) {
    double ratio = (double)input_rate / output_rate;
    int output_len = (int)((double)input_len / ratio);
    output.resize(output_len);
    
    for (int i = 0; i < output_len; ++i) {
        double src_idx = i * ratio;
        int idx0 = (int)src_idx;
        int idx1 = idx0 + 1;
        double frac = src_idx - idx0;
        
        if (idx1 >= input_len) {
            output[i] = input[input_len - 1];
        } else {
            output[i] = (float)((1.0 - frac) * input[idx0] + frac * input[idx1]);
        }
    }
}

Qwen3TTS::Qwen3TTS() = default;

Qwen3TTS::~Qwen3TTS() = default;

bool Qwen3TTS::load_models(const std::string & model_dir,
                           const std::string & tts_model,
                           const std::string & tokenizer_model) {
    int64_t t_start = get_time_ms();
    log_memory_usage("load/start");

    transformer_.unload_model();
    audio_decoder_.unload_model();
    transformer_loaded_ = false;
    decoder_loaded_ = false;

    // Construct model paths — explicit paths override auto-detection
    if (!tts_model.empty()) {
        tts_model_path_ = model_dir + "/" + tts_model;
    } else {
        // Prefer quantized (q8_0) over full-precision (f16)
        std::string q8_path = model_dir + "/qwen3-tts-0.6b-q8_0.gguf";
        std::string f16_path = model_dir + "/qwen3-tts-0.6b-f16.gguf";
        FILE * q8_check = fopen(q8_path.c_str(), "r");
        if (q8_check) {
            fclose(q8_check);
            tts_model_path_ = q8_path;
        } else {
            tts_model_path_ = f16_path;
        }
    }
    if (!tokenizer_model.empty()) {
        decoder_model_path_ = model_dir + "/" + tokenizer_model;
    } else {
        decoder_model_path_ = model_dir + "/qwen3-tts-tokenizer-f16.gguf";
    }
    encoder_loaded_ = false;
    transformer_loaded_ = false;
    decoder_loaded_ = false;

    const char * low_mem_env = std::getenv("QWEN3_TTS_LOW_MEM");
    low_mem_mode_ = low_mem_env && low_mem_env[0] != '\0' && low_mem_env[0] != '0';
    if (low_mem_mode_) {
        fprintf(stderr, "  Low-memory mode enabled (lazy decoder + component unloads)\n");
    }
    
    // Load TTS model (contains text tokenizer + transformer for generation)
    fprintf(stderr, "Loading TTS model from %s...\n", tts_model_path_.c_str());

    // Load text tokenizer from TTS model
    int64_t t_tokenizer_start = get_time_ms();
    {
        GGUFLoader loader;
        if (!loader.open(tts_model_path_)) {
            error_msg_ = "Failed to open TTS model: " + loader.get_error();
            return false;
        }
        
        if (!tokenizer_.load_from_gguf(loader.get_ctx())) {
            error_msg_ = "Failed to load text tokenizer: " + tokenizer_.get_error();
            return false;
        }
        fprintf(stderr, "  Text tokenizer loaded: vocab_size=%d (%lld ms)\n",
                tokenizer_.get_config().vocab_size,
                (long long)(get_time_ms() - t_tokenizer_start));
    }
    log_memory_usage("load/after-tokenizer");
    
    // Speaker encoder is loaded lazily on first voice cloning request.
    fprintf(stderr, "  Speaker encoder: deferred (lazy load)\n");
    
    // Load TTS transformer from TTS model
    int64_t t_transformer_start = get_time_ms();
    if (!transformer_.load_model(tts_model_path_)) {
        error_msg_ = "Failed to load TTS transformer: " + transformer_.get_error();
        return false;
    }
    transformer_loaded_ = true;
    fprintf(stderr, "  TTS transformer loaded: hidden_size=%d, n_layers=%d (%lld ms)\n",
            transformer_.get_config().hidden_size, transformer_.get_config().n_layers,
            (long long)(get_time_ms() - t_transformer_start));
    log_memory_usage("load/after-transformer");
    
    if (!low_mem_mode_) {
        // Load vocoder (audio decoder) from tokenizer model
        fprintf(stderr, "Loading vocoder from %s...\n", decoder_model_path_.c_str());
        int64_t t_decoder_start = get_time_ms();
        if (!audio_decoder_.load_model(decoder_model_path_)) {
            error_msg_ = "Failed to load vocoder: " + audio_decoder_.get_error();
            return false;
        }
        decoder_loaded_ = true;
        fprintf(stderr, "  Vocoder loaded: sample_rate=%d, n_codebooks=%d (%lld ms)\n",
                audio_decoder_.get_config().sample_rate, audio_decoder_.get_config().n_codebooks,
                (long long)(get_time_ms() - t_decoder_start));
        log_memory_usage("load/after-vocoder");
    } else {
        fprintf(stderr, "  Vocoder: deferred (lazy load)\n");
    }
    
    models_loaded_ = true;
    
    int64_t t_end = get_time_ms();
    fprintf(stderr, "All models loaded in %lld ms\n", (long long)(t_end - t_start));
    log_memory_usage("load/end");
    
    return true;
}

tts_result Qwen3TTS::synthesize(const std::string & text,
                                 const tts_params & params) {
    tts_result result;
    
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }
    
    // For basic synthesis without voice cloning, we use a zero speaker embedding
    // This will use the model's default voice characteristics
    std::vector<float> zero_embedding(transformer_.get_config().hidden_size, 0.0f);
    
    return synthesize_internal(text, zero_embedding.data(), params, result);
}

tts_result Qwen3TTS::synthesize_with_voice(const std::string & text,
                                            const std::string & reference_audio,
                                            const tts_params & params) {
    tts_result result;
    
    std::vector<float> ref_samples;
    int ref_sample_rate;
    if (!load_audio_file(reference_audio, ref_samples, ref_sample_rate)) {
        result.error_msg = "Failed to load reference audio: " + reference_audio;
        return result;
    }
    
    const int target_rate = 24000;
    if (ref_sample_rate != target_rate) {
        fprintf(stderr, "Resampling audio from %d Hz to %d Hz...\n", ref_sample_rate, target_rate);
        std::vector<float> resampled;
        resample_linear(ref_samples.data(), (int)ref_samples.size(), ref_sample_rate, resampled, target_rate);
        ref_samples = std::move(resampled);
    }
    
    return synthesize_with_voice(text, ref_samples.data(), (int32_t)ref_samples.size(), params);
}

tts_result Qwen3TTS::synthesize_with_voice(const std::string & text,
                                            const float * ref_samples, int32_t n_ref_samples,
                                            const tts_params & params) {
    tts_result result;
    
    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    if (!encoder_loaded_) {
        if (tts_model_path_.empty()) {
            result.error_msg = "Internal error: missing TTS model path for lazy encoder load";
            return result;
        }
        int64_t t_encoder_load_start = get_time_ms();
        if (!audio_encoder_.load_model(tts_model_path_)) {
            result.error_msg = "Failed to load speaker encoder: " + audio_encoder_.get_error();
            return result;
        }
        encoder_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Speaker encoder lazy-loaded in %lld ms\n",
                    (long long)(get_time_ms() - t_encoder_load_start));
            log_memory_usage("voice/after-encoder-load");
        }
    }
    
    int64_t t_encode_start = get_time_ms();
    std::vector<float> speaker_embedding;

    if (!audio_encoder_.encode(ref_samples, n_ref_samples, speaker_embedding)) {
        result.error_msg = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
        return result;
    }
    result.t_encode_ms = get_time_ms() - t_encode_start;

    if (params.print_progress) {
        fprintf(stderr, "Speaker embedding extracted: %zu floats\n", speaker_embedding.size());
    }

    // ICL mode: also encode reference audio to discrete codec codes and thread
    // them + the reference transcript into the talker prefill.
    if (!params.ref_text.empty()) {
        if (!codec_encoder_loaded_) {
            if (decoder_model_path_.empty()) {
                result.error_msg = "Internal error: missing tokenizer model path for codec encoder";
                return result;
            }
            int64_t t_ce_start = get_time_ms();
            if (!codec_encoder_.load_model(decoder_model_path_)) {
                result.error_msg = "Failed to load Mimi codec encoder: " + codec_encoder_.get_error();
                return result;
            }
            codec_encoder_loaded_ = true;
            if (params.print_timing) {
                fprintf(stderr, "  Codec encoder lazy-loaded in %lld ms\n",
                        (long long)(get_time_ms() - t_ce_start));
            }
        }

        std::vector<int32_t> ref_codes;
        int32_t n_ref_frames = 0;
        int64_t t_ce_encode_start = get_time_ms();
        if (!codec_encoder_.encode(ref_samples, n_ref_samples, ref_codes, n_ref_frames)) {
            result.error_msg = "Failed to encode reference audio: " + codec_encoder_.get_error();
            return result;
        }
        result.t_encode_ms += get_time_ms() - t_ce_encode_start;

        if (params.print_progress) {
            fprintf(stderr, "Reference codes: %d frames x 16 codebooks (ICL mode)\n", n_ref_frames);
        }
        return synthesize_internal(text, speaker_embedding.data(), params, result,
                                   ref_codes.data(), n_ref_frames);
    }

    return synthesize_internal(text, speaker_embedding.data(), params, result);
}

bool Qwen3TTS::extract_speaker_embedding(const float * ref_samples, int32_t n_ref_samples,
                                          std::vector<float> & embedding,
                                          const tts_params & params) {
    if (!models_loaded_) {
        error_msg_ = "Models not loaded";
        return false;
    }

    if (!encoder_loaded_) {
        if (tts_model_path_.empty()) {
            error_msg_ = "Internal error: missing TTS model path for lazy encoder load";
            return false;
        }
        int64_t t_encoder_load_start = get_time_ms();
        if (!audio_encoder_.load_model(tts_model_path_)) {
            error_msg_ = "Failed to load speaker encoder: " + audio_encoder_.get_error();
            return false;
        }
        encoder_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Speaker encoder lazy-loaded in %lld ms\n",
                    (long long)(get_time_ms() - t_encoder_load_start));
        }
    }

    if (!audio_encoder_.encode(ref_samples, n_ref_samples, embedding)) {
        error_msg_ = "Failed to extract speaker embedding: " + audio_encoder_.get_error();
        return false;
    }

    return true;
}

tts_result Qwen3TTS::synthesize_with_embedding(const std::string & text,
                                                const float * embedding, int32_t embedding_size,
                                                const tts_params & params) {
    tts_result result;

    if (!models_loaded_) {
        result.error_msg = "Models not loaded";
        return result;
    }

    if (embedding == nullptr || embedding_size <= 0) {
        result.error_msg = "Invalid speaker embedding";
        return result;
    }

    int32_t expected_size = transformer_.get_config().hidden_size;
    if (embedding_size != expected_size) {
        result.error_msg = "Speaker embedding size mismatch: expected " +
                           std::to_string(expected_size) + " but got " +
                           std::to_string(embedding_size);
        return result;
    }

    return synthesize_internal(text, embedding, params, result);
}

tts_result Qwen3TTS::synthesize_internal(const std::string & text,
                                          const float * speaker_embedding,
                                          const tts_params & params,
                                          tts_result & result,
                                          const int32_t * ref_codes,
                                          int32_t n_ref_frames) {
    int64_t t_total_start = get_time_ms();
    auto sample_memory = [&](const char * stage) {
        process_memory_snapshot mem;
        if (!get_process_memory_snapshot(mem)) {
            return;
        }
        if (result.mem_rss_start_bytes == 0) {
            result.mem_rss_start_bytes = mem.rss_bytes;
            result.mem_phys_start_bytes = mem.phys_footprint_bytes;
        }
        result.mem_rss_end_bytes = mem.rss_bytes;
        result.mem_phys_end_bytes = mem.phys_footprint_bytes;
        if (mem.rss_bytes > result.mem_rss_peak_bytes) {
            result.mem_rss_peak_bytes = mem.rss_bytes;
        }
        if (mem.phys_footprint_bytes > result.mem_phys_peak_bytes) {
            result.mem_phys_peak_bytes = mem.phys_footprint_bytes;
        }
        if (params.print_timing) {
            fprintf(stderr, "  [mem] %-24s rss=%s  phys=%s\n",
                    stage,
                    format_bytes(mem.rss_bytes).c_str(),
                    format_bytes(mem.phys_footprint_bytes).c_str());
        }
    };
    sample_memory("synth/start");
    
    // Step 2: Tokenize input text (with optional voice steering instruction)
    int64_t t_tokenize_start = get_time_ms();
    std::vector<int32_t> text_tokens;
    if (!params.instruction.empty()) {
        text_tokens = tokenizer_.encode_for_tts_with_instruction(text, params.instruction);
    } else {
        text_tokens = tokenizer_.encode_for_tts(text);
    }
    result.t_tokenize_ms = get_time_ms() - t_tokenize_start;
    sample_memory("synth/after-tokenize");
    
    if (text_tokens.empty()) {
        result.error_msg = "Failed to tokenize text";
        return result;
    }
    
    if (params.print_progress) {
        fprintf(stderr, "Text tokenized: %zu tokens\n", text_tokens.size());
        fprintf(stderr, "  Tokens: ");
        for (size_t i = 0; i < std::min(text_tokens.size(), (size_t)10); ++i) {
            fprintf(stderr, "%d ", text_tokens[i]);
        }
        if (text_tokens.size() > 10) fprintf(stderr, "...");
        fprintf(stderr, "\n");
    }
    
    // Step 3: Generate speech codes using TTS transformer
    int64_t t_generate_start = get_time_ms();
    if (!transformer_loaded_) {
        int64_t t_reload_start = get_time_ms();
        if (!transformer_.load_model(tts_model_path_)) {
            result.error_msg = "Failed to reload TTS transformer: " + transformer_.get_error();
            return result;
        }
        transformer_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Transformer reloaded in %lld ms\n",
                    (long long)(get_time_ms() - t_reload_start));
            sample_memory("synth/after-transformer-reload");
        }
    }
    transformer_.clear_kv_cache();
    transformer_.set_f32_acc(params.f32_acc);
    if (params.seed >= 0) {
        transformer_.set_seed((uint32_t)params.seed);
    }

    // ICL mode: tokenize the reference transcript so the talker prefill can
    // align the new text with the reference codes. We encode with the normal
    // encode_for_tts framing and let the transformer trim the 8-token wrap.
    std::vector<int32_t> ref_text_tokens;
    const bool icl_mode = (ref_codes != nullptr && n_ref_frames > 0 && !params.ref_text.empty());
    if (icl_mode) {
        ref_text_tokens = tokenizer_.encode_for_tts(params.ref_text);
    }

    std::vector<int32_t> speech_codes;
    if (!transformer_.generate(text_tokens.data(), (int32_t)text_tokens.size(),
                               speaker_embedding, params.max_audio_tokens, speech_codes,
                               params.language_id, params.repetition_penalty,
                               params.temperature, params.top_k,
                               ref_text_tokens.empty() ? nullptr : ref_text_tokens.data(),
                               (int32_t)ref_text_tokens.size(),
                               icl_mode ? ref_codes : nullptr,
                               icl_mode ? n_ref_frames : 0,
                               params.print_timing)) {
        result.error_msg = "Failed to generate speech codes: " + transformer_.get_error();
        return result;
    }
    result.t_generate_ms = get_time_ms() - t_generate_start;
#ifdef QWEN3_TTS_TIMING
    if (const tts_timing * timing = transformer_.last_timing()) {
        result.detailed_timing = *timing;
        result.has_detailed_timing = true;
    }
#endif
    sample_memory("synth/after-generate");
    
    int n_codebooks = transformer_.get_config().n_codebooks;
    int n_frames = (int)speech_codes.size() / n_codebooks;
    
    if (params.print_progress) {
        fprintf(stderr, "Speech codes generated: %d frames x %d codebooks\n", n_frames, n_codebooks);
    }
    
    if (n_frames == 0) {
        result.error_msg = "No speech codes generated";
        return result;
    }

    if (low_mem_mode_) {
        transformer_.unload_model();
        transformer_loaded_ = false;
        sample_memory("synth/after-transformer-unload");
    }
    
    // Step 4: Decode speech codes to waveform using vocoder
    int64_t t_decode_start = get_time_ms();
    if (!decoder_loaded_) {
        int64_t t_decoder_load_start = get_time_ms();
        if (decoder_model_path_.empty()) {
            result.error_msg = "Internal error: missing vocoder model path";
            return result;
        }
        if (!audio_decoder_.load_model(decoder_model_path_)) {
            result.error_msg = "Failed to load vocoder: " + audio_decoder_.get_error();
            return result;
        }
        decoder_loaded_ = true;
        if (params.print_timing) {
            fprintf(stderr, "  Vocoder lazy-loaded in %lld ms\n",
                    (long long)(get_time_ms() - t_decoder_load_start));
            sample_memory("synth/after-vocoder-load");
        }
    }
    
    // ICL: prepend the reference codes to the generated codes before vocoder
    // decode so the decoder has warm context (matches Qwen's Python reference,
    // qwen3_tts_model.py torch.cat([ref, new])). Without this the vocoder
    // cold-starts and produces ~350ms of noise at the beginning of the output.
    // We slice the ref portion off the decoded wav immediately after.
    std::vector<int32_t> decode_codes_storage;
    const int32_t * decode_codes_ptr = speech_codes.data();
    int32_t decode_n_frames = n_frames;
    if (icl_mode) {
        decode_n_frames = n_ref_frames + n_frames;
        decode_codes_storage.resize((size_t) decode_n_frames * n_codebooks);
        memcpy(decode_codes_storage.data(), ref_codes,
               (size_t) n_ref_frames * n_codebooks * sizeof(int32_t));
        memcpy(decode_codes_storage.data() + (size_t) n_ref_frames * n_codebooks,
               speech_codes.data(), speech_codes.size() * sizeof(int32_t));
        decode_codes_ptr = decode_codes_storage.data();
    }

    if (!audio_decoder_.decode(decode_codes_ptr, decode_n_frames, result.audio)) {
        result.error_msg = "Failed to decode speech codes: " + audio_decoder_.get_error();
        return result;
    }

    // Trim the leading reference portion from the decoded wav.
    if (icl_mode && !result.audio.empty()) {
        size_t total = result.audio.size();
        size_t cut = (size_t) (((int64_t) n_ref_frames * (int64_t) total) / (int64_t) decode_n_frames);
        if (cut < total) {
            result.audio.erase(result.audio.begin(),
                               result.audio.begin() + (ptrdiff_t) cut);
        }
    }
    result.t_decode_ms = get_time_ms() - t_decode_start;
    sample_memory("synth/after-decode");

    if (low_mem_mode_) {
        audio_decoder_.unload_model();
        decoder_loaded_ = false;
        sample_memory("synth/after-vocoder-unload");
    }
    
    result.sample_rate = audio_decoder_.get_config().sample_rate;
    result.success = true;
    result.t_total_ms = get_time_ms() - t_total_start;
    sample_memory("synth/end");
    
    if (params.print_timing) {
        const double audio_sec = result.sample_rate > 0
            ? (double) result.audio.size() / (double) result.sample_rate : 0.0;
        const double wall_sec = (double) result.t_total_ms / 1000.0;
        const double realtime_factor = audio_sec > 0.0 ? wall_sec / audio_sec : 0.0;
        const double x_realtime = wall_sec > 0.0 ? audio_sec / wall_sec : 0.0;
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Tokenization:    %lld ms\n", (long long)result.t_tokenize_ms);
        fprintf(stderr, "  Speaker encode:  %lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Code generation: %lld ms\n", (long long)result.t_generate_ms);
        fprintf(stderr, "  Vocoder decode:  %lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:           %lld ms\n", (long long)result.t_total_ms);
        fprintf(stderr, "  Audio duration:  %.2f s\n", audio_sec);
        fprintf(stderr, "  Throughput:      %.2fx realtime (RTF=%.3f)\n", x_realtime, realtime_factor);
        fprintf(stderr, "\nMemory:\n");
        fprintf(stderr, "  RSS start/end:   %s -> %s\n",
                format_bytes(result.mem_rss_start_bytes).c_str(),
                format_bytes(result.mem_rss_end_bytes).c_str());
        fprintf(stderr, "  RSS peak:        %s\n",
                format_bytes(result.mem_rss_peak_bytes).c_str());
        fprintf(stderr, "  Phys start/end:  %s -> %s\n",
                format_bytes(result.mem_phys_start_bytes).c_str(),
                format_bytes(result.mem_phys_end_bytes).c_str());
        fprintf(stderr, "  Phys peak:       %s\n",
                format_bytes(result.mem_phys_peak_bytes).c_str());
    }
    
    return result;
}

void Qwen3TTS::set_progress_callback(tts_progress_callback_t callback) {
    progress_callback_ = callback;
}

// --- Model metadata & speaker-preset accessors ------------------------------

const std::string & Qwen3TTS::get_model_type() const {
    return transformer_.get_config().model_type;
}

const std::string & Qwen3TTS::get_model_size() const {
    return transformer_.get_config().model_size;
}

bool Qwen3TTS::has_speaker_encoder() const {
    return transformer_.get_config().has_speaker_encoder;
}

const std::vector<std::string> & Qwen3TTS::get_speaker_names() const {
    return transformer_.get_config().speaker_names;
}

const std::vector<int32_t> & Qwen3TTS::get_speaker_ids() const {
    return transformer_.get_config().speaker_ids;
}

const std::vector<std::string> & Qwen3TTS::get_speaker_dialects() const {
    return transformer_.get_config().speaker_dialects;
}

int32_t Qwen3TTS::get_speaker_id(const std::string & name) const {
    const auto & cfg = transformer_.get_config();
    for (size_t i = 0; i < cfg.speaker_names.size(); ++i) {
        if (cfg.speaker_names[i] == name) {
            return cfg.speaker_ids[i];
        }
    }
    return -1;
}

bool Qwen3TTS::get_speaker_embedding(const std::string & name, std::vector<float> & out) {
    int32_t tid = get_speaker_id(name);
    if (tid < 0) {
        error_msg_ = "Unknown speaker preset: " + name;
        return false;
    }
    if (!transformer_.get_codec_embedding_row(tid, out)) {
        error_msg_ = "Failed to read codec_embd row: " + transformer_.get_error();
        return false;
    }
    return true;
}

// WAV file loading (16-bit PCM or 32-bit float)
bool load_audio_file(const std::string & path, std::vector<float> & samples, 
                     int & sample_rate) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open WAV file: %s\n", path.c_str());
        return false;
    }
    
    // Read RIFF header
    char riff[4];
    if (fread(riff, 1, 4, f) != 4 || strncmp(riff, "RIFF", 4) != 0) {
        fprintf(stderr, "ERROR: Not a RIFF file\n");
        fclose(f);
        return false;
    }
    
    uint32_t file_size;
    if (fread(&file_size, 4, 1, f) != 1) {
        fclose(f);
        return false;
    }
    
    char wave[4];
    if (fread(wave, 1, 4, f) != 4 || strncmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "ERROR: Not a WAVE file\n");
        fclose(f);
        return false;
    }
    
    // Find fmt and data chunks
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sr = 0;
    uint16_t bits_per_sample = 0;
    
    while (!feof(f)) {
        char chunk_id[4];
        uint32_t chunk_size;
        
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;
        
        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            if (fread(&audio_format, 2, 1, f) != 1) break;
            if (fread(&num_channels, 2, 1, f) != 1) break;
            if (fread(&sr, 4, 1, f) != 1) break;
            fseek(f, 6, SEEK_CUR);  // Skip byte rate and block align
            if (fread(&bits_per_sample, 2, 1, f) != 1) break;
            
            // Skip any extra format bytes
            if (chunk_size > 16) {
                fseek(f, chunk_size - 16, SEEK_CUR);
            }
        }
        else if (strncmp(chunk_id, "data", 4) == 0) {
            sample_rate = sr;
            
            if (audio_format == 1) {  // PCM
                if (bits_per_sample == 16) {
                    int n_samples = chunk_size / (2 * num_channels);
                    samples.resize(n_samples);
                    
                    std::vector<int16_t> raw(n_samples * num_channels);
                    if (fread(raw.data(), 2, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                        fclose(f);
                        return false;
                    }
                    
                    // Convert to mono float
                    for (int i = 0; i < n_samples; ++i) {
                        float sum = 0.0f;
                        for (int c = 0; c < num_channels; ++c) {
                            sum += raw[i * num_channels + c] / 32768.0f;
                        }
                        samples[i] = sum / num_channels;
                    }
                }
                else if (bits_per_sample == 32) {
                    int n_samples = chunk_size / (4 * num_channels);
                    samples.resize(n_samples);
                    
                    std::vector<int32_t> raw(n_samples * num_channels);
                    if (fread(raw.data(), 4, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                        fclose(f);
                        return false;
                    }
                    
                    // Convert to mono float
                    for (int i = 0; i < n_samples; ++i) {
                        float sum = 0.0f;
                        for (int c = 0; c < num_channels; ++c) {
                            sum += raw[i * num_channels + c] / 2147483648.0f;
                        }
                        samples[i] = sum / num_channels;
                    }
                }
                else {
                    fprintf(stderr, "ERROR: Unsupported bits per sample: %d\n", bits_per_sample);
                    fclose(f);
                    return false;
                }
            }
            else if (audio_format == 3) {  // IEEE float
                int n_samples = chunk_size / (4 * num_channels);
                samples.resize(n_samples);
                
                std::vector<float> raw(n_samples * num_channels);
                if (fread(raw.data(), 4, n_samples * num_channels, f) != (size_t)(n_samples * num_channels)) {
                    fclose(f);
                    return false;
                }
                
                // Convert to mono
                for (int i = 0; i < n_samples; ++i) {
                    float sum = 0.0f;
                    for (int c = 0; c < num_channels; ++c) {
                        sum += raw[i * num_channels + c];
                    }
                    samples[i] = sum / num_channels;
                }
            }
            else {
                fprintf(stderr, "ERROR: Unsupported audio format: %d\n", audio_format);
                fclose(f);
                return false;
            }
            
            fclose(f);
            return true;
        }
        else {
            // Skip unknown chunk
            fseek(f, chunk_size, SEEK_CUR);
        }
    }
    
    fprintf(stderr, "ERROR: No data chunk found\n");
    fclose(f);
    return false;
}

// WAV file saving (16-bit PCM at specified sample rate)
bool save_audio_file(const std::string & path, const std::vector<float> & samples,
                     int sample_rate) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot create WAV file: %s\n", path.c_str());
        return false;
    }
    
    // WAV header parameters
    uint16_t num_channels = 1;
    uint16_t bits_per_sample = 16;
    uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    uint16_t block_align = num_channels * bits_per_sample / 8;
    uint32_t data_size = samples.size() * block_align;
    uint32_t file_size = 36 + data_size;
    
    // Write RIFF header
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    
    // Write fmt chunk
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    uint16_t audio_format = 1;  // PCM
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    uint32_t sr = sample_rate;
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);
    
    // Write data chunk
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    
    // Convert float samples to 16-bit PCM and write
    for (size_t i = 0; i < samples.size(); ++i) {
        // Clamp to [-1, 1] and convert to int16
        float sample = samples[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        int16_t pcm_sample = (int16_t)(sample * 32767.0f);
        fwrite(&pcm_sample, 2, 1, f);
    }
    
    fclose(f);
    return true;
}

} // namespace qwen3_tts
