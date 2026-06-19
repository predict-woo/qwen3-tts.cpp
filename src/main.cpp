#include "pipeline/qwen3_tts.h"
#include "common/benchmark_json.h"
#include "common/speaker_embedding_io.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string basename_of(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

std::string env_or_default(const char * name, const char * fallback) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0' ? std::string(value) : std::string(fallback);
}

std::string infer_quantization(const std::string & model_name) {
    const std::string name = basename_of(model_name);
    const char * known[] = {
        "q2_k", "q3_k", "q4_0", "q4_1", "q4_k", "q5_0", "q5_1",
        "q5_k", "q6_k", "q8_0", "f16", "f32"
    };
    for (const char * q : known) {
        if (name.find(q) != std::string::npos) {
            return q;
        }
    }
    return std::string();
}

qwen3_tts::benchmark_record make_benchmark_record(const qwen3_tts::Qwen3TTS & tts,
                                                   const qwen3_tts::tts_result & result,
                                                   const qwen3_tts::tts_params & params,
                                                   const std::string & mode,
                                                   const std::string & text) {
    qwen3_tts::benchmark_record record;
    record.mode = mode;
    record.backend = env_or_default("QWEN3_TTS_BACKEND", "auto");
    record.device = env_or_default("QWEN3_TTS_DEVICE", "");
    record.thread_count = params.n_threads;
    record.model_type = tts.get_model_type();
    record.model_size = tts.get_model_size();
    record.tts_model = basename_of(tts.get_tts_model_path());
    record.decoder_model = basename_of(tts.get_decoder_model_path());
    record.quantization = infer_quantization(record.tts_model);
    record.text = text;
    record.audio_seconds = result.sample_rate > 0
        ? static_cast<double>(result.audio.size()) / static_cast<double>(result.sample_rate)
        : 0.0;
    record.tokenize_ms = result.t_tokenize_ms;
    record.encode_ms = result.t_encode_ms;
    record.generate_ms = result.t_generate_ms;
    record.decode_ms = result.t_decode_ms;
    record.total_ms = result.t_total_ms;
    record.mem_rss_start_bytes = result.mem_rss_start_bytes;
    record.mem_rss_end_bytes = result.mem_rss_end_bytes;
    record.mem_rss_peak_bytes = result.mem_rss_peak_bytes;
    record.mem_phys_start_bytes = result.mem_phys_start_bytes;
    record.mem_phys_end_bytes = result.mem_phys_end_bytes;
    record.mem_phys_peak_bytes = result.mem_phys_peak_bytes;

#ifdef QWEN3_TTS_TIMING
    if (result.has_detailed_timing) {
        const qwen3_tts::tts_timing & timing = result.detailed_timing;
        record.has_detailed_timing = true;
        record.prefill_build_ms = timing.t_prefill_build_ms;
        record.prefill_forward_ms = timing.t_prefill_forward_ms;
        record.prefill_graph_build_ms = timing.t_prefill_graph_build_ms;
        record.prefill_graph_alloc_ms = timing.t_prefill_graph_alloc_ms;
        record.prefill_compute_ms = timing.t_prefill_compute_ms;
        record.prefill_data_ms = timing.t_prefill_data_ms;
        record.talker_forward_ms = timing.t_talker_forward_ms;
        record.talker_graph_build_ms = timing.t_talker_graph_build_ms;
        record.talker_graph_alloc_ms = timing.t_talker_graph_alloc_ms;
        record.talker_compute_ms = timing.t_talker_compute_ms;
        record.talker_data_ms = timing.t_talker_data_ms;
        record.code_pred_ms = timing.t_code_pred_ms;
        record.code_pred_init_ms = timing.t_code_pred_init_ms;
        record.code_pred_prefill_ms = timing.t_code_pred_prefill_ms;
        record.code_pred_steps_ms = timing.t_code_pred_steps_ms;
        record.code_pred_graph_build_ms = timing.t_code_pred_graph_build_ms;
        record.code_pred_graph_alloc_ms = timing.t_code_pred_graph_alloc_ms;
        record.code_pred_compute_ms = timing.t_code_pred_compute_ms;
        record.code_pred_data_ms = timing.t_code_pred_data_ms;
        record.code_pred_coreml_ms = timing.t_code_pred_coreml_ms;
        record.embed_lookup_ms = timing.t_embed_lookup_ms;
        record.frames = timing.n_frames;
        record.generate_total_ms = timing.t_generate_total_ms;
    }
#endif

    return record;
}

} // namespace

void print_usage(const char * program) {
    fprintf(stderr, "Usage: %s [options] -m <model_dir> -t <text>\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <dir>      Model directory (required)\n");
    fprintf(stderr, "  --tts-model <file>     TTS model filename (overrides auto-detection)\n");
    fprintf(stderr, "  --tokenizer-model <file> Tokenizer/vocoder filename (overrides default)\n");
    fprintf(stderr, "  -t, --text <text>      Text to synthesize (required)\n");
    fprintf(stderr, "  -o, --output <file>    Output WAV file (default: output.wav)\n");
    fprintf(stderr, "  -r, --reference <file> Reference audio for voice cloning\n");
    fprintf(stderr, "  --speaker-embedding <file>  Use precomputed speaker embedding (.json/.bin)\n");
    fprintf(stderr, "  --dump-speaker-embedding <file> Save extracted embedding from --reference\n");
    fprintf(stderr, "  --temperature <val>    Sampling temperature (default: 0.9, 0=greedy)\n");
    fprintf(stderr, "  --top-k <n>            Top-k sampling (default: 50, 0=disabled)\n");
    fprintf(stderr, "  --top-p <val>          Top-p sampling (default: 1.0)\n");
    fprintf(stderr, "  --max-tokens <n>       Maximum audio tokens (default: 2048, ~170s @ 12Hz)\n");
    fprintf(stderr, "  --repetition-penalty <val> Repetition penalty (default: 1.05)\n");
    fprintf(stderr, "  --seed <n>             RNG seed for reproducible output (default: random)\n");
    fprintf(stderr, "  --no-f32-acc           Disable f32 matmul accumulation (faster, less precise)\n");
    fprintf(stderr, "  -i, --instruction <text>  Voice steering instruction (e.g. \"Speak happily\")\n");
    fprintf(stderr, "  --speaker <name>       Use a built-in preset voice (CustomVoice models only)\n");
    fprintf(stderr, "  --list-speakers        List preset voices available in the loaded model and exit\n");
    fprintf(stderr, "  --ref-text <text>      Reference transcript (with -r) for ICL voice cloning\n");
    fprintf(stderr, "  --benchmark-json <file> Write structured benchmark JSON after synthesis\n");
    fprintf(stderr, "  --quiet-progress       Suppress CLI progress/status/timing output for benchmark runs\n");
    fprintf(stderr, "  -l, --language <lang>  Language: en,ru,zh,ja,ko,de,fr,es (default: en)\n");
    fprintf(stderr, "  -j, --threads <n>      Number of threads (default: 4)\n");
    fprintf(stderr, "  -h, --help             Show this help\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Environment:\n");
    fprintf(stderr, "  QWEN3_TTS_BACKEND      Runtime backend override: auto|cuda|cpu\n");
    fprintf(stderr, "  QWEN3_TTS_DEVICE       CUDA device index when backend=cuda (default: 0)\n");
    fprintf(stderr, "  QWEN3_TTS_DECODER_GPU_MAX_FRAMES     Max frames per CUDA vocoder chunk (default: 34)\n");
    fprintf(stderr, "  QWEN3_TTS_DECODER_GPU_CONTEXT_FRAMES Left context per CUDA vocoder chunk (default: 12)\n");
    fprintf(stderr, "  QWEN3_TTS_LOW_MEM      Enable low-memory mode (set to 1)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello, world!\" -o hello.wav\n", program);
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" -r reference.wav -o cloned.wav\n", program);
}

int main(int argc, char ** argv) {
    std::string model_dir;
    std::string tts_model;
    std::string tokenizer_model;
    std::string text;
    std::string output_file = "output.wav";
    std::string reference_audio;
    std::string speaker_embedding_file;
    std::string dump_speaker_embedding_file;
    std::string speaker_preset;
    bool list_speakers = false;
    std::string ref_text;
    std::string benchmark_json_file;
    bool quiet_progress = false;
    
    qwen3_tts::tts_params params;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing model directory\n");
                return 1;
            }
            model_dir = argv[i];
        } else if (arg == "--tts-model") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing TTS model filename\n");
                return 1;
            }
            tts_model = argv[i];
        } else if (arg == "--tokenizer-model") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing tokenizer model filename\n");
                return 1;
            }
            tokenizer_model = argv[i];
        } else if (arg == "-t" || arg == "--text") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing text\n");
                return 1;
            }
            text = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing output file\n");
                return 1;
            }
            output_file = argv[i];
        } else if (arg == "-r" || arg == "--reference") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing reference audio\n");
                return 1;
            }
            reference_audio = argv[i];
        } else if (arg == "--speaker-embedding") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing speaker embedding file\n");
                return 1;
            }
            speaker_embedding_file = argv[i];
        } else if (arg == "--dump-speaker-embedding") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing dump speaker embedding file\n");
                return 1;
            }
            dump_speaker_embedding_file = argv[i];
        } else if (arg == "--speaker") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing speaker name\n");
                return 1;
            }
            speaker_preset = argv[i];
        } else if (arg == "--list-speakers") {
            list_speakers = true;
        } else if (arg == "--ref-text") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing ref-text value\n");
                return 1;
            }
            ref_text = argv[i];
        } else if (arg == "--benchmark-json") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing benchmark JSON file\n");
                return 1;
            }
            benchmark_json_file = argv[i];
        } else if (arg == "--quiet-progress") {
            quiet_progress = true;
        } else if (arg == "--temperature") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing temperature value\n");
                return 1;
            }
            params.temperature = std::stof(argv[i]);
        } else if (arg == "--top-k") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing top-k value\n");
                return 1;
            }
            params.top_k = std::stoi(argv[i]);
        } else if (arg == "--top-p") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing top-p value\n");
                return 1;
            }
            params.top_p = std::stof(argv[i]);
        } else if (arg == "--max-tokens") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing max-tokens value\n");
                return 1;
            }
            params.max_audio_tokens = std::stoi(argv[i]);
        } else if (arg == "--repetition-penalty") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing repetition-penalty value\n");
                return 1;
            }
            params.repetition_penalty = std::stof(argv[i]);
        } else if (arg == "--seed") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing seed value\n");
                return 1;
            }
            params.seed = std::stoi(argv[i]);
        } else if (arg == "--no-f32-acc") {
            params.f32_acc = false;
        } else if (arg == "-i" || arg == "--instruction" || arg == "--instruct") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing instruction text\n");
                return 1;
            }
            params.instruction = argv[i];
        } else if (arg == "-l" || arg == "--language") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing language value\n");
                return 1;
            }
            std::string lang = argv[i];
            if (lang == "en" || lang == "english")       params.language_id = 2050;
            else if (lang == "ru" || lang == "russian")  params.language_id = 2069;
            else if (lang == "zh" || lang == "chinese")  params.language_id = 2055;
            else if (lang == "ja" || lang == "japanese")  params.language_id = 2058;
            else if (lang == "ko" || lang == "korean")   params.language_id = 2064;
            else if (lang == "de" || lang == "german")   params.language_id = 2053;
            else if (lang == "fr" || lang == "french")   params.language_id = 2061;
            else if (lang == "es" || lang == "spanish")  params.language_id = 2054;
            else if (lang == "it" || lang == "italian")  params.language_id = 2070;
            else if (lang == "pt" || lang == "portuguese") params.language_id = 2071;
            else {
                fprintf(stderr, "Error: unknown language '%s'. Supported: en,ru,zh,ja,ko,de,fr,es,it,pt\n", lang.c_str());
                return 1;
            }
        } else if (arg == "-j" || arg == "--threads") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing threads value\n");
                return 1;
            }
            params.n_threads = std::stoi(argv[i]);
        } else {
            fprintf(stderr, "Error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate required arguments
    if (model_dir.empty()) {
        fprintf(stderr, "Error: model directory is required\n");
        print_usage(argv[0]);
        return 1;
    }

    if (text.empty() && !list_speakers) {
        fprintf(stderr, "Error: text is required\n");
        print_usage(argv[0]);
        return 1;
    }

    // Wire --ref-text through to the synth params; the pipeline routes voice
    // cloning through the Mimi ICL path only when both -r and --ref-text are
    // provided.
    if (!ref_text.empty()) {
        if (reference_audio.empty()) {
            fprintf(stderr, "Error: --ref-text requires -r/--reference\n");
            return 1;
        }
        if (!params.instruction.empty()) {
            fprintf(stderr, "Error: --ref-text cannot be combined with --instruction yet\n");
            return 1;
        }
        params.ref_text = ref_text;
    }

    if (quiet_progress) {
        params.print_progress = false;
        params.print_timing = false;
    }

    // Initialize TTS
    qwen3_tts::Qwen3TTS tts;
    
    if (!quiet_progress) {
        fprintf(stderr, "Loading models from: %s\n", model_dir.c_str());
    }
    if (!tts.load_models(model_dir, tts_model, tokenizer_model)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }
    
    // Handle --list-speakers: just print and exit.
    if (list_speakers) {
        const auto & names = tts.get_speaker_names();
        const auto & ids = tts.get_speaker_ids();
        const auto & dialects = tts.get_speaker_dialects();
        fprintf(stderr, "Model type: %s\n", tts.get_model_type().c_str());
        if (names.empty()) {
            fprintf(stderr, "No preset voices available in this model.\n");
            fprintf(stderr, "(Preset voices are only present in CustomVoice / VoiceDesign variants.)\n");
            return 0;
        }
        fprintf(stderr, "Preset voices (%zu):\n", names.size());
        for (size_t i = 0; i < names.size(); ++i) {
            const std::string & d = i < dialects.size() ? dialects[i] : std::string();
            if (!d.empty()) {
                fprintf(stderr, "  %-16s (id=%d, dialect=%s)\n", names[i].c_str(), ids[i], d.c_str());
            } else {
                fprintf(stderr, "  %-16s (id=%d)\n", names[i].c_str(), ids[i]);
            }
        }
        return 0;
    }

    if (!quiet_progress) {
        tts.set_progress_callback([](int tokens, int max_tokens) {
            fprintf(stderr, "\rGenerating: %d/%d tokens", tokens, max_tokens);
        });
    }

    // Generate speech
    qwen3_tts::tts_result result;
    std::string benchmark_mode;

    if (!speaker_preset.empty()) {
        benchmark_mode = "preset";
        // Preset voice from CustomVoice model: resolve name → codec_embd row.
        std::vector<float> embedding;
        if (!tts.get_speaker_embedding(speaker_preset, embedding)) {
            fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
            const auto & names = tts.get_speaker_names();
            if (!names.empty()) {
                fprintf(stderr, "Available presets:");
                for (const auto & n : names) fprintf(stderr, " %s", n.c_str());
                fprintf(stderr, "\n");
            } else {
                fprintf(stderr, "This model has no preset voices (model_type=%s). Use a CustomVoice or VoiceDesign GGUF.\n",
                        tts.get_model_type().c_str());
            }
            return 1;
        }
        if (!quiet_progress) {
            fprintf(stderr, "Synthesizing with preset voice '%s' (%d dims): \"%s\"\n",
                    speaker_preset.c_str(), (int)embedding.size(), text.c_str());
        }
        result = tts.synthesize_with_embedding(text, embedding.data(), (int32_t)embedding.size(), params);
    } else if (!speaker_embedding_file.empty()) {
        benchmark_mode = "speaker_embedding";
        // Use precomputed speaker embedding
        std::vector<float> embedding;
        if (!qwen3_tts::load_speaker_embedding(speaker_embedding_file, embedding)) {
            fprintf(stderr, "Error: failed to load speaker embedding from %s\n", speaker_embedding_file.c_str());
            return 1;
        }
        if (!quiet_progress) {
            fprintf(stderr, "Synthesizing with speaker embedding (%d dims): \"%s\"\n",
                    (int)embedding.size(), text.c_str());
        }
        result = tts.synthesize_with_embedding(text, embedding.data(), (int32_t)embedding.size(), params);
    } else if (reference_audio.empty()) {
        benchmark_mode = "default";
        if (!quiet_progress) {
            fprintf(stderr, "Synthesizing: \"%s\"\n", text.c_str());
        }
        result = tts.synthesize(text, params);
    } else {
        benchmark_mode = ref_text.empty() ? "xvector_voice_clone" : "icl_voice_clone";
        if (!quiet_progress) {
            fprintf(stderr, "Synthesizing with voice cloning: \"%s\"\n", text.c_str());
            fprintf(stderr, "Reference audio: %s\n", reference_audio.c_str());
        }

        // Optionally dump the speaker embedding for reuse
        if (!dump_speaker_embedding_file.empty()) {
            std::vector<float> embedding;
            std::vector<float> ref_samples;
            int ref_rate = 0;
            if (!qwen3_tts::load_audio_file(reference_audio, ref_samples, ref_rate)) {
                fprintf(stderr, "Error: failed to load reference audio\n");
                return 1;
            }
            if (!tts.extract_speaker_embedding(ref_samples.data(), (int32_t)ref_samples.size(), embedding, params)) {
                fprintf(stderr, "Error: failed to extract speaker embedding\n");
                return 1;
            }
            if (qwen3_tts::save_speaker_embedding(dump_speaker_embedding_file, embedding)) {
                if (!quiet_progress) {
                    fprintf(stderr, "Speaker embedding saved to: %s (%d dims)\n",
                            dump_speaker_embedding_file.c_str(), (int)embedding.size());
                }
            } else {
                fprintf(stderr, "Warning: failed to save speaker embedding\n");
            }
        }

        result = tts.synthesize_with_voice(text, reference_audio, params);
    }
    
    if (!result.success) {
        fprintf(stderr, "\nError: %s\n", result.error_msg.c_str());
        return 1;
    }
    
    if (!quiet_progress) {
        fprintf(stderr, "\n");
    }
    
    // Save output
    if (!qwen3_tts::save_audio_file(output_file, result.audio, result.sample_rate)) {
        fprintf(stderr, "Error: failed to save output file: %s\n", output_file.c_str());
        return 1;
    }
    
    if (!quiet_progress) {
        fprintf(stderr, "Output saved to: %s\n", output_file.c_str());
        fprintf(stderr, "Audio duration: %.2f seconds\n",
                (float)result.audio.size() / result.sample_rate);
    }

    if (!benchmark_json_file.empty()) {
        const qwen3_tts::benchmark_record record =
            make_benchmark_record(tts, result, params, benchmark_mode, text);
        std::string error;
        if (!qwen3_tts::write_benchmark_record_json(benchmark_json_file, record, error)) {
            fprintf(stderr, "Error: %s\n", error.c_str());
            return 1;
        }
        if (!quiet_progress) {
            fprintf(stderr, "Benchmark JSON saved to: %s\n", benchmark_json_file.c_str());
        }
    }
    
    // Print timing
    if (params.print_timing) {
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Load:      %6lld ms\n", (long long)result.t_load_ms);
        fprintf(stderr, "  Tokenize:  %6lld ms\n", (long long)result.t_tokenize_ms);
        fprintf(stderr, "  Encode:    %6lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Generate:  %6lld ms\n", (long long)result.t_generate_ms);
        fprintf(stderr, "  Decode:    %6lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:     %6lld ms\n", (long long)result.t_total_ms);
    }
    
    return 0;
}
