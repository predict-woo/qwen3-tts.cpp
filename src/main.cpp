#include "qwen3_tts.h"

#include <cstdio>
#include <chrono>
#include <cstring>
#include <string>

void print_usage(const char * program) {
    fprintf(stderr, "Usage: %s [options] -m <model_dir> -t <text>\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <dir>      Model directory (required)\n");
    fprintf(stderr, "  -t, --text <text>      Text to synthesize (required)\n");
    fprintf(stderr, "  -o, --output <file>    Output WAV file (default: output.wav)\n");
    fprintf(stderr, "  -r, --reference <file> Reference audio for voice cloning\n");
    fprintf(stderr, "  --instruct <text>    Voice design instruct (needs VoiceDesign model)\n");
    fprintf(stderr, "      --ref-text <text>  Transcript (enables ICL cloning)\n");
    fprintf(stderr, "      --ref-code <file>  Pre-computed codec tokens (exact Python encoder)\n");
    fprintf(stderr, "  --temperature <val>    Sampling temperature (default: 0.9, 0=greedy)\n");
    fprintf(stderr, "  --top-k <n>            Top-k sampling (default: 50, 0=disabled)\n");
    fprintf(stderr, "  --top-p <val>          Top-p sampling (default: 1.0)\n");
    fprintf(stderr, "  --max-tokens <n>       Maximum audio tokens (default: 4096)\n");
    fprintf(stderr, "  --repetition-penalty <val> Repetition penalty (default: 1.05)\n");
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
    fprintf(stderr, "  %s -m ./models -t \"Hi\" -r ref.wav --ref-text \"Transcript\" -o icl.wav\n", program);
}

// --stream demo callback: reports chunk arrivals + first-chunk latency
static bool stream_cli_callback(const float * samples, size_t n) {
    static int64_t t0 = 0;
    static int chunk = 0;
    auto now_ms = [] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    if (t0 == 0) t0 = now_ms();
    fprintf(stderr, "[stream] chunk %d: %zu samples (%.2fs audio) at +%.2fs\n",
            ++chunk, n, (double)n / 24000.0, (double)(now_ms() - t0) / 1000.0);
    return true;
}

int main(int argc, char ** argv) {
    std::string model_dir;
    std::string text;
    std::string output_file = "output.wav";
    std::string reference_audio;
    std::string ref_text;
    std::string instruct_text;
    bool stream_mode = false;
    std::string ref_code_file;  // pre-computed codec tokens (bypasses C++ encoder)
    
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
        } else if (arg == "--stream") {
            stream_mode = true;
        } else if (arg == "-r" || arg == "--reference") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing reference audio\n");
                return 1;
            }
            reference_audio = argv[i];
        } else if (arg == "--ref-text") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing reference text\n");
                return 1;
            }
            ref_text = argv[i];
        } else if (arg == "--instruct") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing instruct text\n");
                return 1;
            }
            instruct_text = argv[i];
} else if (arg == "--ref-code") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing ref-code file\n");
                return 1;
            }
            ref_code_file = argv[i];
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
        } else if (arg == "--eos-bias") {
            if (++i >= argc) {
                fprintf(stderr, "Error: missing eos-bias value\n");
                return 1;
            }
            params.eos_bias = std::stof(argv[i]);
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
    
    if (text.empty()) {
        fprintf(stderr, "Error: text is required\n");
        print_usage(argv[0]);
        return 1;
    }
    
    // Initialize TTS
    qwen3_tts::Qwen3TTS tts;
    
    fprintf(stderr, "Loading models from: %s\n", model_dir.c_str());
    if (!tts.load_models(model_dir)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }
    
    // Set progress callback
    tts.set_progress_callback([](int tokens, int max_tokens) {
        fprintf(stderr, "\rGenerating: %d/%d tokens", tokens, max_tokens);
    });
    
    // Generate speech
    qwen3_tts::tts_result result;

    if (!ref_code_file.empty() && ref_text.empty()) {
        fprintf(stderr, "Error: --ref-code requires --ref-text (ICL mode)\n");
        return 1;
    }

    if (!instruct_text.empty()) {
        fprintf(stderr, "Synthesizing with voice design: \"%s\"\n", text.c_str());
        fprintf(stderr, "Instruct: %s\n", instruct_text.c_str());
        result = tts.synthesize_voice_design(text, instruct_text, params);
    } else if (reference_audio.empty()) {
        fprintf(stderr, "Synthesizing: \"%s\"\n", text.c_str());
        result = stream_mode
            ? tts.synthesize_streaming(text, params, stream_cli_callback)
            : tts.synthesize(text, params);
    } else {
        fprintf(stderr, "Synthesizing with voice cloning: \"%s\"\n", text.c_str());
        fprintf(stderr, "Reference audio: %s\n", reference_audio.c_str());
        if (!ref_text.empty()) {
            params.ref_code_path = ref_code_file;
            fprintf(stderr, "ICL mode with transcript: %zu chars\n", ref_text.size());
            result = stream_mode
                ? tts.synthesize_in_context_streaming(text, reference_audio, ref_text, params, stream_cli_callback)
                : tts.synthesize_in_context(text, reference_audio, ref_text, params);
        } else {
            result = stream_mode
                ? tts.synthesize_with_voice(text, reference_audio, params, stream_cli_callback)
                : tts.synthesize_with_voice(text, reference_audio, params);
        }
    }
    
    if (!result.success) {
        fprintf(stderr, "\nError: %s\n", result.error_msg.c_str());
        return 1;
    }
    
    fprintf(stderr, "\n");
    
    // Save output
    if (!qwen3_tts::save_audio_file(output_file, result.audio, result.sample_rate)) {
        fprintf(stderr, "Error: failed to save output file: %s\n", output_file.c_str());
        return 1;
    }
    
    fprintf(stderr, "Output saved to: %s\n", output_file.c_str());
    fprintf(stderr, "Audio duration: %.2f seconds\n", 
            (float)result.audio.size() / result.sample_rate);
    
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
