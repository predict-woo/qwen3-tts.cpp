#include "pipeline/qwen3_tts.h"
#include "common/speaker_embedding_io.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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
    
    // Initialize TTS
    qwen3_tts::Qwen3TTS tts;
    
    fprintf(stderr, "Loading models from: %s\n", model_dir.c_str());
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

    // Set progress callback
    tts.set_progress_callback([](int tokens, int max_tokens) {
        fprintf(stderr, "\rGenerating: %d/%d tokens", tokens, max_tokens);
    });

    // Generate speech
    qwen3_tts::tts_result result;

    if (!speaker_preset.empty()) {
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
        fprintf(stderr, "Synthesizing with preset voice '%s' (%d dims): \"%s\"\n",
                speaker_preset.c_str(), (int)embedding.size(), text.c_str());
        result = tts.synthesize_with_embedding(text, embedding.data(), (int32_t)embedding.size(), params);
    } else if (!speaker_embedding_file.empty()) {
        // Use precomputed speaker embedding
        std::vector<float> embedding;
        if (!qwen3_tts::load_speaker_embedding(speaker_embedding_file, embedding)) {
            fprintf(stderr, "Error: failed to load speaker embedding from %s\n", speaker_embedding_file.c_str());
            return 1;
        }
        fprintf(stderr, "Synthesizing with speaker embedding (%d dims): \"%s\"\n",
                (int)embedding.size(), text.c_str());
        result = tts.synthesize_with_embedding(text, embedding.data(), (int32_t)embedding.size(), params);
    } else if (reference_audio.empty()) {
        fprintf(stderr, "Synthesizing: \"%s\"\n", text.c_str());
        result = tts.synthesize(text, params);
    } else {
        fprintf(stderr, "Synthesizing with voice cloning: \"%s\"\n", text.c_str());
        fprintf(stderr, "Reference audio: %s\n", reference_audio.c_str());

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
                fprintf(stderr, "Speaker embedding saved to: %s (%d dims)\n",
                        dump_speaker_embedding_file.c_str(), (int)embedding.size());
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
