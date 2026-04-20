// Standalone test for AudioCodecEncoder (Mimi ICL encoder).
// Usage: test_codec_encoder <tokenizer.gguf> <reference.wav>

#include "encoder/audio_codec_encoder.h"
#include "pipeline/qwen3_tts.h"  // for load_audio_file

#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <tokenizer.gguf> <reference.wav>\n", argv[0]);
        return 1;
    }
    const char * tok_path = argv[1];
    const char * wav_path = argv[2];

    qwen3_tts::AudioCodecEncoder enc;
    if (!enc.load_model(tok_path)) {
        fprintf(stderr, "load failed: %s\n", enc.get_error().c_str());
        return 1;
    }
    const auto & cfg = enc.get_config();
    fprintf(stderr, "config: sample_rate=%d hidden=%d n_heads=%d n_layers=%d codebook_size=%d n_valid_q=%d\n",
            cfg.sample_rate, cfg.hidden_size, cfg.n_heads, cfg.n_transformer_layers,
            cfg.codebook_size, cfg.n_valid_quantizers);

    std::vector<float> samples;
    int sample_rate = 0;
    if (!qwen3_tts::load_audio_file(wav_path, samples, sample_rate)) {
        fprintf(stderr, "failed to load %s\n", wav_path);
        return 1;
    }
    fprintf(stderr, "loaded %zu samples at %d Hz (%.2f s)\n",
            samples.size(), sample_rate, (double) samples.size() / sample_rate);

    if (sample_rate != 24000) {
        // Simple linear resample to 24 kHz.
        const double ratio = (double) sample_rate / 24000.0;
        const size_t out_n = (size_t) ((double) samples.size() / ratio);
        std::vector<float> resampled(out_n);
        for (size_t i = 0; i < out_n; ++i) {
            double src = (double) i * ratio;
            size_t s0 = (size_t) src;
            size_t s1 = s0 + 1 < samples.size() ? s0 + 1 : s0;
            double frac = src - (double) s0;
            resampled[i] = (float) ((1.0 - frac) * samples[s0] + frac * samples[s1]);
        }
        samples = std::move(resampled);
        fprintf(stderr, "resampled to 24000 Hz: %zu samples\n", samples.size());
    }

    std::vector<int32_t> codes;
    int32_t n_frames = 0;
    if (!enc.encode(samples.data(), (int32_t) samples.size(), codes, n_frames)) {
        fprintf(stderr, "encode failed: %s\n", enc.get_error().c_str());
        return 1;
    }

    const double audio_s = (double) samples.size() / 24000.0;
    const double expected_frames_f = audio_s * 12.5;  // Mimi is 12.5 Hz
    const size_t expected_codes = (size_t) n_frames * cfg.n_valid_quantizers;

    fprintf(stderr, "encoded: %d frames (%.1f expected) × %d codebooks = %zu codes\n",
            n_frames, expected_frames_f, cfg.n_valid_quantizers, codes.size());

    if ((size_t) codes.size() != expected_codes) {
        fprintf(stderr, "FAIL: codes.size() mismatch — got %zu, expected %zu\n",
                codes.size(), expected_codes);
        return 2;
    }

    // Print the first frame's codes.
    fprintf(stderr, "frame[0]:");
    for (int cb = 0; cb < cfg.n_valid_quantizers; ++cb) {
        fprintf(stderr, " %d", codes[cb]);
    }
    fprintf(stderr, "\n");

    // Check all codes are within the codebook.
    int out_of_range = 0;
    for (int32_t c : codes) {
        if (c < 0 || c >= cfg.codebook_size) ++out_of_range;
    }
    if (out_of_range) {
        fprintf(stderr, "FAIL: %d codes out of range [0, %d)\n", out_of_range, cfg.codebook_size);
        return 3;
    }

    fprintf(stderr, "PASS\n");
    return 0;
}
