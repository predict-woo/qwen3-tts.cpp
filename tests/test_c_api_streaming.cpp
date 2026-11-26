/* test_c_api_streaming.cpp — the C API's streaming entry points for voice
 * cloning, against their batch twins.
 *
 * With greedy decoding a route and its streaming twin generate the same
 * speech codes; only the vocoder's windowed decode differs from a decode of
 * the whole buffer. So each pair must give the same audio to within that,
 * and the streamed route must deliver audio before it returns. The three
 * routes share one reference, so they must agree with each other too.
 *
 * Usage: test_c_api_streaming [--model-dir <dir>] [--audio <wav>]
 *   --model-dir  directory holding qwen3-tts-0.6b-*.gguf and
 *                qwen3-tts-tokenizer-f16.gguf (default: models)
 *   --audio      reference recording to clone (default:
 *                examples/readme_clone_input.wav); its first seconds are used */

#include "qwen3tts_c_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int32_t kRate = 24000;
constexpr double kRefSeconds = 6.0;
// Long enough that an utterance spans several deliveries: the first comes
// after about two seconds of generated audio.
const char * const kText = "Streaming keeps the cloned voice, and the listener hears the first "
                           "words long before the whole sentence has been made.";
// A windowed decode measures 26-31 dB against a whole-buffer one; speech
// that differs at all measures below 0 dB. So this separates the two with
// a margin either way.
constexpr double kMinPairSnrDb = 15.0;
constexpr double kMinRouteSnrDb = 30.0;

int g_failures = 0;

bool check(bool ok, const char * what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
    return ok;
}

double rms(const std::vector<float> & x) {
    double acc = 0.0;
    for (float s : x) acc += double(s) * s;
    return x.empty() ? 0.0 : std::sqrt(acc / x.size());
}

// Similarity of two renderings in dB: the first one's energy over the error's.
// Returns a huge value when they are identical.
double snr_db(const std::vector<float> & a, const std::vector<float> & b) {
    const size_t n = std::min(a.size(), b.size());
    // The last half second is left out: the windowed decode is least
    // settled at the end of the buffer.
    const size_t compared = n > static_cast<size_t>(kRate / 2) ? n - kRate / 2 : n;
    double signal = 0.0, error = 0.0;
    for (size_t i = 0; i < compared; ++i) {
        signal += double(a[i]) * a[i];
        const double d = double(a[i]) - b[i];
        error += d * d;
    }
    if (error == 0.0) return 1000.0;
    return signal == 0.0 ? -1000.0 : 10.0 * std::log10(signal / error);
}

std::string snr_text(double db) {
    return db >= 999.0 ? "identical" : std::to_string(db).substr(0, 5) + " dB";
}

double seconds_of(size_t samples) {
    return double(samples) / kRate;
}

std::vector<float> take(Qwen3TtsAudio * audio) {
    std::vector<float> out;
    if (audio) {
        out.assign(audio->samples, audio->samples + audio->n_samples);
        qwen3_tts_free_audio(audio);
    }
    return out;
}

// ── Reference audio ───────────────────────────────────────────────────────

bool read_wav_pcm16(const std::string & path, std::vector<float> & samples,
                    int32_t & rate, std::string & error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        error = "cannot open " + path;
        return false;
    }
    char riff[4], wave[4];
    f.read(riff, 4).ignore(4).read(wave, 4);
    if (!f || std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) {
        error = "not a RIFF/WAVE file: " + path;
        return false;
    }
    bool have_fmt = false;
    uint16_t channels = 0;
    rate = 0;
    while (f) {
        char id[4];
        char len[4];
        f.read(id, 4).read(len, 4);
        if (f.gcount() != 4) break;
        const uint32_t n = uint32_t(uint8_t(len[0])) | uint32_t(uint8_t(len[1])) << 8 |
                           uint32_t(uint8_t(len[2])) << 16 | uint32_t(uint8_t(len[3])) << 24;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            uint8_t fmt[16];
            f.read(reinterpret_cast<char *>(fmt), 16);
            if (n > 16) f.ignore(n - 16);
            if (!f || n < 16) {
                error = "truncated fmt chunk in " + path;
                return false;
            }
            const uint16_t format = uint16_t(fmt[0]) | uint16_t(fmt[1]) << 8;
            channels = uint16_t(fmt[2]) | uint16_t(fmt[3]) << 8;
            rate = int32_t(fmt[4]) | int32_t(fmt[5]) << 8 | int32_t(fmt[6]) << 16 |
                   int32_t(fmt[7]) << 24;
            const uint16_t bits = uint16_t(fmt[14]) | uint16_t(fmt[15]) << 8;
            if (format != 1 || bits != 16 || channels == 0 || rate == 0) {
                error = "only 16-bit PCM is supported, and " + path + " is not";
                return false;
            }
            have_fmt = true;
        } else if (std::memcmp(id, "data", 4) == 0) {
            if (!have_fmt || n % (2u * channels) != 0) {
                error = "malformed data chunk in " + path;
                return false;
            }
            const size_t frames = n / (2u * channels);
            std::vector<int16_t> raw(frames * channels);
            f.read(reinterpret_cast<char *>(raw.data()), std::streamsize(n));
            if (static_cast<uint32_t>(f.gcount()) != n) {
                error = "truncated data chunk in " + path;
                return false;
            }
            samples.resize(frames);
            for (size_t i = 0; i < frames; ++i) {
                int32_t acc = 0;
                for (uint16_t c = 0; c < channels; ++c) acc += raw[i * channels + c];
                samples[i] = float(acc) / float(channels) / 32768.0f;
            }
        } else {
            f.ignore(n);
        }
        if (n % 2 == 1) f.ignore(1);  // chunks are word-aligned
    }
    if (!have_fmt || samples.empty()) {
        error = "no PCM samples in " + path;
        return false;
    }
    return true;
}

bool write_wav_pcm16(const std::string & path, const std::vector<int16_t> & samples) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const uint32_t data_bytes = uint32_t(samples.size() * 2);
    const auto put16 = [&f](uint16_t v) {
        char b[2] = {char(v & 0xff), char(v >> 8)};
        f.write(b, 2);
    };
    const auto put32 = [&f](uint32_t v) {
        char b[4] = {char(v & 0xff), char((v >> 8) & 0xff), char((v >> 16) & 0xff), char(v >> 24)};
        f.write(b, 4);
    };
    f.write("RIFF", 4);
    put32(36 + data_bytes);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    put32(16);
    put16(1);  // PCM
    put16(1);  // mono
    put32(uint32_t(kRate));
    put32(uint32_t(kRate) * 2);
    put16(2);
    put16(16);
    f.write("data", 4);
    put32(data_bytes);
    for (int16_t s : samples) put16(uint16_t(s));
    return bool(f);
}

std::vector<float> resample_linear(const std::vector<float> & in, int32_t from, int32_t to) {
    if (from == to || in.empty()) return in;
    std::vector<float> out(static_cast<size_t>(double(in.size()) * to / from));
    const double step = double(from) / double(to);
    for (size_t i = 0; i < out.size(); ++i) {
        const double pos = i * step;
        const size_t i0 = static_cast<size_t>(pos);
        const size_t i1 = std::min(i0 + 1, in.size() - 1);
        const double frac = pos - double(i0);
        out[i] = float((1.0 - frac) * in[i0] + frac * in[i1]);
    }
    return out;
}

bool has_model(const std::string & dir) {
    const std::filesystem::path base(dir);
    const bool tokenizer = std::filesystem::exists(base / "qwen3-tts-tokenizer-f16.gguf");
    return tokenizer && (std::filesystem::exists(base / "qwen3-tts-0.6b-f16.gguf") ||
                         std::filesystem::exists(base / "qwen3-tts-0.6b-q8_0.gguf"));
}

void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [--model-dir <dir>] [--audio <wav>]\n", prog);
    fprintf(stderr, "  --model-dir <dir>  directory with qwen3-tts-0.6b-*.gguf and the tokenizer GGUF\n");
    fprintf(stderr, "                     (default: models)\n");
    fprintf(stderr, "  --audio <wav>      reference recording to clone; its first %.0f s are used\n",
            kRefSeconds);
    fprintf(stderr, "                     (default: examples/readme_clone_input.wav)\n");
}

// ── Streaming ─────────────────────────────────────────────────────────────

struct Streamed {
    std::vector<float> audio;  // what the callback was given, in order
    int chunks = 0;
    double first_ms = -1.0;
    double last_ms = -1.0;
    double returned_ms = -1.0;
};

struct Probe {
    // stop_after: 0 never stops; n returns 0 from the callback at delivery n
    explicit Probe(int stop_after = 0)
        : started(std::chrono::steady_clock::now()), stop_after(stop_after) {}
    std::chrono::steady_clock::time_point started;
    Streamed out;
    int stop_after;
};

int on_chunk(const float * samples, int32_t n, void * user_data) {
    auto * probe = static_cast<Probe *>(user_data);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - probe->started).count();
    if (probe->out.chunks == 0) probe->out.first_ms = ms;
    probe->out.last_ms = ms;
    ++probe->out.chunks;
    probe->out.audio.insert(probe->out.audio.end(), samples, samples + n);
    return probe->stop_after == 0 || probe->out.chunks < probe->stop_after ? 1 : 0;
}

void finished(Probe & probe) {
    probe.out.returned_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - probe.started).count();
}

// A streamed route against its batch twin: the same speech, delivered before
// the call returned.
bool check_pair(const char * what, const std::vector<float> & batch,
                const Streamed & streamed, const std::vector<float> & returned) {
    printf("\n%s:\n", what);
    printf("  batch %.2f s, streamed %.2f s in %d deliveries; first at %.0f ms, last at %.0f ms, returned at %.0f ms\n",
           seconds_of(batch.size()), seconds_of(returned.size()), streamed.chunks,
           streamed.first_ms, streamed.last_ms, streamed.returned_ms);
    bool ok = true;
    ok &= check(batch.size() > size_t(kRate), "the batch route produces more than a second of audio");
    ok &= check(rms(batch) > 0.005, "the batch route is not silent");
    if (!ok) return false;
    ok &= check(streamed.audio.size() == returned.size(),
                "the returned audio is exactly what the callback was given");
    ok &= check(streamed.chunks >= 2, "audio arrives in more than one delivery");
    ok &= check(streamed.first_ms < streamed.last_ms,
                "the first delivery arrives before the last");
    ok &= check(streamed.last_ms <= streamed.returned_ms, "everything is delivered before the call returns");
    const size_t drift = streamed.audio.size() > batch.size() ? streamed.audio.size() - batch.size()
                                                              : batch.size() - streamed.audio.size();
    ok &= check(drift <= size_t(kRate), "the two routes give the same length to within a second");
    const double db = snr_db(batch, streamed.audio);
    ok &= check(db >= kMinPairSnrDb, "the streamed audio is the batch audio");
    printf("        stream vs batch: %s (at least %.0f dB required)\n", snr_text(db).c_str(), kMinPairSnrDb);
    return ok;
}

bool check_routes(const char * what, const std::vector<float> & a, const std::vector<float> & b) {
    const double db = snr_db(a, b);
    const bool ok = check(db >= kMinRouteSnrDb, what);
    printf("        %s\n", snr_text(db).c_str());
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model_dir = "models";
    std::string audio_path = "examples/readme_clone_input.wav";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    printf("=== C API Streaming Test (voice cloning) ===\n\n");
    if (!has_model(model_dir)) {
        fprintf(stderr, "No model in '%s': it needs qwen3-tts-0.6b-f16.gguf (or -q8_0) and "
                        "qwen3-tts-tokenizer-f16.gguf. See --help.\n", model_dir.c_str());
        return 1;
    }

    // 1. One reference for every route: the first seconds of the recording,
    //    at 24 kHz, quantised to 16-bit samples once. The routes that take
    //    a file read those back; the samples route gets the floats the
    //    engine makes of them, so the routes see one reference and not
    //    three near-copies of one. (They would not be near enough: greedy
    //    decoding turns a sliver of difference into different speech.)
    std::vector<float> ref;
    int32_t ref_rate = 0;
    std::string error;
    if (!read_wav_pcm16(audio_path, ref, ref_rate, error)) {
        fprintf(stderr, "Reference recording: %s\n", error.c_str());
        return 1;
    }
    ref = resample_linear(ref, ref_rate, kRate);
    ref.resize(std::min(ref.size(), size_t(kRefSeconds * kRate)));
    std::vector<int16_t> ref_pcm;
    ref_pcm.reserve(ref.size());
    for (float s : ref) {
        ref_pcm.push_back(int16_t(std::clamp<double>(s, -1.0, 1.0) * 32767.0));
    }
    ref.clear();
    ref.reserve(ref_pcm.size());
    for (int16_t s : ref_pcm) ref.push_back(float(s) / 32768.0f);
    const std::filesystem::path ref_path =
        std::filesystem::temp_directory_path() / "qwen3_tts_c_api_streaming_ref.wav";
    if (!write_wav_pcm16(ref_path.string(), ref_pcm)) {
        fprintf(stderr, "Cannot write %s\n", ref_path.string().c_str());
        return 1;
    }
    printf("Reference: %.1f s of %s\n", seconds_of(ref.size()), audio_path.c_str());
    printf("Text: \"%s\" (greedy decoding, so routes are comparable)\n", kText);

    // 2. Engine and parameters
    Qwen3Tts * tts = qwen3_tts_create(model_dir.c_str(), 4);
    if (!tts) {
        fprintf(stderr, "Failed to load the models in '%s'\n", model_dir.c_str());
        std::filesystem::remove(ref_path);
        return 1;
    }
    Qwen3TtsParams params;
    qwen3_tts_default_params(&params);
    params.temperature = 0.0f;
    params.n_threads = 4;

    // 3. Reference file
    const std::vector<float> file_batch =
        take(qwen3_tts_synthesize_with_voice_file(tts, kText, ref_path.string().c_str(), &params));
    Probe file_probe;
    const std::vector<float> file_streamed = take(qwen3_tts_synthesize_with_voice_file_streaming(
        tts, kText, ref_path.string().c_str(), &params, on_chunk, &file_probe));
    finished(file_probe);
    const bool file_ok = check_pair("Route: reference file", file_batch, file_probe.out, file_streamed);

    // 4. Reference samples
    const std::vector<float> samples_batch = take(qwen3_tts_synthesize_with_voice_samples(
        tts, kText, ref.data(), int32_t(ref.size()), &params));
    Probe samples_probe;
    const std::vector<float> samples_streamed = take(qwen3_tts_synthesize_with_voice_samples_streaming(
        tts, kText, ref.data(), int32_t(ref.size()), &params, on_chunk, &samples_probe));
    finished(samples_probe);
    const bool samples_ok =
        check_pair("Route: reference samples", samples_batch, samples_probe.out, samples_streamed);

    // 5. Speaker embedding, extracted once from the same samples
    std::vector<float> embedding(4096);
    const int32_t embedding_size =
        qwen3_tts_extract_embedding_file(tts, ref_path.string().c_str(), embedding.data(),
                                         int32_t(embedding.size()));
    embedding.resize(embedding_size > 0 ? size_t(embedding_size) : 0);
    bool embedding_ok = check(embedding_size > 0, "the speaker embedding is extracted");
    std::vector<float> embedding_batch, embedding_streamed;
    if (embedding_ok) {
        printf("        %d floats\n", embedding_size);
        embedding_batch = take(qwen3_tts_synthesize_with_embedding(
            tts, kText, embedding.data(), embedding_size, &params));
        Probe embedding_probe;
        embedding_streamed = take(qwen3_tts_synthesize_with_embedding_streaming(
            tts, kText, embedding.data(), embedding_size, &params, on_chunk, &embedding_probe));
        finished(embedding_probe);
        embedding_ok = check_pair("Route: speaker embedding", embedding_batch, embedding_probe.out,
                                  embedding_streamed);

        // 6. An early stop: the callback says 0, is not asked again, and
        //    keeps what it was given. The tail of what was generated by then
        //    is still decoded into the returned audio.
        printf("\nEarly stop on the speaker-embedding route:\n");
        Probe stop_probe{1};
        const std::vector<float> stopped = take(qwen3_tts_synthesize_with_embedding_streaming(
            tts, kText, embedding.data(), embedding_size, &params, on_chunk, &stop_probe));
        finished(stop_probe);
        printf("  stopped after 1 delivery; returned %.2f s at %.0f ms\n",
               seconds_of(stopped.size()), stop_probe.out.returned_ms);
        bool stop_ok = true;
        stop_ok &= check(stop_probe.out.chunks == 1, "the callback is not invoked again after it stops generation");
        stop_ok &= check(!stopped.empty() && stopped.size() >= stop_probe.out.audio.size(),
                         "what was delivered is kept in the returned audio");
        stop_ok &= check(stopped.size() < embedding_streamed.size(), "stopping early ends with less audio");
        embedding_ok &= stop_ok;
    }

    // 7. The three routes agree
    printf("\nThe routes agree with each other:\n");
    bool routes_ok = true;
    if (!file_batch.empty() && !samples_batch.empty()) {
        routes_ok &= check_routes("the same reference as a file and as samples sounds the same",
                                  file_batch, samples_batch);
    } else {
        routes_ok = false;
    }
    if (!file_batch.empty() && !embedding_batch.empty()) {
        routes_ok &= check_routes("a cached embedding sounds the same as the reference it came from",
                                  file_batch, embedding_batch);
    } else {
        routes_ok = false;
    }

    qwen3_tts_destroy(tts);
    std::filesystem::remove(ref_path);

    printf("\n");
    if (file_ok && samples_ok && embedding_ok && routes_ok) {
        printf("=== All tests passed! ===\n");
        return 0;
    }
    fprintf(stderr, "=== %d check(s) failed ===\n", g_failures);
    return 1;
}
