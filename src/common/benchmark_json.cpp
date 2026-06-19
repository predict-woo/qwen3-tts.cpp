#include "common/benchmark_json.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace qwen3_tts {
namespace {

void append_escaped_string(std::ostringstream & out, const std::string & value) {
    out << '"';
    for (unsigned char c : value) {
        switch (c) {
            case '"':  out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c) << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(c);
                }
                break;
        }
    }
    out << '"';
}

void append_key(std::ostringstream & out, bool & first, const char * key) {
    if (!first) {
        out << ',';
    }
    first = false;
    append_escaped_string(out, key);
    out << ':';
}

void append_string(std::ostringstream & out, bool & first,
                   const char * key, const std::string & value) {
    append_key(out, first, key);
    append_escaped_string(out, value);
}

void append_i64(std::ostringstream & out, bool & first, const char * key, int64_t value) {
    append_key(out, first, key);
    out << value;
}

void append_u64(std::ostringstream & out, bool & first, const char * key, uint64_t value) {
    append_key(out, first, key);
    out << value;
}

void append_i32(std::ostringstream & out, bool & first, const char * key, int32_t value) {
    append_key(out, first, key);
    out << value;
}

void append_double(std::ostringstream & out, bool & first, const char * key, double value) {
    append_key(out, first, key);
    if (std::isfinite(value)) {
        out << value;
    } else {
        out << "null";
    }
}

void append_bool(std::ostringstream & out, bool & first, const char * key, bool value) {
    append_key(out, first, key);
    out << (value ? "true" : "false");
}

} // namespace

std::string benchmark_record_to_json(const benchmark_record & r) {
    std::ostringstream out;
    out << std::setprecision(15);

    bool first = true;
    out << '{';
    append_string(out, first, "mode", r.mode);
    append_string(out, first, "backend", r.backend);
    append_string(out, first, "device", r.device);
    append_i32(out, first, "thread_count", r.thread_count);
    append_string(out, first, "model_type", r.model_type);
    append_string(out, first, "model_size", r.model_size);
    append_string(out, first, "tts_model", r.tts_model);
    append_string(out, first, "decoder_model", r.decoder_model);
    append_string(out, first, "quantization", r.quantization);
    append_string(out, first, "text", r.text);
    append_double(out, first, "audio_seconds", r.audio_seconds);

    if (r.audio_seconds > 0.0 && r.total_ms > 0) {
        const double total_seconds = static_cast<double>(r.total_ms) / 1000.0;
        append_double(out, first, "speed_x_realtime", r.audio_seconds / total_seconds);
        append_double(out, first, "wall_rtf", total_seconds / r.audio_seconds);
    }

    append_i64(out, first, "tokenize_ms", r.tokenize_ms);
    append_i64(out, first, "encode_ms", r.encode_ms);
    append_i64(out, first, "generate_ms", r.generate_ms);
    append_i64(out, first, "decode_ms", r.decode_ms);
    append_i64(out, first, "total_ms", r.total_ms);

    append_u64(out, first, "mem_rss_start_bytes", r.mem_rss_start_bytes);
    append_u64(out, first, "mem_rss_end_bytes", r.mem_rss_end_bytes);
    append_u64(out, first, "mem_rss_peak_bytes", r.mem_rss_peak_bytes);
    append_u64(out, first, "mem_phys_start_bytes", r.mem_phys_start_bytes);
    append_u64(out, first, "mem_phys_end_bytes", r.mem_phys_end_bytes);
    append_u64(out, first, "mem_phys_peak_bytes", r.mem_phys_peak_bytes);

    append_bool(out, first, "has_detailed_timing", r.has_detailed_timing);
    if (r.has_detailed_timing) {
        append_double(out, first, "prefill_build_ms", r.prefill_build_ms);
        append_double(out, first, "prefill_forward_ms", r.prefill_forward_ms);
        append_double(out, first, "prefill_graph_build_ms", r.prefill_graph_build_ms);
        append_double(out, first, "prefill_graph_alloc_ms", r.prefill_graph_alloc_ms);
        append_double(out, first, "prefill_compute_ms", r.prefill_compute_ms);
        append_double(out, first, "prefill_data_ms", r.prefill_data_ms);
        append_double(out, first, "talker_forward_ms", r.talker_forward_ms);
        append_double(out, first, "talker_graph_build_ms", r.talker_graph_build_ms);
        append_double(out, first, "talker_graph_alloc_ms", r.talker_graph_alloc_ms);
        append_double(out, first, "talker_compute_ms", r.talker_compute_ms);
        append_double(out, first, "talker_data_ms", r.talker_data_ms);
        append_double(out, first, "code_pred_ms", r.code_pred_ms);
        append_double(out, first, "code_pred_init_ms", r.code_pred_init_ms);
        append_double(out, first, "code_pred_prefill_ms", r.code_pred_prefill_ms);
        append_double(out, first, "code_pred_steps_ms", r.code_pred_steps_ms);
        append_double(out, first, "code_pred_graph_build_ms", r.code_pred_graph_build_ms);
        append_double(out, first, "code_pred_graph_alloc_ms", r.code_pred_graph_alloc_ms);
        append_double(out, first, "code_pred_compute_ms", r.code_pred_compute_ms);
        append_double(out, first, "code_pred_data_ms", r.code_pred_data_ms);
        append_double(out, first, "code_pred_coreml_ms", r.code_pred_coreml_ms);
        append_double(out, first, "embed_lookup_ms", r.embed_lookup_ms);
        append_i32(out, first, "frames", r.frames);
        append_double(out, first, "generate_total_ms", r.generate_total_ms);
    }
    out << "}\n";
    return out.str();
}

bool write_benchmark_record_json(const std::string & path,
                                 const benchmark_record & r,
                                 std::string & error) {
    std::ofstream file(path, std::ios::out | std::ios::trunc);
    if (!file) {
        error = "failed to open benchmark JSON file: " + path;
        return false;
    }

    file << benchmark_record_to_json(r);
    if (!file) {
        error = "failed to write benchmark JSON file: " + path;
        return false;
    }

    error.clear();
    return true;
}

} // namespace qwen3_tts
