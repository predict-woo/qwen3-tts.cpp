#pragma once

#include <cstdint>
#include <string>

namespace qwen3_tts {

struct benchmark_record {
    std::string mode;
    std::string backend;
    std::string device;
    int32_t thread_count = 0;

    std::string model_type;
    std::string model_size;
    std::string tts_model;
    std::string decoder_model;
    std::string quantization;

    std::string text;
    double audio_seconds = 0.0;

    int64_t tokenize_ms = 0;
    int64_t encode_ms = 0;
    int64_t generate_ms = 0;
    int64_t decode_ms = 0;
    int64_t total_ms = 0;

    uint64_t mem_rss_start_bytes = 0;
    uint64_t mem_rss_end_bytes = 0;
    uint64_t mem_rss_peak_bytes = 0;
    uint64_t mem_phys_start_bytes = 0;
    uint64_t mem_phys_end_bytes = 0;
    uint64_t mem_phys_peak_bytes = 0;

    bool has_detailed_timing = false;
    double prefill_build_ms = 0.0;
    double prefill_forward_ms = 0.0;
    double prefill_graph_build_ms = 0.0;
    double prefill_graph_alloc_ms = 0.0;
    double prefill_compute_ms = 0.0;
    double prefill_data_ms = 0.0;
    double talker_forward_ms = 0.0;
    double talker_graph_build_ms = 0.0;
    double talker_graph_alloc_ms = 0.0;
    double talker_compute_ms = 0.0;
    double talker_data_ms = 0.0;
    double code_pred_ms = 0.0;
    double code_pred_init_ms = 0.0;
    double code_pred_prefill_ms = 0.0;
    double code_pred_steps_ms = 0.0;
    double code_pred_graph_build_ms = 0.0;
    double code_pred_graph_alloc_ms = 0.0;
    double code_pred_compute_ms = 0.0;
    double code_pred_data_ms = 0.0;
    double code_pred_coreml_ms = 0.0;
    double embed_lookup_ms = 0.0;
    int32_t frames = 0;
    double generate_total_ms = 0.0;
};

std::string benchmark_record_to_json(const benchmark_record & r);
bool write_benchmark_record_json(const std::string & path,
                                 const benchmark_record & r,
                                 std::string & error);

} // namespace qwen3_tts
