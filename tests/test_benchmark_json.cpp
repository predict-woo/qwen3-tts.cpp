#include "common/benchmark_json.h"

#include <cassert>
#include <string>

int main() {
    qwen3_tts::benchmark_record r;
    r.mode = "default";
    r.backend = "cpu";
    r.text = "hello \"tts\"\nnext";
    r.audio_seconds = 2.0;
    r.total_ms = 1000;
    r.generate_ms = 700;
    r.mem_rss_start_bytes = 1024;
    r.mem_rss_end_bytes = 2048;
    r.has_detailed_timing = true;
    r.prefill_graph_build_ms = 1.25;

    const std::string json = qwen3_tts::benchmark_record_to_json(r);
    assert(json.find("\"mode\":\"default\"") != std::string::npos);
    assert(json.find("hello \\\"tts\\\"\\nnext") != std::string::npos);
    assert(json.find("\"speed_x_realtime\":2") != std::string::npos);
    assert(json.find("\"wall_rtf\":0.5") != std::string::npos);
    assert(json.find("\"mem_rss_start_bytes\":1024") != std::string::npos);
    assert(json.find("\"prefill_graph_build_ms\":1.25") != std::string::npos);

    qwen3_tts::benchmark_record zero;
    zero.audio_seconds = 0.0;
    zero.total_ms = 0;
    const std::string zero_json = qwen3_tts::benchmark_record_to_json(zero);
    assert(zero_json.find("\"speed_x_realtime\"") == std::string::npos);
    assert(zero_json.find("\"wall_rtf\"") == std::string::npos);

    return 0;
}
