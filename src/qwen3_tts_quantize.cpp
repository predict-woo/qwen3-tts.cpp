// qwen3-tts-quantize: Quantize GGUF TTS models from F16/F32 to Q8_0, Q4_K, etc.
//
// Usage: qwen3-tts-quantize --input <path> --output <path> --type <q8_0|q4_k|q4_0|q5_0|q5_1>
//        [--exclude <pattern>]  (keep matching tensors at original precision)

#include "common/gguf_loader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

struct quantize_params {
    std::string input_path;
    std::string output_path;
    enum ggml_type target_type = GGML_TYPE_Q8_0;
    std::vector<std::string> exclude_patterns;
};

static bool matches_exclude(const char * name, const std::vector<std::string> & patterns) {
    for (const auto & pat : patterns) {
        if (strstr(name, pat.c_str())) return true;
    }
    return false;
}

static bool should_quantize(const char * name, int n_dims, const std::vector<std::string> & excludes) {
    if (matches_exclude(name, excludes)) return false;
    // Only quantize 2D weight matrices
    if (n_dims < 2) return false;
    // Skip 3D tensors (conv1d weights — Q8_0 doesn't support 3D)
    if (n_dims >= 3) return false;
    // Keep embeddings, norms, biases, and heads at original precision
    if (strstr(name, "_embd") || strstr(name, "codebook")) return false;
    if (strstr(name, "_norm")) return false;
    if (strstr(name, ".bias")) return false;
    if (strstr(name, "lm_head") || strstr(name, "codec_head")) return false;
    // Keep speaker encoder at original precision (tiny, not worth quantizing)
    if (strstr(name, "spk_enc")) return false;
    return true;
}

static enum ggml_type parse_type(const char * s) {
    if (strcmp(s, "q8_0") == 0 || strcmp(s, "Q8_0") == 0) return GGML_TYPE_Q8_0;
    if (strcmp(s, "q4_0") == 0 || strcmp(s, "Q4_0") == 0) return GGML_TYPE_Q4_0;
    if (strcmp(s, "q4_1") == 0 || strcmp(s, "Q4_1") == 0) return GGML_TYPE_Q4_1;
    if (strcmp(s, "q5_0") == 0 || strcmp(s, "Q5_0") == 0) return GGML_TYPE_Q5_0;
    if (strcmp(s, "q5_1") == 0 || strcmp(s, "Q5_1") == 0) return GGML_TYPE_Q5_1;
    if (strcmp(s, "q4_k") == 0 || strcmp(s, "Q4_K") == 0) return GGML_TYPE_Q4_K;
    if (strcmp(s, "q5_k") == 0 || strcmp(s, "Q5_K") == 0) return GGML_TYPE_Q5_K;
    if (strcmp(s, "q6_k") == 0 || strcmp(s, "Q6_K") == 0) return GGML_TYPE_Q6_K;
    return GGML_TYPE_COUNT; // invalid
}

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s --input <path> --output <path> --type <type>\n", prog);
    fprintf(stderr, "\nSupported types: q8_0, q4_0, q4_1, q5_0, q5_1, q4_k, q5_k, q6_k\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --exclude <pattern>    Keep tensors matching pattern at original precision\n");
    fprintf(stderr, "                         (can be specified multiple times)\n");
}

int main(int argc, char ** argv) {
    quantize_params params;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            params.input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            params.output_path = argv[++i];
        } else if (arg == "--type" && i + 1 < argc) {
            params.target_type = parse_type(argv[++i]);
            if (params.target_type == GGML_TYPE_COUNT) {
                fprintf(stderr, "Error: unknown quantization type '%s'\n", argv[i]);
                return 1;
            }
        } else if (arg == "--exclude" && i + 1 < argc) {
            params.exclude_patterns.push_back(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Error: unknown argument '%s'\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (params.input_path.empty() || params.output_path.empty()) {
        fprintf(stderr, "Error: --input and --output are required\n");
        print_usage(argv[0]);
        return 1;
    }

    // Open input GGUF
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params gguf_params = {};
    gguf_params.no_alloc = false;
    gguf_params.ctx = &meta_ctx;
    struct gguf_context * ctx = gguf_init_from_file(params.input_path.c_str(), gguf_params);
    if (!ctx) {
        fprintf(stderr, "Error: failed to open '%s'\n", params.input_path.c_str());
        return 1;
    }

    int64_t n_tensors = gguf_get_n_tensors(ctx);
    int n_kv = gguf_get_n_kv(ctx);

    fprintf(stderr, "Input: %s\n", params.input_path.c_str());
    fprintf(stderr, "  Tensors: %lld, metadata keys: %d\n", (long long)n_tensors, n_kv);
    fprintf(stderr, "  Target type: %s\n", ggml_type_name(params.target_type));
    fprintf(stderr, "\n");

    // Create output GGUF writer and copy all metadata
    struct gguf_context * out_ctx = gguf_init_empty();
    gguf_set_kv(out_ctx, ctx);

    // Process tensors
    int n_quantized = 0;
    int n_kept = 0;
    size_t total_input_bytes = 0;
    size_t total_output_bytes = 0;
    std::vector<struct ggml_context *> temp_contexts; // keep alive until write

    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx, i);
        struct ggml_tensor * tensor = ggml_get_tensor(meta_ctx, name);
        if (!tensor) {
            fprintf(stderr, "Warning: tensor '%s' not found in meta context, skipping\n", name);
            continue;
        }

        int n_dims = ggml_n_dims(tensor);
        size_t nbytes = ggml_nbytes(tensor);
        total_input_bytes += nbytes;

        if (should_quantize(name, n_dims, params.exclude_patterns) &&
            (tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_F32)) {

            // Quantize: first convert to F32 if needed, then quantize
            int64_t n_elements = ggml_nelements(tensor);

            // Read source data as F32
            std::vector<float> f32_data(n_elements);
            if (tensor->type == GGML_TYPE_F32) {
                memcpy(f32_data.data(), tensor->data, n_elements * sizeof(float));
            } else if (tensor->type == GGML_TYPE_F16) {
                const ggml_fp16_t * fp16 = (const ggml_fp16_t *)tensor->data;
                for (int64_t j = 0; j < n_elements; j++) {
                    f32_data[j] = ggml_fp16_to_fp32(fp16[j]);
                }
            }

            // Quantize
            size_t qbytes = n_elements / ggml_blck_size(params.target_type) * ggml_type_size(params.target_type);
            std::vector<uint8_t> q_data(qbytes);
            int64_t quantized = ggml_quantize_chunk(params.target_type, f32_data.data(),
                                                     q_data.data(), 0, 1, n_elements, nullptr);
            if (quantized <= 0) {
                fprintf(stderr, "Warning: failed to quantize '%s', keeping original\n", name);
                gguf_add_tensor(out_ctx, tensor);
                n_kept++;
                total_output_bytes += nbytes;
                continue;
            }

            // Create quantized tensor
            size_t ctx_size = ggml_tensor_overhead() + qbytes + 64;
            struct ggml_init_params init_params = {};
            init_params.mem_size = ctx_size;
            init_params.mem_buffer = nullptr;
            init_params.no_alloc = false;
            struct ggml_context * tmp_ctx = ggml_init(init_params);
            struct ggml_tensor * q_tensor = ggml_new_tensor(tmp_ctx, params.target_type, n_dims, tensor->ne);
            ggml_set_name(q_tensor, name);
            memcpy(q_tensor->data, q_data.data(), qbytes);

            gguf_add_tensor(out_ctx, q_tensor);
            temp_contexts.push_back(tmp_ctx); // keep alive until write

            total_output_bytes += qbytes;
            n_quantized++;
            fprintf(stderr, "  [Q] %s: %s -> %s (%.1f MB -> %.1f MB)\n",
                    name, ggml_type_name(tensor->type), ggml_type_name(params.target_type),
                    nbytes / 1048576.0, qbytes / 1048576.0);
        } else {
            // Keep as-is
            gguf_add_tensor(out_ctx, tensor);
            n_kept++;
            total_output_bytes += nbytes;
        }
    }

    fprintf(stderr, "\nQuantized: %d tensors, kept: %d tensors\n", n_quantized, n_kept);
    fprintf(stderr, "Size: %.1f MB -> %.1f MB (%.1f%% reduction)\n",
            total_input_bytes / 1048576.0, total_output_bytes / 1048576.0,
            100.0 * (1.0 - (double)total_output_bytes / total_input_bytes));

    // Write output
    fprintf(stderr, "Writing %s...\n", params.output_path.c_str());
    gguf_write_to_file(out_ctx, params.output_path.c_str(), false);

    gguf_free(out_ctx);
    for (auto * tc : temp_contexts) ggml_free(tc);
    gguf_free(ctx);
    ggml_free(meta_ctx);

    fprintf(stderr, "Done.\n");
    return 0;
}
