#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <map>
#include <vector>

namespace qwen3_tts {

// Mimi encoder configuration
struct codec_encoder_config {
    int32_t sample_rate = 24000;
    int32_t hidden_size = 512;
    int32_t n_transformer_layers = 8;
    int32_t n_heads = 8;
    int32_t head_dim = 64;          // hidden_size / n_heads
    int32_t intermediate_size = 2048;
    int32_t n_valid_quantizers = 16; // first 16 of 32 total
    int32_t codebook_size = 2048;
    int32_t codebook_dim = 256;
    float rope_theta = 10000.0f;
    float norm_eps = 1e-5f;
    // CNN downsample strides in encoder order (reversed from upsampling_ratios)
    static constexpr int32_t cnn_strides[4] = {4, 5, 6, 8};
    // total audio downsample: 4*5*6*8 * 2 = 1920
};

// CNN ResUnit weights (ELU -> Conv k=3 -> ELU -> Conv k=1 -> residual add)
struct enc_resunit {
    struct ggml_tensor * conv1_w = nullptr;  // k=3
    struct ggml_tensor * conv1_b = nullptr;
    struct ggml_tensor * conv2_w = nullptr;  // k=1
    struct ggml_tensor * conv2_b = nullptr;
};

// Encoder transformer layer (LayerNorm, GELU MLP, LayerScale, RoPE)
struct enc_tfm_layer {
    struct ggml_tensor * attn_norm_w = nullptr;  // LayerNorm
    struct ggml_tensor * attn_norm_b = nullptr;
    struct ggml_tensor * attn_q_w = nullptr;
    struct ggml_tensor * attn_k_w = nullptr;
    struct ggml_tensor * attn_v_w = nullptr;
    struct ggml_tensor * attn_output_w = nullptr;
    struct ggml_tensor * attn_scale = nullptr;   // LayerScale
    struct ggml_tensor * ffn_norm_w = nullptr;   // LayerNorm
    struct ggml_tensor * ffn_norm_b = nullptr;
    struct ggml_tensor * ffn_up_w = nullptr;     // fc1: 512 -> 2048
    struct ggml_tensor * ffn_down_w = nullptr;   // fc2: 2048 -> 512
    struct ggml_tensor * ffn_scale = nullptr;    // LayerScale
};

// Codec encoder model weights
struct codec_encoder_model {
    codec_encoder_config config;

    // CNN layer 0: input conv [64, 1, 7]
    struct ggml_tensor * input_conv_w = nullptr;
    struct ggml_tensor * input_conv_b = nullptr;

    // 4 CNN stages: ResUnit + Downsample
    // ResUnits at layers 1, 4, 7, 10
    enc_resunit resunits[4];
    // Downsample convs at layers 3, 6, 9, 12
    struct ggml_tensor * ds_conv_w[4] = {};
    struct ggml_tensor * ds_conv_b[4] = {};

    // CNN layer 14: final projection [512, 1024, 3]
    struct ggml_tensor * final_conv_w = nullptr;
    struct ggml_tensor * final_conv_b = nullptr;

    // Transformer layers
    enc_tfm_layer tfm_layers[8];

    // Final downsample: stride 2, kernel 4 (no bias)
    struct ggml_tensor * final_ds_w = nullptr;

    // VQ semantic (1 codebook)
    struct ggml_tensor * vq_sem_in_proj = nullptr;   // [256, 512, 1] 1x1 conv
    struct ggml_tensor * vq_sem_codebook = nullptr;  // [2048, 256]
    struct ggml_tensor * vq_sem_usage = nullptr;     // [2048] cluster_usage
    struct ggml_tensor * vq_sem_out_proj = nullptr;  // [512, 256, 1]

    // VQ acoustic (15 codebooks for TTS, 31 total stored)
    struct ggml_tensor * vq_acou_in_proj = nullptr;
    struct ggml_tensor * vq_acou_codebook[31] = {};
    struct ggml_tensor * vq_acou_usage[31]    = {};
    struct ggml_tensor * vq_acou_out_proj = nullptr;

    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct codec_encoder_state {
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
};

// Encodes an audio waveform into discrete speech codes for ICL voice cloning.
// Ported from khimaros/qwen3-tts.cpp (commit 9c57131, with dec135e bias fix).
class AudioCodecEncoder {
public:
    AudioCodecEncoder();
    ~AudioCodecEncoder();

    // Load encoder weights from tokenizer GGUF (same file as vocoder).
    bool load_model(const std::string & model_path);
    void unload_model();

    // Encode audio to speech codes [n_frames, 16].
    // samples: 24kHz mono normalized [-1, 1]
    // codes:   output flattened [n_frames * 16], row-major
    bool encode(const float * samples, int32_t n_samples,
                std::vector<int32_t> & codes, int32_t & n_frames);

    const codec_encoder_config & get_config() const { return model_.config; }
    const std::string & get_error() const { return error_msg_; }

    // Apply thread count to loaded GGML backends where supported.
    bool set_n_threads(int32_t n_threads);

private:
    struct ggml_cgraph * build_graph(int32_t n_samples);

    struct ggml_tensor * apply_resunit(struct ggml_context * ctx,
                                       struct ggml_tensor * x,
                                       const enc_resunit & ru);

    struct ggml_tensor * apply_tfm_layer(struct ggml_context * ctx,
                                         struct ggml_tensor * x,
                                         const enc_tfm_layer & layer,
                                         int32_t seq_len,
                                         struct ggml_tensor * positions);

    // CPU-side VQ nearest-neighbor encoding.
    // features: [n_frames, 512] continuous encoder output
    void vq_encode(const float * features, int32_t n_frames,
                   std::vector<int32_t> & codes);

    // Find nearest codebook vector, return index.
    int32_t vq_nearest(const float * query, int32_t dim,
                       const void * codebook_data, enum ggml_type cb_type,
                       int32_t codebook_size);

    codec_encoder_model model_;
    codec_encoder_state state_;
    std::string error_msg_;
};

void free_codec_encoder_model(codec_encoder_model & model);

} // namespace qwen3_tts
