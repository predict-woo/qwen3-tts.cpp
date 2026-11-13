/* audio_tokenizer_codec_encoder.h — Mimi encoder (audio → codec tokens)
 *
 * Port of the encoder half of Qwen3-TTS-Tokenizer-12Hz (Mimi variant).
 * The decoder is already implemented in audio_tokenizer_decoder.cpp.
 *
 * Architecture: docs/ENCODER_ARCHITECTURE.md (verified from tensors)
 * - SEANet convs (4 stages, strides 8,6,5,4 → 25Hz, ELU, causal)
 * - Bottleneck (conv.12 512→1024 k16 + conv.14 1024→512 k3, stride 1)
 * - Causal transformer (8L × 512H × 8 heads, LayerNorm+GELU, sliding
 *   window 250, RoPE θ=10000, layer scales)
 * - Post-transformer stride-2 downsample → 12.5Hz
 * - Split RVQ (1 semantic + 15 acoustic, L2 argmin)
 */
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <map>
#include <vector>

namespace qwen3_tts {

class GGUFLoader;

// ─── Configuration ───────────────────────────────────────────────────

struct MimiEncoderConfig {
    // SEANet — strides are REVERSED from upsampling_ratios [8,6,5,4],
    // verified from Python reference intermediate shapes:
    //   conv.0: stride 1 (no temporal reduction, just 1→64 channels)
    //   conv.3: stride 4, conv.6: stride 5, conv.9: stride 6
    //   conv.12 (bottleneck): stride 8 → 25 Hz
    static constexpr int NUM_STAGES = 4;
    static constexpr int CONV_STRIDES[NUM_STAGES] = {1, 4, 5, 6};
    static constexpr int ENTRY_KERNELS[NUM_STAGES] = {7, 8, 10, 12};
    static constexpr int BOTTLENECK_STRIDE = 8;
    static constexpr int RES_KERNEL = 3;     // residual conv kernel
    static constexpr int RES_KERNEL_2 = 1;   // second residual conv kernel
    static constexpr int COMPRESS = 2;       // residual bottleneck factor

    // Bottleneck
    static constexpr int BOTTLENECK_EXPAND_KERNEL = 16;   // conv.12
    static constexpr int BOTTLENECK_CONTRACT_KERNEL = 3;  // conv.14

    // Transformer
    static constexpr int HIDDEN_SIZE = 512;
    static constexpr int NUM_LAYERS = 8;
    static constexpr int NUM_HEADS = 8;
    static constexpr int SLIDING_WINDOW = 250;
    static constexpr float ROPE_THETA = 10000.0f;
    static constexpr float LAYER_SCALE_INIT = 0.01f;
    static constexpr int INTERMEDIATE_SIZE = 2048;  // GELU MLP

    // Downsample
    static constexpr int DOWNSAMPLE_KERNEL = 4;
    static constexpr int DOWNSAMPLE_STRIDE = 2;

    // Quantizer
    static constexpr int CODEBOOK_SIZE = 2048;
    static constexpr int CODEBOOK_DIM = 256;
    static constexpr int NUM_SEMANTIC = 1;
    static constexpr int NUM_ACOUSTIC_VALID = 15;  // layers 0-14 of 31

    // Audio
    int32_t sample_rate = 24000;
    static constexpr float FRAME_RATE = 12.5f;

    int32_t head_dim() const { return HIDDEN_SIZE / NUM_HEADS; }
};

// ─── Model tensors ───────────────────────────────────────────────────

struct MimiEncoderModel {
    MimiEncoderConfig config;
    bool loaded = false;

    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, struct ggml_tensor *> tensors;

    // SEANet entry convs: [stage]
    struct ggml_tensor * conv_entry_w[4] = {};
    struct ggml_tensor * conv_entry_b[4] = {};

    // SEANet residual convs: [stage][conv_idx]
    // conv_idx 0 = blk.1 (dilated), conv_idx 1 = blk.3 (pointwise)
    struct ggml_tensor * res_w[4][2] = {};
    struct ggml_tensor * res_b[4][2] = {};

    // Bottleneck convs (stride 1)
    struct ggml_tensor * bottleneck_expand_w = nullptr;   // 512→1024 k16
    struct ggml_tensor * bottleneck_expand_b = nullptr;
    struct ggml_tensor * bottleneck_contract_w = nullptr; // 1024→512 k3
    struct ggml_tensor * bottleneck_contract_b = nullptr;

    // Transformer: [layer]
    struct ggml_tensor * tfm_q[8] = {};
    struct ggml_tensor * tfm_k[8] = {};
    struct ggml_tensor * tfm_v[8] = {};
    struct ggml_tensor * tfm_o[8] = {};
    struct ggml_tensor * tfm_attn_ln_w[8] = {};
    struct ggml_tensor * tfm_attn_ln_b[8] = {};
    struct ggml_tensor * tfm_attn_scale[8] = {};
    struct ggml_tensor * tfm_ffn_up[8] = {};    // [512, 2048]
    struct ggml_tensor * tfm_ffn_down[8] = {};  // [2048, 512]
    struct ggml_tensor * tfm_ffn_ln_w[8] = {};
    struct ggml_tensor * tfm_ffn_ln_b[8] = {};
    struct ggml_tensor * tfm_ffn_scale[8] = {};

    // Post-transformer downsample
    struct ggml_tensor * downsample_w = nullptr; // [4, 512, 512]

    // Quantizer projections
    struct ggml_tensor * vq_sem_in = nullptr;   // [256, 512]
    struct ggml_tensor * vq_sem_out = nullptr;  // [512, 256]
    struct ggml_tensor * vq_aco_in = nullptr;
    struct ggml_tensor * vq_aco_out = nullptr;

    // Codebooks: [codebook_size, codebook_dim]
    struct ggml_tensor * vq_sem_cb = nullptr;
    struct ggml_tensor * vq_sem_usage = nullptr;  // null = already centroids (v2 GGUF)
    struct ggml_tensor * vq_aco_cb[15] = {};
    struct ggml_tensor * vq_aco_usage[15] = {};   // null = already centroids (v2 GGUF)
};

// ─── Encoder ─────────────────────────────────────────────────────────

class CodecEncoder {
public:
    CodecEncoder() = default;
    ~CodecEncoder() { unload(); }

    bool load(const std::string & model_path);
    void unload();
    bool is_loaded() const { return model_.loaded; }

    /// Encode 24kHz mono f32 audio → codec tokens.
    /// codec_tokens: [n_frames × 16] (1 semantic + 15 acoustic per frame)
    /// Uses the host-side F32 SEANet by default; toggle with set_use_f32()
    /// or QWEN3_TTS_ENCODER_F32=0 for the GGML F16 graph path.
    bool encode(const float * audio, int32_t n_samples,
                std::vector<int32_t> & codec_tokens,
                int32_t & n_frames);

    void set_use_f32(bool enabled) { use_f32_ = enabled; }

    const char * get_error() const { return error_msg_.c_str(); }
    const MimiEncoderConfig & config() const { return model_.config; }
    const std::map<std::string, struct ggml_tensor *> & model_tensors() const { return model_.tensors; }

private:
    bool encode_host_f32(const float * audio, int32_t n_samples,
                         std::vector<int32_t> & codec_tokens, int32_t & n_frames);
    bool encode_ggml(const float * audio, int32_t n_samples,
                     std::vector<int32_t> & codec_tokens, int32_t & n_frames);

    MimiEncoderModel model_;
    std::string error_msg_;
    bool use_f32_ = true;
};

} // namespace qwen3_tts
