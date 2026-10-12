/* audio_tokenizer_codec_encoder.cpp — Mimi encoder implementation
 *
 * Single forward pass: audio → SEANet convs → causal transformer →
 * stride-2 downsample → split RVQ → codec tokens [n_frames × 16].
 *
 * Architecture: docs/ENCODER_ARCHITECTURE.md
 * Conventions: AGENTS.md (C++17, no exceptions, bool returns, etc.)
 */

#include "audio_tokenizer_codec_encoder.h"
#include "gguf_loader.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

namespace qwen3_tts {

#define ENC_LOG(...) do { fprintf(stderr, "[codec-encoder] " __VA_ARGS__); } while (0)

// ─── Helpers ──────────────────────────────────────────────────────────

// Debug: print tensor shape
static void print_shape(const char * label, struct ggml_tensor * t) {
    if (!t) { fprintf(stderr, "  %s: NULL\n", label); return; }
    fprintf(stderr, "  %s: [%lld, %lld, %lld, %lld] type=%d\n",
            label, (long long)t->ne[0], (long long)t->ne[1],
            (long long)t->ne[2], (long long)t->ne[3], (int)t->type);
}


// ─── Host-side F32 convolution (bypasses ggml_conv_1d F16 limitation) ──────
// PR #29 insight: ggml_mul_mat handles F32 weights natively on CPU.
// The F16 limitation is ONLY in ggml_im2col (the conv_1d kernel).
// By doing patch extraction on the host and feeding the result as a
// mul_mat input, we get full F32 precision through the SEANet stages.

/// Plain 1D convolution on the host, F32 throughout.
/// weight: [K, IC, OC] (kernel, in_channels, out_channels)
/// input: [T, IC] (temporal, channels)
/// output: [OW, OC] where OW = (T - effective_k) / stride + 1
static void host_conv1d(
    const float * weight,   // [K * IC * OC]
    const float * bias,     // [OC] or nullptr
    const float * input,    // [T * IC] (T rows of IC channels)
    int32_t T, int32_t IC, int32_t OC,
    int32_t K, int32_t stride, int32_t dilation,
    float * output,          // [OW * OC]
    int32_t * out_len)
{
    int32_t effective_k = (K - 1) * dilation + 1;
    int32_t OW = (T - effective_k) / stride + 1;
    *out_len = OW;

    for (int32_t ow = 0; ow < OW; ow++) {
        for (int32_t oc = 0; oc < OC; oc++) {
            float sum = bias ? bias[oc] : 0.0f;
            for (int32_t ic = 0; ic < IC; ic++) {
                for (int32_t k = 0; k < K; k++) {
                    int32_t input_pos = ow * stride + k * dilation;
                    if (input_pos >= 0 && input_pos < T) {
                        // GGML weight layout: ne=[K, IC, OC] → memory [OC, IC, K]
                        // float_index(k, ic, oc) = oc * IC * K + ic * K + k
                        // input layout: [T, IC] → inp[input_pos*IC + ic]
                        sum += weight[oc * IC * K + ic * K + k] * input[input_pos * IC + ic];
                    }
                }
            }
            output[ow * OC + oc] = sum;
        }
    }
}

/// Read a GGML tensor into a host F32 vector (handles F16→F32 conversion).
static std::vector<float> tensor_to_host_f32(struct ggml_tensor * t) {
    size_t nbytes = ggml_nbytes(t);
    std::vector<float> out;
    if (t->type == GGML_TYPE_F32) {
        out.resize(nbytes / sizeof(float));
        ggml_backend_tensor_get(t, out.data(), 0, nbytes);
    } else if (t->type == GGML_TYPE_F16) {
        size_t n = nbytes / sizeof(uint16_t);
        std::vector<uint16_t> f16(n);
        ggml_backend_tensor_get(t, f16.data(), 0, nbytes);
        out.resize(n);
        for (size_t i = 0; i < n; i++) {
            out[i] = ggml_fp16_to_fp32(f16[i]);
        }
    }
    return out;
}

/// Apply ELU activation element-wise.
static void host_elu(float * data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = data[i] > 0.0f ? data[i] : expf(data[i]) - 1.0f;
    }
}

// Forward declaration (definition at bottom of file)
static int32_t rvq_l2_argmin(
    const float * codebook, int32_t cb_size, int32_t cb_dim, const float * vector);

/// Causal 1D conv (left-pad only) + ELU activation.
/// Weight: [K, Cin, Cout]. Input: [Cin, T].
static struct ggml_tensor * causal_conv_elu(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * w,
    struct ggml_tensor * b,
    int32_t kernel,
    int32_t stride,
    int32_t dilation = 1)
{
    // Input/output layout: [T, C] (ne[0]=temporal, ne[1]=channels)
    // Causal-SAME padding (Mimi/encodec convention):
    //   pad_left = (k - 1) * dil + 1 - stride
    //   pad_right = extra to make output length divide by stride
    int32_t pad_left = (kernel - 1) * dilation + 1 - stride;
    if (pad_left < 0) pad_left = 0;
    // Compute output length to determine right padding
    int32_t T = (int32_t)x->ne[0];
    int32_t out_len = (T + pad_left - (kernel - 1) * dilation - 1) / stride + 1;
    int32_t needed = (out_len - 1) * stride + (kernel - 1) * dilation + 1;
    int32_t pad_right = needed - T - pad_left;
    if (pad_right < 0) pad_right = 0;
    if (pad_left > 0 || pad_right > 0) {
        x = ggml_pad_ext(ctx, x, pad_left, pad_right, 0, 0, 0, 0, 0, 0);
    }
    x = ggml_conv_1d(ctx, w, x, stride, 0, dilation);
    if (b) {
        // Broadcast bias [OC] → [1, OC, 1] across [T, OC, batch]
        int32_t oc = (int32_t)w->ne[2];
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, b, 1, oc, 1));
    }
    return ggml_elu(ctx,  x);
}

/// LayerNorm with weight and bias (Mimi encoder uses this, not RMSNorm).
static struct ggml_tensor * apply_layer_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * w,
    struct ggml_tensor * b,
    float eps)
{
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    if (b) {
        x = ggml_add(ctx, x, b);
    }
    return x;
}

/// SEANet residual block (Mimi pre-activation):
///   ELU(x) → conv1(dilated) → ELU(→) → conv2 → + residual
/// ELU comes BEFORE each conv, not after (Mimi's ResnetBlock uses
/// nn.Sequential(ELU(), Conv1d(...)) — verified from Python model).
static struct ggml_tensor * seanet_res_block(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * w1, struct ggml_tensor * b1,
    struct ggml_tensor * w2, struct ggml_tensor * b2,
    int32_t kernel1, int32_t kernel2,
    int32_t dilation)
{
    struct ggml_tensor * residual = x;
    struct ggml_tensor * h;

    // Pre-activation: ELU first
    h = ggml_elu(ctx, x);

    // conv1 (dilated, causal-SAME: stride=1 so pad_left = k_eff - 1)
    int32_t pad1 = (kernel1 - 1) * dilation;
    if (pad1 > 0) {
        h = ggml_pad_ext(ctx, h, pad1, 0, 0, 0, 0, 0, 0, 0);
    }
    h = ggml_conv_1d(ctx, w1, h, 1, 0, dilation);
    if (b1) {
        int32_t oc = (int32_t)w1->ne[2];
        h = ggml_add(ctx, h, ggml_reshape_3d(ctx, b1, 1, oc, 1));
    }

    // Pre-activation again: ELU before conv2
    h = ggml_elu(ctx, h);

    // conv2 (causal, typically kernel 1)
    int32_t pad2 = kernel2 - 1;
    if (pad2 > 0) {
        h = ggml_pad_ext(ctx, h, pad2, 0, 0, 0, 0, 0, 0, 0);
    }
    h = ggml_conv_1d(ctx, w2, h, 1, 0, 1);
    if (b2) {
        int32_t oc = (int32_t)w2->ne[2];
        h = ggml_add(ctx, h, ggml_reshape_3d(ctx, b2, 1, oc, 1));
    }

    // Residual connection (add to the PRE-activation input)
    return ggml_add(ctx, residual, h);
}

/// One transformer layer: LayerNorm → Self-Attn → LayerNorm → GELU MLP.
static struct ggml_tensor * encoder_transformer_layer(
    struct ggml_context * ctx,
    const MimiEncoderModel & model,
    struct ggml_tensor * x,
    int layer_idx,
    int32_t n_positions)
{
    const auto & cfg = model.config;
    const int n_heads = cfg.NUM_HEADS;
    const int head_dim = cfg.head_dim();
    const int hidden = cfg.HIDDEN_SIZE;
    const float eps = 1e-5f;
    const float inv_sqrt_hd = 1.0f / sqrtf((float)head_dim);

    struct ggml_tensor * residual = x;

    // ── Self-attention ──
    struct ggml_tensor * normed = apply_layer_norm(
        ctx, x, model.tfm_attn_ln_w[layer_idx], model.tfm_attn_ln_b[layer_idx], eps);

    // Q/K/V weights were permuted at load time (kyutai → standard block layout)
    // so the output of mul_mat is already in standard block layout.
    struct ggml_tensor * Q = ggml_mul_mat(ctx, model.tfm_q[layer_idx], normed);
    struct ggml_tensor * K = ggml_mul_mat(ctx, model.tfm_k[layer_idx], normed);
    struct ggml_tensor * V = ggml_mul_mat(ctx, model.tfm_v[layer_idx], normed);

    Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, n_positions);
    K = ggml_reshape_3d(ctx, K, head_dim, n_heads, n_positions);
    V = ggml_reshape_3d(ctx, V, head_dim, n_heads, n_positions);

    // RoPE positions [0, 1, ..., n_positions-1] — unique name per layer
    // (all layers share the same VALUES but need distinct graph nodes)
    char pos_name[32];
    snprintf(pos_name, sizeof(pos_name), "enc_pos_%d", layer_idx);
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_positions);
    ggml_set_name(pos, pos_name);
    ggml_set_input(pos);

    Q = ggml_rope_ext(ctx, Q, pos, nullptr,
                      head_dim, GGML_ROPE_TYPE_NEOX, 0,
                      cfg.ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx, K, pos, nullptr,
                      head_dim, GGML_ROPE_TYPE_NEOX, 0,
                      cfg.ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Permute to [head_dim, n_positions, n_heads]
    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
    K = ggml_permute(ctx, K, 0, 2, 1, 3);
    V = ggml_permute(ctx, V, 0, 2, 1, 3);

    // Attention: KQ = K·Q^T / √d, causal mask, softmax
    struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);
    KQ = ggml_scale(ctx, KQ, inv_sqrt_hd);
    KQ = ggml_diag_mask_inf(ctx, KQ, 0);  // causal
    KQ = ggml_soft_max(ctx, KQ);

    // KQV = V · softmax(KQ)
    V = ggml_cont(ctx, ggml_transpose(ctx, V));
    struct ggml_tensor * KQV = ggml_mul_mat(ctx, V, KQ);
    KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);
    struct ggml_tensor * attn_out = ggml_cont_2d(ctx, KQV, hidden, n_positions);

    // Output projection + layer scale
    attn_out = ggml_mul_mat(ctx, model.tfm_o[layer_idx], attn_out);
    if (model.tfm_attn_scale[layer_idx]) {
        attn_out = ggml_mul(ctx, attn_out, model.tfm_attn_scale[layer_idx]);
    }

    // Residual
    x = ggml_add(ctx, residual, attn_out);
    residual = x;

    // ── MLP (GELU, non-gated: fc1 → GELU → fc2) ──
    normed = apply_layer_norm(
        ctx, x, model.tfm_ffn_ln_w[layer_idx], model.tfm_ffn_ln_b[layer_idx], eps);

    struct ggml_tensor * ff = ggml_mul_mat(ctx, model.tfm_ffn_up[layer_idx], normed);
    ff = ggml_gelu(ctx, ff);
    ff = ggml_mul_mat(ctx, model.tfm_ffn_down[layer_idx], ff);

    if (model.tfm_ffn_scale[layer_idx]) {
        ff = ggml_mul(ctx, ff, model.tfm_ffn_scale[layer_idx]);
    }

    return ggml_add(ctx, residual, ff);
}

// ─── Load from GGUF ────────────────────────────────────────────────────


/// Read a GGML tensor into a float vector, converting from F16 if needed.
static bool read_tensor_as_f32(
    struct ggml_tensor * tensor,
    std::vector<float> & out,
    std::string & error)
{
    size_t nbytes = ggml_nbytes(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        out.resize(nbytes / sizeof(float));
        ggml_backend_tensor_get(tensor, out.data(), 0, nbytes);
    } else if (tensor->type == GGML_TYPE_F16) {
        size_t n_f16 = nbytes / sizeof(uint16_t);
        std::vector<uint16_t> f16_buf(n_f16);
        ggml_backend_tensor_get(tensor, f16_buf.data(), 0, nbytes);
        out.resize(n_f16);
        for (size_t i = 0; i < n_f16; i++) {
            out[i] = ggml_fp16_to_fp32(f16_buf[i]);
        }
    } else {
        error = "unsupported tensor type for read_tensor_as_f32: " +
                std::to_string((int)tensor->type);
        return false;
    }
    return true;
}

/// Read an encoder codebook and normalize EMA sums → centroids.
/// The ORIGINAL tokenizer GGUF stores raw embed_sum (NOT normalized);
/// the decoder normalizes at load time. The encoder must do the same.
/// If the GGUF already contains normalized centroids (v2), this is a no-op
/// (detected by checking if a companion usage tensor exists).
static bool read_encoder_codebook_normalized(
    struct ggml_tensor * codebook,
    struct ggml_tensor * usage,  // may be nullptr (v2 GGUF)
    std::vector<float> & out,
    std::string & error)
{
    if (!read_tensor_as_f32(codebook, out, error)) {
        return false;
    }

    // If no usage tensor, the codebook is already normalized (v2 GGUF)
    if (!usage) {
        return true;
    }

    // Read usage tensor and normalize: centroid = embed_sum / usage
    std::vector<float> usage_data;
    if (!read_tensor_as_f32(usage, usage_data, error)) {
        return false;
    }

    // Codebook shape: [dim, entries] → out is [entries × dim]
    // Usage shape: [entries] (one count per entry)
    int32_t dim = (int32_t)codebook->ne[0];
    int32_t entries = (int32_t)codebook->ne[1];

    if ((int32_t)usage_data.size() != entries) {
        error = "codebook/usage size mismatch";
        return false;
    }

    for (int32_t e = 0; e < entries; e++) {
        float count = usage_data[e];
        if (count < 1e-5f) count = 1e-5f;
        for (int32_t d = 0; d < dim; d++) {
            out[(size_t)e * dim + d] /= count;
        }
    }

    return true;
}


/// Permute a Q/K/V weight from kyutai strided-head layout to standard
/// block layout. Weight is [hidden, hidden] = [512, 512].
/// Kyutai: feature f = d*8 + h → Standard: feature f = h*64 + d.
/// Reshape to [n_heads, head_dim, hidden] → permute(1,0,2) → reshape [hidden, hidden].
static struct ggml_tensor * permute_qkv_weight_kyutai_to_block(
    struct ggml_context * ctx,
    struct ggml_tensor * w,
    int n_heads, int head_dim)
{
    if (!w) return nullptr;
    int hidden = n_heads * head_dim;
    // Reshape [512, 512] → [8, 64, 512] (n_heads fastest, then head_dim, then output)
    // But GGML weight from GGUF is [in=512, out=512] with ne[0]=512(in), ne[1]=512(out)
    // We need to permute the INPUT dimension
    // Actually the weight is [out=512, in=512] in PyTorch, stored as ne[0]=in, ne[1]=out in GGML
    // For the input dim (ne[0]=512): kyutai feature f = d*8 + h
    // We want to remap to block: f' = h*64 + d
    // This is a permutation of the 512 input features
    
    // In GGML, the weight tensor is already loaded and its data is in backend buffer.
    // We can't permute it in-place; we need to create a new tensor.
    // For now, this is handled at the graph level instead.
    return w; // Placeholder — actual permutation happens in the graph
}

bool CodecEncoder::load(const std::string & model_path) {
    // Runtime path toggle: QWEN3_TTS_ENCODER_F32=0 selects the GGML F16
    // graph path (default 1 = host-side F32 on this branch)
    if (const char * env = getenv("QWEN3_TTS_ENCODER_F32")) {
        use_f32_ = (strcmp(env, "0") != 0);
    }
    unload();

    GGUFLoader loader;
    if (!loader.open(model_path)) {
        error_msg_ = loader.get_error();
        return false;
    }

    model_.config.sample_rate = loader.get_u32("qwen3-tts.tokenizer.sample_rate", 24000);

    int64_t n_tensors = loader.get_n_tensors();

    int enc_tensor_count = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (name && strncmp(name, "tok_enc.", 8) == 0) {
            enc_tensor_count++;
        }
    }

    if (enc_tensor_count == 0) {
        error_msg_ = "No encoder tensors (tok_enc.*) found in model";
        return false;
    }

    ENC_LOG("Found %d encoder tensors\n", enc_tensor_count);

    size_t ctx_size = ggml_tensor_overhead() * enc_tensor_count;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_msg_ = "Failed to initialize GGML context";
        return false;
    }

    struct gguf_context * gguf_ctx = loader.get_ctx();
    struct ggml_context * meta_ctx = loader.get_meta_ctx();

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name || strncmp(name, "tok_enc.", 8) != 0) {
            continue;
        }

        struct ggml_tensor * meta_tensor = ggml_get_tensor(meta_ctx, name);
        if (!meta_tensor) {
            continue;
        }

        struct ggml_tensor * tensor = ggml_dup_tensor(model_.ctx, meta_tensor);
        ggml_set_name(tensor, name);
        model_.tensors[name] = tensor;

        std::string sname(name);

        // SEANet entry convs
        if (sname == "tok_enc.conv.0.weight")  model_.conv_entry_w[0] = tensor;
        if (sname == "tok_enc.conv.0.bias")    model_.conv_entry_b[0] = tensor;
        if (sname == "tok_enc.conv.3.weight")  model_.conv_entry_w[1] = tensor;
        if (sname == "tok_enc.conv.3.bias")    model_.conv_entry_b[1] = tensor;
        if (sname == "tok_enc.conv.6.weight")  model_.conv_entry_w[2] = tensor;
        if (sname == "tok_enc.conv.6.bias")    model_.conv_entry_b[2] = tensor;
        if (sname == "tok_enc.conv.9.weight")  model_.conv_entry_w[3] = tensor;
        if (sname == "tok_enc.conv.9.bias")    model_.conv_entry_b[3] = tensor;

        // SEANet bottleneck
        if (sname == "tok_enc.conv.12.weight") model_.bottleneck_expand_w = tensor;
        if (sname == "tok_enc.conv.12.bias")   model_.bottleneck_expand_b = tensor;
        if (sname == "tok_enc.conv.14.weight") model_.bottleneck_contract_w = tensor;
        if (sname == "tok_enc.conv.14.bias")   model_.bottleneck_contract_b = tensor;

        // Residual blocks
        for (int stage = 0; stage < 4; stage++) {
            int layer_idx = stage * 3 + 1;
            for (int ci = 0; ci < 2; ci++) {
                int conv_idx = ci == 0 ? 1 : 3;
                std::string wname = "tok_enc.res." + std::to_string(layer_idx) +
                    ".blk." + std::to_string(conv_idx) + ".weight";
                std::string bname = "tok_enc.res." + std::to_string(layer_idx) +
                    ".blk." + std::to_string(conv_idx) + ".bias";
                if (sname == wname) model_.res_w[stage][ci] = tensor;
                if (sname == bname) model_.res_b[stage][ci] = tensor;
            }
        }

        // Transformer
        for (int layer = 0; layer < 8; layer++) {
            std::string p = "tok_enc.blk." + std::to_string(layer) + ".";
            if (sname == p + "attn_q.weight")         model_.tfm_q[layer] = tensor;
            if (sname == p + "attn_k.weight")         model_.tfm_k[layer] = tensor;
            if (sname == p + "attn_v.weight")         model_.tfm_v[layer] = tensor;
            if (sname == p + "attn_output.weight")    model_.tfm_o[layer] = tensor;
            if (sname == p + "attn_norm.weight")      model_.tfm_attn_ln_w[layer] = tensor;
            if (sname == p + "attn_norm.bias")        model_.tfm_attn_ln_b[layer] = tensor;
            if (sname == p + "attn_scale")            model_.tfm_attn_scale[layer] = tensor;
            if (sname == p + "ffn_up.weight")         model_.tfm_ffn_up[layer] = tensor;
            if (sname == p + "ffn_down.weight")       model_.tfm_ffn_down[layer] = tensor;
            if (sname == p + "ffn_norm.weight")       model_.tfm_ffn_ln_w[layer] = tensor;
            if (sname == p + "ffn_norm.bias")         model_.tfm_ffn_ln_b[layer] = tensor;
            if (sname == p + "ffn_scale")             model_.tfm_ffn_scale[layer] = tensor;
        }

        // Downsample
        if (sname == "tok_enc.downsample.weight") model_.downsample_w = tensor;

        // Quantizer
        if (sname == "tok_enc.vq_semantic.input_proj.weight")  model_.vq_sem_in = tensor;
        if (sname == "tok_enc.vq_semantic.output_proj.weight") model_.vq_sem_out = tensor;
        if (sname == "tok_enc.vq_acoustic.input_proj.weight")  model_.vq_aco_in = tensor;
        if (sname == "tok_enc.vq_acoustic.output_proj.weight") model_.vq_aco_out = tensor;
        if (sname == "tok_enc.vq_semantic.0.codebook") model_.vq_sem_cb = tensor;
        if (sname == "tok_enc.vq_semantic.0.usage") model_.vq_sem_usage = tensor;
        for (int q = 0; q < 15; q++) {
            if (sname == "tok_enc.vq_acoustic." + std::to_string(q) + ".codebook") {
                model_.vq_aco_cb[q] = tensor;
            }
            if (sname == "tok_enc.vq_acoustic." + std::to_string(q) + ".usage") {
                model_.vq_aco_usage[q] = tensor;
            }
        }
    }

    if (!load_tensor_data_from_file(model_path, gguf_ctx, model_.ctx,
                                     model_.tensors, model_.buffer, error_msg_)) {
        error_msg_ = "Failed to load encoder tensor data: " + error_msg_;
        return false;
    }

    model_.loaded = true;
    ENC_LOG("Encoder loaded: %d tensors\n", enc_tensor_count);
    return true;
}

void CodecEncoder::unload() {
    if (model_.buffer) {
        ggml_backend_buffer_free(model_.buffer);
        model_.buffer = nullptr;
    }
    if (model_.ctx) {
        ggml_free(model_.ctx);
        model_.ctx = nullptr;
    }
    model_.tensors.clear();
    model_.loaded = false;
}

// ─── Encode: full forward pass ────────────────────────────────────────

bool CodecEncoder::encode(
    const float * audio,
    int32_t n_samples,
    std::vector<int32_t> & codec_tokens,
    int32_t & n_frames)
{
    // Dispatch: host-side F32 SEANet (default on this branch) or the GGML
    // F16 graph path (the #30 encoder). Toggle with set_use_f32() or
    // QWEN3_TTS_ENCODER_F32=0/1 (default 1).
    return use_f32_ ? encode_host_f32(audio, n_samples, codec_tokens, n_frames)
                    : encode_ggml(audio, n_samples, codec_tokens, n_frames);
}

bool CodecEncoder::encode_host_f32(
    const float * audio,
    int32_t n_samples,
    std::vector<int32_t> & codec_tokens,
    int32_t & n_frames)
{
    if (!model_.loaded) {
        error_msg_ = "encoder not loaded";
        return false;
    }
    if (!audio || n_samples <= 0) {
        error_msg_ = "invalid audio input";
        return false;
    }

    const auto & cfg = model_.config;
    const int32_t hidden = cfg.HIDDEN_SIZE;

    // ── Stage 1: SEANet convolutions (HOST-SIDE, F32) ──
    // This bypasses ggml_conv_1d's F16 limitation entirely.
    // Input: [T=n_samples, C=1]
    std::vector<float> buf(audio, audio + n_samples);

    // Activation dumps for parity debugging (QWEN3_TTS_ENC_ACT_DIR=<dir>)
    const char * act_dir = getenv("QWEN3_TTS_ENC_ACT_DIR");
    auto dump_act = [&](const char * name) {
        if (!act_dir) return;
        std::string p = std::string(act_dir) + "/" + name;
        FILE * f = fopen(p.c_str(), "wb");
        if (f) { fwrite(buf.data(), 4, buf.size(), f); fclose(f); }
    };

    // Helper to run one conv stage: pad → conv → bias → ELU
    auto run_conv = [&](struct ggml_tensor * w, struct ggml_tensor * b,
                        int32_t stride) -> bool {
        if (!w) { error_msg_ = "missing conv weight"; return false; }
        int32_t K = (int32_t)w->ne[0];   // kernel
        int32_t IC = (int32_t)w->ne[1];  // in channels
        int32_t OC = (int32_t)w->ne[2];  // out channels
        int32_t T = (int32_t)buf.size() / IC;

        // Causal-SAME padding. Output length is ceil(T / stride) — the
        // floor formula loses a position when T % stride != 0 (e.g. conv.12
        // with T=1180, stride 8: 148, not 147).
        int32_t pad_left = (K - 1) + 1 - stride;
        if (pad_left < 0) pad_left = 0;
        int32_t OW = (T + stride - 1) / stride;  // ceil(T / stride)
        if (OW < 1) OW = 1;
        int32_t needed = (OW - 1) * stride + K;
        int32_t pad_right = needed - T - pad_left;
        if (pad_right < 0) pad_right = 0;

        // Pad input
        if (pad_left > 0 || pad_right > 0) {
            std::vector<float> padded((size_t)(T + pad_left + pad_right) * IC, 0.0f);
            memcpy(padded.data() + (size_t)pad_left * IC, buf.data(), (size_t)T * IC * sizeof(float));
            buf = std::move(padded);
            T = T + pad_left + pad_right;
        }

        // Read weight and bias to host F32
        std::vector<float> w_f32 = tensor_to_host_f32(w);
        std::vector<float> b_f32 = b ? tensor_to_host_f32(b) : std::vector<float>();

        // Convolution
        std::vector<float> out((size_t)OW * OC);
        int32_t out_len;
        host_conv1d(w_f32.data(), b_f32.empty() ? nullptr : b_f32.data(),
                     buf.data(), T, IC, OC, K, stride, 1,
                     out.data(), &out_len);
        // NOTE: no activation here — Python's MimiConv1d is padding+conv
        // only. ELUs live at explicit positions (after resblocks, after
        // conv.12), applied by the callers.
        buf = std::move(out);
        ENC_LOG("  conv: K=%d IC=%d OC=%d stride=%d → [%d, %d]\n", K, IC, OC, stride, out_len, OC);
        return true;
    };

    // Helper for pre-activation residual block: ELU → conv1 → ELU → conv2 → +residual
    auto run_res_block = [&](struct ggml_tensor * w1, struct ggml_tensor * b1,
                              struct ggml_tensor * w2, struct ggml_tensor * b2) -> bool {
        if (!w1 || !w2) { error_msg_ = "missing res block weight"; return false; }
        int32_t IC = (int32_t)w1->ne[1];
        int32_t T = (int32_t)buf.size() / IC;

        // Save residual
        std::vector<float> residual = buf;

        // Pre-activation: ELU first
        host_elu(buf.data(), buf.size());

        // conv1 (dilated, kernel 3)
        {
            int32_t K1 = (int32_t)w1->ne[0];
            int32_t OC1 = (int32_t)w1->ne[2];
            int32_t dilation = 1;
            int32_t effective_k = (K1 - 1) * dilation + 1;
            int32_t pad_left = effective_k - 1; // causal for dilated conv
            std::vector<float> padded((size_t)(T + pad_left) * IC, 0.0f);
            memcpy(padded.data() + (size_t)pad_left * IC, buf.data(), (size_t)T * IC * sizeof(float));
            std::vector<float> w1_f32 = tensor_to_host_f32(w1);
            std::vector<float> b1_f32 = b1 ? tensor_to_host_f32(b1) : std::vector<float>();
            std::vector<float> out((size_t)T * OC1);
            int32_t out_len;
            host_conv1d(w1_f32.data(), b1_f32.empty() ? nullptr : b1_f32.data(),
                         padded.data(), T + pad_left, IC, OC1, K1, 1, dilation,
                         out.data(), &out_len);
            host_elu(out.data(), out.size());
            buf = std::move(out);
        }

        // conv2 (kernel 1, no padding needed)
        {
            int32_t IC2 = (int32_t)w2->ne[1];
            int32_t OC2 = (int32_t)w2->ne[2];
            std::vector<float> w2_f32 = tensor_to_host_f32(w2);
            std::vector<float> b2_f32 = b2 ? tensor_to_host_f32(b2) : std::vector<float>();
            std::vector<float> out((size_t)T * OC2);
            int32_t out_len;
            host_conv1d(w2_f32.data(), b2_f32.empty() ? nullptr : b2_f32.data(),
                         buf.data(), T, IC2, OC2, 1, 1, 1,
                         out.data(), &out_len);
            // NO ELU after conv2 (pre-activation: ELU already applied before conv1)
            buf = std::move(out);
        }

        // Residual connection
        if (buf.size() == residual.size()) {
            for (size_t i = 0; i < buf.size(); i++) buf[i] += residual[i];
        }
        return true;
    };

    ENC_LOG("SEANet stages (host-side F32):\n");

    // Stage 0: conv.0 (stride 1, k7, 1→64) + res + ELU
    if (!run_conv(model_.conv_entry_w[0], model_.conv_entry_b[0], 1)) return false;
    dump_act("cpp_layer_00.f32");
    if (!run_res_block(model_.res_w[0][0], model_.res_b[0][0],
                       model_.res_w[0][1], model_.res_b[0][1])) return false;
    dump_act("cpp_layer_01.f32");
    host_elu(buf.data(), buf.size()); // post-resblock ELU

    // Stage 1: conv.3 (stride 4, k8, 64→128) + res + ELU
    if (!run_conv(model_.conv_entry_w[1], model_.conv_entry_b[1], 4)) return false;
    dump_act("cpp_layer_03.f32");
    if (!run_res_block(model_.res_w[1][0], model_.res_b[1][0],
                       model_.res_w[1][1], model_.res_b[1][1])) return false;
    dump_act("cpp_layer_04.f32");
    host_elu(buf.data(), buf.size());

    // Stage 2: conv.6 (stride 5, k10, 128→256) + res + ELU
    if (!run_conv(model_.conv_entry_w[2], model_.conv_entry_b[2], 5)) return false;
    dump_act("cpp_layer_06.f32");
    if (!run_res_block(model_.res_w[2][0], model_.res_b[2][0],
                       model_.res_w[2][1], model_.res_b[2][1])) return false;
    dump_act("cpp_layer_07.f32");
    host_elu(buf.data(), buf.size());

    // Stage 3: conv.9 (stride 6, k12, 256→512) + res + ELU
    if (!run_conv(model_.conv_entry_w[3], model_.conv_entry_b[3], 6)) return false;
    dump_act("cpp_layer_09.f32");
    if (!run_res_block(model_.res_w[3][0], model_.res_b[3][0],
                       model_.res_w[3][1], model_.res_b[3][1])) return false;
    dump_act("cpp_layer_10.f32");
    host_elu(buf.data(), buf.size());

    // Bottleneck expand: conv.12 (stride 8, k16, 512→1024), then ELU
    // (Python layer_13 is a bare ELU between conv.12 and conv.14)
    if (!run_conv(model_.bottleneck_expand_w, model_.bottleneck_expand_b, 8)) return false;
    host_elu(buf.data(), buf.size());
    dump_act("cpp_layer_13.f32");  // conv.12 + ELU (Python layer_13 = the ELU)

    // Bottleneck contract: conv.14 (stride 1, k3, 1024→512) NO ELU
    // The contract output goes directly to the transformer — no activation.
    {
        struct ggml_tensor * cw = model_.bottleneck_contract_w;
        struct ggml_tensor * cb = model_.bottleneck_contract_b;
        int32_t cK = (int32_t)cw->ne[0];
        int32_t cIC = (int32_t)cw->ne[1];
        int32_t cOC = (int32_t)cw->ne[2];
        int32_t cT = (int32_t)buf.size() / cIC;
        int32_t cpad = cK - 1;
        std::vector<float> cpadded((size_t)(cT + cpad) * cIC, 0.0f);
        memcpy(cpadded.data() + (size_t)cpad * cIC, buf.data(), (size_t)cT * cIC * sizeof(float));
        std::vector<float> cw_f32 = tensor_to_host_f32(cw);
        std::vector<float> cb_f32 = cb ? tensor_to_host_f32(cb) : std::vector<float>();
        std::vector<float> cout((size_t)cT * cOC);
        int32_t clen;
        host_conv1d(cw_f32.data(), cb_f32.empty() ? nullptr : cb_f32.data(),
                     cpadded.data(), cT + cpad, cIC, cOC, cK, 1, 1,
                     cout.data(), &clen);
        buf = std::move(cout);
        dump_act("cpp_layer_14.f32");
        ENC_LOG("  contract (no ELU): [%d, %d]\n", clen, cOC);
    }

    int32_t seanet_T = (int32_t)buf.size() / hidden;
    ENC_LOG("SEANet output: [%d, %d]\n", seanet_T, hidden);

    // ── Stage 2: Transformer (GGML graph — mul_mat only, no F16 limitation) ──
    // The transformer uses only ggml_mul_mat, ggml_norm, ggml_rope, etc.
    // No convolutions → F32 works natively per PR #29's insight.
    {
        int32_t n_pos = seanet_T;
        ENC_LOG("Transformer: %d positions × %d hidden\n", n_pos, hidden);

        // Build graph
        size_t ctx_size = 256 * 1024 * 1024; // 256 MB for transformer graph
        struct ggml_init_params gparams = {
            /*.mem_size =*/ ctx_size,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc =*/ true,
        };
        struct ggml_context * ctx0 = ggml_init(gparams);
        if (!ctx0) { error_msg_ = "failed to init transformer ctx"; return false; }

        // Input: [hidden, n_pos] (channel-major for mul_mat)
        struct ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden, n_pos);
        ggml_set_name(inp, "seanet_out");
        ggml_set_input(inp);

        // Positions for RoPE
        struct ggml_tensor * pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
        ggml_set_name(pos, "tfm_pos");
        ggml_set_input(pos);

        struct ggml_tensor * x = inp;
        std::vector<struct ggml_tensor *> tfm_layer_outs;
        std::vector<struct ggml_tensor *> tfm_attn_outs;
        std::vector<struct ggml_tensor *> dbg_tensors;
        std::vector<std::string> dbg_names;
        std::vector<struct ggml_tensor *> dbg_weight_tensors;
        std::vector<std::string> dbg_weight_names;

        // 8 transformer layers
        for (int layer = 0; layer < 8; layer++) {
            // LayerNorm
            struct ggml_tensor * normed_raw = ggml_norm(ctx0, x, 1e-5f);
            struct ggml_tensor * normed = ggml_mul(ctx0, normed_raw, model_.tfm_attn_ln_w[layer]);
            if (model_.tfm_attn_ln_b[layer]) {
                normed = ggml_add(ctx0, normed, model_.tfm_attn_ln_b[layer]);
            }
            if (act_dir && layer == 0) {
                ggml_set_output(normed_raw);
                dbg_tensors.push_back(normed_raw);
                dbg_names.push_back("cpp_norm_raw.f32");
            }

            if (act_dir && layer == 0) {
                ggml_set_output(normed);
                dbg_tensors.push_back(normed);
                dbg_names.push_back("cpp_normed.f32");
                // also capture the RUNTIME weight bytes for verification
                dbg_weight_tensors.push_back(model_.tfm_q[layer]);
                dbg_weight_names.push_back("cpp_runtime_qw.f32");
            }

            // Q/K/V projections
            struct ggml_tensor * Q = ggml_mul_mat(ctx0, model_.tfm_q[layer], normed);
            struct ggml_tensor * K = ggml_mul_mat(ctx0, model_.tfm_k[layer], normed);
            struct ggml_tensor * V = ggml_mul_mat(ctx0, model_.tfm_v[layer], normed);

            int32_t n_heads = cfg.NUM_HEADS;
            int32_t head_dim = cfg.head_dim();

            Q = ggml_reshape_3d(ctx0, Q, head_dim, n_heads, n_pos);
            K = ggml_reshape_3d(ctx0, K, head_dim, n_heads, n_pos);
            V = ggml_reshape_3d(ctx0, V, head_dim, n_heads, n_pos);

            // RoPE
            struct ggml_tensor * Q_pre_rope = Q;
            struct ggml_tensor * K_pre_rope = K;
            Q = ggml_rope_ext(ctx0, Q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0,
                               cfg.ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            K = ggml_rope_ext(ctx0, K, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0,
                               cfg.ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            struct ggml_tensor * Q_rope = Q;
            struct ggml_tensor * K_rope = K;
            struct ggml_tensor * V_raw = V;
            if (act_dir && layer == 0) {
                ggml_set_output(Q_pre_rope); ggml_set_output(K_pre_rope);
                ggml_set_output(Q_rope); ggml_set_output(K_rope);
                ggml_set_output(V_raw);
                for (auto * t2 : {Q_pre_rope, K_pre_rope, Q_rope, K_rope, V_raw}) {
                    dbg_tensors.push_back(t2);
                }
                for (const char * n2 : {"cpp_q_pre_rope.f32", "cpp_k_pre_rope.f32",
                                        "cpp_q_rope.f32", "cpp_k_rope.f32", "cpp_v_raw.f32"}) {
                    dbg_names.push_back(n2);
                }
            }

            // Permute to [head_dim, n_pos, n_heads]
            Q = ggml_permute(ctx0, Q, 0, 2, 1, 3);
            K = ggml_permute(ctx0, K, 0, 2, 1, 3);
            V = ggml_permute(ctx0, V, 0, 2, 1, 3);

            // Attention: KQ = K^T · Q / √d, causal mask, softmax
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            KQ = ggml_scale(ctx0, KQ, 1.0f / sqrtf((float)head_dim));
            KQ = ggml_diag_mask_inf(ctx0, KQ, 0);
            KQ = ggml_soft_max(ctx0, KQ);

            // KQV = V · softmax(KQ)
            V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
            KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            struct ggml_tensor * attn_out = ggml_cont_2d(ctx0, KQV, hidden, n_pos);

            // Output projection + layer scale
            attn_out = ggml_mul_mat(ctx0, model_.tfm_o[layer], attn_out);
            if (model_.tfm_attn_scale[layer]) {
                attn_out = ggml_mul(ctx0, attn_out, model_.tfm_attn_scale[layer]);
            }

            // Residual
            x = ggml_add(ctx0, x, attn_out);
            if (act_dir) {
                ggml_set_output(x);
                tfm_attn_outs.push_back(x);
            }

            // FFN: LayerNorm → fc1 → GELU → fc2 (+ layer scale)
            struct ggml_tensor * normed2 = ggml_norm(ctx0, x, 1e-5f);
            normed2 = ggml_mul(ctx0, normed2, model_.tfm_ffn_ln_w[layer]);
            if (model_.tfm_ffn_ln_b[layer]) {
                normed2 = ggml_add(ctx0, normed2, model_.tfm_ffn_ln_b[layer]);
            }
            struct ggml_tensor * ff = ggml_mul_mat(ctx0, model_.tfm_ffn_up[layer], normed2);
            ff = ggml_gelu_erf(ctx0, ff);
            ff = ggml_mul_mat(ctx0, model_.tfm_ffn_down[layer], ff);
            if (model_.tfm_ffn_scale[layer]) {
                ff = ggml_mul(ctx0, ff, model_.tfm_ffn_scale[layer]);
            }
            x = ggml_add(ctx0, x, ff);
            if (act_dir) {
                ggml_set_output(x);
                tfm_layer_outs.push_back(x);
            }
        }

        ggml_set_output(x);
        struct ggml_cgraph * gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, x);

        // Setup backend
        std::string be_err;
        ggml_backend_t backend = init_preferred_backend("CodecEncoder-TF", &be_err);
        if (!backend) { error_msg_ = "backend init: " + be_err; ggml_free(ctx0); return false; }

        ggml_backend_t backend_cpu = nullptr;
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        }
        std::vector<ggml_backend_t> backends = {backend};
        if (backend_cpu) backends.push_back(backend_cpu);

        constexpr int MAX_NODES = 8192;
        ggml_backend_sched_t sched = ggml_backend_sched_new(
            backends.data(), nullptr, (int)backends.size(), MAX_NODES, false, true);
        if (!sched || !ggml_backend_sched_alloc_graph(sched, gf)) {
            error_msg_ = "transformer graph alloc failed";
            if (sched) ggml_backend_sched_free(sched);
            release_preferred_backend(backend);
            ggml_free(ctx0);
            return false;
        }

        // Set SEANet output as input. buf is [T, hidden] row-major, which IS
        // the memory layout of a [hidden, T] ggml tensor (ne0=hidden is the
        // fastest dimension) — copy directly, no transpose.
        ggml_backend_tensor_set(inp, buf.data(), 0, (size_t)hidden * n_pos * sizeof(float));

        // Set positions
        {
            std::vector<int32_t> pos_data(n_pos);
            for (int32_t i = 0; i < n_pos; i++) pos_data[i] = i;
            ggml_backend_tensor_set(pos, pos_data.data(), 0, (size_t)n_pos * sizeof(int32_t));
        }

        if (act_dir) {
            ggml_set_output(inp);
            dbg_tensors.push_back(inp);
            dbg_names.push_back("cpp_tfm_input.f32");
        }

        // Compute
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            error_msg_ = "transformer compute failed";
            ggml_backend_sched_reset(sched);
            ggml_backend_sched_free(sched);
            release_preferred_backend(backend);
            ggml_free(ctx0);
            return false;
        }

        // Read output: [hidden, T] ggml tensor memory == [T, hidden] row-major
        {
            buf.resize((size_t)n_pos * hidden);
            ggml_backend_tensor_get(x, buf.data(), 0, (size_t)hidden * n_pos * sizeof(float));
        }

        // Runtime weight byte dumps (verify loader correctness)
        if (act_dir && !dbg_weight_tensors.empty()) {
            for (size_t di = 0; di < dbg_weight_tensors.size(); ++di) {
                struct ggml_tensor * dt = dbg_weight_tensors[di];
                const size_t n_el = ggml_nelements(dt);
                std::vector<float> dbuf(n_el);
                ggml_backend_tensor_get(dt, dbuf.data(), 0, ggml_nbytes(dt));
                std::string pth = std::string(act_dir) + "/" + dbg_weight_names[di];
                FILE * df = fopen(pth.c_str(), "wb");
                if (df) { fwrite(dbuf.data(), 4, n_el, df); fclose(df); }
            }
        }

        // Debug tensor dumps (layer 0 attention internals)
        if (act_dir && !dbg_tensors.empty()) {
            for (size_t di = 0; di < dbg_tensors.size(); ++di) {
                struct ggml_tensor * dt = dbg_tensors[di];
                const size_t n_el = ggml_nelements(dt);
                std::vector<float> dbuf(n_el);
                ggml_backend_tensor_get(dt, dbuf.data(), 0, n_el * sizeof(float));
                std::string pth = std::string(act_dir) + "/" + dbg_names[di];
                FILE * df = fopen(pth.c_str(), "wb");
                if (df) { fwrite(dbuf.data(), 4, n_el, df); fclose(df); }
            }
        }

        // Per-layer parity dumps
        if (act_dir) {
            std::vector<float> tbuf((size_t)hidden * n_pos);
            for (size_t li = 0; li < tfm_attn_outs.size(); ++li) {
                ggml_backend_tensor_get(tfm_attn_outs[li], tbuf.data(), 0,
                                        (size_t)hidden * n_pos * sizeof(float));
                std::vector<float> rowmajor((size_t)n_pos * hidden);
                for (int32_t h = 0; h < hidden; h++) {
                    for (int32_t t = 0; t < n_pos; t++) {
                        rowmajor[(size_t)t * hidden + h] = tbuf[(size_t)h * n_pos + t];
                    }
                }
                char name[32];
                snprintf(name, sizeof(name), "cpp_attn_%02zu.f32", li);
                std::string pth = std::string(act_dir) + "/" + name;
                FILE * af = fopen(pth.c_str(), "wb");
                if (af) { fwrite(rowmajor.data(), 4, rowmajor.size(), af); fclose(af); }
            }
            for (size_t li = 0; li < tfm_layer_outs.size(); ++li) {
                ggml_backend_tensor_get(tfm_layer_outs[li], tbuf.data(), 0,
                                        (size_t)hidden * n_pos * sizeof(float));
                // transpose [hidden, T] -> [T, hidden] and write via dump path
                std::vector<float> rowmajor((size_t)n_pos * hidden);
                for (int32_t h = 0; h < hidden; h++) {
                    for (int32_t t = 0; t < n_pos; t++) {
                        rowmajor[(size_t)t * hidden + h] = tbuf[(size_t)h * n_pos + t];
                    }
                }
                char name[32];
                snprintf(name, sizeof(name), "cpp_tfm_%02zu.f32", li);
                std::string p = std::string(act_dir) + "/" + name;
                FILE * f = fopen(p.c_str(), "wb");
                if (f) { fwrite(rowmajor.data(), 4, rowmajor.size(), f); fclose(f); }
            }
        }

        ggml_backend_sched_reset(sched);
        ggml_backend_sched_free(sched);
        release_preferred_backend(backend);
        ggml_free(ctx0);
        ENC_LOG("Transformer done: [%d, %d]\n", n_pos, hidden);
        dump_act("cpp_transformer.f32");
    }

    // ── Stage 3: Downsample (stride 2, k4) — host-side ──
    {
        std::vector<float> w_f32 = tensor_to_host_f32(model_.downsample_w);
        int32_t K = (int32_t)model_.downsample_w->ne[0]; // 4
        int32_t IC = (int32_t)model_.downsample_w->ne[1]; // 512
        int32_t OC = (int32_t)model_.downsample_w->ne[2]; // 512
        int32_t T = (int32_t)buf.size() / IC;
        int32_t stride = 2;

        // Causal-SAME with REPLICATE padding (Python: pad_mode='replicate').
        // Output count is ceil(T / stride); left/right padding replicates the
        // first/last frame. (Zero padding gives a small systematic mismatch.)
        int32_t pad_left = K - 1 + 1 - stride; // 2
        int32_t OW_ds = (T + stride - 1) / stride;
        int32_t pad_right = (OW_ds - 1) * stride + K - (T + pad_left);
        if (pad_right < 0) pad_right = 0;
        std::vector<float> padded((size_t)(T + pad_left + pad_right) * IC, 0.0f);
        // left: replicate frame 0
        for (int32_t p = 0; p < pad_left; ++p) {
            memcpy(padded.data() + (size_t)p * IC, buf.data(), (size_t)IC * sizeof(float));
        }
        memcpy(padded.data() + (size_t)pad_left * IC, buf.data(), (size_t)T * IC * sizeof(float));
        // right: replicate last frame
        for (int32_t p = 0; p < pad_right; ++p) {
            memcpy(padded.data() + (size_t)(pad_left + T + p) * IC,
                   buf.data() + (size_t)(T - 1) * IC, (size_t)IC * sizeof(float));
        }

        std::vector<float> out((size_t)OW_ds * OC);
        int32_t out_len;
        host_conv1d(w_f32.data(), nullptr, padded.data(),
                     T + pad_left + pad_right, IC, OC, K, stride, 1,
                     out.data(), &out_len);
        buf = std::move(out);
        ENC_LOG("Downsample output: [%d, %d]\n", out_len, OC);
        dump_act("cpp_downsample.f32");
    }

    n_frames = (int32_t)buf.size() / hidden;

    // ── Stage 4: RVQ quantization (host-side, already F32) ──
    codec_tokens.resize((size_t)n_frames * 16);

    // Semantic path
    {
        std::vector<float> cb_data;
        if (!read_encoder_codebook_normalized(model_.vq_sem_cb, model_.vq_sem_usage,
                                              cb_data, error_msg_)) {
            return false;
        }
        int32_t cb_dim = cfg.CODEBOOK_DIM; // 256

        // Project to codebook dim: input [n_frames, 512] × proj [512, 256] → [n_frames, 256]
        std::vector<float> proj_w = tensor_to_host_f32(model_.vq_sem_in);
        std::vector<float> latents((size_t)n_frames * cb_dim, 0.0f);
        for (int32_t f = 0; f < n_frames; f++) {
            for (int32_t d = 0; d < cb_dim; d++) {
                float sum = 0.0f;
                for (int32_t h = 0; h < hidden; h++) {
                    sum += buf[(size_t)f * hidden + h] * proj_w[(size_t)d * hidden + h];
                }
                latents[(size_t)f * cb_dim + d] = sum;
            }
        }

        for (int32_t f = 0; f < n_frames; f++) {
            const float * vec = latents.data() + (size_t)f * cb_dim;
            codec_tokens[(size_t)f * 16 + 0] = rvq_l2_argmin(
                cb_data.data(), cfg.CODEBOOK_SIZE, cb_dim, vec);
        }
    }

    // Acoustic path (15 quantizers, residual-chained)
    {
        int32_t cb_dim = cfg.CODEBOOK_DIM;
        std::vector<float> proj_w = tensor_to_host_f32(model_.vq_aco_in);
        std::vector<float> residuals((size_t)n_frames * cb_dim, 0.0f);

        // Project to codebook dim
        for (int32_t f = 0; f < n_frames; f++) {
            for (int32_t d = 0; d < cb_dim; d++) {
                float sum = 0.0f;
                for (int32_t h = 0; h < hidden; h++) {
                    sum += buf[(size_t)f * hidden + h] * proj_w[(size_t)d * hidden + h];
                }
                residuals[(size_t)f * cb_dim + d] = sum;
            }
        }

        for (int q = 0; q < 15; q++) {
            if (!model_.vq_aco_cb[q]) continue;
            std::vector<float> cb_data;
            if (!read_encoder_codebook_normalized(model_.vq_aco_cb[q], model_.vq_aco_usage[q],
                                                   cb_data, error_msg_)) {
                return false;
            }
            for (int32_t f = 0; f < n_frames; f++) {
                float * vec = residuals.data() + (size_t)f * cb_dim;
                int32_t idx = rvq_l2_argmin(cb_data.data(), cfg.CODEBOOK_SIZE, cb_dim, vec);
                codec_tokens[(size_t)f * 16 + 1 + q] = idx;
                if (q < 14) {
                    const float * entry = cb_data.data() + (size_t)idx * cb_dim;
                    for (int32_t d = 0; d < cb_dim; d++) vec[d] -= entry[d];
                }
            }
        }
    }

    ENC_LOG("F32 encoder: %d samples → %d frames × 16 codebooks\n", n_samples, n_frames);
    return true;
}

// ─── RVQ L2 argmin ────────────────────────────────────────────────────

static int32_t rvq_l2_argmin(
    const float * codebook,
    int32_t cb_size,
    int32_t cb_dim,
    const float * vector)
{
    float best_dist = 1e30f;
    int32_t best_idx = 0;
    for (int32_t i = 0; i < cb_size; i++) {
        const float * entry = codebook + (size_t)i * cb_dim;
        float dist = 0.0f;
        for (int32_t d = 0; d < cb_dim; d++) {
            const float diff = entry[d] - vector[d];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    return best_idx;
}

// ── GGML F16 graph path (ported from feat/in-context-cloning) ──
bool CodecEncoder::encode_ggml(
    const float * audio,
    int32_t n_samples,
    std::vector<int32_t> & codec_tokens,
    int32_t & n_frames)
{
    if (!model_.loaded) {
        error_msg_ = "encoder not loaded";
        return false;
    }
    if (!audio || n_samples <= 0) {
        error_msg_ = "invalid audio input";
        return false;
    }

    const auto & cfg = model_.config;

    // ── Build the computation graph ──
    size_t compute_size = 512 * 1024 * 1024; // 512 MB should be enough for encoder
    struct ggml_init_params gparams = {
        /*.mem_size   =*/ compute_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(gparams);
    if (!ctx0) {
        error_msg_ = "failed to init compute context";
        return false;
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // Input: [1, n_samples] (mono audio)
    struct ggml_tensor * inp_audio = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_samples);
    ggml_set_name(inp_audio, "audio");
    ggml_set_input(inp_audio);

    // Reshape to [T, C] = [n_samples, 1] for ggml_conv_1d
    struct ggml_tensor * x = ggml_reshape_2d(ctx0, inp_audio, n_samples, 1);
    ENC_LOG("input: "); print_shape("x", x);

    // ── SEANet stages ──
    for (int stage = 0; stage < 4; stage++) {
        const int stride = MimiEncoderConfig::CONV_STRIDES[stage];
        const int kernel = MimiEncoderConfig::ENTRY_KERNELS[stage];

        // Strided entry conv (causal-SAME, NO ELU — it comes after the res block)
        {
            int32_t pad_left = (kernel - 1) + 1 - stride;
            if (pad_left < 0) pad_left = 0;
            int32_t T = (int32_t)x->ne[0];
            int32_t out_len = (T + pad_left - (kernel - 1) - 1) / stride + 1;
            int32_t needed = (out_len - 1) * stride + kernel;
            int32_t pad_right = needed - T - pad_left;
            if (pad_right < 0) pad_right = 0;
            if (pad_left > 0 || pad_right > 0) {
                x = ggml_pad_ext(ctx0, x, pad_left, pad_right, 0, 0, 0, 0, 0, 0);
            }
            x = ggml_conv_1d(ctx0, model_.conv_entry_w[stage], x, stride, 0, 1);
            if (model_.conv_entry_b[stage]) {
                int32_t oc = (int32_t)model_.conv_entry_w[stage]->ne[2];
                x = ggml_add(ctx0, x, ggml_reshape_3d(ctx0, model_.conv_entry_b[stage], 1, oc, 1));
            }
        }

        // Residual block (1 per stage, pre-activation)
        int layer_idx = stage * 3 + 1;
        x = seanet_res_block(ctx0, x,
                             model_.res_w[stage][0], model_.res_b[stage][0],
                             model_.res_w[stage][1], model_.res_b[stage][1],
                             3, 1,   // kernel1=3, kernel2=1
                             1);      // dilation=1
        // Post-resblock ELU (the flat layer list has: Conv → ResBlock → ELU)
        x = ggml_elu(ctx0, x);
        ENC_LOG("after SEANet stage %d: ", stage); print_shape("x", x);
    }

    // After SEANet: [512, T/960] at 25 Hz

    // ── Bottleneck ──
    // conv.12: stride 8 (the main temporal reduction to 25 Hz)
    x = causal_conv_elu(ctx0, x,
                        model_.bottleneck_expand_w,
                        model_.bottleneck_expand_b,
                        16, MimiEncoderConfig::BOTTLENECK_STRIDE);  // k16, stride 8

    // conv.14 (bottleneck contract, k3, stride 1) — NO ELU, causal-SAME
    {
        int32_t pad = 3 - 1 + 1 - 1;  // (k-1)*dil + 1 - stride = 2 + 0 = 2
        if (pad > 0) {
            x = ggml_pad_ext(ctx0, x, pad, 0, 0, 0, 0, 0, 0, 0);
        }
        x = ggml_conv_1d(ctx0, model_.bottleneck_contract_w, x, 1, 0, 1);
        if (model_.bottleneck_contract_b) {
            int32_t oc = (int32_t)model_.bottleneck_contract_w->ne[2];
            x = ggml_add(ctx0, x,
                ggml_reshape_3d(ctx0, model_.bottleneck_contract_b, 1, oc, 1));
        }
    }
    ENC_LOG("after bottleneck: "); print_shape("x", x);
    // Compute mean/std for comparison with Python reference
    // (Python: seanet_L14 mean=0.144544 std=3.495283)
    {
        // We can't easily read the tensor data before computation (symbolic),
        // so we'll add this after graph_compute instead.
    }

    // ── Transformer ──
    // Transpose from [T, C] to [C, T] for mul_mat-based transformer
    int n_pos = ggml_nelements(x) / cfg.HIDDEN_SIZE;
    ENC_LOG("before transformer: n_pos=%d\n", n_pos);
    {
        int ch = cfg.HIDDEN_SIZE;
        x = ggml_reshape_2d(ctx0, x, n_pos, ch);      // [T, C] → [T, C]
        x = ggml_transpose(ctx0, x);                    // [C, T]
        x = ggml_cont(ctx0, x);                         // make contiguous
    }
    for (int layer = 0; layer < 8; layer++) {
        x = encoder_transformer_layer(ctx0, model_, x, layer, n_pos);
    }
    // Transpose back from [C, T] to [T, C] for the downsample conv
    {
        x = ggml_transpose(ctx0, x);
        x = ggml_cont(ctx0, x);
    }

    // ── Post-transformer downsample (stride 2, k4) ──
    // Causal-SAME: pad_left = (4-1) + 1 - 2 = 2, pad_right = 1
    // (Python uses pad_mode="replicate", bias=False)
    {
        x = ggml_pad_ext(ctx0, x, 2, 1, 0, 0, 0, 0, 0, 0);
        x = ggml_conv_1d(ctx0, model_.downsample_w, x, 2, 0, 1);
    }

    // After downsample conv: [T', C] format (ne[0]=T', ne[1]=512)
    ENC_LOG("after downsample: "); print_shape("x", x);
    // Transpose to [C, T'] for mul_mat quantizer projections
    x = ggml_transpose(ctx0, x);
    x = ggml_cont(ctx0, x);
    // Now [C=512, T'=n_frames] with ne[0]=512, ne[1]=n_frames

    n_frames = ggml_nelements(x) / cfg.HIDDEN_SIZE;

    // ── Quantizer projections ──
    // Projection weights are stored as conv-style [1, IC, OC] — reshape to [IC, OC]
    struct ggml_tensor * sem_in_2d = ggml_reshape_2d(ctx0, model_.vq_sem_in,
        cfg.HIDDEN_SIZE, cfg.CODEBOOK_DIM);  // [512, 256]
    struct ggml_tensor * aco_in_2d = ggml_reshape_2d(ctx0, model_.vq_aco_in,
        cfg.HIDDEN_SIZE, cfg.CODEBOOK_DIM);  // [512, 256]

    // Semantic path: [IC=512, OC=256] × [IC=512, T'] → [OC=256, T']
    struct ggml_tensor * sem_latent = ggml_mul_mat(ctx0, sem_in_2d, x);
    ggml_set_name(sem_latent, "sem_latent");

    // Acoustic path: [IC=512, OC=256] × [IC=512, T'] → [OC=256, T']
    struct ggml_tensor * aco_latent = ggml_mul_mat(ctx0, aco_in_2d, x);
    ggml_set_name(aco_latent, "aco_latent");

    // Set outputs for readback
    ggml_set_output(sem_latent);
    ggml_set_output(aco_latent);

    // Build graph
    ggml_build_forward_expand(gf, sem_latent);
    ggml_build_forward_expand(gf, aco_latent);

    // ── Allocate and run (scheduler pattern from decoder) ──
    std::string backend_err;
    ggml_backend_t backend = init_preferred_backend("CodecEncoder", &backend_err);
    if (!backend) {
        error_msg_ = "failed to init backend: " + backend_err;
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }

    std::vector<ggml_backend_t> backends;
    backends.push_back(backend);
    if (backend_cpu) backends.push_back(backend_cpu);

    constexpr int ENC_MAX_NODES = 8192;
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends.data(), nullptr, (int)backends.size(), ENC_MAX_NODES, false, true);
    if (!sched) {
        error_msg_ = "failed to create scheduler";
        release_preferred_backend(backend);
        ggml_free(ctx0);
        return false;
    }

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        error_msg_ = "failed to allocate graph";
        ggml_backend_sched_free(sched);
        release_preferred_backend(backend);
        ggml_free(ctx0);
        return false;
    }

    // Set input audio
    ggml_backend_tensor_set(inp_audio, audio, 0, (size_t)n_samples * sizeof(float));

    // Set RoPE positions for ALL layers
    {
        std::vector<int32_t> pos_data(n_pos);
        for (int i = 0; i < n_pos; i++) pos_data[i] = i;
        for (int layer = 0; layer < 8; layer++) {
            char name[32];
            snprintf(name, sizeof(name), "enc_pos_%d", layer);
            struct ggml_tensor * pos_t = ggml_graph_get_tensor(gf, name);
            if (pos_t) {
                ggml_backend_tensor_set(pos_t, pos_data.data(), 0,
                                         (size_t)n_pos * sizeof(int32_t));
            }
        }
    }

    // Compute
    ENC_LOG(">>> calling ggml_backend_sched_graph_compute...\n");
    fflush(stderr);
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "graph computation failed";
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_free(sched);
        release_preferred_backend(backend);
        ggml_free(ctx0);
        return false;
    }
    ENC_LOG("<<< graph_compute returned SUCCESS\n");
    fflush(stderr);

    // ── Debug: read back intermediate activations for comparison ──
    {
        // Read the tensor AFTER bottleneck (named from the graph)
        // We'll read from the quantizer outputs for now (sem_latent)
        // and compare with Python's downsample output
    }

    // ── Readback latent representations ──
    // Use ggml_graph_get_tensor to get the actual computed tensor
    // (the scheduler may redirect output tensors).
    struct ggml_tensor * sem_out = ggml_graph_get_tensor(gf, "sem_latent");
    struct ggml_tensor * aco_out = ggml_graph_get_tensor(gf, "aco_latent");
    if (!sem_out || !aco_out) {
        error_msg_ = "output tensors not found in graph";
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_free(sched);
        release_preferred_backend(backend);
        ggml_free(ctx0);
        return false;
    }

    ENC_LOG("quantizer outputs:\n");
    print_shape("sem_out", sem_out);
    print_shape("aco_out", aco_out);

    int32_t cb_dim = MimiEncoderConfig::CODEBOOK_DIM;
    size_t sem_bytes = (size_t)sem_out->ne[1] * (size_t)sem_out->ne[0] * sizeof(float);
    size_t aco_bytes = (size_t)aco_out->ne[1] * (size_t)aco_out->ne[0] * sizeof(float);
    ENC_LOG("readback: sem_bytes=%zu, aco_bytes=%zu\n", sem_bytes, aco_bytes);

    std::vector<float> sem_data((size_t)sem_out->ne[0] * sem_out->ne[1]);
    std::vector<float> aco_data((size_t)aco_out->ne[0] * aco_out->ne[1]);

    ENC_LOG("ggml_nbytes(sem_out)=%zu, ggml_nbytes(aco_out)=%zu\n",
            ggml_nbytes(sem_out), ggml_nbytes(aco_out));
    ENC_LOG("sem_out strides: nb=[%zu, %zu, %zu, %zu]\n",
            sem_out->nb[0], sem_out->nb[1], sem_out->nb[2], sem_out->nb[3]);
    ENC_LOG("buffer=%p, data=%p\n", (void*)sem_out->buffer, sem_out->data);
    fflush(stderr);

    // Use ggml_nbytes as the authoritative size
    sem_bytes = ggml_nbytes(sem_out);
    aco_bytes = ggml_nbytes(aco_out);

    // On CPU backend: read directly from the tensor's data pointer.
    // The scheduler may not expose these through ggml_backend_tensor_get
    // correctly for scheduler-allocated output tensors.
    {
        ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
        bool is_cpu = !dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
        if (is_cpu && sem_out->data && aco_out->data) {
            memcpy(sem_data.data(), sem_out->data, sem_bytes);
            memcpy(aco_data.data(), aco_out->data, aco_bytes);
            ENC_LOG("readback via direct memcpy\n");
        } else {
            ggml_backend_tensor_get(sem_out, sem_data.data(), 0, sem_bytes);
            ggml_backend_tensor_get(aco_out, aco_data.data(), 0, aco_bytes);
            ENC_LOG("readback via backend API\n");
        }
    }
    // Update n_frames from the actual tensor shape
    n_frames = (int32_t)sem_out->ne[1];

    // Debug: compare with Python reference stats
    {
        float mean = 0.0f, sum_sq = 0.0f;
        size_t n = sem_data.size();
        for (float v : sem_data) { mean += v; sum_sq += v * v; }
        mean /= n;
        float var = sum_sq / n - mean * mean;
        float std = sqrtf(var > 0 ? var : 0);
        ENC_LOG("sem_latent: n=%zu mean=%.6f std=%.6f (Py: mean=-0.0598 std=4.851)\n",
                n, mean, std);
    }

    // Cleanup scheduler
    ggml_backend_sched_reset(sched);
    ggml_backend_sched_free(sched);
    release_preferred_backend(backend);
    ggml_free(ctx0);

    // ── RVQ quantization (CPU, outside the GGML graph) ──
    codec_tokens.resize((size_t)n_frames * 16);

    // Semantic path: 1 quantizer
    {
        std::vector<float> cb_data;
        {
            if (!read_encoder_codebook_normalized(model_.vq_sem_cb, model_.vq_sem_usage,
                     cb_data, error_msg_)) {
                ENC_LOG("semantic codebook read failed: %s\n", error_msg_.c_str());
                return false;
            }
        }
        for (int f = 0; f < n_frames; f++) {
            const float * vec = sem_data.data() + (size_t)f * cb_dim;
            codec_tokens[(size_t)f * 16 + 0] = rvq_l2_argmin(
                cb_data.data(), MimiEncoderConfig::CODEBOOK_SIZE, cb_dim, vec);
        }
    }

    // Acoustic path: 15 quantizers, residual-chained
    {
        for (int q = 0; q < 15; q++) {
            if (!model_.vq_aco_cb[q]) continue;

            std::vector<float> cb_data;
            if (!read_encoder_codebook_normalized(model_.vq_aco_cb[q], model_.vq_aco_usage[q],
                     cb_data, error_msg_)) {
                ENC_LOG("acoustic codebook %d read failed: %s\n", q, error_msg_.c_str());
                return false;
            }

            for (int f = 0; f < n_frames; f++) {
                float * vec = aco_data.data() + (size_t)f * cb_dim;
                int32_t idx = rvq_l2_argmin(
                    cb_data.data(), MimiEncoderConfig::CODEBOOK_SIZE, cb_dim, vec);

                codec_tokens[(size_t)f * 16 + 1 + q] = idx;

                // Subtract the quantized value (residual for next quantizer)
                if (q < 14) {
                    const float * entry = cb_data.data() + (size_t)idx * cb_dim;
                    for (int d = 0; d < cb_dim; d++) {
                        vec[d] -= entry[d];
                    }
                }
            }
        }
    }

    ENC_LOG("Encoded %d samples → %d frames × 16 codebooks\n", n_samples, n_frames);
    return true;
}

} // namespace qwen3_tts
