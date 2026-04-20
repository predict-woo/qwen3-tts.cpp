// Mimi audio codec encoder for ICL voice cloning.
// Ported from khimaros/qwen3-tts.cpp commit 9c57131 (with dec135e bias fix).
// Encodes 24kHz mono audio into discrete speech codes compatible with the
// talker's codec vocabulary — the "codes" path for CustomVoice-style cloning.

#include "encoder/audio_codec_encoder.h"
#include "common/gguf_loader.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define ENC_MAX_NODES 16384

namespace qwen3_tts {

constexpr int32_t codec_encoder_config::cnn_strides[4];

// Original Python layer indices for conv and resunit tensors in GGUF.
static constexpr int CONV_LAYERS[]    = {0, 3, 6, 9, 12, 14};
static constexpr int RESUNIT_LAYERS[] = {1, 4, 7, 10};

AudioCodecEncoder::AudioCodecEncoder() = default;

AudioCodecEncoder::~AudioCodecEncoder() {
    unload_model();
}

void AudioCodecEncoder::unload_model() {
    free_codec_encoder_model(model_);

    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend) {
        release_preferred_backend(state_.backend);
        state_.backend = nullptr;
    }
    if (state_.backend_cpu) {
        ggml_backend_free(state_.backend_cpu);
        state_.backend_cpu = nullptr;
    }
    state_.compute_meta.clear();
}

bool AudioCodecEncoder::load_model(const std::string & model_path) {
    unload_model();

    GGUFLoader loader;
    if (!loader.open(model_path)) {
        error_msg_ = loader.get_error();
        return false;
    }

    model_.config.sample_rate          = loader.get_u32("qwen3-tts-tokenizer.sample_rate", 24000);
    model_.config.codebook_size        = loader.get_u32("qwen3-tts-tokenizer.codebook_size", 2048);
    model_.config.hidden_size          = loader.get_u32("qwen3-tts-tokenizer.encoder.hidden_size", 512);
    model_.config.n_transformer_layers = loader.get_u32("qwen3-tts-tokenizer.encoder.num_layers", 8);
    model_.config.n_heads              = loader.get_u32("qwen3-tts-tokenizer.encoder.num_heads", 8);
    model_.config.n_valid_quantizers   = loader.get_u32("qwen3-tts-tokenizer.encoder.valid_quantizers", 16);
    model_.config.codebook_dim         = loader.get_u32("qwen3-tts-tokenizer.encoder.codebook_dim", 256);
    model_.config.head_dim             = model_.config.hidden_size / model_.config.n_heads;

    int64_t n_tensors = loader.get_n_tensors();
    int enc_count = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (name && strncmp(name, "tok_enc.", 8) == 0) enc_count++;
    }

    if (enc_count == 0) {
        error_msg_ = "no encoder tensors (tok_enc.*) found in model";
        return false;
    }

    size_t ctx_size = ggml_tensor_overhead() * enc_count;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_msg_ = "failed to initialize ggml context";
        return false;
    }

    struct gguf_context * gguf_ctx = loader.get_ctx();
    struct ggml_context * meta_ctx = loader.get_meta_ctx();

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = loader.get_tensor_name(i);
        if (!name || strncmp(name, "tok_enc.", 8) != 0) continue;

        std::string sname(name);
        if (sname.find(".initialized") != std::string::npos) continue;

        struct ggml_tensor * meta = ggml_get_tensor(meta_ctx, name);
        if (!meta) continue;

        struct ggml_tensor * tensor = ggml_dup_tensor(model_.ctx, meta);
        ggml_set_name(tensor, name);
        model_.tensors[name] = tensor;

        int idx = 0, sub = 0, n_consumed = -1;
        auto full_match = [&](const char * fmt, int * a, int * b) -> bool {
            n_consumed = -1;
            int r = b ? sscanf(name, fmt, a, b, &n_consumed)
                      : sscanf(name, fmt, a, &n_consumed);
            return r >= (b ? 2 : 1) && n_consumed > 0 && name[n_consumed] == '\0';
        };

        if (sname == "tok_enc.downsample.weight") {
            model_.final_ds_w = tensor;
            continue;
        }

        if (sname == "tok_enc.vq_semantic.input_proj.weight")  { model_.vq_sem_in_proj = tensor; continue; }
        if (sname == "tok_enc.vq_semantic.output_proj.weight") { model_.vq_sem_out_proj = tensor; continue; }
        if (sname == "tok_enc.vq_acoustic.input_proj.weight")  { model_.vq_acou_in_proj = tensor; continue; }
        if (sname == "tok_enc.vq_acoustic.output_proj.weight") { model_.vq_acou_out_proj = tensor; continue; }

        if (full_match("tok_enc.vq_semantic.%d.codebook%n", &idx, nullptr)) {
            if (idx == 0) model_.vq_sem_codebook = tensor;
            continue;
        }
        if (full_match("tok_enc.vq_semantic.%d.usage%n", &idx, nullptr)) {
            if (idx == 0) model_.vq_sem_usage = tensor;
            continue;
        }
        if (full_match("tok_enc.vq_acoustic.%d.codebook%n", &idx, nullptr)) {
            if (idx >= 0 && idx < 31) model_.vq_acou_codebook[idx] = tensor;
            continue;
        }
        if (full_match("tok_enc.vq_acoustic.%d.usage%n", &idx, nullptr)) {
            if (idx >= 0 && idx < 31) model_.vq_acou_usage[idx] = tensor;
            continue;
        }

        if (full_match("tok_enc.conv.%d.weight%n", &idx, nullptr)) {
            if      (idx == CONV_LAYERS[0]) model_.input_conv_w   = tensor;
            else if (idx == CONV_LAYERS[1]) model_.ds_conv_w[0]   = tensor;
            else if (idx == CONV_LAYERS[2]) model_.ds_conv_w[1]   = tensor;
            else if (idx == CONV_LAYERS[3]) model_.ds_conv_w[2]   = tensor;
            else if (idx == CONV_LAYERS[4]) model_.ds_conv_w[3]   = tensor;
            else if (idx == CONV_LAYERS[5]) model_.final_conv_w   = tensor;
            continue;
        }
        if (full_match("tok_enc.conv.%d.bias%n", &idx, nullptr)) {
            if      (idx == CONV_LAYERS[0]) model_.input_conv_b   = tensor;
            else if (idx == CONV_LAYERS[1]) model_.ds_conv_b[0]   = tensor;
            else if (idx == CONV_LAYERS[2]) model_.ds_conv_b[1]   = tensor;
            else if (idx == CONV_LAYERS[3]) model_.ds_conv_b[2]   = tensor;
            else if (idx == CONV_LAYERS[4]) model_.ds_conv_b[3]   = tensor;
            else if (idx == CONV_LAYERS[5]) model_.final_conv_b   = tensor;
            continue;
        }

        if (full_match("tok_enc.res.%d.blk.%d.weight%n", &idx, &sub)) {
            for (int s = 0; s < 4; ++s) {
                if (idx == RESUNIT_LAYERS[s]) {
                    if      (sub == 1) model_.resunits[s].conv1_w = tensor;
                    else if (sub == 3) model_.resunits[s].conv2_w = tensor;
                    break;
                }
            }
            continue;
        }
        if (full_match("tok_enc.res.%d.blk.%d.bias%n", &idx, &sub)) {
            for (int s = 0; s < 4; ++s) {
                if (idx == RESUNIT_LAYERS[s]) {
                    if      (sub == 1) model_.resunits[s].conv1_b = tensor;
                    else if (sub == 3) model_.resunits[s].conv2_b = tensor;
                    break;
                }
            }
            continue;
        }

        if (sscanf(name, "tok_enc.blk.%d.", &idx) == 1 && idx >= 0 && idx < 8) {
            auto & L = model_.tfm_layers[idx];
            if      (sname.find(".attn_norm.weight") != std::string::npos) L.attn_norm_w   = tensor;
            else if (sname.find(".attn_norm.bias")   != std::string::npos) L.attn_norm_b   = tensor;
            else if (sname.find(".attn_q.weight")    != std::string::npos) L.attn_q_w      = tensor;
            else if (sname.find(".attn_k.weight")    != std::string::npos) L.attn_k_w      = tensor;
            else if (sname.find(".attn_v.weight")    != std::string::npos) L.attn_v_w      = tensor;
            else if (sname.find(".attn_output.weight") != std::string::npos) L.attn_output_w = tensor;
            else if (sname.find(".attn_scale")       != std::string::npos) L.attn_scale    = tensor;
            else if (sname.find(".ffn_norm.weight")  != std::string::npos) L.ffn_norm_w    = tensor;
            else if (sname.find(".ffn_norm.bias")    != std::string::npos) L.ffn_norm_b    = tensor;
            else if (sname.find(".ffn_up.weight")    != std::string::npos) L.ffn_up_w      = tensor;
            else if (sname.find(".ffn_down.weight")  != std::string::npos) L.ffn_down_w    = tensor;
            else if (sname.find(".ffn_scale")        != std::string::npos) L.ffn_scale     = tensor;
            continue;
        }
    }

    if (!load_tensor_data_from_file(model_path, gguf_ctx, model_.ctx,
                                    model_.tensors, model_.buffer, error_msg_,
                                    GGML_BACKEND_DEVICE_TYPE_CPU)) {
        return false;
    }

    // Mimi stores codebooks as running embed_sum + cluster_usage; recover
    // the actual centroids by dividing each row by its usage count. This is
    // an in-place edit of the backend buffer after upload.
    {
        const float epsilon = 1e-5f;
        std::vector<ggml_fp16_t> cb_buf;
        std::vector<float> usage_buf;

        auto normalize = [&](struct ggml_tensor * codebook, struct ggml_tensor * usage) {
            if (!codebook || !usage) return;
            const int64_t dim = codebook->ne[0];
            const int64_t sz  = codebook->ne[1];
            const size_t cb_elems = (size_t) dim * sz;

            cb_buf.resize(cb_elems);
            usage_buf.resize(sz);
            ggml_backend_tensor_get(codebook, cb_buf.data(), 0, cb_elems * sizeof(ggml_fp16_t));
            ggml_backend_tensor_get(usage, usage_buf.data(), 0, sz * sizeof(float));

            for (int64_t e = 0; e < sz; ++e) {
                float u = usage_buf[e];
                if (u < epsilon) u = epsilon;
                float inv_u = 1.0f / u;
                for (int64_t d = 0; d < dim; ++d) {
                    size_t m = (size_t) d + (size_t) e * dim;
                    float v = ggml_fp16_to_fp32(cb_buf[m]);
                    cb_buf[m] = ggml_fp32_to_fp16(v * inv_u);
                }
            }
            ggml_backend_tensor_set(codebook, cb_buf.data(), 0, cb_elems * sizeof(ggml_fp16_t));
        };

        normalize(model_.vq_sem_codebook, model_.vq_sem_usage);
        for (int i = 0; i < 31; ++i) {
            normalize(model_.vq_acou_codebook[i], model_.vq_acou_usage[i]);
        }
    }

    // Init compute backend.
    state_.backend = init_preferred_backend("AudioCodecEncoder", &error_msg_);
    if (!state_.backend) return false;

    ggml_backend_dev_t device = ggml_backend_get_device(state_.backend);
    const char * dev_name = device ? ggml_backend_dev_name(device) : "Unknown";
    fprintf(stderr, "  AudioCodecEncoder backend: %s\n", dev_name);

    if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        state_.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!state_.backend_cpu) {
            error_msg_ = "failed to init CPU fallback for AudioCodecEncoder";
            return false;
        }
    }

    std::vector<ggml_backend_t> backends;
    backends.push_back(state_.backend);
    if (state_.backend_cpu) backends.push_back(state_.backend_cpu);
    state_.sched = ggml_backend_sched_new(backends.data(), nullptr,
                                          (int) backends.size(), ENC_MAX_NODES, false, true);
    if (!state_.sched) {
        error_msg_ = "failed to create backend scheduler";
        return false;
    }

    state_.compute_meta.resize(ggml_tensor_overhead() * ENC_MAX_NODES + ggml_graph_overhead());
    return true;
}

// --- graph building --------------------------------------------------------

struct ggml_tensor * AudioCodecEncoder::apply_resunit(struct ggml_context * ctx,
                                                     struct ggml_tensor * x,
                                                     const enc_resunit & ru) {
    // ELU -> Conv k=3 -> ELU -> Conv k=1 -> residual add
    struct ggml_tensor * residual = x;

    x = ggml_elu(ctx, x);

    // Causal conv k=3: left_pad = 2.
    x = ggml_pad_ext(ctx, x, 2, 0, 0, 0, 0, 0, 0, 0);
    x = ggml_conv_1d(ctx, ru.conv1_w, x, 1, 0, 1);
    int64_t out_ch = ru.conv1_w->ne[2];
    if (ru.conv1_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, ru.conv1_b, 1, out_ch, 1));
    }

    x = ggml_elu(ctx, x);

    // Conv k=1: no padding needed.
    x = ggml_conv_1d(ctx, ru.conv2_w, x, 1, 0, 1);
    out_ch = ru.conv2_w->ne[2];
    if (ru.conv2_b) {
        x = ggml_add(ctx, x, ggml_reshape_3d(ctx, ru.conv2_b, 1, out_ch, 1));
    }

    return ggml_add(ctx, residual, x);
}

struct ggml_tensor * AudioCodecEncoder::apply_tfm_layer(struct ggml_context * ctx,
                                                       struct ggml_tensor * x,
                                                       const enc_tfm_layer & layer,
                                                       int32_t seq_len,
                                                       struct ggml_tensor * positions) {
    const auto & cfg = model_.config;
    const int n_heads = cfg.n_heads;
    const int head_dim = cfg.head_dim;

    // Pre-attention LayerNorm (with bias).
    struct ggml_tensor * residual = x;
    struct ggml_tensor * normed = ggml_norm(ctx, x, cfg.norm_eps);
    normed = ggml_mul(ctx, normed, layer.attn_norm_w);
    if (layer.attn_norm_b) normed = ggml_add(ctx, normed, layer.attn_norm_b);

    // QKV projections.
    struct ggml_tensor * Q = ggml_mul_mat(ctx, layer.attn_q_w, normed);
    struct ggml_tensor * K = ggml_mul_mat(ctx, layer.attn_k_w, normed);
    struct ggml_tensor * V = ggml_mul_mat(ctx, layer.attn_v_w, normed);

    Q = ggml_reshape_3d(ctx, Q, head_dim, n_heads, seq_len);
    K = ggml_reshape_3d(ctx, K, head_dim, n_heads, seq_len);
    V = ggml_reshape_3d(ctx, V, head_dim, n_heads, seq_len);

    Q = ggml_rope_ext(ctx, Q, positions, nullptr,
                      head_dim, GGML_ROPE_TYPE_NEOX, 0,
                      cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx, K, positions, nullptr,
                      head_dim, GGML_ROPE_TYPE_NEOX, 0,
                      cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
    K = ggml_permute(ctx, K, 0, 2, 1, 3);
    V = ggml_permute(ctx, V, 0, 2, 1, 3);

    struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);
    KQ = ggml_scale(ctx, KQ, 1.0f / sqrtf((float) head_dim));
    // Mimi's encoder_transformer is causal (uses create_causal_mask).
    KQ = ggml_diag_mask_inf_inplace(ctx, KQ, 0);
    KQ = ggml_soft_max(ctx, KQ);

    V = ggml_cont(ctx, ggml_transpose(ctx, V));
    struct ggml_tensor * KQV = ggml_mul_mat(ctx, V, KQ);
    KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);
    struct ggml_tensor * attn_out = ggml_cont_2d(ctx, KQV, n_heads * head_dim, seq_len);

    attn_out = ggml_mul_mat(ctx, layer.attn_output_w, attn_out);

    if (layer.attn_scale) {
        attn_out = ggml_mul(ctx, attn_out, layer.attn_scale);
    }

    x = ggml_add(ctx, residual, attn_out);
    residual = x;

    // Pre-FFN LayerNorm.
    normed = ggml_norm(ctx, x, cfg.norm_eps);
    normed = ggml_mul(ctx, normed, layer.ffn_norm_w);
    if (layer.ffn_norm_b) normed = ggml_add(ctx, normed, layer.ffn_norm_b);

    // fc1 -> GELU -> fc2 (no SwiGLU).
    struct ggml_tensor * ffn_out = ggml_mul_mat(ctx, layer.ffn_up_w, normed);
    ffn_out = ggml_gelu(ctx, ffn_out);
    ffn_out = ggml_mul_mat(ctx, layer.ffn_down_w, ffn_out);

    if (layer.ffn_scale) {
        ffn_out = ggml_mul(ctx, ffn_out, layer.ffn_scale);
    }

    return ggml_add(ctx, residual, ffn_out);
}

struct ggml_cgraph * AudioCodecEncoder::build_graph(int32_t n_samples) {
    const auto & cfg = model_.config;

    struct ggml_init_params params = {
        state_.compute_meta.size(),
        state_.compute_meta.data(),
        true,
    };
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, ENC_MAX_NODES, false);

    // Input: mono audio [n_samples, 1, 1].
    struct ggml_tensor * inp = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_samples, 1, 1);
    ggml_set_name(inp, "audio_input");
    ggml_set_input(inp);

    struct ggml_tensor * cur = inp;

    // CNN encoder.
    // Layer 0: input conv k=7, s=1 (causal pad 6).
    cur = ggml_pad_ext(ctx0, cur, 6, 0, 0, 0, 0, 0, 0, 0);
    cur = ggml_conv_1d(ctx0, model_.input_conv_w, cur, 1, 0, 1);
    if (model_.input_conv_b) {
        int64_t ch = model_.input_conv_w->ne[2];
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.input_conv_b, 1, ch, 1));
    }

    int32_t seq_len = n_samples;

    // Mimi causal conv pads both left (K-S) and right (to make pre-conv length
    // a multiple of stride so output = ceil(L/S)).
    auto extra_pad = [](int32_t L, int32_t S) {
        return ((L + S - 1) / S) * S - L;
    };

    // 4 stages: ResUnit -> ELU -> Downsample.
    for (int s = 0; s < 4; ++s) {
        cur = apply_resunit(ctx0, cur, model_.resunits[s]);
        cur = ggml_elu(ctx0, cur);

        int32_t stride = cfg.cnn_strides[s];
        int32_t left_pad = stride;
        int32_t right_pad = extra_pad(seq_len, stride);

        cur = ggml_pad_ext(ctx0, cur, left_pad, right_pad, 0, 0, 0, 0, 0, 0);
        cur = ggml_conv_1d(ctx0, model_.ds_conv_w[s], cur, stride, 0, 1);
        int64_t out_ch = model_.ds_conv_w[s]->ne[2];
        if (model_.ds_conv_b[s]) {
            cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.ds_conv_b[s], 1, out_ch, 1));
        }
        seq_len = (seq_len + stride - 1) / stride;
    }

    cur = ggml_elu(ctx0, cur);

    // Layer 14: final projection k=3, s=1 (causal pad 2).
    cur = ggml_pad_ext(ctx0, cur, 2, 0, 0, 0, 0, 0, 0, 0);
    cur = ggml_conv_1d(ctx0, model_.final_conv_w, cur, 1, 0, 1);
    if (model_.final_conv_b) {
        int64_t ch = model_.final_conv_w->ne[2];
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model_.final_conv_b, 1, ch, 1));
    }

    // Transformer encoder.
    int64_t cnn_seq_len = cur->ne[0];
    cur = ggml_reshape_2d(ctx0, cur, cnn_seq_len, cfg.hidden_size);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, cnn_seq_len);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    for (int i = 0; i < cfg.n_transformer_layers; ++i) {
        cur = apply_tfm_layer(ctx0, cur, model_.tfm_layers[i], (int32_t) cnn_seq_len, positions);
    }

    // Final downsample (stride 2, kernel 4).
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = ggml_reshape_3d(ctx0, cur, cnn_seq_len, cfg.hidden_size, 1);

    // Mimi's self.downsample uses pad_mode="replicate".
    int32_t final_right_pad = extra_pad((int32_t) cnn_seq_len, 2);
    struct ggml_tensor * first = ggml_cont(ctx0,
        ggml_view_3d(ctx0, cur, 1, cur->ne[1], cur->ne[2], cur->nb[1], cur->nb[2], 0));
    struct ggml_tensor * left_rep = ggml_concat(ctx0, first, first, 0);
    if (final_right_pad > 0) {
        struct ggml_tensor * last = ggml_cont(ctx0,
            ggml_view_3d(ctx0, cur, 1, cur->ne[1], cur->ne[2],
                         cur->nb[1], cur->nb[2], (cnn_seq_len - 1) * cur->nb[0]));
        cur = ggml_concat(ctx0, left_rep, cur, 0);
        for (int i = 0; i < final_right_pad; ++i) {
            cur = ggml_concat(ctx0, cur, last, 0);
        }
    } else {
        cur = ggml_concat(ctx0, left_rep, cur, 0);
    }
    cur = ggml_conv_1d(ctx0, model_.final_ds_w, cur, 2, 0, 1);

    int64_t final_seq = cur->ne[0];
    cur = ggml_reshape_2d(ctx0, cur, final_seq, cfg.hidden_size);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    ggml_set_name(cur, "encoder_output");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// --- VQ encoding (CPU) -----------------------------------------------------

int32_t AudioCodecEncoder::vq_nearest(const float * query, int32_t dim,
                                      const void * codebook_data, enum ggml_type cb_type,
                                      int32_t codebook_size) {
    int best_idx = 0;
    float best_dist = 1e30f;

    for (int i = 0; i < codebook_size; ++i) {
        float dist = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float cb_val;
            if (cb_type == GGML_TYPE_F16) {
                cb_val = ggml_fp16_to_fp32(((const ggml_fp16_t *) codebook_data)[i * dim + d]);
            } else {
                cb_val = ((const float *) codebook_data)[i * dim + d];
            }
            float diff = query[d] - cb_val;
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    return best_idx;
}

void AudioCodecEncoder::vq_encode(const float * features, int32_t n_frames,
                                  std::vector<int32_t> & codes) {
    const auto & cfg = model_.config;
    const int32_t dim = cfg.codebook_dim;       // 256
    const int32_t hidden = cfg.hidden_size;     // 512
    const int n_codes = cfg.n_valid_quantizers; // 16
    codes.resize((size_t) n_frames * n_codes);

    std::vector<float> sem_in_proj((size_t) dim * hidden);
    std::vector<float> acou_in_proj((size_t) dim * hidden);
    std::vector<float> sem_out_proj((size_t) hidden * dim);
    std::vector<float> acou_out_proj((size_t) hidden * dim);

    auto download_f32 = [](struct ggml_tensor * t, float * buf, size_t n) {
        if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(n);
            ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < n; ++i) buf[i] = ggml_fp16_to_fp32(tmp[i]);
        } else {
            ggml_backend_tensor_get(t, buf, 0, n * sizeof(float));
        }
    };

    download_f32(model_.vq_sem_in_proj,   sem_in_proj.data(),   (size_t) dim * hidden);
    download_f32(model_.vq_acou_in_proj,  acou_in_proj.data(),  (size_t) dim * hidden);
    download_f32(model_.vq_sem_out_proj,  sem_out_proj.data(),  (size_t) hidden * dim);
    download_f32(model_.vq_acou_out_proj, acou_out_proj.data(), (size_t) hidden * dim);

    std::vector<float> sem_cb_f32((size_t) cfg.codebook_size * dim);
    const float * sem_cb_ptr = nullptr;
    if (model_.vq_sem_codebook->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> sem_cb_fp16((size_t) cfg.codebook_size * dim);
        ggml_backend_tensor_get(model_.vq_sem_codebook, sem_cb_fp16.data(), 0,
                                (size_t) cfg.codebook_size * dim * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < sem_cb_f32.size(); ++i) {
            sem_cb_f32[i] = ggml_fp16_to_fp32(sem_cb_fp16[i]);
        }
        sem_cb_ptr = sem_cb_f32.data();
    } else {
        ggml_backend_tensor_get(model_.vq_sem_codebook, sem_cb_f32.data(), 0,
                                (size_t) cfg.codebook_size * dim * sizeof(float));
        sem_cb_ptr = sem_cb_f32.data();
    }

    std::vector<std::vector<float>> acou_cbs(15);
    for (int cb = 0; cb < 15; ++cb) {
        if (!model_.vq_acou_codebook[cb]) continue;
        acou_cbs[cb].resize((size_t) cfg.codebook_size * dim);
        if (model_.vq_acou_codebook[cb]->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp((size_t) cfg.codebook_size * dim);
            ggml_backend_tensor_get(model_.vq_acou_codebook[cb], tmp.data(), 0,
                                    (size_t) cfg.codebook_size * dim * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < acou_cbs[cb].size(); ++i) {
                acou_cbs[cb][i] = ggml_fp16_to_fp32(tmp[i]);
            }
        } else {
            ggml_backend_tensor_get(model_.vq_acou_codebook[cb], acou_cbs[cb].data(), 0,
                                    (size_t) cfg.codebook_size * dim * sizeof(float));
        }
    }

    std::vector<float> proj_buf(dim);
    std::vector<float> quantized(dim);
    std::vector<float> dequant(hidden);

    auto apply_proj = [&](const float * proj_w, const float * input,
                          float * output, int out_dim, int in_dim) {
        for (int o = 0; o < out_dim; ++o) {
            float sum = 0.0f;
            for (int i = 0; i < in_dim; ++i) {
                sum += proj_w[o * in_dim + i] * input[i];
            }
            output[o] = sum;
        }
    };

    for (int f = 0; f < n_frames; ++f) {
        const float * feat = features + (size_t) f * hidden;

        // Semantic quantizer (code 0).
        apply_proj(sem_in_proj.data(), feat, proj_buf.data(), dim, hidden);

        int32_t sem_idx = 0;
        float best_dist = 1e30f;
        for (int i = 0; i < cfg.codebook_size; ++i) {
            float dist = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = proj_buf[d] - sem_cb_ptr[i * dim + d];
                dist += diff * diff;
            }
            if (dist < best_dist) { best_dist = dist; sem_idx = i; }
        }
        codes[f * n_codes + 0] = sem_idx;

        // (residual chain intentionally unused: Mimi's acoustic RVQ runs on
        // the original pre-VQ features, not on the semantic residual.)
        for (int d = 0; d < dim; ++d) quantized[d] = sem_cb_ptr[sem_idx * dim + d];
        apply_proj(sem_out_proj.data(), quantized.data(), dequant.data(), hidden, dim);

        // Acoustic quantizers (codes 1-15) on pre-VQ features.
        std::vector<float> acou_proj(dim);
        apply_proj(acou_in_proj.data(), feat, acou_proj.data(), dim, hidden);

        std::vector<float> acou_residual(dim);
        for (int d = 0; d < dim; ++d) acou_residual[d] = acou_proj[d];

        for (int cb = 0; cb < 15 && cb < n_codes - 1; ++cb) {
            if (acou_cbs[cb].empty()) {
                codes[f * n_codes + 1 + cb] = 0;
                continue;
            }

            int32_t acou_idx = 0;
            best_dist = 1e30f;
            for (int i = 0; i < cfg.codebook_size; ++i) {
                float dist = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    float diff = acou_residual[d] - acou_cbs[cb][i * dim + d];
                    dist += diff * diff;
                }
                if (dist < best_dist) { best_dist = dist; acou_idx = i; }
            }
            codes[f * n_codes + 1 + cb] = acou_idx;

            for (int d = 0; d < dim; ++d) {
                acou_residual[d] -= acou_cbs[cb][acou_idx * dim + d];
            }
        }
    }
}

// --- public API ------------------------------------------------------------

bool AudioCodecEncoder::encode(const float * samples, int32_t n_samples,
                               std::vector<int32_t> & codes, int32_t & n_frames) {
    if (!model_.ctx) {
        error_msg_ = "model not loaded";
        return false;
    }

    struct ggml_cgraph * gf = build_graph(n_samples);

    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "failed to allocate encoder graph";
        return false;
    }

    struct ggml_tensor * inp = ggml_graph_get_tensor(gf, "audio_input");
    if (!inp) {
        error_msg_ = "missing audio_input tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }
    ggml_backend_tensor_set(inp, samples, 0, (size_t) n_samples * sizeof(float));

    struct ggml_tensor * pos = ggml_graph_get_tensor(gf, "positions");
    if (pos) {
        int32_t pos_len = (int32_t) pos->ne[0];
        std::vector<int32_t> pos_data(pos_len);
        for (int i = 0; i < pos_len; ++i) pos_data[i] = i;
        ggml_backend_tensor_set(pos, pos_data.data(), 0, (size_t) pos_len * sizeof(int32_t));
    }

    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "encoder graph compute failed";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }

    struct ggml_tensor * out = ggml_graph_get_tensor(gf, "encoder_output");
    if (!out) {
        error_msg_ = "missing encoder_output tensor";
        ggml_backend_sched_reset(state_.sched);
        return false;
    }

    int32_t hidden = (int32_t) out->ne[0];
    n_frames = (int32_t) out->ne[1];

    std::vector<float> features((size_t) hidden * n_frames);
    ggml_backend_tensor_get(out, features.data(), 0, features.size() * sizeof(float));

    ggml_backend_sched_reset(state_.sched);

    // Features are laid out [hidden, n_frames] column-major. Re-pack to
    // [n_frames, hidden] row-major for the VQ step.
    std::vector<float> features_row((size_t) n_frames * hidden);
    for (int f = 0; f < n_frames; ++f) {
        for (int h = 0; h < hidden; ++h) {
            features_row[f * hidden + h] = features[h + f * hidden];
        }
    }

    vq_encode(features_row.data(), n_frames, codes);
    return true;
}

void free_codec_encoder_model(codec_encoder_model & model) {
    if (model.buffer) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    model.tensors.clear();
}

} // namespace qwen3_tts
