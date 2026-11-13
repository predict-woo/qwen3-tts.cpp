# Mimi Encoder Architecture (verified from tensors + config)

## Confirmed tensor inventory (193 tok_enc tensors in v2 GGUF)

### SEANet stages (4 stages, strides 8,6,5,4 → 25Hz)

| Stage | Entry conv | Channels | Kernel | Res blocks |
|-------|-----------|----------|--------|------------|
| 0 | conv.0 | 1→64 | 7 | res.1.blk.{1,3}: 64→32→64, k3+k1 |
| 1 | conv.3 | 64→128 | 8 | res.4.blk.{1,3}: 128→64→128, k3+k1 |
| 2 | conv.6 | 128→256 | 10 | res.7.blk.{1,3}: 256→128→256, k3+k1 |
| 3 | conv.9 | 256→512 | 12 | res.10.blk.{1,3}: 512→256→512, k3+k1 |

All convs are **causal** (`use_causal_conv: true`), activation is **ELU**.

### SEANet bottleneck (stride 1, still 25Hz)

| Conv | Channels | Kernel | Purpose |
|------|----------|--------|---------|
| conv.12 | 512→1024 | 16 | Wide-context expansion |
| conv.14 | 1024→512 | 3 | Final contraction (last_kernel_size=3) |

### Transformer (8 layers, 25Hz, 512 hidden)

- LayerNorm **with bias** (not RMSNorm)
- Self-attention: 8 heads, head_dim=64, sliding_window=250, RoPE θ=10000
- MLP: **GELU** non-gated (fc1→fc2, intermediate=2048, no gate_proj)
- Layer scales on both attn and FFN outputs (initial_scale=0.01)
- Causal attention

### Post-transformer downsample

| Tensor | Shape | Stride | Rate |
|--------|-------|--------|------|
| tok_enc.downsample | [4, 512, 512] | 2 | 25Hz→12.5Hz |

### Quantizer (split RVQ)

| Path | Input proj | Codebooks | Output proj |
|------|-----------|-----------|-------------|
| Semantic | 512→256 | 1 layer, 2048×256 | 256→512 |
| Acoustic | 512→256 | 15 of 31 layers, 2048×256 each | 256→512 |

Both paths quantize the **same input** (parallel, not sequential).
Residual chaining happens **within** each path.
Codebook search: L2 argmin (not cosine).
Output: [n_frames × 16] tokens (1 semantic + 15 acoustic).

## Config (from speech_tokenizer/config.json)

```json
{
  "upsampling_ratios": [8, 6, 5, 4],
  "num_residual_layers": 1,
  "kernel_size": 7,
  "last_kernel_size": 3,
  "residual_kernel_size": 3,
  "compress": 2,
  "use_causal_conv": true,
  "hidden_size": 512,
  "num_hidden_layers": 8,
  "num_attention_heads": 8,
  "sliding_window": 250,
  "rope_theta": 10000.0,
  "hidden_act": "gelu",
  "layer_scale_initial_scale": 0.01,
  "num_filters": 64,
  "codebook_size": 2048,
  "codebook_dim": 256,
  "num_quantizers": 32,
  "num_semantic_quantizers": 1,
  "encoder_valid_num_quantizers": 16
}
```

## Forward pass

```
[1, T] @ 24kHz
  ↓ SEANet stage 0 (stride 8) → [64, T/8]
  ↓ SEANet stage 1 (stride 6) → [128, T/48]
  ↓ SEANet stage 2 (stride 5) → [256, T/240]
  ↓ SEANet stage 3 (stride 4) → [512, T/960] @ 25Hz
  ↓ conv.12 (stride 1, expand) → [1024, T/960]
  ↓ conv.14 (stride 1, contract) → [512, T/960]
  ↓ Transformer (8 layers) → [512, T/960] @ 25Hz
  ↓ downsample (stride 2) → [512, T/1920] @ 12.5Hz
  ↓ Split RVQ → [n_frames × 16] tokens @ 12.5Hz
```

Where n_frames = T / 1920 (for T samples at 24kHz).
