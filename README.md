# llm-quantization-from-scratch

From-scratch implementation of LLM quantization algorithms in pure PyTorch — no library quantization APIs. Built to understand the math behind each method, not just call a function.

**Model:** Llama 3 8B | **Device:** A100 80GB | **Eval:** WikiText-2 Perplexity

---

## Methods Implemented

| Method | Bits | Granularity | Status | Perplexity | Δ vs BF16 |
|---|---|---|---|---|---|
| BF16 Baseline | 16 | — | ✅ Done | 6.14 | — |
| PTQ (W8A16) | INT8 | Per-channel | ✅ Done | 6.15 | +0.01 |
| SmoothQuant (W8A16) | INT8 | Per-channel | ✅ Done | 6.15 | +0.01 |
| GPTQ | INT4 | Per-group (g=128) | 🔄 In Progress | — | — |
| AWQ | INT4 | Per-group (g=128) | 🔄 In Progress | — | — |
| FP8 | FP8 | Per-tensor | ⏳ Planned | — | — |

> **GPTQ and FP8** experiments are still in progress due to GPU availability and compute limitations. GPTQ requires long per-layer calibration runs (~90 min on A100), and FP8 requires H100 hardware which is not consistently available on Colab Pro.

---

## What's Implemented

### 1. Quantization Primitives
- Symmetric and asymmetric quantization
- Per-tensor, per-channel, and per-group scale computation
- `quantize` / `dequantize` with full dtype and broadcasting support
- Verified on [4096, 4096] weight matrices matching Llama 3 8B dimensions

### 2. PTQ — Post-Training Quantization (W8A16)
- Per-channel symmetric INT8 weight quantization
- In-place layer replacement with `QuantizedLinear`
- CPU offload during quantization to avoid VRAM spikes
- 225 Linear layers quantized, model size: 16.1 GB → 8.56 GB

### 3. SmoothQuant (W8A16)
- Forward hook-based calibration to collect per-channel activation statistics
- Outlier detection: up to 3472x max/mean ratio in MLP layers
- Smoothing factor computation: `s_j = max|X_j|^α / max|W_j|^(1-α)`, α=0.5
- Correct absorption: LayerNorm gamma ÷ s, Linear weight columns × s
- Shared s per LayerNorm (q/k/v projections share one smoothing factor)
- 160 layers smoothed, 65 skipped (o_proj, down_proj — no preceding LayerNorm)

### 4. GPTQ (In Progress)
- Hessian computation: H = 2XX^T with diagonal damping
- Cholesky decomposition of H for stable H⁻¹ computation
- Column-by-column INT4 quantization with OBS weight updates
- Per-group scales (g=128) for INT4 precision

### 5. AWQ (Planned)
- Salience scoring: `salience(j) = E[|x_j|] × max|W[:,j]|`
- Per-channel scaling of salient input channels before INT4 quantization
- Scale absorption into preceding LayerNorm

### 6. FP8 (Planned, requires H100)
- FP8 E4M3 weight and activation quantization
- Requires hardware FP8 support (H100/Ada Lovelace)

---

## Key Implementation Details

**Why CPU offload for quantization?**
Llama 3 8B in BF16 uses ~16 GB VRAM. Creating intermediate quantization tensors on GPU pushes over budget. All quantization math runs in FP32 on CPU; only the final INT8/INT4 result moves back to GPU.

**Why shared smoothing factors per LayerNorm?**
q_proj, k_proj, v_proj all receive the same LayerNorm output X. They must share the same s — otherwise the LayerNorm gamma gets multiplied multiple times, exploding activations. We compute one s per LayerNorm using the element-wise max of w_max across all sharing projections.

**Why skip o_proj and down_proj in SmoothQuant?**
SmoothQuant requires absorbing s into a preceding LayerNorm. o_proj and down_proj have no preceding LayerNorm — applying one-sided weight division without the corresponding activation compensation destroys the model output.

---

## Setup

```bash
# Requires Colab Pro with A100 (40GB+) for Llama 3 8B
# H100 required for FP8 experiments

pip install transformers datasets huggingface_hub
```

```python
from huggingface_hub import login
login()  # requires access to meta-llama/Meta-Llama-3-8B
```

---

## Results

Evaluated on WikiText-2 test set, sequence length 2048.

```
Method                 Perplexity   Delta vs BF16
--------------------------------------------------
BF16 Baseline                6.14            0.00
PTQ W8A16 (INT8)             6.15           +0.01
SmoothQuant (W8A16)          6.15           +0.01
GPTQ (INT4)                   —          in progress
AWQ (INT4)                    —          in progress
```

## Memory Savings

| Method | Model Size | Reduction |
|---|---|---|
| BF16 Baseline | 16.1 GB | — |
| PTQ W8A16 (INT8) | 8.56 GB | 47% |
| SmoothQuant (W8A16) | 8.56 GB | 47% |
| GPTQ (INT4) | ~4.5 GB (est.) | ~72% |
| AWQ (INT4) | ~4.5 GB (est.) | ~72% |

INT8 methods cut model size roughly in half by storing weights as 1-byte integers instead of 2-byte BF16. INT4 methods (GPTQ, AWQ) cut it to ~¼ — enabling Llama 3 8B to fit on a single 6GB consumer GPU at INT4, versus requiring a 24GB+ card at BF16.

PTQ and SmoothQuant both achieve near-lossless INT8 compression — 2x memory reduction with only +0.01 perplexity degradation on Llama 3 8B.

---

## References

- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
