# Qwen3.5-0.8B W8 inference benchmark

RXLA implements the real text topology of `Qwen/Qwen3.5-0.8B`: 24 decoder
layers, with three gated-delta linear-attention layers followed by one full
attention layer in each group of four. The implementation includes causal
depthwise convolution, recurrent delta state, Q/K normalization, gated full
attention, partial text RoPE, zero-centered RMSNorm and tied embeddings.

This target is text-only. The checkpoint's vision tower and the optional MTP
layer are intentionally excluded. Prefill currently constructs the gated-delta
recurrence at trace time for a static sequence length; persistent decode state
and dynamic-length batching remain future work.

## Quantized checkpoint

Two-dimensional weights use RXLA's symmetric per-output-row, per-128-element
group W8 format. U8 values encode signed INT8 by adding 128, while F32 group
scales are stored separately. U8 remains the PJRT input type and dequantization
is visible to XLA next to the contraction; the host never expands the checkpoint
to F32. An optimized-HLO dump of the measured sequence-1 executable placed U8
conversion, offset, group scaling and contraction reduction in the same fused
computation; this verifies fusion for that shape, not every backend or shape.

Download the official checkpoint, then convert only its language model:

```sh
uv run --no-sync python benchmarks/quantize_qwen3_5_w8.py \
  model.safetensors model-w8.safetensors --group-size 128
```

On the measured checkpoint this reduced the file from 1.7 GB to 742 MB. The W8
file also contains F32 normalization, convolution and recurrent parameters.
This is an RXLA interchange format, not GGUF, GPTQ or AWQ.

## Reproduction

Build and run the RXLA benchmark:

```sh
cargo build --release -p rxla-safetensors --example qwen3_5_bench \
  --features model

target/release/examples/qwen3_5_bench \
  "$PJRT_CUDA_PLUGIN_PATH" model-w8.safetensors \
  --sequence 16 --warmup 2 --iterations 10 \
  --output rxla-qwen3.5-s16.f32
```

Run the official BF16 Transformers implementation using the project `uv`
environment:

```sh
uv run --no-sync python benchmarks/qwen3_5_torch.py MODEL_DIRECTORY \
  --sequence 16 --warmup 2 --iterations 10 \
  --compare rxla-qwen3.5-s16.f32
```

## RTX 5080 snapshot

Measured September 15, 2026 on an NVIDIA GeForce RTX 5080 with the JAX CUDA 13
PJRT plugin, CUDA runtime 13.0, driver 13.4 and cuDNN 9.24. Both measurements
use batch 1, sequence 16, synchronized execution and no KV/recurrent cache.

| Runner | Storage | mean | p50 | p95 | tokens/s |
| --- | --- | ---: | ---: | ---: | ---: |
| RXLA/PJRT/XLA | grouped W8 | 10.535 ms | 10.572 ms | 10.792 ms | 1518.7 |
| Transformers eager fallback | BF16 | 44.512 ms | 44.507 ms | 44.581 ms | 359.5 |

RXLA checkpoint loading took 501 ms and compilation took 40.9 s. The static
recurrent expansion produces register-spill warnings at sequence 16, so compile
time and longer-prefill scaling are known optimization targets. At sequence 1,
RXLA measured 2.393 ms, or 417.8 token/s, after a 2.54 s compilation.

Transformers reported that FLA and `causal-conv1d` fast paths were unavailable
and used its ordinary Torch implementation. Therefore the 4.2x difference is a
comparison against that explicit fallback, not a claim against optimized Qwen
runtimes such as vLLM, SGLang or llama.cpp.

Against official BF16 logits for identical token IDs, W8 produced maximum and
mean absolute differences of 0.606823 and 0.047905; the BF16 mean absolute logit
was 1.825396. Per-position top-1 agreement was 15/16. These numbers establish a
basic quantized correctness check, not language-quality or perplexity parity.
