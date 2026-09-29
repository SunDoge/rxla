# Qwen-Image-2.1 inference benchmark

Measured on 2026-09-29 on an NVIDIA GeForce RTX 5080 (16 GiB). Another
long-lived process occupied about 1 GiB during every run. Both implementations
used cached prompt embeddings with shape `[1, 43, 4096]`, 512×512 output, four
Euler steps, seed 42, and the official VAE.

This is not a same-dtype kernel shootout:

- Official Diffusers used the released BF16 transformer (about 14.2 GB of
  checkpoint payload).
- RXLA used its grouped-W8 checkpoint (about 7.6 GB) with BF16 compute.

The comparison therefore answers the deployment question—how fast can each
documented/implemented path produce an image on this 16 GiB card—while the
resident-transformer diagnostic separately exposes raw DiT performance.

## Versions

- PyTorch `2.14.0+cu130`
- Transformers `5.17.0`
- Diffusers `0.41.0.dev0`, git commit
  `4ac08e940880fa906c36b944a42f420db4e24b26`
- Official pipeline: `QwenImage21Pipeline`
- Official constrained-memory mode: `enable_model_cpu_offload()`, as shown in
  the [Qwen-Image-2.1 model README](https://huggingface.co/Qwen/Qwen-Image-2.1)
- RXLA CUDA backend: JAX CUDA 13 PJRT plugin

The official benchmark requirements are recorded in uv's `qwen-bench` group;
the Diffusers git revision is pinned by `uv.lock` because released Diffusers
0.35 does not contain Qwen-Image-2.1.

## Results

| Measurement | Median | Samples | Peak CUDA allocation |
|---|---:|---|---:|
| Official BF16, model CPU offload, complete image | 7.582 s | 7.582, 7.607, 7.437 s | 14.453 GB |
| Official BF16, model CPU offload, latent only | 7.211 s | 7.212, 7.211, 7.204 s | 14.453 GB |
| Official BF16, resident transformer, latent only | 0.634 s | 0.634, 0.634, 0.635 s | 14.453 GB |
| Official BF16, resident transformer, KV cache disabled | 0.647 s | 0.647, 0.647, 0.648 s | 14.431 GB |
| RXLA W8, fresh process, complete image generation stage | 3.129 s | 3.129, 3.108, 3.154 s | not recorded |
| RXLA W8, resident DiT execution only | 0.904 s | 0.906, 0.904, 0.903 s | not recorded |

The RXLA generation timer begins after PJRT client creation and includes model
tracing, checkpoint upload, executable-cache lookup, four denoising steps, VAE
checkpoint upload and decode, pixel conversion, and PNG writing. Each RXLA
sample above came from a new process, so it is a conservative approximation of
a persistent server. The official samples exclude initial CPU model loading but
include every request's CPU→GPU model offload.

At this operating point:

- RXLA complete-image latency is **2.42× faster** than the official documented
  16-GiB offload path (`7.582 / 3.129`), a 58.7% reduction.
- Official resident BF16 DiT execution is **1.43× faster** than RXLA W8 DiT
  execution (`0.904 / 0.634`). RXLA's deployment win currently comes from
  fitting quantized weights and avoiding the official path's repeated 14-GB
  transfer, not from faster transformer kernels.
- Official prefix KV caching accounts for only about 2% in this short,
  43-token/four-step case (`0.647 s` without versus `0.634 s` with cache). It
  does not explain most of the RXLA compute gap.
- A complete official BF16 resident pipeline does not fit comfortably alongside
  the existing 1-GiB GPU process. The resident-transformer row intentionally
  skips VAE decode and is a compute diagnostic, not a deployable full-pipeline
  result on this card.

## Batch throughput

RXLA now batches the complete transformer graph and uses an independently
bounded VAE micro-batch. This matters on 16-GiB devices: batch 8 fits through
the W8 transformer, while decoding all eight images in one VAE invocation asks
XLA's BFC allocator for a contiguous 6.75-GiB block and fails. The default
`--vae-batch-size 4` decodes the same eight latents in two launches without
reducing transformer utilization.

The DiT-to-VAE boundary remains device-resident: latent denormalization,
micro-batch slicing and final-batch padding are compiled as a small StableHLO
bridge. An A/B run produced byte-identical PNGs for all eight samples. Its
measured latency is neutral at this latent size (9.08 s versus 9.04 s), but it
removes a host synchronization point required for overlapping stages in a
persistent serving pipeline.

These RXLA measurements are warm disk-cache, fresh-process runs. Consequently
they still include roughly one second of checkpoint loading plus tracing and
cache lookup. PNG writes are included. The official batch 2/4 rows are
single measured requests without warmup; they are deployment-path comparisons,
not a same-dtype kernel benchmark.

| Implementation | Batch | DiT execute | VAE | End-to-end | Throughput |
|---|---:|---:|---:|---:|---:|
| RXLA W8 | 1 | 0.968 s | 1.264 s | 3.787 s | 0.264 img/s |
| RXLA W8 | 2 | 1.319 s | 1.491 s | 4.367 s | 0.458 img/s |
| RXLA W8 | 4 | 2.470 s | 1.935 s | 5.909 s | 0.677 img/s |
| RXLA W8, VAE micro-batch 4 | 8 | 4.845 s | 2.681 s | 9.042 s | 0.885 img/s |
| Official BF16 model offload | 1 | — | — | 7.582 s | 0.132 img/s |
| Official BF16 model offload | 2 | — | — | 11.682 s | 0.171 img/s |
| Official BF16 model offload | 4 | — | — | 13.902 s | 0.288 img/s |

RXLA batch 8 delivers **3.35×** the batch-1 image throughput. It is also
**3.07×** the throughput of the official batch-4 offload path on this machine,
with the important caveat that RXLA uses W8 weights while the official model is
BF16. For latency-sensitive traffic, batch 1 remains appropriate; a serving
scheduler should choose batch 4 or 8 when throughput is the priority.

## Reproduction

Official complete-image path:

```bash
uv run --group qwen-bench python \
  scripts/benchmark_qwen_image_2_1_official.py \
  --output image \
  --residency model-offload \
  --kv-cache on \
  --steps 4 \
  --warmup 1 \
  --iterations 3
```

Official resident-transformer diagnostic:

```bash
uv run --group qwen-bench python \
  scripts/benchmark_qwen_image_2_1_official.py \
  --output latent \
  --residency resident-transformer \
  --kv-cache on \
  --steps 4 \
  --warmup 1 \
  --iterations 3
```

RXLA complete image (repeat serially; do not run multiple GPU samples in
parallel):

```bash
target/release/examples/qwen_image_2_1_generate \
  .venv/lib/python3.14/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so \
  .cache/qwen-image-2.1/transformer_w8.safetensors \
  .cache/qwen-image-2.1/vae/diffusion_pytorch_model.safetensors \
  .cache/qwen-image-2.1/vae_config.json \
  .cache/qwen-image-2.1/conditioning_chair.safetensors \
  .cache/qwen-image-2.1/output.png \
  --width 512 --height 512 --steps 4 \
  --batch-size 8 --vae-batch-size 4 \
  --memory-fraction 0.9 --preallocate false \
  --cache-dir .cache/qwen-image-2.1/compile-cache
```

## Next optimization target

The primary remaining performance target is the resident DiT gap, not offload.
Profiling should focus on grouped-W8 dequantization/GEMM, attention lowering and
layout conversions. Closing the 0.27-second four-step compute gap would improve
both resident and offloaded serving without sacrificing RXLA's memory advantage.
