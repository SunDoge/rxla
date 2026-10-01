# Qwen-Image-2.1 inference benchmark

The original results below use four steps. See [40-step measurements](#40-step-measurements)
for the 2026-10-01 default-step workload and a weight-matched PyTorch reference.

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

### Persistent pipeline

The generation example can retain its compiled DiT, prompt context, DiT and VAE
weights, VAE bridge, and VAE decoder across requests with `--requests`. On the
same card, three batch-8 requests produced:

| Phase | Request 0 | Request 1 | Request 2 |
|---|---:|---:|---:|
| DiT | 4.760 s | 4.696 s | 4.705 s |
| VAE | 1.855 s | 1.715 s | 1.714 s |
| Complete request | 6.616 s | 6.412 s | 6.421 s |

Pipeline initialization took 2.356 s once. Steady-state median throughput is
**1.246 img/s**, 40.8% above the fresh-process batch-8 result of 0.885 img/s.
This is **4.33×** the measured throughput of the official BF16 batch-4 model
offload path. A seed-42 persistent request produced byte-identical images to
the earlier one-shot device-bridge path.

## 40-step measurements

Measured on 2026-10-01 on the same RTX 5080, with about 1,036 MiB already used
by another process. Resolution is 512×512, cached prompt embeddings are
`[1, 43, 4096]`, and guidance uses the single conditional path (the official
pipeline's default `true_cfg_scale=1`). Forty steps match the installed official
pipeline's default. Runs are serial, each with one warmup and three measured
requests. RXLA retains its pipeline and uses VAE micro-batch 4.
Each batch repeats the same cached prompt; these runs do not measure mixed
prompt lengths or concurrent requests.

| Implementation | Batch | Median request | Throughput | Measured request samples |
|---|---:|---:|---:|---|
| RXLA W8, resident pipeline | 1 | 6.937 s | 0.144 img/s | 6.916, 6.937, 6.957 s |
| RXLA W8, resident pipeline | 8 | 49.164 s | 0.163 img/s | 49.362, 49.164, 49.115 s |
| Diffusers, same RXLA W8 checkpoint, resident pipeline | 1 | 15.512 s | 0.064 img/s | 15.502, 15.512, 15.512 s |
| Diffusers, same RXLA W8 checkpoint, VAE micro-batch 2 | 8 | 62.024 s | 0.129 img/s | 62.011, 62.040, 62.024 s |
| Official BF16, model CPU offload | 1 | 12.769 s | 0.078 img/s | 13.320, 12.769, 12.661 s |

The weight-matched reference replaces Diffusers' bias-free Linear modules with
eager grouped-W8 dequantization (`(U8 - 128) * F32 scale`) followed by BF16 GEMM.
It loads the exact 7,559,831,552-byte RXLA transformer payload and checks the
entire state dictionary strictly. This is an executable same-checkpoint
reference, not a tuned quantization library or a `torch.compile` comparison.
Both implementations produce visually plausible images for the cached chair
prompt; different RNGs, seeds of saved requests and intermediate arithmetic
mean this does not establish numerical equivalence or a quality ranking.
RXLA batch 1 and the first image of batch 8, at the same final-request seed,
have mean absolute RGBA byte difference 2.783/255 and maximum difference
151/255. Batching is therefore not bit-exact over forty denoising steps.

RXLA is **2.24× faster** than this same-weight eager reference at batch 1 and
**1.84× faster** than the official BF16 offload deployment path. For resident
compute only, official BF16 takes 6.316 s (samples 6.321, 6.315, 6.316 s),
versus RXLA W8 DiT's 6.718 s: RXLA takes about **6.4% longer**. The official
diagnostic skips VAE and retains prefix KV caching; these are different weight
formats and execution paths, so this is not an isolated GEMM comparison.
At batch 8 RXLA delivers **26.2% more throughput** than the same-weight eager
reference. The PyTorch run requires VAE micro-batch 2 and
`PYTORCH_ALLOC_CONF=expandable_segments:True`, whereas RXLA uses micro-batch 4
and PJRT's BFC allocator. This is a comparison of the working deployment
configurations, not identical VAE batching or allocator settings.

RXLA batch 8 improves image throughput by only **12.9%** over batch 1 at forty
steps. Its DiT median is 47.430 s and VAE/download median is 1.730 s; batch 1
uses 6.718 s and 0.219 s respectively. Denoising dominates this workload, so
amortizing VAE and initialization cannot reproduce the large throughput gains
in the earlier four-step runs. Batch-8 request latency is about 49 s, not the
6.15 s per-image amortized cost.

Timing boundaries: RXLA request timing includes denoising, VAE and output
download, and excludes initialization and PNG encoding. Diffusers includes
denoising and VAE with device-completion synchronization but leaves output on
GPU; output download and PNG encoding are excluded. Neither includes text
encoding or network/server overhead. Initialization and the first RXLA request
are excluded; batch 1's first request took 15.245 s and batch 8's took 49.500 s.

Memory measurements use different counters and must not be compared directly:

- Sampling `nvidia-smi` every approximately 200 ms found 15,699 MiB total GPU
  usage for both RXLA runs, 14,663 MiB above the baseline. This includes CUDA
  context and retained BFC allocator regions, including free blocks, and is not
  the peak live tensor allocation or a minimum required VRAM estimate.
- PyTorch batch-1 active-allocation peaks were 10,069,532,160 bytes (9.38 GiB)
  for the W8 resident pipeline and 14,455,545,344 bytes (13.46 GiB) for BF16
  offload. Those exclude cached allocator blocks and other processes.
- PyTorch W8 batch 8 with micro-batch 2 peaked at 11,863,838,208 active bytes
  (11.05 GiB), with the expandable allocator enabled.

Reproduce the RXLA matrix:

```bash
cargo build --release -p rxla-safetensors --features model,disk-cache \
  --example qwen_image_2_1_generate
uv run --group qwen-bench python scripts/benchmark_qwen_image_2_1_rxla.py \
  --steps 40 --batches 1,8 --warmup 1 --iterations 3
```

The wrapper writes logs, `results.json` and final-request PNGs to
`.cache/qwen-image-2.1/benchmark-40steps/`. Sampling errors and failed/incomplete
generation requests are recorded rather than reported as valid benchmarks.

Reproduce the weight-matched reference:

```bash
uv run --group qwen-bench python scripts/benchmark_qwen_image_2_1_official.py \
  --steps 40 --batch-size 1 --warmup 1 --iterations 3 \
  --transformer-weights rxla-w8 --residency resident-pipeline --kv-cache on

PYTORCH_ALLOC_CONF=expandable_segments:True \
uv run --group qwen-bench python scripts/benchmark_qwen_image_2_1_official.py \
  --steps 40 --batch-size 8 --vae-batch-size 2 --warmup 1 --iterations 3 \
  --transformer-weights rxla-w8 --residency resident-pipeline --kv-cache on
```

For the BF16 deployment reference use `--transformer-weights bf16 --residency
model-offload`. For the BF16 compute diagnostic use `--transformer-weights bf16
--residency resident-transformer --output latent`. `--save-image PATH` optionally
saves the first image of the final measured image request after the timer ends.

The PyTorch same-W8 batch-8 pipeline exhausted memory with a full-batch VAE and
again with VAE micro-batch 4. The latter failed inside VAE normalization with
12.20 GiB of active allocations and another 1.12 GiB allocation requested.
RXLA's measured batch-8 path completes with micro-batch 4. These failed runs
provide no valid complete-request latency samples.

## Four-step reproduction

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
  --batch-size 8 --vae-batch-size 4 --requests 3 \
  --memory-fraction 0.9 --preallocate false \
  --cache-dir .cache/qwen-image-2.1/compile-cache
```

## IR staging experiment

On 2026-10-01 we extracted timestep embedding/modulation and final output scale
from the same traced Qwen model using explicit named SSA cuts. The preparation
ABI contains only model input 1 (timestep); the execution ABI contains inputs
0/2/3 (sample, step size, prompt) plus prepared vectors. This is an IR partition,
not a separately maintained model implementation. The original program remains
available for the A/B reference. See [API and restrictions](IR-STAGING.md).

Same RTX 5080, W8 checkpoint, BF16 projections, 512×512, 40 steps, conditional
path, one warmup and three measured requests, run serially with the same binary:

| Path | Batch | Median request | DiT median | Request samples |
|---|---:|---:|---:|---|
| Original | 1 | 6.961 s | 6.740 s | 6.938, 6.961, 6.969 s |
| Prepared schedule | 1 | 6.989 s | 6.767 s | 6.972, 6.989, 6.996 s |
| Original | 8 | 49.296 s | 47.558 s | 49.272, 49.296, 49.300 s |
| Prepared schedule | 8 | 48.992 s | 47.257 s | 49.122, 48.992, 48.980 s |

Preparation took 31.480 ms for batch 1 and 47.958 ms for batch 8. Its cached
F32 tensor payload is 5.625 MiB / 27.5 MiB respectively for all forty steps,
excluding allocator overhead and step-size buffers. Preparation is charged to
pipeline initialization, not steady-state requests. Step-size buffers are also
uploaded once in the staged path; the baseline still uploads both scalars each
step, so this is a serving-path experiment rather than an isolated kernel test.

Batch 1's final seed-45 latent has 65,536 F32 values and is bit-exact against
the original after forty steps. Batch 8's 524,288 values are also bit-exact;
all nine final PNG files match byte-for-byte against their corresponding
references. Sampled whole-GPU peaks are unchanged between paths: 15,699 MiB
at batch 1 and 15,701 MiB at batch 8, including the 1,036 MiB background usage
and retained BFC allocator regions. This is not a live-tensor memory counter.
The tiny CPU test
checks four chained steps at batch 1/2 with both F32 and BF16 projections.
These checks establish equivalence for these inputs, not for every possible
compiler configuration.

There is **no substantial demonstrated acceleration**: batch 1 is about 0.4%
slower, within the observed run spread; batch 8 is about 0.6% faster in this
single serial A/B, insufficient to claim a stable gain. Staging remains opt-in
(`--prepare-schedule`), and no
memory-saving claim is made. It adds reusable infrastructure for larger
request-invariant computations; it does not address the dominant attention and
large projection kernels in this workload.

Reproduce with matching seed, prompt, weights, resolution and step count:

```bash
uv run --group qwen-bench python scripts/benchmark_qwen_image_2_1_rxla.py \
  --batches 1,8 --steps 40 --seed 42 \
  --output-dir .cache/qwen-image-2.1/staging-reference
uv run --group qwen-bench python scripts/benchmark_qwen_image_2_1_rxla.py \
  --batches 1,8 --steps 40 --seed 42 --prepare-schedule \
  --output-dir .cache/qwen-image-2.1/staging-prepared \
  --reference-dir .cache/qwen-image-2.1/staging-reference
```

The wrapper exports the last request's little-endian F32 latent and can record
bit equality, maximum/mean absolute error and RMS error against a reference
folder. Diagnostic latent download and PNG encoding are outside request timing.
Actual experiment artifacts are in `staging-baseline-40steps/` (batch 1),
`staging-baseline-batch8-40steps/` (batch 8), and `staging-latent-40steps/`
(both staged batches), under `.cache/qwen-image-2.1/`.

## Next optimization target

The primary remaining performance target is the resident DiT gap, not offload.
Profiling should focus on grouped-W8 dequantization/GEMM, attention lowering and
layout conversions. Closing the 0.27-second four-step compute gap would improve
both resident and offloaded serving without sacrificing RXLA's memory advantage.

### Buffer-donation experiment

RXLA now supports explicit input/output aliases and ownership-consuming PJRT
execution, but the Qwen denoising latent is deliberately not donated. On the
RTX 5080 batch-8 workload, forcing latent input 0 to alias output 0 constrained
XLA's global buffer assignment: execution peaked at 9.07 GiB and then requested
another 5.63 GiB, exhausting the 14 GiB allocator pool. The ordinary executable
completes the same workload.

Donation remains useful for small, isolated update programs and is verified on
the CPU PJRT backend. It should be selected from measured buffer-assignment
results, not inferred merely because an input and output have identical types.
