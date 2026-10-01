# Qwen-Image-2.1 causal text-prefix staging

RTX 5080 16 GiB, 2026-10-01. Grouped-W8 weights, BF16 compute, resident
transformer, 512×512, 40 Euler steps, conditional generation, the same cached
43-token prompt as the [throughput experiments](QWEN-IMAGE-2.1-THROUGHPUT.md).
Each timing run uses one warmup and three measured requests (seeds 42–45).
GPU runs are serial. Timing includes DiT, VAE and image download, but excludes
initialization, final diagnostic latent download and PNG encoding.

## Implemented optimization

`QwenImage21Config::factor_text_prefix(true)` separates text/image computation
while preserving checkpoint parameter names and sharing the same weight leaves.
Text queries attend only to the causal text prefix; image queries still attend
to text plus the complete image. Text uses zero-time modulation, constructed
independently of the runtime timestep. This dependency separation matters:
slicing a zero-time row out of a joint timestep-dependent computation does not
make its SSA dependency closure independent of timestep.

Each transformer block exposes typed `prefix_key` / `prefix_value` taps after
key normalization/RoPE and value projection. `text_prefix_taps()` returns these
paths. Existing `AppliedModel::compile_staged` cuts the IR into:

- Preparation: conditioning and shared parameters → 32 layers of text K/V.
- Execution: latent, timestep, step size, shared parameters and prepared K/V
  → next latent. No conditioning input or text MLP/attention remains in this ABI.

The example's `--prepare-text-prefix` prepares once per conditioning/weight
binding and reuses the buffers across all steps and same-prompt requests.
There is no approximate reuse of image activations, skipped layer or shortened
schedule. The cache must be rebuilt when conditioning or weights/LoRA change.
`BoundModelStages` rejects intermediates belonging to another weight binding.
This is model-level causal factoring followed by the generic IR staging pass,
not an automatic attention-pattern rewrite.

64 F32 buffers occupy 43 MiB at batch 1, 172 MiB at batch 4. Preparation took
65.506 ms / 86.783 ms respectively in the first cold-compilation runs. This
cost is included in pipeline initialization, not each measured request. The
same prompt is repeated across a batch; no cross-request lookup/LRU or mixed
prompt-length serving has been added.

## Steady-state results

| Path | Request median | DiT median | Images/s |
|---|---:|---:|---:|
| Joint baseline, rerun | 6.958 s | 6.737 s | 0.1437 |
| Factored but recomputed each step | 7.412 s | 7.191 s | 0.1349 |
| Factored and prepared once | 6.687 s | 6.465 s | 0.1495 |

Prepared-prefix request time decreased 3.9%, throughput increased **4.1%**.
Factoring alone is slower: duplicating small text projections each step costs
more than it saves. Staging removes that repeated branch. No benefit is claimed
from simply reducing IR instruction counts.

| Batch 4 path | Request median | DiT median | Images/s |
|---|---:|---:|---:|
| Joint baseline, rerun | 25.028 s | 24.162 s | 0.1598 |
| Factored and prepared once | 23.540 s | 22.677 s | 0.1699 |

Batch-4 request time decreased 5.9%, throughput increased **6.3%**. VAE/download
stayed approximately 0.86 s. Baseline samples: 25.028, 25.050, 24.931 s;
prepared: 23.510, 23.568, 23.540 s. Artifacts:
`.cache/qwen-image-2.1/{prefix-baseline-batch4-40steps,text-prefix-batch4-40steps}/`.
These are sequential trials, not randomized repetitions or a multi-prompt
quality/serving benchmark. Sampled whole-GPU peaks were 15,699 MiB baseline
and 15,707 MiB prepared; retained BFC regions obscure live-allocation differences.

Artifacts: `.cache/qwen-image-2.1/{prefix-baseline-40steps,
factored-prefix-40steps,text-prefix-40steps}/`. Request samples:

- Baseline: 6.945, 6.958, 6.961 s.
- Factored: 7.393, 7.412, 7.422 s.
- Prepared: 6.665, 6.687, 6.697 s.

## Numerical limitations: opt-in, not bitwise equivalent

Splitting joint GEMMs changes matrix shapes, kernel selection, fusion and BF16
rounding. Cutting a compiled program can also change fusion/precision boundaries.
The mathematical dependency factoring is exact; GPU floating-point output is
**not** bitwise-equivalent, and this is not a validated quality-neutral default.

For the final batch-1 latent after 40 steps (65,536 F32 values, seed 45):

| Comparison | Max absolute error | Mean absolute error | RMS error |
|---|---:|---:|---:|
| Prepared vs original joint reference | 2.26660 | 0.151532 | 0.230260 |
| Recomputed factored vs prepared | 1.46484 | 0.0308135 | 0.0763320 |
| Prepared vs joint, batch 4 (262,144 values) | 2.51953 | 0.0487880 | 0.105558 |

Both generated images are plausible chair images but have visible local
differences. This single prompt is not an image-quality evaluation. Therefore
factoring/staging remain **off by default**, with explicit CLI opt-in.

Tiny native CPU tests compare original joint vs prepared text branches for
dense/W8 and F32/BF16 projection computation. Separate four-step chain tests
reuse one prefix with changing timestep/latent, for batch 1/2 and F32/BF16;
they compare staged execution to the uncached factored program with absolute
tolerance 1e-5. Host-only tests verify matching parameter leaves, typed tap
shapes and valid IR cuts without creating a runtime. These tests do not turn
the full CUDA numerical differences above into a bitwise guarantee.

## Kernel profiling: optimize GEMMs, not softmax

Nsight Systems traced two cached-executable requests. The second request's
40-step DiT range was isolated using the first timestep GEMM and the final
Euler-output kernel, checked against optimized HLO. This excludes initial
weights/autotuning and VAE. Profiling is diagnostic, not timing evidence for
the benchmark table.

The new profiler range was also exercised on the prepared-prefix path:
`prefix-steady-profile.{nsys-rep,sqlite}` contains only the second request's
DiT. It contains 1,280 image cuDNN attention calls (32 layers × 40 steps),
no text-softmax kernel and no VAE convolution. Dominant GEMM families still
consume about 78.5% of kernel time; this optimization does not remove the
remaining matrix-multiplication bottleneck. Nsight capture-stop processing
perturbs request wall timing, so its logged generation times are not included
in the performance tables.

| Kernel family in original DiT | GPU kernel time | Share |
|---|---:|---:|
| MLP value/gate `gemm_fusion_dot_602` | 2376.627 ms | 35.46% |
| CUTLASS BF16 GEMM | 1212.496 ms | 18.09% |
| Combined Q/K `gemm_fusion_dot_885` | 837.858 ms | 12.50% |
| Other hidden projections `gemm_fusion_dot_598` | 804.740 ms | 12.01% |
| W8 dequantization `loop_multiply_fusion_251` | 414.480 ms | 6.18% |
| Image cuDNN attention | 284.680 ms | 4.25% |
| W8 dequantization `loop_multiply_fusion_250` | 227.916 ms | 3.40% |
| W8 dequantization `loop_multiply_fusion_258` | 209.171 ms | 3.12% |

The four dominant GEMM families total about 78%; the three dequantization
families about 12.7%. Text softmax is tiny. CPU enqueue/transfer optimization
alone cannot remove this GPU compute time. Artifact:
`.cache/qwen-image-2.1/steady-profile.{nsys-rep,sqlite}`. The selected range is
10,571,543,997–17,399,754,176 ns in that trace; kernel durations are summed,
not a wall-time partition when streams overlap.

Controlled GEMM policy experiments did not improve performance:

| DiT compiler policy | Request median | DiT median |
|---|---:|---:|
| `library`: disable Triton GEMM and GEMM fusion v2 | 7.145 s | 6.924 s |
| `triton-v1`: enable Triton, disable fusion v2 | 6.990 s | 6.767 s |
| `cublas`: library policy + CUBLASLT/NATIVE_EMITTER backends | 7.148 s | 6.927 s |

These policies use per-compiler PJRT options and distinct executable cache
keys, not process-global flags. `auto` remains default. The example/wrapper
expose `--gemm-policy`; benchmark reports also record `XLA_FLAGS`. A backend
list containing only CUBLASLT failed compilation for non-GEMM fusions;
CUBLASLT/NATIVE_EMITTER with fusion enabled still failed some GEMM fusions.
Failed candidates have no timing result. Disabling both GEMM fusion paths
made the restricted backend experiment compile, but it was slower. Library
and cublas final latents also differ from the joint reference; triton-v1 was
bit-exact in this batch-1 trial. No default policy was changed.

See [OpenXLA flags guidance](https://openxla.org/xla/flags_guidance) and the
[backend selection schema](https://github.com/openxla/xla/blob/main/xla/xla.proto)
for plugin-dependent option semantics. A policy request is not a guarantee
that every contraction maps to one particular kernel family.

## Reproduction

```sh
cargo build --release -p rxla-safetensors --features model,disk-cache \
  --example qwen_image_2_1_generate
uv run --group qwen-bench python scripts/benchmark_qwen_image_2_1_rxla.py \
  --batches 1,4 --output-dir .cache/qwen-image-2.1/prefix-reference
uv run --group qwen-bench python scripts/benchmark_qwen_image_2_1_rxla.py \
  --batches 1,4 --prepare-text-prefix \
  --output-dir .cache/qwen-image-2.1/prefix-prepared \
  --reference-dir .cache/qwen-image-2.1/prefix-reference
# --factor-text-prefix recomputes the separated branch for diagnostic comparison.
```

For uncontaminated warm DiT capture, the Rust example now supports
`--profile-request 1 --requests 2 --cuda-runtime /trusted/path/libcudart.so.13`.
Run it under `nsys profile --capture-range=cudaProfilerApi
--capture-range-end=stop --trace=cuda --cuda-graph-trace=node --sample=none
--cpuctxsw=none`. The range waits for the final latent before stopping capture,
then executes VAE outside it. An RAII guard stops profiling on error; explicit
start/stop errors are returned. Profiled timings are not normal benchmark
timings. In this environment the runtime is
`.venv/lib/python3.14/site-packages/nvidia/cu13/lib/libcudart.so.13`.
