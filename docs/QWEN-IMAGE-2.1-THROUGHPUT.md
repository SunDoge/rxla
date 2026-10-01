# Qwen-Image-2.1 throughput experiments

Follow-up: [causal text-prefix staging and kernel profiling](QWEN-IMAGE-2.1-PREFIX-CACHE.md).

Measured 2026-10-01 on RTX 5080 16 GiB, with 1,036 MiB background GPU usage.
RXLA grouped-W8 weights, BF16 compute, 512×512, cached `[1,43,4096]` prompt,
conditional path, resident weights and VAE micro-batch limit 4. GPU runs are
serial, with one warmup and three measured requests. Request timing includes
DiT, VAE and output download, but excludes initialization, diagnostic latent
export and PNG encoding. Requests use seeds 42–45; comparisons use final seed 45.
These are steady-state, same-prompt batches, not concurrent serving or p95 tests.

## Batch sweep: forty steps

| Batch | VAE tail | Request median | DiT median | VAE/download median | Images/s |
|---:|---|---:|---:|---:|---:|
| 1 | No tail | 6.961 s | 6.740 s | 0.221 s | 0.1437 |
| 2 | No tail | 12.861 s | 12.428 s | 0.434 s | 0.1555 |
| 4 | No tail | 25.047 s | 24.182 s | 0.862 s | 0.1597 |
| 6 | Pad 2 to 4 | 37.496 s | 35.764 s | 1.729 s | 0.1600 |
| 6 | Exact 4+2 | 37.068 s | 35.760 s | 1.305 s | 0.1619 |
| 8 | No tail | 49.296 s | 47.558 s | 1.735 s | 0.1623 |

Batch 1/8 references are from the preceding same-configuration staging A/B;
batch 2/4/6 are new measurements. This is not a randomized batch-size trial.
Going from batch 1 to 8 improves throughput by about 13%, not eightfold.
Batch 2 gains about 8% at the cost of nearly doubling request latency. Batch 4
captures most of the measured batching gain. Exact-tail batch 6 approaches
batch-8 throughput with a shorter whole-batch completion time.

Artifacts under `.cache/qwen-image-2.1/`:

- Batch 1 reference: `staging-baseline-40steps/`.
- Batch 8 reference: `staging-baseline-batch8-40steps/`.
- Batch 2/4/6 padded reference: `batch-sweep-40steps/`.
- Exact-tail batch 6: `exact-tail-40steps/`.

## Implemented improvement: exact VAE tail

`MicroBatchPlan::split_exact` keeps the tail's real shape. Qwen compiles at most
one additional VAE shape and binds it to the same shared weight buffers: no
second weight upload. For batch 6, VAE executes 4+2 rather than 4+4 with two
discarded padded results. Shape executables use the normal compilation cache.

VAE/download median decreased **24.5%**, from 1.729 to 1.305 seconds; end-to-end
throughput increased **1.15%**. DiT timing stayed essentially unchanged. The
first four PNG files match byte-for-byte. The last two differ by at most one
RGBA byte level, with mean byte errors 0.0000229 and 0.0000162. All 393,216
latent F32 values match bit-for-bit. This does not imply every backend/shape
will produce identical VAE rounding.

Exact tails are enabled by default in the Qwen example. Restore padding with
`--exact-vae-tail false`, or `--pad-vae-tail` in the Python benchmark wrapper.
The public micro-batch planner still offers both methods; other consumers are
not automatically changed. Sampled whole-GPU usage remained 15,701 MiB in both
batch-6 runs, including background usage and retained BFC regions. No reduction
in peak live tensor allocation is claimed.

```sh
uv run --group qwen-bench python scripts/benchmark_qwen_image_2_1_rxla.py \
  --batches 6 --steps 40 --pad-vae-tail \
  --output-dir .cache/qwen-image-2.1/tail-reference
uv run --group qwen-bench python scripts/benchmark_qwen_image_2_1_rxla.py \
  --batches 6 --steps 40 \
  --output-dir .cache/qwen-image-2.1/tail-exact \
  --reference-dir .cache/qwen-image-2.1/tail-reference
```

## Projection fusion: no demonstrated throughput gain

Opt-in `QwenImage21Config::fuse_qkv` / `fuse_mlp` concatenate bias-free
projection weights, perform one contraction, and split its output. W8 weights
and scales concatenate before dequantization. Original parameter paths, shapes
and storage types are retained, so the same checkpoint loads without repacking.
These are explicit model-construction experiments, not an automatic IR pass.

### Batch 1, forty steps

| Projections | Request median | DiT median | Images/s |
|---|---:|---:|---:|
| Original reference | 6.961 s | 6.740 s | 0.1437 |
| QKV combined | 6.956 s | 6.734 s | 0.1438 |
| MLP combined | 7.139 s | 6.915 s | 0.1401 |
| Both combined | 7.032 s | 6.810 s | 0.1422 |

All three batch-1 final latents are bit-exact against the reference. QKV's
0.07% difference is not a reliable gain; MLP fusion is slower in this trial.
Artifacts: `fusion-qkv-40steps/`, `fusion-mlp-40steps/`, `fusion-all-40steps/`.

### Batch 4 screening, four steps

| Projections | Request median | DiT median | Images/s |
|---|---:|---:|---:|
| Original | 3.261 s | 2.401 s | 1.2266 |
| QKV combined | 3.284 s | 2.423 s | 1.2180 |
| MLP combined | 3.273 s | 2.410 s | 1.2221 |
| Both combined | 3.266 s | 2.409 s | 1.2247 |

This short schedule screens kernel performance, not image quality or the
forty-step workload. No path improves throughput. MLP-only final latent is
bit-exact. QKV/both differ: maximum absolute error 0.091796875, mean absolute
error 0.00161492 and RMS error 0.00456221. Mathematical projection fusion is
not a guarantee of bitwise equivalence under different BF16 GPU kernels.
Artifacts: `fusion-screen-{baseline,qkv,mlp,all}/`.

Both fusion options remain **off by default**. Tiny CPU tests check separate
and combined variants with F32/BF16 and dense/W8 storage, plus unchanged schema
and lowered contraction counts. No speed benefit is inferred from dot counts.

### What optimized HLO shows

The original batch-1 program already combines Q/K into a projection of width
8,192. Its optimized `gemm_fusion_dot.885_computation` concatenates the weights
for ABI arguments 19/21 (block-0 Q/K) and performs the shared contraction.
XLA therefore already supplies part of the proposed horizontal fusion.

| Batch-1 module | Original | Both combined |
|---|---:|---:|
| Pre-optimization HLO dot instructions | 296 | 200 |
| Optimized HLO dot instructions, including fusion bodies | 231 | 167 |
| Static buffer-assignment total, including inputs | 7,783,073,322 B | 7,876,339,242 B |

The fused module reduces dot instructions but adds about **88.9 MiB** to its
static allocation plan. This plan excludes the rest of the pipeline and is
neither a kernel-count measurement nor sampled peak live VRAM. These structural
facts do not isolate the cause of its timing; kernel-level profiling is still
needed before attributing the slowdown to a particular allocation or GEMM.

Full module dumps, excluding autotuning submodules, are in:

```text
.cache/qwen-image-2.1/fusion-hlo/{baseline,all}/
  module_0000.main.before_optimizations.txt
  module_0000.main.sm_12.0a_gpu_after_optimizations.txt
  module_0000.main.sm_12.0a_gpu_after_optimizations-memory-usage-report.txt
```

Dumps use `XLA_FLAGS="--xla_dump_to=/absolute/directory --xla_dump_hlo_as_text"`
and two steps to compile the same per-step program. Autotuning can emit large
scratch-allocation warnings for rejected candidates; those are not the final
program's memory requirement. Do not use these cold diagnostic runs as the
steady-state timing samples above.

## Reproduction and next target

```sh
cargo build --release -p rxla-safetensors --features model,disk-cache \
  --example qwen_image_2_1_generate
uv run --group qwen-bench python scripts/benchmark_qwen_image_2_1_rxla.py \
  --batches 2,4,6 --steps 40
# Add --fuse-qkv and/or --fuse-mlp for the experimental variants.
```

Current practical choices are small batches for latency, larger batches for
queued throughput, and exact VAE tails. The next substantial compute target
remains DiT. Request-invariant text-prefix computation/KV preparation is now
implemented as a separate opt-in staging experiment; see the follow-up above.
