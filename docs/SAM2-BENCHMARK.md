# SAM 2.1 Hiera-small benchmark

These are local single-GPU measurements of the image trunk only. They are not
end-to-end SAM 2 prompt encoding or mask decoding results, and should not be
treated as general backend performance guarantees.

## Environment

- NVIDIA GeForce RTX 5080
- CUDA 13 PJRT plugin from the locked JAX environment
- cuDNN 9.24
- input: one deterministic nonzero `1024 x 1024` RGB image
- checkpoint: official SAM 2.1 Hiera-small weights
- two warmups and ten timed executions

## Results

| Implementation | Precision | Compile | Mean | P50 | P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| RXLA, original | F32 | 33.825 s | 30.873 ms | 31.046 ms | 31.216 ms |
| RXLA, compute demotion only | F16 | 15.626 s | 10.690 ms | 10.761 ms | 11.005 ms |
| RXLA, native F16 parameter ABI | F16 | 14.854 s | 8.038 ms | 8.096 ms | 8.424 ms |
| RXLA, F32 reductions + direct launches + GEMM fusion v2 | F16 | 14.555 s | 7.412 ms | 7.417 ms | 7.688 ms |
| RXLA, patterned input recheck (20 runs) | F16 | 15.181 s | 7.509 ms | 7.494 ms | 7.806 ms |
| PyTorch eager | F32 | n/a | 24.266 ms | 24.251 ms | 24.456 ms |
| `torch.compile` default | F32 | 10.758 s | 22.723 ms | 22.707 ms | 22.838 ms |
| `torch.compile` default | F16 autocast | 12.569 s | 6.039 ms | 5.972 ms | 6.234 ms |
| `torch.compile` patterned input recheck (20 runs) | F16 autocast | 12.665 s | 6.000 ms | 5.979 ms | 6.097 ms |

The native F16 parameter ABI is important: converting the F32 checkpoint once
while loading removes roughly 176 MiB of parameter casts from every execution.
It improves the F16 path by 24.8% and the complete RXLA optimization improves
latency by 3.84x over the original F32 run. RXLA remains 33.1% slower than the
equivalent non-autotuned PyTorch F16 result, so this is progress rather than a
parity claim.

The later target-aware configuration narrows that gap substantially. In the
current same-input 20-run recheck RXLA is 25.2% slower. It is supplied
through PJRT `CompileOptionsProto.env_option_overrides`, not process-global
`XLA_FLAGS`: `--direct-launches` disables command buffers and
`--gemm-fusion-v2` opts into the experimental XLA pass. The latter is kept
explicit because the loaded PJRT plugin owns its compatibility and semantics.

## Numerical alignment

A deterministic nonzero NHWC input is now shared by both implementations.
RXLA can export the four stages or all 16 post-block debug taps, and the Python
oracle reports max/mean absolute error, mean relative error, and cosine.

F32 agrees with the official PyTorch implementation through every stage:

| Stage | Max absolute error | Mean absolute error | Cosine |
| --- | ---: | ---: | ---: |
| 0 | 3.81e-6 | 2.01e-7 | 1.000000000000 |
| 1 | 3.52e-6 | 3.65e-7 | 0.999999999999 |
| 2 | 2.81e-5 | 1.39e-6 | 0.999999999994 |
| 3 | 2.49e-5 | 7.63e-7 | 0.999999999991 |

RXLA F16 with F32 reduction accumulation remains close to RXLA F32, with final
stage mean absolute error `7.13e-4` and cosine `0.999991823878`. PyTorch F16
autocast is closer to F32 (`4.41e-4`, `0.999996036337`) because it uses an
operator-specific mixed-precision policy rather than demoting the complete
graph. Final mask-level validation and an IR mixed-precision policy remain open.

`Cx::tap` retains a named SSA intermediate without making it a normal model
output. `AppliedModel::tap`, `outputs_with_taps`, `prepare_tensors`, and
`compile_tensors` can then materialize only the requested debug roots. Hiera
registers `image_encoder.trunk.blocks.N.output` for block-level bisection.

F16 produced a checksum of `-1287909.161656380`, versus
`-1287919.772190087` for RXLA F32. This aggregate is a useful smoke signal, not
an accuracy proof; mask-level validation is still required.

## Reproduction

RXLA defaults to F16 for this CUDA benchmark. F32 and BF16 remain explicit:

```console
cargo run -p rxla-safetensors --release --features model \
  --example sam2_hiera_bench -- \
  .venv/lib/python3.14/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so \
  /home/me/.cache/rxla/models/sam2.1_hiera_small.safetensors \
  --precision f16 --direct-launches --gemm-fusion-v2 \
  --warmup 2 --iterations 10
```

The PyTorch baseline deliberately uses ordinary `torch.compile`; kernel search
is available only through the explicit `--mode max-autotune` option:

```console
uv run python benchmarks/sam2_hiera_torch.py \
  /home/me/.cache/rxla/models/sam2.1_hiera_small.pt \
  /path/to/facebookresearch/sam2 \
  --mode compile --precision f16 --warmup 2 --iterations 10
```

## Kernel attribution

Nsight Systems 2026.5.1 successfully captured both implementations. Compile
and autotuning activity was excluded: RXLA rows start after the last XLA
redzone kernel, while the PyTorch capture consists of the initial execution,
five warmups and 50 timed executions. Profiler wall times are inflated, so the
useful comparison is GPU kernel time per execution.

| Implementation | GPU kernel time | Kernels/run | GEMM | Fused attention/softmax | Layout/other |
| --- | ---: | ---: | ---: | ---: | ---: |
| RXLA | 7.181 ms | 303 | 4.842 ms | 1.395 ms | 0.944 ms |
| `torch.compile` | 5.699 ms | 289 | 2.941 ms | 1.480 ms | 1.278 ms |

RXLA's standalone Triton softmax kernels account for only 0.191 ms/run. The
main gap is that PyTorch FlashAttention covers all 16 Hiera blocks, whereas
RXLA currently sends only three long-sequence blocks to cuDNN FMHA. The 13
window-attention blocks lower to two GEMMs plus softmax, explaining the extra
26 GEMM launches and most of the 1.48 ms kernel-time gap. Lowering the cuDNN
threshold to include 196-token windows was measured and rejected because it
increased end-to-end latency; a suitable short-attention fusion/kernel is the
next optimization target.
