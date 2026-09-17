# SAM 2.1 Hiera-small benchmark

These are local single-GPU measurements of the image trunk only. They are not
end-to-end SAM 2 prompt encoding or mask decoding results, and should not be
treated as general backend performance guarantees.

## Environment

- NVIDIA GeForce RTX 5080
- CUDA 13 PJRT plugin from the locked JAX environment
- cuDNN 9.24
- input: one zero-filled `1024 x 1024` RGB image
- checkpoint: official SAM 2.1 Hiera-small weights
- two warmups and ten timed executions

## Results

| Implementation | Precision | Compile | Mean | P50 | P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| RXLA, original | F32 | 33.825 s | 30.873 ms | 31.046 ms | 31.216 ms |
| RXLA, compute demotion only | F16 | 15.626 s | 10.690 ms | 10.761 ms | 11.005 ms |
| RXLA, native F16 parameter ABI | F16 | 14.854 s | 8.038 ms | 8.096 ms | 8.424 ms |
| PyTorch eager | F32 | n/a | 24.266 ms | 24.251 ms | 24.456 ms |
| `torch.compile` default | F32 | 10.758 s | 22.723 ms | 22.707 ms | 22.838 ms |
| `torch.compile` default | F16 autocast | 12.569 s | 6.039 ms | 5.972 ms | 6.234 ms |

The native F16 parameter ABI is important: converting the F32 checkpoint once
while loading removes roughly 176 MiB of parameter casts from every execution.
It improves the F16 path by 24.8% and the complete RXLA optimization improves
latency by 3.84x over the original F32 run. RXLA remains 33.1% slower than the
equivalent non-autotuned PyTorch F16 result, so this is progress rather than a
parity claim.

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
  --precision f16 --warmup 2 --iterations 10
```

The PyTorch baseline deliberately uses ordinary `torch.compile`; kernel search
is available only through the explicit `--mode max-autotune` option:

```console
uv run python benchmarks/sam2_hiera_torch.py \
  /home/me/.cache/rxla/models/sam2.1_hiera_small.pt \
  /path/to/facebookresearch/sam2 \
  --mode compile --precision f16 --warmup 2 --iterations 10
```

An Nsight Systems capture was attempted with 2026.3.2, but it contained target
metadata and no CUDA activity tables. No kernel attribution is claimed from
that capture; the remaining gap needs a working CUPTI trace or XLA GPU profile.
