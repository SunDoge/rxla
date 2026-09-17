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
  --precision f16 --cudnn-attention --direct-launches --gemm-fusion-v2 \
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

### Isolated JAX attention backends

JAX 0.11.1's public `jax.nn.dot_product_attention` was measured on the same RTX
5080 using device-side timing. Unlike the upstream Mosaic kernels, its cuDNN
implementation supports every relevant `sm120` Hiera shape and lowers to
`__cudnn$fmhaSoftmax`.

| B, T, H, D | JAX cuDNN | JAX XLA | PyTorch Flash SDPA |
| --- | ---: | ---: | ---: |
| 1, 256, 2, 64 | 0.0079 ms | **0.0053 ms** | 0.0159 ms |
| 25, 196, 4, 96 | 0.0433 ms | 0.0539 ms | **0.0395 ms** |
| 1024, 64, 1, 96 | **0.0293 ms** | 0.0688 ms | 0.0444 ms |
| 1024, 16, 2, 96 | 0.0475 ms | **0.0283 ms** | 0.0731 ms |
| 25, 49, 8, 96 | **0.0077 ms** | 0.0234 ms | 0.0179 ms |

Run the two frameworks separately so their allocators and CUDA contexts do not
interfere:

```console
uv run python benchmarks/attention_backends.py jax
uv run python benchmarks/attention_backends.py torch
```

The isolated numbers show that a new FlashAttention kernel is not automatically
the best solution for short windows: XLA wins at T=16 and T=256, while cuDNN is
already very competitive elsewhere. They do not overturn the end-to-end Hiera
result above. Layout conversions, fusion boundaries and 13 repeated call sites
can make an individually fast cuDNN kernel regress the whole model. The next
optimization should therefore make attention lowering shape-aware and measure
the complete trunk, rather than forcing one backend for every attention op.

RXLA's cuDNN emitter now reproduces JAX's logical FMHA ABI instead of only
copying its backend configuration. RXLA attention remains `[B,H,T,D]` in the
semantic IR; lowering transposes Q/K/V to JAX's `[B,T,H,D]`, uses the same dot
dimension numbers and requests JAX's `[3,1,2,0]` result layout. XLA can fold
those logical transposes into layout assignment. With the established
long-sequence policy, the complete trunk measured 7.435 ms P50 versus the prior
7.494 ms recheck. The 0.8% difference is small and should not be presented as a
large speedup, but the custom-call ABI is no longer an RXLA-specific variant.

Compute precision no longer selects an attention backend as a side effect.
Without `--cudnn-attention`, F16 Hiera emits ordinary StableHLO and lets XLA
optimize it; that path measured 8.245 ms P50 because XLA does not recover FMHA
from the decomposed attention pattern. Explicit cuDNN measured 7.435 ms. The
flag therefore records a deliberate target policy without contaminating the
backend-independent attention IR or silently coupling it to F16 computation.

Forcing cuDNN onto the individually favorable 49/64/196-token shapes was also
tested and rejected: complete-trunk P50 regressed to 7.793 ms even though the
isolated kernels were faster. The resulting stages remained numerically close
to the previous F16 outputs (final-stage cosine 0.9999939), so this is an
end-to-end layout/fusion cost rather than a correctness failure. Backend choice
must remain an IR compilation decision informed by the surrounding graph.

### Complete JAX trunk control

`benchmarks/sam2_hiera_jax.py` independently implements the complete Hiera-small
trunk in JAX using the same safetensors weights, NHWC input, native F16
parameters and four stage outputs. Its position resize deliberately reproduces
RXLA/PyTorch bicubic semantics (half-pixel centers, edge padding and four taps
with `a=-0.75`) instead of using JAX's different generic bicubic convention.

| Implementation | Attention policy | Compile | P50 |
| --- | --- | ---: | ---: |
| RXLA | explicit long-sequence cuDNN | 14.14 s | 7.44–7.53 ms |
| JAX | long-sequence cuDNN, short XLA | 18.39 s | 7.710 ms |
| JAX | cuDNN for every attention | 15.32 s | 7.785 ms |
| JAX | XLA decomposition only | 18.87 s | 9.104 ms |

The JAX hybrid output agrees closely with the saved RXLA F16 control: stage 0
cosine is 0.99999966 and stage 3 cosine is 0.99999270. JAX therefore provides
no hidden frontend optimization that explains RXLA's gap to PyTorch; on this
model RXLA is slightly faster under the same broad attention policy. Both XLA
frontends remain around 1.5 ms behind `torch.compile`, consistent with the
Nsight attribution to short-window attention and GEMM/layout interactions.

Reproduce the JAX control with:

```console
uv run python benchmarks/sam2_hiera_jax.py \
  /home/me/.cache/rxla/models/sam2.1_hiera_small.safetensors \
  --attention hybrid --warmup 2 --iterations 10 \
  --compare-prefix /tmp/rxla-sam2-f16-acc32
```
