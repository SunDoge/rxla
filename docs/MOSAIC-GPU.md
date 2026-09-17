# Mosaic GPU kernel artifacts

RXLA can execute StableHLO containing JAX Mosaic GPU kernels without Python or
JAX in the serving process. This is an experimental kernel-authoring path, not
a runtime dependency or a second model frontend.

## Deployment shape

Kernel development uses the locked `uv` environment to turn Pallas/Mosaic GPU
code into a StableHLO `mosaic_gpu_v2` custom call. Production needs only:

- the generated StableHLO or RXLA executable artifact;
- the CUDA PJRT plugin;
- `libmosaic_gpu_runtime.so`, loaded globally before compilation or executable
  restoration; and
- the CUDA driver and ordinary system C/C++ runtime libraries.

The runtime shipped by the local JAX CUDA 13 wheel is 424 KiB and exports the
two host symbols used by the generated kernel: `mosaic_gpu_launch_kernel` and
`mosaic_gpu_init_tma_desc`. The validation process did not import Python or
JAX. `NativeSupportLibrary::load_global` intentionally retains the mapping for
the process lifetime because compiled executables keep references to its host
symbols.

Generate and validate the probe artifact:

```console
uv run python benchmarks/mosaic_gpu_add.py /tmp/rxla-mosaic-add.mlir
cargo run -p rxla-core --release --example mosaic_gpu -- \
  .venv/lib/python3.14/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so \
  .venv/lib/python3.14/site-packages/jax_cuda13_plugin/libmosaic_gpu_runtime.so \
  /tmp/rxla-mosaic-add.mlir
```

The RTX 5080/CUDA 13 validation compiled the artifact through RXLA PJRT,
executed it, downloaded the result and checked every element.

## Compatibility and security

Mosaic embeds versioned MLIR bytecode in StableHLO. Treat the generator JAX
version, PJRT plugin, Mosaic runtime, target compute capability and CUDA stack
as one artifact compatibility tuple. A production manifest should fingerprint
all of them, and cache namespaces must change when any member changes.

StableHLO and native support libraries are executable input, not untrusted
model data. Only load artifacts and libraries produced by a trusted build
pipeline. Failure to load the Mosaic runtime is a deployment error; RXLA must
not silently fall back to a slower or numerically different kernel.

## Current performance decision

The SAM 2 Hiera trunk remains approximately 1.5 ms behind ordinary
`torch.compile` F16 on the local RTX 5080. Profiling attributes the difference
primarily to 13 short window-attention blocks: PyTorch uses its own Flash
kernel, while XLA emits GEMM/softmax/GEMM. cuDNN FMHA threshold and BTNH-layout
experiments both regressed latency, so they were rejected.

Mosaic is viable for a future specialized short-attention kernel, but the
existing example assumes batch one and aligned sequence/head dimensions and is
not a drop-in implementation for Hiera's window shapes. A custom kernel should
be justified by end-to-end serving measurements, artifact portability and
maintenance cost—not by winning one isolated benchmark.

## FlashAttention architecture probe

The upstream JAX implementations were tested unchanged on the local RTX 5080
(`sm_120a`) with F16 Q/K/V shaped `[1, 256, 2, 64]` and 64-element query and
key/value blocks:

| Upstream implementation | Intended architecture | RTX 5080 result |
| --- | --- | --- |
| `mosaic/gpu/examples/flash_attention.py` | Hopper `sm90a` | compilation rejected `nvvm.wgmma.fence.aligned` on `sm_120a` |
| `pallas/ops/gpu/attention_mgpu.py` | Hopper `sm90a` | same WGMMA rejection |
| Tokamax 0.0.14 `implementation="mosaic"` | `sm90` and datacenter Blackwell `sm100` | explicitly rejected `sm120` |

This is an upstream kernel architecture gap, not an RXLA custom-call gap. A
minimal Mosaic elementwise kernel already compiles and executes on `sm_120a`,
but JAX's Hopper attention uses WGMMA instructions that the Mosaic compiler
does not legalize for `sm_120a`. Tokamax has a newer TCGEN05 implementation for
B200-class `sm100`, but its dispatcher deliberately accepts only `sm90` and
`sm100`.

JAX's older Pallas GPU attention API is deprecated in favor of OpenXLA
Tokamax. Do not fork the Hopper example as RXLA's long-term kernel. Recheck
Tokamax first when it adds `sm120`; until then, an RTX 5080 kernel requires an
`sm120` implementation (or another backend such as cuTile/Triton). Tokamax was
installed temporarily for this probe and then removed because it added over 60
Python packages while providing no usable Mosaic implementation on this GPU.

The intended project boundary remains:

- Python/JAX/Mosaic for kernel development, validation, tuning and offline
  artifact generation;
- Rust for RXLA IR, compilation caches, PJRT execution, scheduling and the
  production serving process.

Python-generated kernels must cross the same typed semantic-op and artifact
manifest boundary as native kernels. Python is therefore a build frontend, not
a requirement in the serving hot path.
