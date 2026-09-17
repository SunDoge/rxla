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
