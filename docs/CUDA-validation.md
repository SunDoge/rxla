# CUDA PJRT validation (2026-09-13)

Direct Rust HLO protobuf → PJRT execution was verified on an NVIDIA GeForce
RTX 5080 (compute capability 12.0a), driver 615.71.09. No Python interpreter,
MLIR library, binding regeneration or XLA source compilation was used.
This is a correctness smoke, not a performance benchmark or full GPU test suite.

## Pinned native artifacts

The CUDA plugin matches the CPU plugin release used by the development gate.
Archives were SHA256-verified before extraction into a task-local directory.

| Artifact | SHA256 |
| --- | --- |
| [ZML CUDA PJRT 202609101243.20.1.7ca6884ea2cb](https://mirror.zml.ai/plugins/202609101243.20.1.7ca6884ea2cb/zml-cuda-linux-amd64.tar.zst) | `2ef0e28330d22ce3039adb1de0d5dabf8bd7126988f9b61ba427946d6c60cb93` |
| [NVRTC 13.3.33](https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-13.3.33-archive.tar.xz) | `9e8f78278215babd1236b137252424ca7912c185bd093201f5d97f7dd763b74a` |
| [cuDNN 9.24.0.43 CUDA 13](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.24.0.43_cuda13-archive.tar.xz) | `63f1900222c69ee7e94583408181ccdb988dc2833531ce6bde0df43bbdd04a6d` |

Provenance: [ZML CUDA dependencies](https://github.com/zml/zml/blob/master/platforms/cuda/cuda.bzl),
[NVIDIA CUDA manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.3.0.json),
[NVIDIA cuDNN manifest](https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.24.0.json).

The machine's CUDA toolkit is `/usr/local/cuda-13.3` (including `ptxas`). Other
NVIDIA shared libraries were reused from an existing Python environment's
`nvidia/cu13/lib` and `nvidia/nccl/lib` directories; this does not invoke Python.
This is a mixed dependency environment, not a self-contained distribution.
The plugin reported runtime 13.2.0, toolkit 13.3.0 and DNN 9.24.0.
Installed cuDNN 9.20 and NVRTC 13.2 alone failed at dynamic loading: missing
`libcudnn_engines_tensor_ir.so.9` and `libnvrtc-builtins.so.13.3`.

## Repeating the selected checks

A repeatable gate is now available as `sh scripts/check-cuda.sh --offline`
from the standalone repository root, after setting the native environment below.
It requires a real trusted plugin path, verifies CUDA via `cuda_clients`, then
runs a small serialized selection of PJRT transfer and core tensor tests.
It does not download dependencies, run full models or claim complete CUDA
coverage.
Any test failure stops the script; no numerical tolerance is relaxed for CUDA.
On the pinned RTX 5080 environment the complete gate passed: 20 selected tests
(12 native GPU tests, eight host tests), two-live-client example, parameter-group
training and default CNN training. Ordinary/transposed convolutions and mixed
derivatives through fourth order passed their existing references. Shell syntax,
missing-plugin and unknown-argument rejection were checked; passing the CPU
plugin fails in CUDA client creation instead of reporting GPU success.

From the standalone repository root, adapt the paths below to trusted extracted
artifacts. These temporary paths record the tested environment, not permanent
installation locations. Do not add CUDA stub-library directories to the search path.

```sh
export PJRT_PLUGIN_PATH=/tmp/xla-cuda-pjrt.UA6WlN/libzml_cuda.so
export LD_LIBRARY_PATH=/tmp/xla-cuda-pjrt.UA6WlN/cudnn-linux-x86_64-9.24.0.43_cuda13-archive/lib:/tmp/xla-cuda-pjrt.UA6WlN/cuda_nvrtc-linux-x86_64-13.3.33-archive/lib:/home/me/venvs/py312/lib/python3.12/site-packages/nvidia/cu13/lib:/home/me/venvs/py312/lib/python3.12/site-packages/nvidia/nccl/lib:/usr/local/cuda-13.3/lib64
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-13.3
ldd "$PJRT_PLUGIN_PATH"
cargo run --offline -p rxla-core --example matmul
cargo test --offline -p rxla-core --test threefry --test training_acceptance --test trainable_embedding -- --ignored --test-threads=1
cargo run --offline -p rxla-train --example train_parameter_groups
cargo run --offline -p rxla-train --example train_cnn
```

## Observed results

- `matmul`: client metadata reports `cuda`, selected device RTX 5080; output
  `[23.0, 29.0, 50.0, 65.0]` passes the example assertion.
- `threefry`: three native tests pass, covering known-answer bits, counter carry,
  wrap, conditional updates and resume.
- `trainable_embedding`: tied lookup/projection gradients and updates pass the
  independent reference checks.
- `training_acceptance`: shared microbatch acceptance across RNG, BatchNorm,
  accumulation and optimizer passes, including unchanged state on rejection.
- `train_parameter_groups`: 200 microbatches, 100 joint updates, two rejected
  retries, 6/6 predictions and two compilations. Final microbatch loss
  `0.0022984482`, F64 reference `0.0022984856040791555`; per-state checks pass.
- Default `train_cnn`: seed 42, max pooling; loss `0.68227524 → 0.0015132788`,
  8/8 predictions, one compilation. This invocation does not test the complete
  seed/pooling/accumulation matrix.

No numerical tolerances were changed for these runs. The separate CPU
`scripts/check.sh --offline --with-plugin` gate also passes, including the
new owned metadata test, all-target Clippy and independent downstream execution.

## Remaining limits

The plugin's default BFC allocator reserves about 11.67 GiB at client creation.
Run the default-loading GPU processes sequentially. The full CPU gate includes
multiple-client and child-process tests and is not yet a safe GPU gate: it does
not opt into the explicit allocator configuration described below.

The plugin logs an NVML `Not Supported` diagnostic at startup but these selected
operations complete successfully. The diagnostic's underlying cause has not
been established. These small checks do not establish CUDA TinyLlama/OCR correctness,
throughput, memory efficiency, quantization, or parity with JAX/llama.cpp.
The subsequent full TinyLlama correctness run is recorded separately in the
[model report](../benchmarks/rust-xla-tinyllama-results.md#cuda-complete-model-validation).
The complete OCR graph also has a [CUDA follow-up](../benchmarks/rust-xla-vision-status.md#cuda-follow-up-2026-09-13):
synthetic cases pass with HIGHEST convolution precision, while real-image cases
still fail the unchanged strict ORT agreement gate.

## Explicit client allocation options

`Client::load_with_options` accepts a typed `ClientOptions` builder for all
five PJRT option types (string, int64, int64 list, float, bool). Names and types
are plugin-specific: this is not a portable allocator schema. Empty options
retain defaults. Repeated `set` calls replace the prior value; empty names fail
before loading native code. Other validation belongs to the plugin. No JAX
environment variables are translated.

With the CUDA environment above:

```sh
cargo run --offline -p rxla-core --example cuda_clients
```

This example passes `allocator="bfc"`, `preallocate=false`, and
`memory_fraction=0.25` directly to client creation. Two clients remain alive
together and perform six successful sequential matrix multiplications. The
plugin reports a growth limit of 3.89 GiB per ordinary BFC allocator instead of
preallocating 11.67 GiB. This is a reported pool limit, not a measured process
memory peak or a hard cap on all GPU allocations: the plugin separately reports
a collective pool limit of 12,530,483,200 bytes. No collectives were exercised.
Disabling preallocation does not imply freeing cached BFC allocations after
every operation. Parallel kernels and allocator performance remain unmeasured.
`Client::load` continues to use unmodified plugin defaults.
