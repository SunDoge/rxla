# Product vision and roadmap

RXLA's primary product target is a distributed image-generation inference
stack for Stable Diffusion and Flux. CUDA is the first-class platform. TPU is a
second-layer backend that should remain possible through portable IR, but it
must not delay a complete and competitive CUDA path.

The intended product serves image-generation requests across multiple GPUs and
multiple hosts. It also trains a LoRA quickly from user-provided images and
applies one or more adapters without maintaining a separate model definition or
execution stack.

This document records direction, not implemented capability. Current evidence
and limitations remain in [SCOPE.md](SCOPE.md) and the benchmark documents.

## Product loop

The target is one coherent workflow:

```text
user images
  -> preprocessing and latent encoding
  -> LoRA training
  -> named adapter artifact
  -> load, combine, or cache adapters per request
  -> distributed SD/Flux inference
  -> VAE decode
  -> generated images
```

Training and inference should share the Tensor API, model definition, parameter
identities, Pliron IR, PJRT runtime, kernel backends, weights, and compilation
cache. RXLA should not grow unrelated training and serving representations.

## Programming and compiler model

The public programming model remains MLX-like lazy tensors. Ordinary tensor
expressions build typed SSA-backed IR; materialization is explicit at `eval`
boundaries. Users should not have to construct compiler graphs directly.

Parameters are scoped effects declared where their dependencies are visible.
A dimension inferable from an available tensor should not be duplicated in
module configuration. Different effect interpretations create parameters,
look them up for inference, or expose only selected trainable parameters to
autodiff and an optimizer.

Pliron is the semantic and transformation layer. It must retain operations such
as attention, LoRA application, collectives, structured control flow, and state
long enough to optimize them. StableHLO is a portable backend boundary rather
than the only level at which RXLA can understand a model.

The same model program should support separate transformation pipelines:

```text
model program
  + inference transforms
      -> constant folding, fusion, attention lowering, donation and placement
  + training transforms
      -> selected autodiff, rematerialization, optimizer fusion and collectives
```

Host-driven and IR-resident denoising loops should both be possible. The former
supports development and dynamic policies; the latter can remove repeated host
dispatch and synchronization when the schedule is compilable.

The long-term scripting boundary is a declarative training and serving plan,
not a per-operator foreign-function binding. Thin TypeScript/Deno and Python
frontends may eventually compose model code, data sources, augmentation,
optimizer steps, and scheduling policy into RXLA IR, while compiled Rust
components own execution. The Rust runtime remains free to run input work,
transfers, compilation, and devices in parallel; a scripting-language runtime
must never become the per-batch executor.

Data loading belongs to the overall program IR, but not directly to StableHLO.
File reads, decoding, shuffling, prefetch, and transfer are typed host effects
in an upper scheduling/data dialect. Pure tensor augmentation and model regions
can then lower to StableHLO for CPU or accelerator execution. This separation
preserves reproducible RNG and transformation opportunities without pretending
that backend tensor IR is an I/O runtime. This boundary also makes the system
agent-friendly: an agent can construct, inspect, transform, and launch a typed
plan without generating resource-management or concurrency code. Scripting
frontends are intentionally deferred until the Rust IR and execution semantics
are stable.

## Backend priorities

### Layer 1: CUDA

CUDA receives the complete performance path first:

- StableHLO/XLA supplies broad operator coverage and general fusion.
- cuDNN and cuBLASLt supply mature library implementations where appropriate.
- Custom kernels cover measured bottlenecks instead of duplicating the whole
  backend. Tile-oriented Rust kernels are preferred for attention, normalization,
  GEMM epilogues and fused MLPs; lower-level kernels may cover KV-cache updates,
  sampling, quantization and irregular indexing.
- PJRT FFI/custom calls must consume PJRT-owned buffers and the execution stream
  without a second CUDA context, hidden copies, or implicit synchronization.
- NCCL or backend collectives provide single-host and multi-host communication.
- CUDA Graphs, buffer donation, asynchronous transfers and executable caching
  are applied only with profiling evidence.
- BF16 and FP16 are baseline formats. FP8 and quantized weights follow workload
  correctness and performance measurements rather than API speculation.

Every custom CUDA lowering needs either a portable StableHLO fallback or an
explicit capability error. CUDA-specific details must not leak into the public
Tensor or model API.

### Layer 2: TPU

TPU is the second backend. Initially, preserving a credible path matters more
than feature or performance parity:

- keep portable operations expressible as valid StableHLO;
- represent device meshes, sharding and collectives independently of CUDA;
- preserve a portable fallback for custom CUDA operations;
- keep weights and public APIs independent of CUDA memory layouts;
- validate TPU economics with complete workloads before promising cost savings.

CUDA product milestones take precedence. TPU implementation begins after the
single-GPU inference and LoRA loops are complete enough to benchmark honestly.

## LoRA as an IR-level capability

LoRA is not merely an offline SafeTensors edit. RXLA should understand adapter
semantics so that a linear operation can be rewritten from

```text
y = x W
```

to

```text
y = x W + scale * (x A) B
```

and then choose among a live adapter branch, merged weights, a cached merge, or
a backend-specific fused lowering. The serving system should eventually support
weighted adapter composition and different adapters within a batch when that is
profitable.

Parameter state needs at least these semantic classes:

- **frozen**: base-model parameters excluded from gradients;
- **trainable**: selected LoRA parameters differentiated and updated;
- **optimizer state**: moments, counters and loss-scaling state;
- **constant**: immutable compile-time or device-resident values.

The first training target is deliberately narrow: LoRA parameters only,
reverse-mode differentiation only for their dependencies, Adam/AdamW, gradient
accumulation, mixed precision and rematerialization. A general-purpose trainer
is not a prerequisite.

## Distributed serving direction

Distributed execution is workload-driven rather than a checkbox. Candidate
strategies include data parallel image requests, attention-head or tensor
parallelism, sequence/context parallelism, pipeline placement, parallel
classifier-free-guidance branches, and separately scheduled VAE decode. The
compiler and scheduler should select among them using topology, memory and
request-shape information.

The longer-term serving advantage should come from the complete system:

- keep base weights resident while adapters move independently;
- tier LoRA artifacts across GPU memory, host memory and NVMe;
- prefetch adapters asynchronously and cache hot merged forms;
- batch requests with different resolutions, step counts and adapters;
- schedule denoising, text encoding and VAE stages across devices;
- cache executables by semantic program, shapes, dtypes, backend, architecture
  and lowering policy;
- balance throughput, tail latency and memory pressure under an explicit SLO.

## Milestones

### 1. Single-GPU inference loop

Run a real SD or Flux checkpoint end to end on CUDA and produce a normal image.
Establish pinned correctness, image-quality, warm-latency, throughput,
compilation-time and peak-memory comparisons against a PyTorch Diffusers
baseline. Keep the complete denoising path in the Pliron pipeline without a
legacy graph fallback.

### 2. LoRA loop

Train a useful adapter from user images, save it, restore it, and apply it during
inference using the same model definition. Verify that frozen base parameters do
not acquire gradients or optimizer state. Measure training time, peak memory and
adapter-switch overhead.

### 3. CUDA performance

Profile the end-to-end workload before replacing kernels. Prioritize attention,
normalization, quantized matrix multiplication and repeated denoising dispatch.
Accept custom kernel and CUDA Graph paths only when timelines prove that they do
not introduce copies, extra contexts or synchronization and they improve a
declared workload metric.

### 4. Multi-GPU and multi-host serving

Start with single-host multi-GPU execution, then add multi-host collectives and
topology-aware planning. Validate generated outputs as well as throughput,
latency distribution, communication overlap, failure behavior and peak memory.

### 5. TPU validation

Run the same portable model and representative serving workload through a TPU
PJRT backend. Compare complete cost per generated image under target latency and
quality constraints; theoretical accelerator FLOPS alone do not establish the
backend's value.

## Decision rule and non-goals

A proposed feature is high priority when it measurably improves SD/Flux image
generation, LoRA training or application, distributed serving, memory use, or
the usability of those workflows. Compatibility features that unlock existing
models and weights also qualify. Other general tensor-framework work can wait.

The near-term project is not trying to provide every PyTorch/JAX API, train every
model family, make all PJRT backends equally fast, replace every vendor library,
or claim production readiness from synthetic kernels. Text and vision examples
remain useful compiler tests, but they do not define the product roadmap.
