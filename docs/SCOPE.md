# Current XLA tensor library scope

The active direction is a Rust tensor library built on **Pliron + StableHLO + PJRT**,
not completion of the historical IREE feature list. This page records the
current boundary and outstanding work; it is not a declaration of completion.
The detailed API guide is [README.md](README.md).

The product target and ordering of future work are recorded separately in the
[product vision and roadmap](VISION-ROADMAP.md). In short, RXLA prioritizes
CUDA-based Stable Diffusion and Flux inference, LoRA training/application and
multi-GPU/multi-host serving, while preserving TPU as the second backend layer.

## Accepted architecture requirement

The Pliron SSA IR must support inspection, transformation, planning and staged
execution, not only HLO generation. The target architecture uses the same
validated execution-plan boundary for single-device, manually partitioned and
eventually automatically planned execution. Semantic IR, physical placement,
state commits and backend executables remain distinct.

The primary future model API declares parameters as scoped tracing effects at
their tensor use sites, rather than requiring module constructors to duplicate
inferred dimensions in configuration. See [parameter-effect design](PARAMETER-EFFECT-DESIGN.md).

[Execution-plan design and acceptance gates](EXECUTION-PLAN-DESIGN.md) defines
the required information preservation, training/remote failure semantics,
compilation budgets and staged validation. This is an accepted design constraint,
not implemented distributed support: even the unified one-stage plan remains
pending. `Program` snapshots retain StableHLO plus planning facts derived from
Pliron; `LoweredProgram` is the format-tagged backend boundary rather than a
second computational graph. Existing tested capabilities below are unchanged.

## Architecture and evidence

The [Tensor/storage refactor](TENSOR-STORAGE-DESIGN.md) adds pointer-sized shared
descriptors, SmallVec shape metadata and managed U8/F16/F32/I32/BF16 host/native
input bindings. Index and Output are compatibility aliases for runtime-dtype
Tensor, not separate representations. Shape operations
preserve dtype, unsupported arithmetic is rejected, and autodiff remains F32-only.
Tensor is now thread-affine; prepared snapshots remain the cross-thread path.
CPU/CUDA managed-input and Tensor-first lazy evaluation paths pass. Device views,
device DLPack import/export and donation remain pending.

The development gate has an explicit `--quick` host-only mode for default-feature
library checks/tests/Clippy. Native unit tests stay ignored and plugin paths are
unset; it skips integration tests, examples, optional features and downstream
validation. This is an iteration aid, not a substitute for the full gates below.

```text
Rust Tensor / modules / stateful programs
  → private Pliron SSA → verified StableHLO
  → trusted dynamically loaded PJRT plugin → CPU or CUDA
```

| Layer | Present implementation | Evidence / boundary |
| --- | --- | --- |
| Rust tensor graph | Static shapes, F32 tensors and I32 indexing/state, frozen BF16 storage with explicit F32 conversion, NN compositions and differentiation | [tensor source](crates/rxla-core/src/lib.rs), CPU gate; GPU gate covers a selected subset, not every operator |
| Model composition | Scoped parameter effects with use-site shape inference; legacy trainable modules remain isolated | [effect model](../crates/rxla-nn/src/lib.rs), [legacy module](../crates/rxla-train/src/module.rs); no automatic capture of arbitrary Rust mutation |
| Stateful execution | Parameters, optimizer/BatchNorm/RNG/KV state, compiled programs and owning sessions | [state](crates/rxla-core/src/state.rs); pure state-in/state-out lowering, synchronous non-donating execution |
| Training | Autodiff, plain SGD/Momentum/Adam(W), explicit gradient/proposal APIs, accumulation, parameter groups, guarded state updates | [optimizers](crates/rxla-train/src/optim.rs), training examples; no distributed trainer or universal transform support |
| Detection geometry | Graph-native xyxy/cxcywh conversion, area, aligned/pairwise IoU and aligned GIoU with gradients; explicit host class-agnostic/class-aware NMS | [boxes](crates/rxla-core/src/boxes.rs), [host utilities](crates/rxla-core/src/vision.rs), F64/gradient tests and [resident box fitting](crates/rxla-train/examples/train_boxes.rs); no GPU NMS, general detector or detection dataset training |
| Compilation | Typed Pliron SSA lowers to StableHLO MLIR for PJRT; optimized backend HLO remains diagnostic-only | [Pliron IR](../crates/rxla-core/src/pliron_ir.rs), [runtime](../crates/rxla-pjrt/src/runtime.rs); no second frontend HLO-proto representation or arbitrary external-code compilation in the tensor API |
| Cache / artifacts | In-memory reuse, optional disk executable cache and serialization | [compiler](crates/rxla-core/src/compiler.rs), [disk cache](crates/rxla-core/src/disk_cache.rs); artifacts trusted, compatibility namespace operator-supplied |
| Native ownership | Thread-affine owned clients, buffers and executables; owned submit/wait handles retain in-flight resources; readiness polling and pending-output pipelines; explicit typed creation options | [PJRT runtime](crates/rxla-pjrt/src/runtime.rs), [submission tests](crates/rxla-core/tests/submit.rs); Drop waits, no cancellation/Future/unsafe Send/Sync or automatic cross-client sharing |
| Checkpoints | SafeTensors loading, mixed F32/BF16 module bindings/export, canonical-name loading, streaming no-overwrite module files and training-state snapshots | [weights](crates/rxla-safetensors/src/lib.rs); model layouts and state schemas remain explicit; BF16 storage is not trainable BF16 state |
| Development build | Pregenerated bindings/protos and a separate maintainer xtask | CPU gates isolate native source paths; Rust dependencies and a compatible plugin still need provision |

F32 matmul and convolution request HIGHEST operand precision. This avoids the
observed reduced-precision CUDA defaults in the tested models; it does not
promise bitwise equivalence or choose the fastest algorithm on every device.

## What has actually run

- [CPU development gate](../scripts/check.sh): default/cache/module tests,
  doctests, examples, Clippy and independent downstream execution when explicitly
  opted into the native plugin. Host-only mode leaves native tests ignored.
- [CUDA development gate](../scripts/check-cuda.sh): selected crates/rxla-core/gradient,
  random/state and training tests, explicit growth-mode clients and workers.
  [Native setup and limitations](CUDA-validation.md). It is not the full CPU suite.
- [Heterogeneous gate](../scripts/check-xla-heterogeneous.sh): opt-in dual-plugin
  CPU/CUDA staged transfers and bounded execution, plus host-copy tests on each
  backend. Requires both trusted plugins and existing CUDA dependencies; it is
  separate from single-plugin checks and makes no throughput claim.
- [CPU placement gate](../scripts/check-xla-cpu-placement.sh): opt-in default and
  disk-cache checks with two logical CPU devices in separate native processes.
  It fixes its child XLA flags, verifies selected-device compile/restore and
  wrong-placement rejection, then tests separate device cache entries and scoped
  trimming. This is not multi-GPU/replicated execution and is outside the default
  single-device development gate.
- [TinyLlama](../benchmarks/rust-xla-tinyllama-results.md): full 22-layer FP32 model,
  one fixed prompt, 21 greedy tokens matching Transformers on CPU and CUDA;
  first eight full-logit vectors checked. Cross-process CUDA executable cache
  reuse and opt-in optimized weight loading are tested. Frozen BF16 storage with
  F32 arithmetic also passes the reference; order-balanced RTX 5080 replay tests
  measured about 1.62–1.63x decode throughput versus F32 storage for that prompt.
  Allocator diagnostics show lower active/peak usage but unchanged default pool
  preallocation. These are scoped single-model results, not serving performance.
  A shared full-model builder also passes eight-token chunk prefill versus scalar
  execution for all prompt logits and final KV, sharing uploaded weight buffers.
  Resident prefill-to-decode handoff and greedy continuation also pass the
  fixed-prompt checks. Scalar tail handling avoids padding and extra tail-shape
  compilations. Additional full-model CUDA fixtures cover 5/41-token prompts,
  a 103-token prompt with a seven-token scalar tail, and a 57-token bilingual
  prompt. The latter two each match all 24 Transformers greedy tokens and the
  first eight full-logit vectors, with maximum errors below 1.6e-5. These remain
  a handful of fixed cases at capacity 128; warmed prefill throughput remains
  unverified. The prefill diagnostic's explicit generation budget also has
  one-token and 64-token CUDA/reference checks; reports distinguish EOS from
  length truncation. Configurable capacity defaults to 128, with a separate
  capacity-256 run matching all 64 reference tokens after a 177-token prompt.
  A capacity-2048 repeated-text stress case also matches all 34 reference tokens
  after a 2014-token prompt, ending at position 2047 (the final prediction is not
  consumed). It is not a representative long-context or semantic-quality suite.
  See the linked results for per-case
  storage and precision details.
- [Qwen3.5-0.8B](QWEN3.5-BENCHMARK.md): the actual hybrid text topology runs
  from a 742 MB grouped-W8 checkpoint on CUDA, including gated-delta and full
  attention layers. A sequence-16 RTX 5080 benchmark and BF16 Transformers
  fallback comparison are recorded. It is currently static-shape text prefill,
  without persistent decode state, the vision tower, MTP, tokenizer or a
  language-quality evaluation.
- OCR: the legacy exported detector graph executes all 242 nodes. New
  [PP-OCRv6 CPU host-stage work](PP-OCRV6-CPU-BENCHMARK.md) establishes batched
  recognition preprocessing and CTC decoding, but is not yet an end-to-end
  PP-OCRv6 Medium result.
  Synthetic tensor comparisons pass. Real-image strict ORT agreement still
  fails; complete-map FP64 diagnosis gives lower aggregate errors for Rust CUDA
  on two images. No complete OCR accuracy or general ONNX importer claim.

No result above establishes llama.cpp/JAX speed parity, full YOLO support,
arbitrary model support, or a production-ready distributed RL framework.

The [CNN training example](crates/rxla-train/examples/train_cnn.rs) has an opt-in
`--residual` variant: spatial convolution, a two-layer 1x1 convolutional residual
branch, pooling and a classifier. Three seeds each train for 250 Momentum SGD
steps on eight synthetic horizontal/vertical-line images on CPU/CUDA, reusing one
compiled graph. Checks require correct labels, reduced loss and learned weights
in both residual convolutions. Separate linear residual tests compare every SGD
step with F64 formulas, including shared-weight gradient summation and one update
per parameter. Neither is a real-dataset vision benchmark or full ResNet claim.

The opt-in [heterogeneous example](crates/rxla-core/examples/heterogeneous.rs) also runs
CPU -> CUDA -> CPU subgraphs in one process, with explicit synchronous host
staging and three pipeline compilations reused over 17 requests each at retained-task
capacities one and three, with identical outputs. This establishes a small
cross-backend correctness path, not automatic graph partitioning, direct device
copies, overlapped execution or model-level speedup.
It now also compiles the same prepared model snapshot on CPU for per-request
comparison with CUDA, adding one reference executable. Repeated prepared lookups
retain separate client-local handles and compilation counters; this validates
cross-backend snapshot reuse, not executable portability or universal bit equality.

The explicit `Buffer::copy_to_client_via_host_with_limit` API rejects oversized
host payloads before allocation/download/upload; this is per-transfer accounting,
not an aggregate scheduler quota. A separate quiescent-session example copies
F32/I32 state CPU -> CUDA -> CPU and verifies continuation and source isolation.
`Session::copy_state_to_client_via_host` now copies every resident slot with a
per-tensor host payload limit, retaining schema identities but excluding fixed
input/weight bindings. Partial-copy failure leaves the source usable; successful
copies are independent even on the same client. This remains synchronous and
does not impose an aggregate device-memory quota.
`Session::copy_state_to_client_via_host_with_limits` additionally accepts a
total payload limit. It queries every resident buffer's native host-download
size before any payload copy, checks per-tensor limits and checked-sum total,
then copies sequentially. This bounds transfer volume, not peak memory or device
allocation; aliases count once per slot and fixed bindings remain excluded.
The real CPU/CUDA state example checks rejection at eleven total bytes and
successful copying at exactly twelve bytes in both directions, retaining source
state and verifying continued execution after rejection. The full heterogeneous
gate passes with these checks; this is correctness evidence, not a memory-usage
or transfer-performance benchmark.
The `rxla-safetensors` `worker_state` example reuses named SafeTensors state trees to
send only host checkpoint bytes across CPU -> CUDA -> CPU worker threads. It
joins each source before starting its destination, independently rebuilds graphs
with changed slot order, and checks nine F32/I32 updates. This is synchronous
checkpoint-based continuation with whole-checkpoint host storage, not live
migration or a new cross-thread capability for native PJRT handles.
Its named tree also preserves Threefry key/counter state: all random output
words over nine steps match independent uninterrupted CPU execution across
low-word carry, with restored counters checked after each worker boundary.
CPU-only and CPU/CUDA worker runs pass; this is not distributed RNG stream
assignment or a general deterministic-training guarantee.
The [placement benchmark](HETEROGENEOUS-BENCHMARK.md) measures host-to-host
latency with resident weights: two runs favor whole CPU on small shapes and whole
GPU on the largest tested shape; splitting loses to whole GPU in all cases.
This motivates explicit subgraph placement, not automatic operator-level routing.
The checked-in PJRT API restricts native CopyToDevice/CopyToMemory to one client;
they cannot directly bridge our separately loaded CPU/CUDA clients. Native
`Client::load_on_device` now fixes an addressable-device index at client creation;
the two-logical-CPU example validates raw-HLO and high-level Tensor assignment,
memory-cache reuse and optional disk restoration on either device. Tensor
compilation targets the selected global device ID; disk namespaces must distinguish
placement (`DiskCache::new_for_client` derives a device-specific namespace from
an explicit compatibility key), and foreign-device native artifacts are rejected.
The shared-directory example validates fresh-client hits and device-scoped trimming.
Same-client device
views/copies and multi-device execution remain open work. Creating another client
does not share an allocator or provide zero-copy transport.

## Open work that affects the intended library

Selected CUDA regression through `b4e325fd15` (2026-09-13) passed after the
worker panic-ordering fix and detection geometry additions:
`check-xla-cuda.sh --offline`. This includes coordinate conversions, aligned and
pairwise IoU, GIoU, their gradient checks, 200 panic-ordering iterations, native
workers, checkpoint restoration and the independent stateful consumer. IoU and
GIoU resident fitting reached losses about 6.99e-6 and 0.00171 respectively,
with one compilation per variant. Log: `/tmp/xla-detection-worker-full-cuda.log`.
This is selected correctness coverage, not the entire CPU suite, a full model
benchmark, production detector training or distributed/multi-GPU execution.

CPU regression after box geometry/training additions (2026-09-13) exposed a
worker panic ordering race in the optional-feature async pool test: the running
reply disconnected before the admission receiver was dropped. Handler panics
are now caught only to close admission first, then resumed so shutdown still
reports panic. A 200-iteration regression verifies immediate stopped admission
after the failing response. Full `check-xla.sh --offline --with-plugin` then
passed, including default/optional features, both box-training variants,
Clippy, runtime-only package subprocess tests/dependency isolation and the
independent stateful consumer. Failed log: `/tmp/xla-detection-full-cpu.log`;
successful rerun: `/tmp/xla-detection-worker-fixed-full-cpu.log`. This checkpoint
does not rerun CUDA model execution or establish distributed scheduling.

Selected CUDA regression checkpoint through `779863346d` (2026-09-13):
`check-xla-cuda.sh --offline` completed successfully after axis-reordering APIs,
attention layout refactoring and total state-transfer budgets. Coverage includes
flatten/unflatten/move_axis/swap_axes, state snapshots, workers, training and
checkpoint restoration, native cache checks and the independent stateful
consumer. Log: `/tmp/xla-axis-state-full-cuda.log`. The consumer executed three
stateful calls with one compilation. This is the audited selected GPU suite,
not every CPU test, a new full TinyLlama run, multi-GPU coverage or a performance
comparison. Separate attention and finite-difference gradient tests also passed
on CUDA after the refactor (`/tmp/xla-attention-axis-api-cuda.log`).

Full CPU regression checkpoint through `a721d4c0a0` (2026-09-13):
`check-xla.sh --offline --with-plugin` completed successfully after prepared
output metadata, canonical checkpoint preflight, flatten/unflatten and total
state-transfer budgets. Default and optional cache/training tests, training
examples, both Clippy configurations, runtime-only dependency isolation and
independent worker-owned state execution passed. Log:
`/tmp/xla-total-budget-full-cpu.log`. The preceding heterogeneous gate also
passed (`/tmp/xla-state-total-budget-heterogeneous.log`); this checkpoint does
not rerun the full selected CUDA gate or full-model performance benchmarks.

Selected CUDA regression checkpoint through `52433ba696` (2026-09-13):
`check-xla-cuda.sh --offline` completed successfully after prepared state
metadata, canonical module/state checkpoint preflight and output signature APIs.
The gate includes state/worker execution, training examples, BF16 and optimizer
checkpoint restoration, native cache checks and independent downstream execution.
Log: `/tmp/xla-state-preflight-full-cuda.log`. It is the selected correctness
suite, not the full CPU suite, full-model performance or multi-GPU coverage.

Regression checkpoint through `87c9d91c65` (2026-09-13): full CPU
`check-xla.sh --offline --with-plugin` passed after PreparedStateGraph, prepared
schema/parameter inspection and checkpoint preflight extraction. It covers both
default and optional cache/module configurations, training and checkpoint tests,
both Clippy configurations, runtime-only dependency isolation and the independent
caller-prepared/worker-executed state consumer. Log:
`/tmp/xla-state-preflight-full-cpu.log`. Recent targeted CUDA state/weight tests
and the heterogeneous gate also passed, but this checkpoint is not a fresh full
selected-CUDA regression or model-performance measurement.

`LoweredProgram` and `PreparedStateGraph` provide explicit immutable lowered
programs for reuse across client-local compilers. Ordinary programs carry
StableHLO; state snapshots retain hidden update
roots, slot schemas and optional compact input mappings; they do not contain
resident buffers or replace checkpoints. Their host-only metadata supports
module checkpoint preflight before native compilation. These additions reduce
repeated frontend work, not native compilation complexity, and do not close the
broader model/operator/deployment work listed below.

Regression checkpoint through `63fb334f17` (2026-09-13): full CPU
`check-xla.sh --offline --with-plugin` and selected CUDA
`check-xla-cuda.sh --offline` both passed after compile-time statistics,
LoweredProgram cache-path refactoring and shared lowered worker programs.
Coverage includes existing state/training/checkpoint/cache tests, the prepared
input-ABI test, native worker examples and independent downstream execution.
CPU additionally checks default/optional-feature Clippy and runtime-only
dependency isolation. Local logs: `/tmp/xla-prepared-full-{cpu,cuda}.log`.
This does not rerun full-model benchmarks or establish multi-GPU execution.

Regression checkpoint through `3c38611f46` (2026-09-13): the full CPU
`check-xla.sh --offline --with-plugin` gate passed after static metadata/public
Result, I32 singleton-axis APIs and deployment tensor-size preflight changes.
Both default and optional cache/module configurations, both Clippy configurations,
runtime-only dependency isolation and seven loader unit tests passed. The
independent Tensor consumer executed three stateful calls with one compilation.
The I32 layout target was also run separately on CUDA (four tests passed).
This checkpoint does not imply a full CUDA regression or a new model benchmark.
Local full CPU log: `/tmp/xla-api-deployment-full-cpu.log`.

Sequence composition now includes stable `logaddexp`, default doubling and
opt-in tree `logcumsumexp`, zero-safe differentiable `cumprod`, and affine scans
with zero or explicit initial state. Tests cover signed VJPs, selected nonzero
Hessian-vector cases, symbolic chunk carry, and resident carry across independent
Session calls. The [prefix benchmark](LOGSCAN-BENCHMARK.md) records the measured
CPU runtime versus compilation tradeoff; tree scanning is not an automatic
backend choice. The `gae` example separates bootstrap and trace boundaries and
detaches rollout targets, with CPU/CUDA F64-reference checks. These compositions
do not provide arbitrary loop bodies, cross-call backpropagation or an RL trainer.

Regression checkpoint through `e50dea8799` (2026-09-13): the full CPU
`check-xla.sh --offline --with-plugin` and selected CUDA
`check-xla-cuda.sh --offline` gates both completed successfully after these scan,
streaming-state and GAE additions. This includes existing training, checkpoint,
cache and independent-downstream checks; CPU additionally verifies both Clippy
configurations and runtime-only dependency isolation. No full-model performance
or multi-GPU run is implied by this checkpoint.

Regression checkpoint (2026-09-13, implementation through `70358430d0`):
`check-xla.sh --offline --with-plugin` and `check-xla-cuda.sh --offline` both
completed successfully with the local trusted CPU/CUDA plugins. This includes
the new shared-producer native pool and complete session-copy checks, default
and optional-cache CPU tests, Clippy, checkpoint/training coverage selected by
each gate, runtime-only dependency isolation and independent downstream execution.
It does not rerun full-model benchmarks, establish performance, or close the
remaining items below. The complete-state API's CPU/CUDA round-trip example was
also run separately when introduced.

A subsequent full CPU/default/cache gate and selected CUDA gate also passed on
2026-09-13 through `718f92fbf7`, after adding explicit single/multi-input diagonal
derivatives and the custom-activation SGD example. The gates include the new
derivative tests, automatic retention boundary/native-restore checks, checkpoint
and training tests, and independent downstream execution; the CPU gate also runs
both Clippy configurations and runtime-only dependency isolation. No full-model
LLM/OCR benchmark or multi-GPU validation was rerun for this checkpoint.

1. **Reusable execution ergonomics:** at the native layer, `submit`/`wait`,
   `is_ready` and borrowed pending output
   handles support bounded in-flight inference and device-data pipelines on one
   thread/client. [The bounded example](crates/rxla-core/examples/inflight.rs) does not
   provide a byte-based memory quota or an event-driven executor. A
   [small release benchmark](PIPELINE-BENCHMARK.md) found lower median latency
   for split submissions than split synchronous calls, while a whole graph stayed
   faster on each tested backend. It is not full-model throughput evidence or a
   reason to split graphs that could be optimized together.
2. **Compiler/operator extensibility:** define a supported extension boundary
   and derivative contracts. `Tensor::with_gradient_of` now provides explicit
   same-shape surrogate derivatives for Rust compositions, including VJP and
   higher-order composition; it is not arbitrary custom VJP registration or an
   external-kernel ABI. Only the surrogate's final output must match the forward
   value: a rectangular matmul surrogate now has CPU/CUDA coverage for differently
   shaped inputs, dense signed VJPs, cross derivatives and forward-only pruning.
   This composes existing AD rules, rather than registering a custom VJP body.
   `Tensor::with_elementwise_derivative` also accepts an
   explicit same-shape diagonal derivative: reverse mode multiplies the incoming
   cotangent by it and higher derivatives follow that expression. Native CPU/CUDA
   checks cover the chosen VJP and second derivatives; forward-only snapshots
   prune derivative-only edges. This is not a full matrix Jacobian/custom VJP.
   The multi-input `with_elementwise_derivatives` form additionally checks
   cross-partials, repeated-input accumulation and shared intermediate ancestors
   on CPU/CUDA, without changing the same-shape diagonal contract.
   `train_custom_activation` connects a custom softsign/gain Module to resident
   guarded SGD: CPU/CUDA runs compare every prediction, loss and parameter with
   an independent F64 reference over 200 steps, reusing one compiled plan. This
   is synthetic training coverage, not a model/performance benchmark.
   HLO protobuf generation and optimized-HLO inspection
   do not alone provide arbitrary MLIR passes or FlashAttention kernels.
3. **Broader model execution:** reduced precision/quantization, dedicated LLM
   prefill, donation/alias policy, multiple prompts and real vision applications.
   Profile representative workloads before promising backend performance.
4. **Deployment and cache management:** reproducible plugin/dependency setup,
   compatibility fingerprints and strict concurrent disk-cache quotas. Explicit
   oldest-modified trimming of validated namespace/flags entries is available;
   opt-in post-publication automatic trimming reuses this namespace-scoped policy.
   Neither enforces a concurrent total-directory quota. Current pinned downloads
   and explicit namespaces are tested procedures, not a complete distributor.
5. **Stateful API composition:** preserve convenient module use while specifying
   batch/vectorization, differentiability and shared state rules. Explicit
   `State<F32/I32>` identities, owned update groups, exclusive transaction scopes
   and `impl_state_tree!` named discovery are present. Named trees connect to
   complete-state checkpoint APIs; Threefry, Adam, Momentum SGD, BatchNorm and
   gradient accumulators expose named resident state. Composed-tree native tests
   cover restored optimizer/statistics updates and partial weighted/unweighted
   accumulation windows, including rejected batches and changed slot registration
   order. These are same-client correctness tests, not cross-environment recovery.
   Macro field coverage, canonical-name evolution, initialization and optimizer/RNG
   configuration remain explicit application responsibilities, not automatic capture.
   `ThreefrySequence::split_keys` reservations and guarded child `reset_key_if`
   are available; replica assignment, domain separation and cross-tree stream
   management remain caller policy.
   Existing guarded state updates are not arbitrary Rust side-effect tracing.

These are outstanding areas, not a new narrowed objective or an assertion that
every historical IREE feature must be ported. Keep the implementation and the
evidence proportional to each claim.
