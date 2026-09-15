# Execution planning as an architectural requirement

Status: logical sharding constraints and validated one-stage `ExecutionPlan`
landed (2026-09-14). Pliron is replacing the original semantic snapshot as the
canonical transformation and planning IR; distributed execution remains
incomplete.
The IR must support inspection, transformation, planning and staged execution,
not merely serve as temporary input to HLO generation. New graph/compiler/state
work must preserve this direction. Public type names and wire formats are not
frozen by this document. See [current evidence and limits](SCOPE.md).

The [MLX layout/storage review](MLX-LAYOUT-REVIEW.md) adds required separation of
logical shape, layout constraints and actual storage. Plans must account for
strided/tiled layouts, aliases, completion and explicit layout conversions;
unknown layout must not be represented as known contiguous storage. These are
design constraints, not implemented native view or donation support.

## Boundary and existing foundation

```text
Rust model + explicit state
  -> Pliron module
  -> placement / partition planning
  -> validated execution plan
  -> per-stage HLO compilation and explicit communication
  -> local or remote executors -> PJRT
```

The intended single-device path is a one-stage plan, not a separate architecture.
Manual placement, heuristics and future automatic search must produce the same
plan representation. Automatic search is not required for ordinary execution.

`crates/rxla-core/src/pliron_ir.rs` defines the typed SSA dialect. Its
module/region/block structure is the canonical computation representation.
Compact planning facts (SSA value count, parameters, roots and constraints)
are extracted directly from Pliron before lowering. `LoweredProgram` is an
immutable StableHLO backend program, while `PreparedStateGraph` retains state
schema and hidden update metadata. Neither is a remote wire format yet.
Lowering walks Pliron SSA directly and emits verified StableHLO.
`PlanningPolicy::SingleDevice` rejects multi-device constraints instead of
ignoring them. The bounded auto policy is represented in the same planner API,
but meshes larger than one device are rejected until an executable partition
and communication plan exists.
Current workers and client-local native handles are execution building blocks,
not a distributed executor. No existing API gains distributed semantics through
this decision alone.

## Required invariants

1. **Separate logical values from storage.** Logical shape/dtype and semantic
   constraints belong in typed Pliron attributes. Preserve raw dtype identities
   in the IR; backend lowering, rather than the type representation, declares
   which element types it currently supports. Physical byte strides and device
   buffers remain storage concerns. Logical value identity must not depend on a
   buffer address or worker. A future distributed
   value can map to several physical shards. Local IDs must be remapped when
   serializing; raw pointers, process-local owner IDs and Rc handles are not a
   portable identity scheme.
2. **Preserve information before lowering.** The planning snapshot must retain
   output ordering, shared parameter aliases, state reads/updates, explicit RNG
   dependencies, source/module provenance and user stage/placement constraints.
   Provenance is advisory; a stage boundary is an explicit constraint, not an
   accidental consequence of calling a module. Transformations must preserve
   numerical and state semantics within documented tolerances.
3. **Separate computation, policy and mechanism.** Tensor math must not require
   a concrete worker. Placement, mesh topology and sharding are planning inputs;
   executors implement validated plans. Keep ordinary APIs concrete and use
   owned/smart-pointer metadata where useful; do not encode device meshes or
   distributed schedules in pervasive Rust generics.
4. **Make execution dependencies explicit.** Plans describe stage inputs/results,
   transfers, resharding, collective groups/order, completion dependencies and
   buffer lifetimes. Submission is not device completion. Memory reuse must
   wait for all consumers, including transfers. Async host tasks alone do not
   establish these guarantees. Unsupported transport/collectives must fail
   capability checks, never silently become an expensive or incorrect fallback.
5. **Plan training state, not only forward operations.** Represent backward,
   accumulation and optimizer dependencies; preserve tied-weight gradient
   summation. Start with synchronous steps using a fixed parameter version and
   publish new state only after the complete step succeeds. Distributed atomic
   publication requires its own protocol; existing local Session behavior does
   not provide it. Initially fail explicitly and recover from a committed
   checkpoint rather than transparently retry stateful work. Define request IDs,
   state versions and failure outcomes before enabling remote mutation/retries.
6. **Declare extension capabilities.** Future custom operators need explicit
   shape/dtype, effect, differentiation and partitioning contracts. Unknown
   sharding behavior remains an opaque unpartitioned region where supported,
   or is rejected. Never infer purity, gradients or safe splitting from an HLO
   opcode string or arbitrary external code.
7. **Validate before dispatch.** Check signatures, graph/plan references, stage
   dependencies, shard coverage, state ownership, capability compatibility and
   resource limits. Reject cycles in the initial finite-step task DAG. Future
   loops require explicit semantics. Remote services additionally need bounded
   decoding, authentication and isolation of untrusted compilation/execution;
   existing trusted native artifact loading is not a security boundary.

## Compilation and iteration budget

Keep checked-in bindings/protos and maintainer-only code generation. This
architecture must not introduce normal-build LLVM, MLIR or XLA source builds.
The runtime-only consumer must remain independent of frontend/planner tooling.
Do not add a planner crate or a network dependency solely for placeholder APIs;
extract components once a tested boundary exists.

Provide three eventual policies: fast single-device/manual planning by default,
bounded deployment optimization, and opt-in offline tuning. Never silently run
expensive profiling/search on a normal compile call. Preserve a valid baseline
plan when an optimization budget is exhausted.

Cache semantic preparation, plans, per-stage native compilation and measurements
separately. Keys must account for relevant graph/state ABI, shapes, dtypes,
precision, constraints, planner/compiler versions, topology and target/plugin
compatibility. Keep dynamic weight values out of code keys unless specialized
as constants. Native executable portability is not implied by graph portability.
Record preparation, planning, compilation, transfer and execution costs separately.
Partitioning may reduce fusion and increase communication; more stages are not
automatically faster or cheaper to compile.

## Implementation sequence and acceptance gates

The first milestone is in progress; later stages remain pending. Logical
constraints are retained, but no planner may yet claim physical sharding.

| Milestone | Required evidence before advancing |
| --- | --- |
| Pliron semantic module and one-stage plan (dialect/constraint portion implemented) | Plugin-free validation tests; program unaffected by later graph edits; preserved output/input ABI, aliases and hidden updates; existing CPU/CUDA correctness gates; compile/cache counters and measured overhead against current path |
| Stable Diffusion inference workload | The complete denoising path remains in Pliron without legacy graph fallback; fixed model, resolution, batch, sampler and step count; output tolerance plus warm latency, throughput and peak device-memory comparisons against pinned PyTorch/JAX baselines |
| Manual two-stage local execution | Forward and gradient equivalence to unsplit graph; shared weights and RNG/state checks; bad-plan rejection before dispatch; explicit lifetimes and failure cleanup; measured transfer/fusion cost |
| Remote resident session | Versioned bounded protocol; graph/weights uploaded once and cache reuse demonstrated; local/remote result equivalence; disconnect, duplicate request, stale version and worker-loss tests; explicit commit-unknown outcome where necessary, no exactly-once claim without durable recovery |
| Homogeneous two-GPU data parallelism | Actual multi-device PJRT and collective support proven; deterministic reference update with appropriate tolerances and loss normalization; group/order checks, failure tests, memory and communication measurements |
| Pipeline and tensor parallelism | Microbatch/version correctness, activation lifetimes, resharding and real multi-GPU correctness/performance; support explicitly limited to tested plans/operators |
| Automatic plan selection | Bounded search over validated plans; measured cost-model error; comparison with manual and unsplit baselines including compilation amortization; reproducible saved plans |

Logical CPU devices and CPU/CUDA host-staged copies can test some planning and
transport behavior; they cannot establish multi-GPU collective correctness or
speedup. A prebuilt PJRT plugin must be probed for needed functionality before
depending on it. XLA containing auto-sharding source does not establish that our
plugin exports the required controls. No custom XLA build is an implicit fallback.

## Review gate for future changes

- Does the change erase semantic/state/provenance information needed before HLO?
- Does it unnecessarily bind logical Tensor math to a physical device or worker?
- Can manual and future automatic planning share the resulting contract?
- Are completion, ownership, failures and any state commits explicit?
- Are unsupported extension/partition capabilities rejected or conservatively bounded?
- Are compilation/search costs opt-in or budgeted, with meaningful cache keys?
- Are implemented evidence and proposed distributed behavior clearly separated?

This is inspired by [Alpa's architecture](https://github.com/alpa-projects/alpa/blob/main/docs/architecture/overview.rst):
inter-operator planning, intra-operator planning and runtime orchestration are
separate responsibilities. We adopt that separation, not a claim of Alpa's
automatic optimization or a commitment to its implementation dependencies.
