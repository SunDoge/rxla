# Stateful API design constraints

Status: state is now represented in the source IR by `rxla.state_input`,
`rxla.state_read`, and `rxla.state_write`. The user-facing `StatefulModel` /
`StateCx` API assigns stable scoped names while tracing; lowering discharges the
effects into ordinary ABI inputs and hidden final-state outputs. `StateGraph`,
`StateProgram`, and `Session` remain the lower-level execution machinery.
Module derivation, RNG state, and several transformation policies below are still
design work.

## IR effect model

This follows JAX's Ref/discharge split rather than treating Rust mutation as the
program. A state declaration introduces a logical identity and one tensor SSA
version. Reads and writes are explicit IR operations, so analysis and later
transforms can distinguish state from ordinary dataflow. The semantic snapshot
also preserves these operations; compilation caches and transformation pipelines
therefore do not erase the effect metadata accidentally.

Before StableHLO emission, discharge makes the program pure:

```text
state_input(id, path) -> state_read(id) -> computation -> state_write(id)
       |                                                     |
       +-- ordinary ABI argument             hidden result --+
```

Read/write markers lower to typed identity dataflow and the final SSA version is
returned as a hidden result. The runtime installs new resident buffers only after
all visible and hidden results have executed and passed validation. Dropping a
`StateStep` does not execute or commit it, and its exclusive borrow prevents two
overlapping transactions on one `StatefulSession`.

The first public API is deliberately small: declare state where its shape is
known with `cx.state(name, shape, dtype)`, use `StateValue::read/write`, group
paths with `cx.scope`, and evaluate through `session.call(inputs)?.eval(runtime)`.
One session currently owns one normalized input-program specialization. Reachable
lazy input dataflow is imported into the stateful IR, so preprocessing and the
state transition compile as one executable; only the materialized leaf bindings
remain runtime arguments. Later calls may supply new leaf values through the same
normalized expression structure. A changed expression or root layout is a new
specialization and is currently rejected rather than silently running stale code.

`StateValue::copy_`, `add_`, `sub_`, and `mul_` provide familiar in-place
spelling. They mean “read the current SSA version and emit `state_write`”; they do
not eagerly mutate a `Tensor`, alias-visible host storage, or a PJRT buffer. This
keeps ordinary tensors immutable while allowing update-heavy model code to remain
compact and fully transformable.

Tensor-content updates use Tensor-first APIs. `Tensor::slice_copy_` rebinds the
mutable Rust handle to a `stablehlo.dynamic_update_slice` result, while
`Tensor::index_add_` builds scatter-add and rebinds the handle. Cloned handles
retain their previous SSA value. `StateValue::slice_copy_` and `index_add_`
compose the same tensor operations with a state write, rather than routing users
through `Graph`. Physical buffer reuse is still chosen by XLA aliasing/donation;
the underscore spelling guarantees the handle/state transition, not allocation
identity.

With the optional `disk-cache` feature, `StateGraph::compile` can reuse native
code across processes through `Compiler`. It still reconstructs the current
schema and slot identities locally. The cache stores code and tensor signatures,
not session contents or fixed weight values. A cross-process regression test
verifies fresh initial states/weights, independent sessions, paired indexed
state updates, failed-input atomicity and ownership after caller handles drop.

A second fresh-process disk-cache regression covers mixed F32 data/I32 position
state with no visible outputs. The parent compiles once; the child rebuilds local
slot/parameter identities and restores code with zero backend compilations. Each
process chooses different starting position, data and fixed weights. Two sessions
advance independently; invalid input and wrong replacement dtype preserve all
state, and execution works after caller graph/compiler/client handles are dropped.
This does not serialize sessions or validate compatibility across plugin versions.

## User model

Models use `rxla_nn` parameter effects. Parameters are declared beside the tensor
values that determine their shapes. Tracing records stable paths, shapes, dtypes
and shared identities in `ParamSchema`; `ParameterSelection` chooses trainable
subsets without a parallel Rust object tree or manual visitation implementation.

Checkpoint binding consumes the schema directly. Mutable caches remain explicit
state effects and immutable assets remain constants, so neither is confused with
trainable parameters.

Users should operate on Rust modules and sessions rather than manually flattening
state tuples for every call. Keep immutable inference weights shareable and put
request-local KV storage and random streams in a separately owned session.
Mutation requires exclusive session access. Training owns its parameter updates
and optimizer state separately; not every tensor field is a differentiable parameter.

The implemented `Session::bind_inputs` retains selected read-only runtime inputs
as `Rc<Buffer>` owners, allowing sessions to share weights while their mutable
state remains independently owned. The low-level binding indices refer to visible
input registration order. `StateGraph::parameter` now registers an F32 input with
a `Parameter` identity and symbolic `tensor()` accessor. Clones preserve identity,
so tied weights share one input rather than registering copies.
`Session::bind_parameters` rejects foreign-schema, duplicate and post-compilation
handles, then reuses fixed-input validation and replacement. It replaces the full
fixed set (including earlier numeric bindings), not a partial merge. Handles are
process-local, not checkpoint names or portable serialized identities. Module
derivation, named loading and differentiable parameter selection remain future
work. Rebinding is fully validated before commit and never changes the compiled
graph or state slots. It is not a training parameter-update/differentiation API.

The shared execution plan precomputes visible-input-to-parameter positions once.
`bind_inputs` indexes that map instead of scanning all interleaved parameters for
each binding: setup is O(P) per plan, and binding bookkeeping is O(V+B) per call
for P total parameters, V visible inputs and B supplied bindings. Native buffer
metadata validation still occurs; this is not a measured throughput claim.
Tests cover a 1024-input interleaved map and live execution with 64 interleaved
state/input pairs, reversed full binding and sparse rebinding.

Recognized field roles are parameter, mutable tensor state, random stream, and
static configuration. Dynamic state values are runtime arguments, not embedded
constants or compilation-key contents. Shape/dtype, state schema and static
configuration participate in specialization. Shared parameter identity must be
preserved by loading, flattening and eventual derive macros.

## Functionalization boundary

`state_i32` registers an integer slot; reads and writes use ordinary I32
`Tensor` values. `write_many` atomically records mixed-dtype updates without a
parallel output representation. Existing F32 methods remain available but reject
I32 slots. Plan metadata tracks dtype as well as shape for initial/replacement state
and all result buffers. Session commits only after every typed result validates.
The stateful_cache example records data writes and position increments together;
callers pass only new data. Counter overflow is explicit wrapping I32 arithmetic,
and cache bounds/clamping remain caller policy, not automatic append validation.
Mixed result failure injection covers wrong output dtype before commit; it does
not establish native-device fault recovery. Integer RNG state is not implemented.

State reads/writes must go through tracked slots; do not claim to trace arbitrary
Rust field mutations, global variables, I/O, or tensor-dependent Rust branches.
During graph construction a slot read returns its current symbolic version and
a write advances that version. It must not mutate live device state. Later reads
within that construction observe the updated symbolic version.

`StateGraph::write_many` commits a subset of symbolic slot versions together.
It validates the entire batch first and rejects duplicate slot identities rather
than applying last-writer-wins behavior. Failed batches leave every slot version
unchanged. Inputs are preconstructed tensor values, allowing simultaneous swaps;
unmentioned slots are retained. This does not roll back previously created graph
nodes or arbitrary Rust side effects. The TinyLlama example and masked-decode
test use it through `KvCache::update_at` for paired K/V updates. Host tests check failed-batch atomicity and a
live CPU test checks repeated swaps with an unchanged third state slot.

`KvCache` is the first domain-specific state facade, not a general Module system.
It registers two equal-shaped F32 slots and records paired slice updates using
`update_at(&mut self, &mut StateGraph, keys, values, starts)`. Successful updates
return full symbolic caches for attention; failed validation advances neither
slot. Session initialization and inspection still use its slot accessors.
Positions are explicit and inherit dynamic-update-slice clamping: there is no
automatic append position, capacity error, mask, eviction, or paging policy.
Host tests cover foreign graphs, malformed indices/shapes and partial-build
failure; native tests cover hidden updates, independent sessions, clamping,
failed-input preservation and masked multi-step attention against F64 reference.

Compilation materializes an execution plan mapping registered state slots to
hidden parameters and final state outputs. Every committed state version is an
output root for reachability pruning, even if unrelated to user-visible results.
Repeated writes may discard earlier versions only when there are no remaining
data dependencies. The plan validates slot identity, shape, dtype and client
before execution; these checks cannot depend on Rust field names alone.

Execution binds existing resident buffers and installs all returned state only
after successful completion and validation. With the current non-donating
runtime, failures before commit must leave old slots intact. This is a local
logical commit, not rollback of arbitrary external effects. Future donation or
asynchronous execution requires a separately specified ownership/failure protocol;
`&mut Tensor` alone does not prove unique ownership of its native buffer.

## Transformation rules that must be explicit

State-backed module parameters are created explicitly with `trainable_parameter`.
Their Parameter handle retains both a start-of-step symbolic tensor and a state
slot identity for initialization/optimizer writes. Input and state storage form
separate identity domains within one graph, so collectors cannot accidentally
tie an input parameter and a resident parameter with the same numeric index.
Module methods do not implicitly reread StateGraph after a recorded optimizer
write; use `read(state_slot)` for a later symbolic version. This keeps the
ordinary Module signature while making version behavior explicit.

The experimental scalar-loss `Tensor::grad` now supports smooth graph arithmetic,
matmul, layout transforms and sum. The caller explicitly chooses graph input
leaves (including initial state reads) for differentiation and separately records
optimizer state updates. `train_linear` demonstrates this with explicit SGD;
it is not an automatic transformation of Session, mutable Module fields, or
arbitrary stateful Rust code. Unsupported reachable operators fail rather than
silently dropping their gradient. The broader policies below remain necessary.

- Gradients: select differentiable parameters; carry non-differentiable state as
  auxiliary results rather than silently treating everything as a parameter.
- Recomputation/checkpointing: replay tensor computation without committing state
  twice. Random operations replay the same logical keys, not a newly advanced stream.
- Vectorization: shared writes require an explicit policy (independent state,
  supported reduction, or rejection), never implicit last-writer-wins.
- Randomness: explicit stream ownership/splitting; no hidden process-global RNG.
- Concurrency: no overlapping exclusive state update. Independent sessions may
  eventually schedule concurrently. Persistent PJRT plugin, client, buffer and
  executable handles are `Send + Sync`, following PJRT's thread-safe native
  handle contract; graph-owned sessions remain thread-affine because their
  transactional state uses `Rc`. In-flight submission handles are also still
  thread-affine. Native execution cancellation and in-flight memory budgeting
  remain separate scheduling concerns.
- External effects: not ordinary dead-code-eliminable tensor operations. Ordering
  and retry behavior need separate contracts before adding callbacks or I/O.

Named device randomness is available directly through the state interpreter:

```rust
let model = StatefulModel::new(|cx: &mut StateCx, xs: &[Tensor]| {
    let mut rng = cx.rng("dropout")?;
    Ok(vec![rng.dropout(&xs[0], 0.5)?.output])
});
let mut session = model.session().rng_seed("dropout", 42)?;
```

`StateRng` currently provides raw blocks, uniform, normal, Bernoulli, and
dropout draws. Each name owns four scalar I32 states (`key0`, `key1`,
`counter_low`, and `counter_high`). Reborrowing the same name continues its
symbolic stream. `StateCx::finish` records the final counter update as part of
the state transaction; a counter wrap rejects the whole stream advancement.
The session injects the seed without exposing `StateGraph` or requiring a host
random-number upload. Use `rng_state` when exact raw key words or a nonzero
counter are required.

The public model-building path now uses the same `rxla_nn::Cx` for parameters,
resident state, and RNG effects:

```rust
let model = Model::new(|cx: &mut Cx| {
    let x = cx.input(&[batch, width])?;
    let y = cx.linear("head", &x, classes)?;
    let steps = cx.state("steps", &[], DType::I32)?;
    steps.write(cx, &steps.read(cx)?.wrapping_add_scalar(1)?)?;
    let noise = cx.rng("sampling")?.normal_f32(y.shape())?;
    Ok(y.add(&noise)?)
});
```

`Model::trace` discovers all three effect classes in one schema. Stateless
models retain `prepare`/`compile`; a model declaring state uses
`prepare_stateful`/`compile_stateful`, then `compiled.session()` for named
initialization, including `rng_seed`. The resulting `ModelSession::run`
validates structured inputs and reconstructs structured outputs while retaining
state between calls. Custom transformations with a different visible-output ABI
can explicitly use `compile_stateful_program` and the raw session API. The older
`rxla_core::StatefulModel` and
`StateCx` remain a compatibility layer for lazy-Tensor call sites; new model
code should not combine that context with `rxla_nn::Cx`.

## First implementation verification targets

Start with a state slot plus compiled-session wrapper, not a Monad-heavy public
API. Tests must cover a counter/accumulator with no visible state return; repeated
calls without recompilation; multiple slots with read-after-write; updates not
used by visible results surviving pruning; independent sessions sharing a plan;
shape/client mismatch leaving prior state unchanged; and old-state preservation
on execution failure. Only then add module derivation and broader transforms.

Current tests cover symbolic slot validation, hidden counters, multiple writes,
independent sessions, input shape/client rejection, and commit-boundary failures.
Execution-error and malformed-output tests inject failures at the Rust execution
boundary; they do not claim native-device fault recovery. The `stateful_cache`
example validates dynamic updates with F32 visible inputs, resident F32 data and
an I32 position counter, and four invocations of one compiled program.

`Session::replace_state` atomically replaces the complete state by slot identity
and returns the old owned buffers for restoration or transfer to another session
of the same program. Validation reuses the initial session binding checks; wrong
counts, duplicate/foreign slots, shape/dtype/client mismatches leave the current
session intact. Rejected replacement buffers are consumed and dropped. This is
an ownership exchange, not a deep snapshot, and performs no graph execution,
compilation, or tensor data transfer. Metadata validation can call PJRT. The live
test also resumes both exchanged states independently using one compiled plan.

`Session::into_state(self)` additionally supports consuming a session without
allocating a replacement state. It returns slot/buffer pairs in registration
order, usable by `StateProgram::session` or `Session::replace_state` under the
same identity/type/client checks. Fixed input bindings are released and are not
part of the returned state; resumption must rebind any retained shared weights.
This is resident ownership transfer, not a clone, serialized checkpoint, host
offload, or cross-thread migration. Native tests cover mixed F32/I32 state,
exact integer positions above F32 precision, resumed execution with rebinding,
old visible-output lifetime, buffers surviving plan/client handle drops, and
empty state. No new backend compilation is required to resume.

### Conditional state transitions

`StateGraph::write_many_if(&scalar_mask, &updates)` records one condition
across mixed F32/I32 state slots. It broadcasts the scalar mask to each slot and
selects between the proposed value and the version current at this call, then
advances all selected symbolic versions together. Zero keeps old values; nonzero
(including NaN) chooses proposed values, matching numeric tensor selection.
Use a comparison-produced 0/1 mask when NaN must not count as an affirmative
condition. Conditions must be graph-local scalars. Slot identity, dtype, shape,
and duplicate checks happen before any version advances; omitted slots are kept.

This does not skip proposed computations, gate visible outputs, or catch native
execution failures. It provides a dataflow state-update guard, not general
transactional side effects. The native regression composes a runtime enabled
flag with a position bound: accepted writes advance F32 cache and I32 position
together, disabled writes preserve both despite a NaN proposed value, and writes
beyond capacity preserve both despite clamped slice semantics. The test uses one
compiled program, not a full batched LLM or a scheduling benchmark.
