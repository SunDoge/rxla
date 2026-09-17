# RXLA implementation and API notes

This document preserves detailed API behavior, implementation decisions, and
incremental validation notes. Commands are run from the repository root. For a
short introduction, start with the [project README](../README.md).

This is an evolution log rather than the canonical API reference. Older
sections may retain names or limitations that were subsequently replaced; use
the current crate documentation and `SCOPE.md` when they disagree.

Start with [current scope and remaining work](SCOPE.md) for a compact overview;
this guide contains detailed APIs and incremental verification history.

The primary value API is split into three layers. `Tensor` carries logical
dtype/shape and optional layout/storage; `Tracer` owns explicit IR construction
for reusable programs; `Runtime` owns PJRT clients, compilation caches and
dispatch. A `Program` retains a lowered snapshot plus compact SSA planning
facts, not its source graph or output Tensor handles. Those facts are extracted
directly from the Pliron module. Ordinary
`Tensor` operations follow an MLX-style lazy workflow: concrete values are lazy
leaves, operations build a private expression graph, and `Tensor::eval` or
`Runtime::eval` is the explicit materialization boundary. It accepts a Tensor,
an array, or a tuple such as `runtime.eval((&x, &y))`. `Tracer` and
`Program` remain available for reusable computations called with different
inputs. `Program::run` handles
F32 host values and `Program::run_buffers` handles resident F32/I32/BF16 buffers,
while `Program::run_tensors` accepts stored Tensor inputs and returns Tensor
outputs owning PJRT storage and copied physical-layout metadata. Those result
Tensors can continue participating in lazy expressions. Application code does not
manually own the private mutable graph. `Tracer` is the explicit construction API;
`Compiler` accepts tracers or immutable lowered snapshots.

Concrete host values do not require a public Graph or Tracer:

```rust,ignore
use rxla::{DType, Runtime, Tensor};

let x = Tensor::from_slice([2], DType::F32, [1.0, 2.0])?;
let y = Tensor::builder([2], DType::F32)?.from_vec(vec![3.0, 4.0])?;
let z = (&x + &y)?.exp()?;
let mut runtime = unsafe { Runtime::load("libpjrt_plugin.so")? };
z.eval(&mut runtime)?;
```

Run the same end-to-end facade path with a trusted plugin:

```sh
cargo run -p rxla --example basic -- --plugin /path/to/libpjrt_plugin.so
```

`PJRT_PLUGIN_PATH=/path/to/libpjrt_plugin.so` can supply the argument instead.

With the `disk-cache` feature, the same single-plugin path can restore native
executables across runtime instances without exposing `Client` or `Compiler`:

```rust,ignore
let mut runtime = unsafe {
    Runtime::load_with_cache(
        plugin,
        cache_directory,
        plugin_build_id,
        256 * 1024 * 1024,
    )?
};
```

The compatibility key identifies the exact plugin/driver environment. CPU and
CUDA backends use independent keys; heterogeneous runtimes configure caching
per backend with `Runtime::builder().cached_backend(...)`.

Host construction is dtype-explicit. `from_slice` copies, while the builder's
`from_vec` retains the allocation. `unsafe from_raw_parts` accepts an allocation
deleter that runs exactly once after the final storage owner is released.
`Tensor::try_from(dlpark::versioned::Dlpack)` and its legacy counterpart import
compact CPU tensors without copying and retain the producer's DLPack deleter.
This includes `image::ImageBuffer` values exported through dlpark; U8, F16,
BF16, F32, and I32 host representations are preserved.

`eval` materializes the existing Tensor identity: callers may ignore its
convenience return value, and all clones of that Tensor observe the resulting
storage and physical layout. Re-evaluating an already materialized value is a
no-op. Expression construction remains immutable and lazy.

Fallible Rust operators are also available for compact lazy expressions:

```rust,ignore
let z = ((&x + &y)? * &x)?.exp()?;
z.eval(&mut runtime)?;
```

Logical sharding constraints are attached before lowering and retained by each
`Program`. A `Mesh` names logical axes and a `PartitionSpec` maps tensor
dimensions to them; neither embeds PJRT clients, device pointers, or worker
types in Tensor math. Pliron is the sole semantic IR and planner boundary.

`Tracer` construction emits operations directly into a real
Pliron `builtin.module` region/block. Programs derive parameter identities,
output roots, value count and
sharding constraints from Pliron SSA. SPMD planning transforms the already
lowered backend module from those facts.
Typed dialect operations lower directly from Pliron SSA to verified StableHLO;
there is no computational HLO-proto or parallel semantic-node fallback.
The implementation is split by responsibility under `pliron_ir/`: dialect
schemas, typed attribute codecs, analysis, StableHLO conversion and integration
tests remain separate from the IR owner and builder. Semantic operation
descriptors are consumed immediately by `pliron_ir/builder.rs`; they are not
retained as a second graph representation. Autodiff derives its short-lived
analysis view directly from Pliron SSA.
Ranked tensor types store shape and element type as typed Pliron attributes,
not identifier-formatted strings. Element types retain the raw PJRT dtype value
so the IR remains open to backend types beyond the currently lowered
F32/I32/BF16 set.
Native coverage includes elementwise activation math, reshape/broadcast,
transpose/reverse, static slice/pad, concatenate, reductions and cumulative
sum, comparisons/masks/select, batched matmul, attention composition and grouped
NHWC Conv2d with stride, dilation and asymmetric padding.
Device-side I32 coverage includes iota coordinates and bitwise/shift operations,
providing the primitive path needed by counter-based RNG and sampling code.

Every compile now passes through a validated `ExecutionPlan`. The default
`PlanningPolicy::SingleDevice` produces one stage and rejects any constraint
requiring a larger mesh before the compiler cache or PJRT is touched.
`PlanningPolicy::Auto(AutoShardingOptions)` has an explicit mesh and portable
search-effort hint, quantized to XLA's O0–O3 effort levels. Its placement table
marks user constraints as `ShardingDecision::Explicit` and unconstrained values
as `ShardingDecision::Auto`; it does not fabricate a final replicated layout
before XLA's search runs. Plans expose whether SPMD lowering is required.
Plans compile with XLA auto-SPMD and execute through replicated program
boundaries, including ordinary lazy `Runtime::eval`; compile options are part of
the executable-cache key. Explicit replicated and multi-axis partition constraints
lower to `OpSharding`. Parameter constraints are placed after a replicated
ingress copy, and result values feed replicated egress copies, so the public
Tensor ABI remains complete logical arrays while XLA inserts internal reshards.
Tile assignment reorders physical device IDs from mesh-axis order into tensor
partition-axis order and represents unused mesh axes as a replicated tile
dimension.

Tensor dtype unification (2026-09-13): the former `Index` type was folded into
`Tensor` and removed. Use `Tracer::input_dtype(shape, dtype)`
for F32/I32/BF16 inputs; all share managed storage and one-pointer descriptors.
Shape transforms/gathers preserve dtype. Mixed arithmetic is rejected, and
`to_f32()` explicitly converts I32/BF16. NN arithmetic/autodiff remain F32-only;
BF16 arithmetic is not yet enabled. State reads/updates use the same Tensor API,
with dtype checks at graph construction or state write/commit time.

Architecture requirement: the IR must support inspection, transformation,
planning and staged execution, with a shared execution-plan boundary from one
device to future distributed execution. Read the
[design constraints and staged acceptance gates](EXECUTION-PLAN-DESIGN.md)
before changing graph, compilation, state or placement interfaces. This direction
does not mean the distributed planner/runtime is implemented.

The [MLX reference review](MLX-LAYOUT-REVIEW.md) records adopted layout/view,
storage ownership and completion constraints, with concrete PJRT validation
gates. Logical Tensor metadata currently remains shape/dtype, not physical strides.

`Buffer::memory_layout()` queries owned physical-layout diagnostics through the
pinned legacy PJRT API. CPU/CUDA `layout_probe` validates transpose/reshape
outputs and scalar/empty/singleton uploads. This does not expose native views or
promise zero-copy; see the review's validation checkpoint and remaining limits.

For host layout metadata, `rxla_pjrt::{Shape, ByteStrides}` use private
const-generic `SmallVec<i64, 5>` storage. `ByteStrides::from_elements` performs
checked conversion when an adapter starts with element strides. `StridedLayout`
checks byte spans and `HostView` provides bounded read-only byte access and
explicit packing. These do not create native PJRT views or implement DLPack;
semantic IR shape vectors remain unchanged.

`Tensor` now uses a one-pointer immutable descriptor with inline shape metadata
and optional managed F32/I32/BF16 input storage. See [ownership and migration notes](TENSOR-STORAGE-DESIGN.md)
and `crates/rxla-core/examples/managed_tensor.rs`. Native-backed Tensor handles are
Arc-backed and `Send + Sync`; send prepared snapshots to
workers instead. Binding and uploads remain explicit, not automatic evaluation.

Storage retains type-erased bytes plus an explicit dtype: `Storage::host(dtype,
owner)`. Tensor descriptors expose `dtype()` and validate it against storage,
including same-width mismatches. `Storage::upload` dispatches F32/I32/BF16 without
float conversion of integer/BF16 payloads. Existing graph arithmetic APIs are not
yet unified across these dtypes.

For a short host-only edit loop, from the Rust workspace use:

```sh
sh scripts/check.sh --offline --quick
```

JAX reference values and canonical StableHLO are reproducible through the
project's locked uv environment rather than a developer-specific Python
installation:

```sh
uv run --frozen python scripts/jax_oracle.py broadcast --emit output
uv run --frozen python scripts/jax_oracle.py broadcast --emit stablehlo
```

The oracle is an explicit differential-development tool; Rust tests remain
self-contained and do not install Python packages or invoke Python implicitly.

`scripts/check.sh` checks formatting, all workspace targets, library tests and
Clippy. Pass `--offline` when Cargo dependencies are already cached.

For custom graph/module code, `rxla_tensor::Result<T>` is the public alias for
`std::result::Result<T, rxla_tensor::Error>`. `Tensor`, `Index` and mixed `Output`
handles all expose `shape()`, `ndim()`, `numel()` and `is_empty()`. These inspect
validated static metadata only: no graph mutation, device query, compilation or
synchronization. Scalars have rank zero and one element, while any zero extent
makes a tensor empty. Element counts are not byte counts or native memory usage.

`Tensor::cumprod(axis)` computes inclusive cumulative products without changing
shape. It reuses the work-efficient tree scan behind `logcumsumexp_tree`; a private
function pointer selects the graph-building combine operation, with no native
callback or additional generic specialization. Empty tensors and singleton axes
are identity. Forward and higher-order derivatives are compositions of products,
slices and concatenations, not division by the input, so finite zero values are
supported. CPU/CUDA tests check multiple zeros, negatives, odd/even lengths and
all axes of small matrices against independent prefix-polynomial VJP/Hessian
formulas. HLO checks verify no divide in those derivative graphs and fewer than
2N multiplied elements in a length-N forward scan at N=8 and N=4096. F32 products
may round differently from a sequential loop; nonfinite arithmetic may yield NaN.
No long-sequence cumprod performance or overflow-avoidance guarantee is made.

`offset.affine_scan(&multiplier, axis)` returns the inclusive recurrence
`s[t] = multiplier[t] * s[t-1] + offset[t]`, with zero initial state and first
output exactly `offset[0]` (the first multiplier is unused). Both tensors must
share graph and shape; other axes represent independent sequences. A tree
composes affine pairs without division. For discounted returns, reverse rewards
and discounts, scan, then reverse the result:

```rust,ignore
let returns = rewards.flip(&[axis])?
    .affine_scan(&discounts.flip(&[axis])?, axis)?
    .flip(&[axis])?;
```

Supply discounts such as `gamma * (1 - terminal)` explicitly. CPU/CUDA tests
verify finite recurrence values and both input VJPs against sequential F64
adjoints, plus terminal-boundary isolation of returns and reward gradients.
Empty/singleton axes and zero/negative coefficients are covered. F32 tree
association may differ from a sequential loop. A zero coefficient is not a
nonfinite-value mask: ordinary `0 * NaN` remains NaN. This is an elementwise
affine recurrence, not a general loop/scan body or a distributed RL trainer.

`offset.affine_scan_from(&multiplier, &initial, axis)` adds an explicit initial
state with shape equal to the input shape minus the scan axis. Unlike the
zero-initial convenience form, its first multiplier participates. Take the last
state slice, reshape away the scan axis, and pass it to the next symbolic block;
gradients flow through the carry unless explicitly detached. The method does
not mutate a Session slot or implicitly save the final state. An empty scan
returns an empty tensor rather than the initial state. CPU/CUDA tests compare a
five-step scan with a 2+3 symbolic split and independent F64 forward/adjoint
formulas for multipliers, offsets and initial state. This checks graph-level
composition, not separately scheduled streaming executions.

A separate CPU/CUDA Session test now executes six real three-step blocks with
resident carry and an accepted-block counter. Four accepted calls jointly update
both slots; two rejected calls still return candidate outputs but retain state.
A malformed input leaves both slots unchanged, and transferring `into_state()`
to a replacement Session midway preserves continuation. Every output and state
matches sequential F64 reference arithmetic, with one backend compilation and a
subsequent memory-cache hit. These are synchronous same-device calls, not
cross-call automatic backpropagation, asynchronous overlap or throughput evidence.

The `gae` example composes generalized advantage estimates from these primitives:
`delta = reward + bootstrap_discount * next_value - value`, followed by a reverse
affine scan with independently supplied trace discounts. Its chosen rollout
policy removes bootstrap at true termination, retains bootstrap at time-limit
truncation, and stops the trace at either boundary or rollout end. Callers must
provide the transition's actual next observation value, not an auto-reset
observation's value. Zero discounts do not sanitize nonfinite inputs.
Both advantages and value targets are explicitly detached. CPU/CUDA runs check
four `[2, 5]` synthetic rollouts against F64 within `2e-5`, terminal/truncation
behavior, zero target gradients into value inputs, and one compile/three cache
hits. Run `cargo run --offline -p rxla-core --example gae` with a trusted plugin.
This is a tensor-composition example, not an environment adapter, PPO trainer,
advantage-normalization policy or distributed rollout system.

`Tensor::logaddexp(&other)` computes a stable elementwise log-sum of exponentials
with explicit broadcasting. `Tensor::logcumsumexp(axis)` computes inclusive
prefix log-sums while preserving shape. It composes a doubling scan of stable
`logaddexp`, slices and concatenations: O(log N) graph stages and O(N log N)
elementwise work, not an N-by-N prefix matrix or a promised single fused kernel.
Empty tensors and length-one axes are identity; invalid axes are rejected.
Negative infinity acts as a mask, positive infinity and NaN propagate through
affected prefixes. Nonfinite-result derivatives are undefined and may be NaN.
Finite-input reverse mode and higher derivatives use ordinary graph composition.
CPU/CUDA checks compare all axes of small tensors against stable F64 prefix and
gradient formulas, including values near +/-1000, empty/singleton cases and the
zero Hessian-times-ones identity for the summed prefixes. Separate graph checks
confirm logarithmic graph growth (the tree variant has 5 versus 23 concatenations
for lengths 8 versus 4096). Throughput at long
sequence measurements are recorded in [LOGSCAN-BENCHMARK.md](LOGSCAN-BENCHMARK.md):
the stable scan passes wide-input forward checks through length 4096 but costs
more than a bounded-input-only direct-exp comparator, especially on CPU. These
noisy host-inclusive timings are not a full-model throughput claim.
The binary operation uses `max + log1p(exp(-abs(x-y)))` for finite results,
with an explicit nonfinite-result fallback. This preserves small increments:
the earlier shifted-exp sum rounded `logaddexp(0, -20)` to zero; regression tests
now check that case and `(0, -40)` against F64 with relative tolerance. Explicit
sigmoid partials avoid differentiating max/abs tie selection and preserve the
correct second derivative at equal finite inputs. CPU/CUDA tests cover these
precision cases alongside the cumulative scan's existing derivative checks.

`logcumsumexp_tree(axis)` is an explicit alternative using pair reduction,
recursive scanning, and prefix reconstruction. It reduces total elementwise
work to O(N), still with O(log N) graph depth. It is not the default: local
length-4096 measurements improved CPU execution by about 2.2x but increased
compilation cost, and did not improve CUDA execution. Both algorithms have
CPU/CUDA forward/gradient checks on odd/even, singleton, empty and multiple-axis
shapes. Their floating-point rounding can differ; choose using workload-specific
measurements, not the complexity bound alone.
Both scan implementations also pass signed-cotangent VJP and nonuniform
Hessian-vector tests for `[2, 17]` and `[2, 128]` on CPU/CUDA. The independent
F64 oracle explicitly forms each prefix's softmax weights and weighted Hessian
action; elementwise errors stay below `3e-5`. This tests nonzero second-order
responses, not only the constant-shift invariance direction. It does not measure
gradient performance or establish precision for arbitrary long sequences.

`forward.with_gradient_of(&surrogate)` is an explicit derivative-substitution
boundary for same-shaped, same-graph F32 expressions. Forward returns `forward`
without evaluating surrogate-only computations or adding/subtracting surrogate
values. Reverse mode routes the cotangent only to `surrogate`; other uses of
`forward` retain their original derivative. For example,
`x.gt_mask(&zero)?.with_gradient_of(&x)?` gives a hard forward threshold with an
identity straight-through derivative. This is an intentionally chosen estimator,
not the true derivative of the threshold. Unsupported operations on the
surrogate's backward path still fail preflight.

VJP and higher-order AD use the resulting backward graph, including dependencies
of its incoming cotangent. For `y = x.with_gradient_of(&(s*s))` and loss `y*y/2`,
the s-gradient is `2*s*y` and its s-derivative is `2*y + 4*s*s`, not just the
surrogate Hessian. Forward compilation snapshots omit the surrogate edge;
pruned compilation also omits surrogate-only inputs, while ordinary compilation
retains declared input ABI. Neither mode mutates the original graph or removes
its later AD capability. This does not provide arbitrary custom-VJP callbacks,
opaque native kernels, custom effects, or automatic differentiation of external
HLO edits. Native tests cover VJP, higher derivatives, straight-through masks,
nonfinite forward bits and input pruning.

`forward.with_elementwise_derivative(&x, &derivative)` instead directly supplies
the diagonal local derivative `dy/dx`. All three tensors must have the same shape
and graph. The backward contribution is `cotangent * derivative`, routed through
`x`; other forward dependencies do not receive a contribution through this
wrapper. For example, a chosen cubic rule can be expressed without integrating
the derivative into a surrogate:

```rust,ignore
let local_derivative = x.mul(&x)?.mul_scalar(3.)?;
let y = forward.with_elementwise_derivative(&x, &local_derivative)?;
```

The forward value is untouched and derivative-only work is pruned from forward
compilation. Higher AD differentiates the local derivative expression as well as
the cotangent: for loss `y*y/2`, the x-gradient is `3*x*x*y` and its derivative
is `9*x^4 + 6*x*y`. Use `detach()` on a derivative coefficient only when its
higher-order derivative should intentionally be suppressed. This is an explicit
elementwise/diagonal rule, not a general VJP for matrix operations, reductions or
broadcasts, and is not automatically inferred from edited external HLO. It adds
no runtime callbacks or LLVM/MLIR dependency. CPU/CUDA tests verify VJP and second
derivatives, while host checks reject graph/shape mismatches and verify pruning.

For multiple inputs, use `forward.with_elementwise_derivatives(&[(&x, &dx),
(&z, &dz)])`. The list must be nonempty and every input/partial must match the
forward graph and shape. Repeated inputs add their partials; distinct intermediate
inputs sharing an ancestor also accumulate through ordinary reverse mode. For
example, partials `(x, z)` and `(z, x)` define a product-like rule without changing
the forward value. For loss `y*y/2`, its x-gradient is `y*z` and the mixed
x/z derivative is `y + x*z`. CPU/CUDA tests verify this cross derivative,
repeated-input accumulation, shared ancestors and forward-only pruning. This is
still a collection of diagonal partials, not a general custom VJP callback.
Native CPU/CUDA cache checks also verify that different local derivative rules
share the same forward executable when their lowered forward HLO is identical,
while distinct backward graphs produce separate cache entries and gradients.
A derivative-only `log(-1)` does not contaminate forward execution; the identity
forward preserves negative zero, infinities and the tested NaN payload bitwise.
This does not promise bitwise NaN preservation for arbitrary arithmetic graphs.

Run `cargo run --offline -p rxla-train --example train_custom_activation` with a
trusted `PJRT_PLUGIN_PATH` to exercise these rules in training. Its user-defined
`Module` computes `gain * softsign(weight * x + bias)` and explicitly declares
partials for the activation input and broadcast gain. Ordinary AD handles the
upstream affine expression and scalar broadcast reductions. All three resident
parameters train with guarded plain SGD. Over 200 steps, every prediction,
pre-update loss and post-update weight matches an independent F64 formula within
`1e-5`, including regularly rejected updates; the loss falls below one tenth of
its starting value. The training transition is prepared once. At step 100 the
example obtains a cache-hit plan from that snapshot and recreates its Session
with `into_state()`, continuing against the same uninterrupted F64 reference.
CPU and CUDA pass with one graph compilation and unchanged compilation time on
the prepared cache hit. This is same-client resident ownership transfer, not a
serialized checkpoint or differentiation across session calls. The example
downloads diagnostics each step and is not a throughput or real-dataset benchmark.

For explicit child RNG streams, `ThreefrySequence::split_keys(count)` reserves
one consecutive block per key, returning two I32 `[count]` tensors. It is the
documented `blocks([count])` policy, not JAX's split convention. Zero count
consumes nothing; oversized counts fail without advancing the proposed cursor.
Commit the parent sequence, select a key pair with `Index::narrow`
and reshape its words to scalars, then call
`child.reset_key_if(&mut graph, [&key0, &key1], &accepted)?` using the parent's
**returned** acceptance mask. This updates the child's key and zero counter
together; begin a new child sequence after the reset. Reset is a symbolic
conditional write and invalidates previously begun child sequences even if its
runtime condition later rejects. Invalid shape/owner/condition errors do not
change symbolic state versions.

No automatic child allocation, replica assignment, key uniqueness across trees,
statistical independence or cryptographic guarantee is provided. Ordinary draws
and splits share the parent's counter space, with no domain-separation tag.
Repeated parent key/counter intentionally reproduces keys; never use split
outputs when their reservation was rejected. The guarded reset follows normal
StateGraph predicate semantics, including NaN-as-true; caller validity checks
remain explicit. Native tests compare keys and child draws bit-for-bit with an
independent integer Threefry implementation, including low-word carry, parent
wrap rejection, empty reservations and mixed ordinary/split draws. This is a
single-client graph test, not a distributed rollout scheduler.

`Index::slice(starts, limits, strides)` and `Index::narrow(axis, start, length)`
provide static I32 slicing with the same shared bounds validation as F32 Tensor
slicing. Strides must be positive, limits are exclusive, and negative indexing
or automatic clipping is not performed. Scalar identity slices and empty regions
are supported. Existing dtype-generic HLO slice lowering preserves all I32 bits;
no F32 conversion or new native operation is introduced. A split key word can
now be selected directly with `keys[0].narrow(0, i, 1)?.reshape(&[])?`.

`Index::dynamic_slice` and `dynamic_update_slice` support runtime scalar starts
with static sizes, sharing F32 region validation and dtype-generic HLO lowering.
Values remain exact I32, including bits not representable by F32. Each start
clamps to `[0, dimension-size]`; negative starts do not wrap. Scalar and empty
regions are supported. Updates return a new symbolic value: they do not mutate
or donate the original buffer or implicitly commit resident state. For token
history append, explicitly test capacity and guard both history and position
writes together; otherwise clamping can overwrite the final valid region.
Native tests cover extreme I32 starts, full-word values, source preservation,
scalar/empty regions, and a one-compilation resident history that accepts four
tokens then rejects a fifth without changing history or position. This is not
a complete decoding scheduler or a claim of in-place updates.

F32 `floor`, `ceil`, `round` (nearest, ties away from zero) and
`round_ties_even` (nearest, ties to even) lower to native HLO rounding operations;
see [XLA rounding semantics](https://openxla.org/xla/operation_semantics#round).
They retain shape/dtype, including scalars and empty tensors. Their reverse-mode
contract explicitly stops the gradient everywhere, including discontinuities
and nonfinite inputs/cotangents; it does not claim a mathematical derivative
at jumps. A straight-through or clipped estimator must be requested separately.
For example, form `clipped = x.clamp(-1., 1.)?`, then
`clipped.mul_scalar(2.)?.round_ties_even()?.mul_scalar(0.5)?`
and attach `.with_gradient_of(&clipped)?`. This gives F32 fake quantization with
step 0.5 and the existing clamp boundary gradient (0.5 at either bound).
It does not create INT8 storage, integer matmul, scale calibration or faster
inference. Native tests compare all four rules to Rust f32 operations, exercise
signed zero/NaN/infinity/large values/tie neighbors, zero first and higher
derivatives, and the explicit clipped-surrogate composition.

The workspace targets `rxla-core` as its primary frontend and uses XLA/PJRT as
the execution backend.
No Python, MLIR library, protoc, libclang or XLA source build is needed for a
normal build. Generated Rust bindings and protobuf types are checked in.

Typed state identities are available without shape generics or proc macros:

```rust,ignore
let mean = State::<F32>::new(&mut graph, &[features])?;
let steps = State::<I32>::new(&mut graph, &[])?;
let next = steps.read(&graph)?.wrapping_add_scalar(1)?;
steps.write(&mut graph, &next)?;
```

`State<F32>` and `State<I32>` both read/write `Tensor`.
Dtype mismatches fail at write/commit time. `from_slot(&graph, slot)` validates an
existing erased slot before wrapping it; `as_slot()`/`into_slot()` bridge to
Session initialization, checkpoint and optimizer APIs. Clones alias the same
identity. Shapes/graph ownership remain dynamically checked. `write_if` guards
one slot using existing mask semantics (zero rejects, nonzero including NaN
accepts); use `write_many_if` for an atomic mixed-slot proposal, not
a sequence of typed writes that might fail partway. These operations record
symbolic state, not arbitrary Rust side effects or live device mutations.
CPU/CUDA tests cover exact I32 counter overflow, accepted/rejected F32/I32 updates,
erased-slot validation and session-state migration with one compilation.

`StateUpdates` collects mixed typed proposals without borrowing/mutating the
graph until explicit commit:

```rust,ignore
StateUpdates::new()
    .with(&mean, &next_mean)
    .with(&steps, &next_steps)
    .commit_if(&mut graph, &finite)?;
```

`with` moves and returns the builder, without cloning previously collected
proposals. `set(&mut self, ...)` remains available for loops/conditional collection;
both forms may be mixed. Both check dtype through the sealed `StateDType` mapping; shape, graph and
duplicate-identity validation is deferred to commit. `commit` and `commit_if`
consume the group and apply the existing all-or-nothing symbolic update rules.
Repeated identities are rejected, not overwritten. Dropping the group discards
it. Candidate values are captured at `set`; they are not reread at commit, so
swaps are simultaneous. A rejected runtime condition retains versions current
at commit. Graph nodes built while preparing proposals and arbitrary host side
effects are not rolled back. There is no automatic commit on Drop or live-device
transaction. The update group adds no proc macro or shape-specialized types.

For stronger coordination, `graph.transaction()` returns a `StateTransaction`
holding an exclusive mutable borrow of that StateGraph until commit or drop:

```rust,ignore
let mut tx = graph.transaction();
let next_mean = tx.read(&mean)?.add(&delta)?;
let next_steps = tx.read(&steps)?.wrapping_add_scalar(1)?;
tx.set(&mean, &next_mean)?.set(&steps, &next_steps)?;
tx.commit_if(&finite)?;
// Or: graph.transaction().with(&mean, &next_mean)?.with(&steps, &next_steps)?.commit_if(&finite)?;
```

Declare inputs/state and prepare external values such as `delta`/`finite` before
entering the scope. Tensor expressions can still be built from values read inside.
No mutable graph access is exposed by the transaction. Reads keep seeing entry
versions because proposals are staged and outside state mutation is borrow-checked.
`set` validates the full proposed set immediately; an error removes only the new
proposal, leaving earlier proposals available for explicit commit or abandonment.
In contrast, a failed consuming `with` drops the entire scope. Commit validates
again and consumes the scope, preventing reuse. Invalid conditions and Drop leave
all symbolic slots unchanged. Compile-fail doctests check exclusive borrowing and
single consumption; host tests check entry reads, simultaneous swaps, errors and
abandonment; CPU/CUDA tests compare scoped, grouped and individual guarded updates.
This is graph-construction coordination, not a runtime lock, database transaction,
automatic host-side mutation capture or rollback of already-built tensor nodes.

`impl_state_tree!` generates named state visitation for ordinary structs:

```rust,ignore
struct ModelState { mean: State<F32>, steps: State<I32>, layers: Vec<LayerState> }
impl_state_tree!(ModelState {
    state mean => "mean",
    state steps => "steps",
    trees layers => "layers",
});
let entries = state_tree::states(&model_state)?;
let slots: Vec<_> = entries.iter().map(|e| e.slot.clone()).collect();
let session = program.session(program.zero_state(&slots)?)?;
```

Supported entries are `state`, `optional_state`, `tree`, `optional_tree` and
`trees`; typed leaves expose `as_slot`, nested objects implement the object-safe
`state_tree::StateTree` trait. Collection keeps declaration/list order, rejects
empty path segments and duplicate names, and deduplicates shared slot identities
with alias names. Slot identity is process-local; list reordering and optional
field presence can change names. Use explicit stable paths for durable schemas.
Omitted fields are not discovered: program/session layout validation must still
check completeness, membership and dtype/shape. The macro does not initialize
state, encode a checkpoint, generate updates or discover trainable parameters.
Zero initialization above is an explicit caller choice and is inappropriate for
some optimizer/RNG/model states. Generic structs can implement the trait manually;
this declarative macro adds no proc-macro dependency or native compilation.

With `rxla-weights/training`, `write_state_tree`, `save_state_tree_new` and
`Checkpoint::load_state_tree` directly accept a StateTree. They delegate to the
existing complete-state checkpoint APIs, using first-visit paths as keys and
saving/uploading shared slots only once. Missing/foreign slots or incompatible
headers are rejected by the existing preflight; writers are untouched on schema
validation failure. Save-new retains the no-overwrite publication contract;
streaming can still leave partial output on I/O or download errors. Loading returns
buffers for explicit Session creation/replacement and never mutates a live session.
Aliases are not encoded or used as fallback keys. Stable canonical names and
explicit application schema metadata remain required; use the old explicit mapping
API for migrations. Fixed inputs, host state and executable code are not included.
CPU/CUDA tests restore into a newly registered graph in reversed slot order, check
exact I32 continuation and one upload per shared identity, and reject incomplete
trees before file creation, writer changes or checkpoint payload reads.

References (`&T`/`&mut T`), `Box<T>`, `Rc<T>` and `Arc<T>` forward StateTree,
including unsized `dyn StateTree`. Wrapping a subtree does not add a path segment
or clone its slots; canonical names and aliases are unchanged. This supports
type-erased heterogeneous `Vec<Box<dyn StateTree>>` fields with the existing
`trees` macro entry and optional pointers with `optional_tree`. Optional presence
and vector indices still affect the schema. Pointer forwarding adds no locks or
Send/Sync guarantees; in particular, Arc does not make a thread-affine tree safe
to send across threads. Trees must remain acyclic for recursive visitation.

`module::Parameterized` and `module::Module` forward through the same pointer
types, including `dyn Parameterized` and `dyn Module` respectively. A boxed
parameterized model can be passed directly to optimizer discovery; Rc-owned
modules can be reused in Sequential while retaining tied parameter aliases.
Pointer wrappers add no parameter-path prefixes and no device-side dispatch:
`forward` dispatch happens during graph construction. This does not turn
arbitrary Parameterized components into single-input Module implementations.

`Sequential::default().with(layer).with(Activation::Relu).with(head)` constructs
a network without writing `Box::new` at each layer. `with` consumes and returns
the same fixed Sequential type; only its small boxing helper is generic, not the
network structure. It can be mixed with `push(Box<dyn Module>)`, which avoids
reboxing modules that are already boxed. Append order determines parameter paths,
and shape/graph compatibility is checked by `forward`, not by `with`. This adds
no initialization, compilation, execution or automatic train/eval mode.

`Residual::new(branch)` computes `branch(x) + x`;
`.with_skip(projection)` replaces the identity with an explicit skip module.
Both paths remain inside the same compiled graph and support autodiff. Output
shapes must match exactly: no implicit broadcasting, activation or normalization.
Parameter names use `branch.*` and `skip.*`; sharing an Rc-owned module between
paths preserves tied aliases and sums both gradient contributions. Residual uses
boxed modules rather than encoding the entire network in a nested generic type.
This is module composition, not a complete ResNet or Transformer implementation.
The native residual tests also train identity-skip and tied-branch linear models
for 80 SGD steps each, checking every pre-update loss and resident weight update
against independent F64 formulas. Each graph compiles once; shared parameters
receive summed gradients but only one optimizer update. CPU/CUDA checks are a
linear regression smoke test, not full residual-network training evidence.
`cargo run -p rxla-train --example train_cnn -- --residual` extends the synthetic
horizontal/vertical-line classifier with `Conv1x1 -> SiLU -> Conv1x1 + identity`
between its spatial convolution and pooling. The final branch convolution starts
at zero. Three deterministic seeds each train for 250 Momentum SGD steps, checking
all eight labels, loss reduction and learning in both residual convolutions. One
compiled training graph is reused across the seeds on CPU/CUDA. This is a small
end-to-end convolutional training check, not ResNet accuracy or throughput data.

`x.apply(&linear)?.relu()?.apply(&head)?` interleaves modules and tensor operators
without importing the Module trait just to call `forward`. `apply` borrows an
object-safe Module and delegates directly to its forward method: no new graph
boundary, parameter copy, compilation or implicit state context is introduced.
Box/Rc/Arc-owned modules use the same pointer forwarding described above. Shape
and graph errors propagate unchanged; stateful and multi-input components retain
their explicit APIs. Native tests cover gradients through nested tied modules
and parameter rebinding without recompilation.
The runnable `train_xor` example uses `Sequential::with`, trainable Linear layers
and `Tensor::apply` end-to-end while retaining explicit guarded SGD updates and
session initialization. Linear weights are `[out, in]`, unlike a raw right-hand
matmul matrix `[in, out]`; transpose existing initialization data when migrating.
Its unchanged 200-step checks require all four labels correct, loss below 0.03,
hidden-layer learning and exactly one training compilation on CPU/CUDA. It then
builds a separate inference-only graph, snapshots learned weights by canonical
parameter path and binds independent buffers. After dropping the training
session, repeated inference checks labels and all logits against independent F64
host math. The inference plan has only the feature input (no labels or dynamic
weights) and requires one additional compilation. This is a same-client in-memory
handoff, not a serialized deployment package or cross-process test.
`optim::sgd_step_if(&mut graph, &model, &loss, &rate, &mask)` provides its
explicit guarded SGD update without manually collecting gradients and writes.
All selected resident weights commit together; tied weights update once and no
momentum slots are added. Zero/-0 rejects; nonzero/NaN accepts, matching other
state masks. The mask does not skip computation, guard visible outputs or imply
finite gradients: the example deliberately retains its existing loss-finite-only
policy. Invalid masks and stale parameter versions are rejected before writes.

For candidate-weight validation or shared acceptance with RNG/statistics, use
`optim::prepare_sgd_step(&graph, &model, &loss, &rate)` to obtain an
`OptimizerUpdate` without recording writes or allocating optimizer state:

```rust,ignore
let update = optim::prepare_sgd_step(&graph, &model, &loss, &rate)?;
let accept = loss.is_finite_mask()?.mul(&update.finite_mask()?)?;
update.commit_if(&mut graph, &accept)?;
```

Dropping the candidate performs no update; commit rejects changed source state.
Candidate finiteness checks output weights, not every intermediate gradient or
unrelated state. Native tests reject an overflowing candidate despite a finite
zero loss, then retry successfully, with tied weights and no extra state slots.
Ordinary `sgd_step`/`sgd_step_if` reuse this preparation path without implicitly
adding finite checks.

`optim::prepare_sgd_gradients(&graph, &model, &gradients, &rate)` accepts explicit
gradients without autodiff, enabling clipping or accumulation without a dummy
momentum optimizer. Gradients follow unique resident parameter discovery order;
count, shape, graph and current parameter versions are checked, but same-shaped
permutations cannot be detected. Its OptimizerUpdate can be passed directly to
`AccumulationProposal::commit_with_optimizer_if` so ready windows update weights
and clear sums/count together. Native tests exercise two windows with tied weights
and a rejected NaN microbatch, checking every weight, sum and I32 count. Only
model weights plus accumulator sums/count are initialized—no optimizer slots.

For checkpoints produced by `write_module`/`serialize_module`,
`Checkpoint::load_named_module_for_program(&program, &model)` removes identity
mapping boilerplate. First-visit canonical parameter paths are checkpoint keys;
all tied aliases resolve to that same key and upload once. The caller's model is
discovered once and a captured selection feeds the existing full program/type/
header preflight. Bind returned inputs and initialize resident states explicitly;
optimizer/non-parameter state and application metadata are still separate.
Canonical-name changes require the existing explicit-mapping API, not heuristic
alias lookup. The native export test performs a disk roundtrip from a trained
mixed-storage module into independent inference and rejects foreign parameters
before payload reads/uploads.

`save_module_new(path, &session, &model, metadata)` streams those same canonical
parameters to a newly published file, reusing `save_buffers_new`'s no-overwrite
and durability contract. Unlike `serialize_module`, it does not retain the entire
encoded model in host memory; peak payload storage still includes the largest
parameter. Module names and session bindings are checked before file creation.
An existing destination is preserved, including if training has advanced since
the first save. Only module parameters are exported—not optimizer/RNG state or
executable code. Use complete state checkpoints for exact training resume.
The mixed-storage native test also streams BF16 fixed weights and F32 resident
parameters with schema metadata, verifies raw BF16 bytes with the upstream
SafeTensors reader, preserves an existing destination on a second save, and
restores by canonical name into a reordered graph. Explicit metadata validation
rejects the wrong version before payload I/O; two unique parameters require two
uploads (12 bytes), and inference survives dropping the source session. This
does not imply BF16 training or automatic metadata-policy enforcement.

### Explicit heterogeneous execution

Native `PJRT_Buffer_CopyToDevice` is not a cross-plugin replacement for host
staging: the checked-in [PJRT header](../vendor/xla/pjrt/c/pjrt_c_api.h) restricts it
to another device within the same client and specifies an error for copying to
the original device. `CopyToMemory` has the analogous same-client restriction.
Our CPU and CUDA plugins create separate clients. Supporting a native same-client
multi-device path requires coherent device selection for buffers and executable
placement. `Client::load` / `load_with_options` select the first addressable
device for uploads. Native callers can use unsafe
`Client::load_on_device(path, options, addressable_device_index)` to fix a
different upload device at creation. The index refers to the plugin's device
list, not a global ID/CUDA ordinal. Raw `compile_hlo` options must assign the
selected device ID reported by `info()`. The high-level Tensor `Compiler` now
emits a one-replica, one-partition assignment to that selected global device ID.
This also applies to direct Tensor graph compilation; it does not shard a graph.
Every call creates a separate native client/allocator, not a shared device view;
on GPUs this can reserve substantial memory even when selecting one device.
Do not pass a device pointer from another client/plugin to these native calls or
silently substitute them for `copy_to_client_via_host`.
Compiled and deserialized executables are checked against the client's
addressable-device set. A single-device executable must match the selected
default device; a multi-device executable uses `execute_sharded` with one input
and output list per executable device. `buffer_f32_on_device` and the typed
variants upload shards through the same native client. Empty or null device
lists and foreign device identities are rejected, with loaded executables owned
for cleanup on failure.

`crates/rxla-core/examples/cpu_placement.rs` exercises this against real native compiler
output, not only synthetic API tables. With two logical CPU devices it rejects a
program assigned only to the non-selected device, then compiles and executes a
two-replica program with different inputs on each device. It also compiles,
executes, serializes and restores a selected-device program on the same client.
It runs that sequence for indices zero and one, rejects an out-of-range index,
and then repeats index one to check that failed selection did not poison loading.
High-level Tensor compilation executes twice with one memory-cache hit on each
device. With `--features disk-cache`, a fresh Compiler restores the same graph
without a backend compilation, using an explicit per-device cache namespace.
Restoring device zero's native artifact on device one is required to fail.
Run in a separate process from other CPU tests (the flag changes device topology):

```sh
PJRT_PLUGIN_PATH=/trusted/path/libzml_cpu.so \
XLA_FLAGS=--xla_force_host_platform_device_count=2 \
cargo run --offline -p rxla-core --example cpu_placement
```

This verifies explicit CPU device selection, replicated multi-device execution,
and placement rejection. It does not claim multi-GPU or SPMD partitioned
execution and remains outside the ordinary single-device gate.

The reusable opt-in gate runs both default and `disk-cache` variants, including
shared-directory device separation and scoped trimming:

```sh
PJRT_CPU_PLUGIN_PATH=/trusted/path/libzml_cpu.so \
  sh scripts/check-xla-cpu-placement.sh --offline
```

It deliberately replaces `XLA_FLAGS` only inside its child processes with
`--xla_force_host_platform_device_count=2`; the invoking shell remains unchanged.
Relative plugin paths are resolved before entering the workspace. Missing plugin
files and unsupported arguments fail before Cargo. The native example must also
identify a CPU backend with at least two addressable devices. No plugin downloads,
binding regeneration or same-client multi-device execution are added by the gate.

The opt-in [placement benchmark](HETEROGENEOUS-BENCHMARK.md) compares whole CPU,
whole GPU and CPU→GPU→CPU execution with identical host input/output boundaries.
Two local release runs favor whole CPU on small shapes and whole GPU on the
largest tested shape; splitting is slower than whole GPU in every case. This
supports explicit workload-based placement, not automatic per-operator splitting.

`Buffer::copy_to_client_via_host(&destination)` explicitly downloads and uploads
a new buffer on the destination client's selected device, preserving shape and
F32/I32/BF16 dtype. BF16 is transferred as raw bits. Even same-client copies use
host staging and return a new buffer. The source is borrowed, not consumed;
failure does not replace it or mutate a session. Temporary host storage is one
full tensor; this is not peer-to-peer copying, zero-copy, async transfer or a
memory quota. Native single-client tests cover scalar negative zero, empty
shapes, integer extremes and BF16 bit patterns. The heterogeneous example checks
I32 and BF16 CPU-to-CUDA-to-CPU round trips as well as the F32 compute pipeline.

Use `copy_to_client_via_host_with_limit(&destination, max_host_bytes)` when a
transfer must fit a host payload budget. It queries the native download size and
rejects over-budget copies before allocating/downloading the payload or uploading
the destination. Exact limits are accepted; zero permits empty tensors. This is
per-copy payload accounting, not a limit on plugin memory, allocator overhead,
device memory or aggregate concurrent transfers. The unlimited convenience method
retains its existing behavior. The compute pipeline uses explicit eight-byte
limits at both stage boundaries; native tests check exact and one-byte-short
limits for all supported dtypes, including a zero-byte empty copy.
A plugin-free synthetic API test reports an impossible native payload size and
verifies that all three dtypes reject it with zero payload-download and upload
calls. This also checks that the budget uses the native download requirement,
not merely the logical shape's byte count.
Download vectors now use fallible capacity reservation: an impossible capacity
or allocator-reported failure returns a `host download allocation` error before
payload transfer. The synthetic test also covers an unlimited copy whose native
size exceeds Vec's supported capacity. This cannot prevent OS overcommit/OOM
killing, plugin-internal allocation failures or all process-wide allocation
failures; explicit payload budgets remain important.

`crates/rxla-core/examples/heterogeneous.rs` loads trusted CPU and CUDA plugins into one
process and runs three separately compiled graphs: CPU affine preprocessing,
CUDA matmul and CPU reduction. Seventeen requests at each retained-task capacity
(one and three) match an independent host reference and each other bit-for-bit;
The same prepared matmul snapshot is additionally compiled for CPU and its
output checked against CUDA for every request. Both runs therefore share three
CPU compilations (including this reference executable) and one GPU compilation.
Repeated prepared lookups reuse each client's own handle without increasing
compile time; the CPU and CUDA executables are distinct. Exact agreement is for
these small dyadic inputs, not a general cross-backend floating-point guarantee. Each
inter-stage value is downloaded then uploaded to the destination client. Foreign
CPU buffers are explicitly rejected by the GPU executable's input validation.
The bounded single-thread loop submits GPU jobs before preparing later CPU
inputs; pending handles retain their inputs after local buffers are dropped.
FIFO draining may cause head-of-line blocking. Capacity counts retained jobs,
not necessarily unfinished GPU work or a byte quota. Transfers/postprocessing
remain synchronous, and native submission may block. This is a small correctness
example, not OCR/YOLO performance, proven CPU/GPU overlap, automatic partitioning,
peer-to-peer copies or a reusable pipeline scheduler.

With the CUDA dependencies configured as in [CUDA validation](CUDA-validation.md):

```sh
PJRT_CPU_PLUGIN_PATH=/trusted/path/libzml_cpu.so \
PJRT_CUDA_PLUGIN_PATH=/trusted/path/libzml_cuda.so \
cargo run --offline -p rxla-core --example heterogeneous
```

Both plugins must be trusted and compatible with this process and host. The
example checks backend metadata instead of allowing two CPU clients to masquerade
as heterogeneous execution. It is opt-in and not run by the single-plugin gates.
For the complete dedicated check, replace the cargo command above with
`sh scripts/check-xla-heterogeneous.sh --offline` from the Rust workspace root.
It validates both plugin paths before running Cargo, executes the mixed-backend
example, then runs scalar/empty/payload host-transfer tests on each backend
sequentially. CUDA shared-library dependencies must already be provisioned;
the script does not download plugins or regenerate bindings. Do not run it beside
another GPU check because the default allocator may reserve most device memory.

The same gate runs `heterogeneous_state`: one pruned PreparedStateGraph is compiled
separately for CPU and CUDA, and a quiescent session's complete F32/I32 state is copied to
the other client with `Session::copy_state_to_client_via_host`. Continuation across CPU -> CUDA
-> CPU matches uninterrupted CPU execution and explicit host references,
including I32 wraparound and rejected updates. Remote execution leaves the source
session unchanged, and the returned CPU session works after dropping CUDA owners.
The source graph and symbolic tensors are dropped before either compilation;
both programs retain the same compact visible input mapping `[1]`. Repeated
prepared compilation hits each backend's own cache without increasing compile
time. The prepared transition is then dropped before session execution, checking
that native programs retain their code and schema independently of it.
This is explicit copied state, not cross-backend executable reuse, automatic
fallback, in-flight live migration or a general floating-point equivalence claim.
The example retains the same graph-local slot identities on both compiled plans;
independently reconstructed graphs require an explicit named-state mapping.

The gate also runs `rxla-weights --features training --example worker_state`.
It uses `write_state_tree` into a `Vec<u8>` and `Checkpoint::new(Cursor::new(bytes))`
with `load_state_tree` to continue across CPU -> CUDA -> CPU worker threads.
Each source worker is shut down and joined before the next starts; only host
checkpoint bytes cross threads. Each destination independently builds and compiles
its graph, validates explicit schema metadata, and restores by canonical names.
The CUDA stage reverses slot registration order to verify that numeric slot IDs
are not the serialization contract. All nine updates are checked against a host
reference, including I32 wraparound. No files or new serialization format are
needed. This stages a complete checkpoint in RAM and does not provide live
migration, shared native handles, memory quotas or concurrent-device speedup.
The tree also carries Threefry keys and its 64-bit counter. Each step emits three
two-word random blocks; all nine steps match a separate uninterrupted CPU worker
exactly, including low-word counter carry and both restored counter words. The
counter and numeric state updates share the RNG acceptance predicate. The schema
metadata is now `worker-counter-rng/v2`; this example does not silently upgrade
old counter-only checkpoints or establish general floating-point replay across
backends.

`random::ThreefryState` implements StateTree with local names `key0`, `key1`,
`counter_low`, `counter_high`. Include it with `tree rng => "rng"` rather than
manually listing its four slots. Cloned RNG handles remain aliases and are saved
once. This does not pick a seed, change reservation/commit policy, or guarantee
independent streams. Initialize keys/counters explicitly and validate application
schema metadata when restoring. The named-tree checkpoint test resumes a three-
block draw/accumulation graph with reordered state registration, checks both I32
output words and all saved state against uninterrupted execution, and checks the
counter against host arithmetic across low-word carry and rejected updates.
CPU and CUDA runs use new graphs on the same client; cross-process/hardware
portability is not established by this test.

`optim::Adam` (including its AdamW configuration) implements StateTree for all
selected resident parameters, first/second moments and the two shared beta powers.
Canonical paths are `parameters.<parameter-path>.value`, `.m`, `.v`, plus
`beta1_power`/`beta2_power`; parameter aliases also alias their moment slots.
Paths are captured once at optimizer construction, not rediscovered from a mutated
model. The state collector saves each tied identity once. Frozen input weights,
unselected parameters, other model/RNG state and optimizer options are excluded.
Compose an outer tree for additional state; persist/validate the optimizer policy,
learning-rate schedule and application schema explicitly. Do not zero all Adam
state: use initial_state for moments/powers and initialize parameters separately.
The AdamW checkpoint test rebuilds a tied-parameter training graph and checks
subsequent losses and every weight/moment/power bit against uninterrupted runs,
including rejected steps. This is a small same-client correctness test, not a
distributed trainer or automatic hyperparameter/configuration restore.

`optim::MomentumSgd` implements the same StateTree contract with paths
`parameters.<parameter-path>.value` and `.velocity`. Tied parameter paths alias
both slots; discovery happens once at construction. Only selected resident
weights and their velocities are included, not other model state or momentum/
learning-rate configuration. Initialize velocities with `zero_state` and weights
separately, and persist policy metadata explicitly. The native checkpoint test
restores tied crates/rxla-weights/velocities into a new graph, checks accepted/rejected-step
continuation against both uninterrupted execution and independent F64 arithmetic,
and verifies that aliases require only two uploads.

`optim::GradientAccumulator` implements StateTree for
`parameters.<parameter-path>.sum`, I32 `count`, and optional F32 `weight_sum`.
Parameter aliases share sums. Model weights and optimizer state are deliberately
excluded: compose it with an optimizer in an outer tree for complete training
checkpoints. Preserve window size and weighted/unweighted policy in application
metadata. The native test restores a partial three-microbatch window into a graph
with reordered registration, rejects a NaN microbatch, and completes successive
windows. Both weighting modes match uninterrupted state bit-for-bit and host
references for counts, sums, weights and total mass on CPU/CUDA (same client).

`Tensor::unbind(axis)` and `Index::unbind(axis)` remove a selected axis and
return one graph value per index, in ascending order. For `[batch, features]`,
`x.unbind(0)?` produces `batch` tensors of shape `[features]`; unbinding a vector
produces scalars. An empty selected axis returns an empty Vec, while scalar inputs
and absent axes are errors. Other zero-size axes are preserved. This is static
graph construction (slice plus reshape), not a host download or a guaranteed
zero-copy view. F32 derivatives propagate through the slices and accumulate when
reused; I32 keeps exact bits and is not differentiable. `Tensor::stack(&parts,
axis)` reverses the shape operation for nonempty parts. For an empty selected
axis, there are no parts from which stack could recover the original shape.
Native tests cover all axes, scalar/empty outputs, exact I32 extremes, stack
roundtrips, and first/second derivatives with shared and unused slices.

`Index::stack(&parts, axis)` and `Index::concatenate(&parts, axis)` also reassemble
I32 data without F32 conversion, preserving RNG words and large token/index values.
Stack inserts an axis and requires equal input shapes; concatenate joins an
existing axis and requires matching other dimensions. Both require nonempty
input lists from one graph. Scalars may be stacked but not concatenated; empty
tensors are allowed and concatenated axis lengths use checked addition.
These are nondifferentiable graph operations, not in-place buffer mutations.
Native tests exercise every axis, stack/unbind roundtrips, singleton lists and
exact I32 extremes, including values beyond F32's exact integer range.

Selected CUDA execution and stateful training checks now pass on an RTX 5080;
see [pinned dependencies, reproduction commands and limits](CUDA-validation.md).
This is separate from the CPU development gate and is not a GPU benchmark.
For explicit plugin-specific creation parameters, use `Client::load_with_options`
and the typed `ClientOptions::set` builder; the `cuda_clients` example demonstrates
two live GPU clients with BFC growth enabled instead of default preallocation.

`Buffer::copy_to<T: Element>` downloads into a caller-owned typed slice, so
repeated result retrieval can reuse host storage instead of allocating a new
output Vec. It requires the exact dtype and element count, rejects mismatches
before writing, and waits for native transfer completion before returning.
BF16 uses `half::bf16`, which preserves every encoding without passing through
F32. Native transfer failures may partially modify the destination. This is a
synchronous API, not pinned-memory or async transfer, and does not establish a
latency improvement over `to_vec::<T>()`.

`Executable::submit(&[&buffer])` returns an owned `PendingExecution`;
`pending.wait()` waits for device completion and returns the output buffers or
an execution error. Multiple submissions can be retained and waited in a different
order. Each pending handle retains shared native ownership of the executable,
inputs, outputs and client, so original handles can be dropped before waiting.
Inputs remain non-donatable. Dropping a pending handle waits and discards its
outputs/errors; it does not cancel execution. Forgetting it leaks these owners
rather than freeing in-flight resources. Submission itself may block in the plugin.
This is not a Future or a Send/Sync handle and does not prove kernel overlap or
better throughput. `execute` remains synchronous (submit then wait), and Session
state commits remain synchronous. Shared ownership and submission bookkeeping add
host allocations/reference counts; no zero-overhead claim is made.
The `submit` native tests cover reverse waits, shared inputs, dropping original
owners, signature rejection, implicit Drop waits and zero-input execution on CPU
and the selected CUDA gate.

`PendingExecution::is_ready()` queries device completion without waiting or
consuming the handle. `true` means completed, not necessarily successful: use
`wait()` to retrieve execution errors and outputs. A query error does not consume
the task or release its owners. Plugins without this optional capability can still
use submit/wait. No wakeup is registered; use bounded/backed-off polling or check
between other work rather than busy-spinning on an executor thread. Readiness
queries do not turn this thread-affine handle into a Future.

`PendingExecution::outputs()` borrows the output buffer handles immediately,
without a host completion wait. Feed them into another executable on the same
client to build a device-data pipeline:

```rust,ignore
let first = encoder.submit(&[&input])?;
let second = head.submit(&[&first.outputs()[0]])?;
let result = second.wait()?;
first.wait()?; // observe the upstream task's status as well
```

PJRT tracks input-buffer dependencies; a handle is not proof of ready or valid
data. Downstream submissions retain their inputs independently, and host downloads
still synchronize. Keep upstream pending handles until you want to wait/check
their statuses: dropping one still waits and discards its errors. This does not
fuse separately compiled graphs, commit Session state, or promise kernel overlap.
Native tests cover a three-stage mixed F32/I32 chain and shared intermediate
fan-out without intermediate host waits/downloads, including exact I32 overflow.
For a release-mode same-device comparison against split synchronous execution
and a whole graph, see [pipeline measurements](PIPELINE-BENCHMARK.md). The measured
small example favored whole-graph execution; submit is not a reason to split a
graph that can otherwise be compiled together.

Run `cargo run -p rxla-core --example inflight` with a trusted
`PJRT_PLUGIN_PATH` for a bounded single-thread inference scheduling example.
It compiles once, shares resident weights across 17 requests, retains at most
three pending tasks, prefers ready tasks, and checks every result by request ID.
If no task is ready it waits for the oldest instead of busy-polling; head-of-line
blocking remains possible. Per-request host inputs are uploaded synchronously,
and outputs are downloaded before admitting more work. The bound counts retained
tasks, not GPU memory bytes or simultaneously running kernels; result storage
grows with the request count (bounded to 1024 in the example). It requires the
optional readiness API and is neither an async executor nor a throughput claim.
CPU/CUDA native example tests cover zero requests, capacity one, partial final
windows and fewer requests than slots; invalid limits fail before plugin loading.

Device noise includes `random::normal_f32_from_bits([&word0, &word1])` and
`ThreefrySequence::normal_f32(shape)`. Both compose Box–Muller from existing
tensor operations; no new native kernel or host RNG is involved. The sequence
reserves one block per output element (scalar one, empty zero), uses both words,
and commits only through the existing explicit commit API. Invalid construction
does not advance its proposed cursor. For location/scale noise, compose
`mean + stddev * noise`; no parameter validation or state commit is implicit.
The high-23-bit midpoint radius grid avoids log(0), but bounds the radius to
about 5.77: this is an approximate finite-grid normal distribution, not exact
Gaussian tails or cross-backend bitwise reproducibility. The sine companion is
discarded. Boundary checks use an independent F64 formula; fixed-seed moment
checks are smoke tests, not a statistical quality certification.

The normal-noise native tests also build broadcasted
`sample = mean + exp(log_std) * noise` and a half-squared-error sum. First
derivatives and both summed-gradient Hessian rows match host F64 formulas
conditioned on the returned noise. This verifies pathwise derivatives through
the location/scale parameters, not differentiation through integer RNG state
or an exact expectation gradient. The same compiled program returns noise,
samples, loss and derivatives; a rejected step replays them exactly, accepted
steps consume only the six forward-draw blocks, and the next accepted step
uses new noise. No extra RNG draw is constructed by autodiff.

`value.normal_log_prob(&mean, &log_std)` evaluates elementwise continuous
Gaussian log density, with log standard deviation rather than log variance.
All three operands must have the same graph/shape and are differentiable.
Broadcast explicitly, sum action/event dimensions explicitly, and negate for
negative log likelihood. For a detached-action score-function objective use
`action.detach()?.normal_log_prob(&mean, &log_std)?`; retaining the action graph
instead preserves its pathwise dependence. Native tests distinguish these two
derivatives and compare values, first derivatives and diagonal second derivatives
against F64 formulas, including scalar and empty tensors. Runtime nonfinite
values/overflow are not clamped or repaired. No tanh Jacobian correction, RNG
draw or state update is implicit. This continuous density is not the probability
mass function of the finite-grid normal sampler.

The `train_normal` example also connects training to inference: after fitting,
it builds a separate inference graph with only observations, mean and log-standard
deviation inputs, then serializes that executable. All old device resources are
released before a fresh client restores the trusted bytes and uploads just the
two learned parameters. Three new observations match an independent F64 density
reference, and restored outputs match pre-export outputs bit-for-bit on the tested
CPU and CUDA plugins. The workflow has one training compilation and one inference
compilation; restore makes no explicit compiler call. This is an in-memory,
same-process/fresh-client handoff, not a disk checkpoint package, cross-process
deployment or a portability guarantee. Optimizer state is not an inference input.
The existing CPU example test and CUDA example run cover this handoff.

The `train_normal` example fits a two-dimensional diagonal Gaussian to four
fixed observations using joint negative log likelihood (sum event dimensions,
mean over the batch). It checks all 400 Adam steps against independent F64
loss/gradient/parameter/moment/beta-power calculations, converging to mean
`[2, -1]` and standard deviation `[1, 2]`. A NaN batch is rejected with every
model/optimizer state bit unchanged, followed by an in-process state move into
a fresh session. One training compilation serves all training calls. CPU and selected CUDA gates
include this example. It uses fixed data, not sampled actions or an RL environment;
state transfer is not a disk checkpoint, and reference downloads make it unsuitable
as a throughput benchmark. Its explicit finite-loss guard is not a universal
finite-gradient policy for arbitrary models.

```sh
PJRT_PLUGIN_PATH=/path/to/trusted/plugin.so cargo run --offline \
  --manifest-path Cargo.toml \
  -p rxla-train --example train_normal
```

## Repeatable development checks

From the standalone repository root:

```sh
# No native plugin is loaded; reports native tests as ignored.
sh scripts/check.sh --offline
# Explicit CUDA gate; first configure native dependencies from CUDA-validation.md.
# PJRT_PLUGIN_PATH=/path/to/trusted/libzml_cuda.so sh scripts/check-cuda.sh --offline
# Explicit opt-in to loading/executing a trusted compatible CPU plugin.
PJRT_PLUGIN_PATH=/path/to/trusted/libzml_cpu.so \
  sh scripts/check.sh --offline --with-plugin
```

The script checks default and `disk-cache` configurations, backend tests/doctests,
the OCR numerical-gate host tests, all-target Clippy and the independent downstream
build. In plugin mode it also runs the independent consumer's three stateful
calls, checking its public-API integration and single compilation. It uses locked dependency resolution and intentionally nonexistent
IREE/LLVM/libclang paths, without building legacy packages or regenerating bindings.
Omit `--offline` if dependency downloads are needed. Relative plugin paths are
resolved before switching directories; file existence does not establish trust.

Host mode unsets an inherited `PJRT_PLUGIN_PATH` to avoid accidentally opting into
native execution. Plugin mode includes ignored tests, including child-process
artifact/cache tests. This is not full-model OCR/TinyLlama validation, a GPU test
matrix, binding regeneration verification or a benchmark. Those retain their
separate documented commands. The script does not enable the optimized-loader
development configuration or silently turn skipped tests into passed tests.

## Crates

- `rxla`: small application facade re-exporting the core tensor, NN, IR and
  PJRT APIs. Models, training, checkpoints and ONNX remain separate crates.
- `rxla-xla-proto`: upstream protobuf definitions and generated prost messages.
- `rxla-pjrt`: dynamic plugin loading, synchronous owned client/buffer/executable
  handles. Persistent handles use `Arc` and follow PJRT's thread-safe native
  contract; in-flight event owners remain thread-affine.
- `rxla-ir`: typed Pliron SSA, verification and StableHLO lowering.
- `rxla-nn`: scoped parameter effects and shape-inferred layer builders.
- `rxla-cache`: backend-agnostic atomic file artifact storage.
- `rxla-core`: tensor construction, transforms, state and PJRT execution;
  it has no dependency on model definitions.
- `rxla-train`: functional optimizer transforms for parameter-effect models.
- `rxla-models`: reusable effect-based inference models.
- `xtask`: maintainer-only bindings/protobuf generation.
- `rxla-weights`: optional seek-based safetensors reader and F32 weight uploads;
  see [crates/rxla-safetensors/README.md](../crates/rxla-safetensors/README.md). Not a dependency of `rxla-core`.

Tensor operations build typed Pliron SSA and are verified before lowering.
Compilation exports legal StableHLO MLIR through typed dialect conversion; XLA
protobufs remain control-plane and post-compilation diagnostic types, not a
second frontend computation representation.

`Tensor::dot_general(rhs, lhs_contract, rhs_contract, lhs_batch, rhs_batch)`
expresses paired contraction/batch axes without manual layout plumbing. Dimensions
must match exactly (no implicit broadcasting); axes must be unique and batch
and contraction axes disjoint within each operand. Output order is the listed
batch axes, remaining lhs axes in original order, then remaining rhs axes.
For example, `[B,M,K]` and `[K,B,N]` use `(&[2], &[0], &[0], &[1])` and produce
`[B,M,N]`. `tensordot(rhs, lhs_axes, rhs_axes)` supplies no batch axes; empty axis
lists form an outer product, and contracting every axis produces a scalar.
Implementation composes transpose, reshape and batched matmul, preserving its
HIGHEST F32 operand precision and higher-order derivatives without a new native
dependency or code generator. Backend layout/fusion is not guaranteed. CPU/CUDA
tests compare batched values, both operand gradients and a second derivative
against dense F64 reference calculations, plus multiple-axis, scalar, outer and
empty-contraction cases. This is not a contraction-order planner.

For label-based notation, `lhs.einsum("bmk,kbn->bmn", &rhs)` wraps these
operations in a two-operand parser. Outputs must be explicit after `->`;
case-sensitive ASCII letter labels identify axes, and ASCII whitespace is ignored.
Repeated input labels select diagonals (`"ii,i->i"`), omitted labels are summed,
and output labels must be unique and occur in an input. Scalars use empty label
lists (`"i,->i"` or `",->"`). Labels with the same name must have exactly equal
dimensions, including repeats within one operand: there is no size-one implicit
broadcasting. Ellipses, implicit output notation and more than two operands are
rejected rather than guessed.

The parser runs when constructing the graph, not per execution. It composes
diagonal extraction, one-sided reductions, `dot_general` and output transpose,
so existing HIGHEST contraction precision and AD rules apply without adding a
native dependency. This does not implement a multi-operand optimization planner
or guarantee a particular floating-point summation order. Native CPU/CUDA tests
compare ten equations to independent dense F64 label enumeration and check both
operand gradients plus a second derivative for a diagonal contraction.
An empty summed dimension yields zeros without evaluating products, including
when the other operand contains NaN/Inf. A regression covers both operand orders
and zero gradients: factoring this case into an empty sum times the other operand
would incorrectly produce NaNs. This does not promise identical nonfinite behavior
for every nonempty algebraic rearrangement or backend optimization.

The former `train_bilinear` example demonstrated a user-defined
trainable operator assembled entirely in Rust:

```rust,ignore
// left [B,I], weight [O,I,J], right [B,J] -> prediction [B,O]
left.einsum("bi,oij->boj", weight.tensor())?
    .einsum("boj,bj->bo", right)?
```

Its small model registers a resident weight through `impl_parameterized!`, uses
ordinary automatic differentiation and `sgd_step`, and initializes explicit zero
state with `StateProgram::zero_state`. One compiled session runs 100 updates.
Every step checks predictions, loss, weight gradients and updated state against
independent dense F64 loops; the final weights must recover the synthetic target.
Both CPU and CUDA development scripts include it. This demonstrates the supported
**composition** extension path, not an external native kernel/custom-VJP interface,
LLM training, or a throughput claim.

```sh
PJRT_PLUGIN_PATH=/path/to/trusted/plugin.so cargo run --offline \
  -p rxla-train --example train_bilinear
```

For reuse, `module::Bilinear` provides `new`, `trainable` and `from_parameters`,
with `[out,left,right]` weights and optional `[out]` bias. Like Linear, `new`
registers input-backed parameters and `trainable` registers resident parameters;
neither implicitly initializes weight values. `weight()`/`bias()` expose the
identities, and its `Parameterized` implementation visits stable `weight`/`bias`
names for optimizers and checkpoint traversal.

```rust,ignore
let layer = rxla_train::module::Bilinear::trainable(&mut graph, 16, 8, 4, true)?;
let output = layer.forward(&left, &right)?; // [...,16], [...,8] -> [...,4]
```

Leading dimensions must match exactly; vectors and empty batches are permitted,
with no implicit broadcasting. Cross-graph parameters/inputs and incompatible
feature dimensions are rejected. Because it accepts two inputs, `forward` is an
inherent method, not an implementation of the unary `Module` trait. Native tests
check multi-axis batch output, bias and SGD updates to both parameter groups;
the earlier example remains a demonstration of writing the composition yourself.

The weights crate's `bilinear_roundtrip` regression also exercises the complete
training-to-inference parameter path: save after one SGD update, inspect exact
F32 shapes/bytes with the independent SafeTensors reader, continue training the
source, then load the snapshot into an independently constructed input-backed
Bilinear graph. CPU/CUDA runs reproduce the saved model's unbatched output exactly.
Swapping the weight/bias checkpoint mapping is rejected before payload reads or
uploads. This verifies parameter snapshots, not serialization of the optimizer
or a portable compiled executable.

`tensor.diagonal(offset, axis1, axis2)` extracts a diagonal across any two distinct
axes and places the diagonal dimension last, after remaining axes in their original
order. Positive offsets move along axis2; negative offsets along axis1. For example,
`[B,M,N].diagonal(1, 1, 2)` returns `[B,min(M,N-1)]` when N is positive.
`tensor.trace(offset, axis1, axis2)` sums that final dimension, returning `[B]` in
this example. Offsets beyond either matrix edge produce an empty diagonal and a
zero trace; even `i64::MIN` is handled without signed-negation overflow. Distinct
valid axes are required. Empty matrix dimensions are supported.

These operations compose transpose, flattening and a strided slice, rather than
generating one graph node per diagonal element or allocating a dense mask. They
are symbolic values, not mutable/aliased memory views, and backend layout/copy
choices are not guaranteed. Existing slice AD supplies first and higher-order
derivatives. Native CPU/CUDA tests cover nonadjacent/reversed axes, signed offsets,
batched values/traces, empty dimensions, sparse gradients and second derivatives.

The inverse construction is `vector.diag_embed(offset)`: `[...,N]` becomes
`[...,M,M]`, where `M=N+abs(offset)`. Matrix axes are appended; transpose the
result explicitly if another order is needed. Positive offsets place values above
the main diagonal, negative offsets below. For example `[a,b].diag_embed(1)` is
`[[0,a,0],[0,0,b],[0,0,0]]`. Scalars and overflowing dimensions are rejected;
an empty vector with offset 2 yields a zero 2x2 matrix. This operation allocates
a dense matrix, not a sparse representation.

Implementation uses zero padding, reshape and slice, avoiding multiplication by
an identity mask: NaN/Inf diagonal values therefore do not contaminate off-diagonal
zeros. No new dependency or native operator is introduced. CPU/CUDA tests cover
batched main/positive/negative diagonals, nonfinite values, empty inputs, first
and second derivatives. Backend physical allocation/fusion is still its decision.

Before lowering, a pure reachability pass removes unused computation branches and
densely renumbers a compilation snapshot, preserving output order and all declared
parameters (including unused ones). The original graph and tensor handles remain
unchanged. Removing dead computation branches therefore also stabilizes cache keys
against irrelevant graph additions. The pass is iterative for deep unrolled graphs.

`Client::memory_stats()` separately provides owned allocator diagnostics for the
selected device. `bytes_in_use` is mandatory on success; optional counters use
`Option<i64>` (including peak usage, reserved bytes and allocator pool bytes).
Missing API slots and unsupported backends return errors, never fake zero usage.
The pinned CPU plugin reports this operation unsupported; CUDA tests check a
completed 4 MiB upload increases active usage and peaks cover current usage.
Queries do not reset peaks or synchronize unrelated outstanding work. Counters
are backend allocator diagnostics, not process RSS or total board VRAM, and
pool/reserved memory must not be confused with live tensor allocations. See the
[PJRT C API contract](https://github.com/openxla/xla/blob/main/xla/crates/rxla-pjrt/c/pjrt_c_api.h).
TinyLlama reports snapshots before/after weight loading, after compilation and
after execution, outside timed intervals; unsupported diagnostics are recorded
without preventing inference. Its after-weight snapshot also includes KV state.

The low-level PJRT crate recognizes `DType::BF16`; `half::bf16` implements its
sealed `Element` trait, so ordinary `client.buffer(shape, &[bf16])`,
`buffer.to_vec::<bf16>()`, and `buffer.copy_to(&mut [bf16])` preserve native
storage without rounding through F32. Raw encodings remain available explicitly
through `bf16::from_bits` and `bf16::to_bits` at serialization/test boundaries.
`rxla-core` directly re-exports `half::{f16, bf16}` and uses those types for
`TensorElement`; they are not RXLA bit wrappers and do not belong to the PJRT
backend layer. Consequently `Tensor::from_slice` and `TensorBuilder::from_vec`
accept native half values while still checking the explicitly declared dtype.
Native tests round-trip all 65536 bit patterns (including NaN payloads), scalars
and empty tensors, reject wrong-type reads and invalid shapes, and verify buffers
keep their client alive. CPU and selected CUDA gates include these tests.

The safe Tensor arithmetic/AD remains F32 (with separate I32 indexing), but the trusted external HLO wrapper
now accepts BF16 host arrays and preserves their input metadata in serialized
executables and the disk cache. BF16 state checkpoint loading remains explicitly
unsupported rather than silently converted (safe StateGraph has no BF16 slots).
Existing F16/BF16 *file-to-F32*
weight loading is unchanged. Adding a public DType variant also means downstream
exhaustive matches must handle BF16. No model mixed-precision speedup or reduced
model memory consumption has been measured by these transfer tests.

For BF16 weight files, `Checkpoint::read_bf16(name)` returns a `HostBf16` with
shape and native `Vec<half::bf16>` values; `upload_bf16(client, name)` uploads
that representation directly. F16/F32/I32 files are rejected by these methods, not
implicitly cast. Normal `read_f32`/`upload_f32` conversion behavior is unchanged.
`write_buffers` and `save_buffers_new` now preserve BF16 buffers alongside F32
and I32 in the same safetensors file. BF16 offsets and transfer statistics count
two bytes per element. Export retains no-clobber publication and streaming
little-endian encoding with a 64 KiB scratch buffer.
Host tests check exact special-value encodings and rejection before payload I/O;
CPU/CUDA tests round-trip all BF16 bit patterns through a mixed checkpoint,
including scalar/empty tensors and NaN payloads, and verify typed reads,
byte counts and no-overwrite behavior. No dependency or binding generation was
added. This is weight/buffer I/O, not a BF16 optimizer or safe mixed-precision AD.

For frozen weights, `Tracer::input_bf16_as_f32(shape)` and the corresponding
`StateGraph` method declare a BF16 input and return an explicit HLO conversion
to an ordinary F32 Tensor. Bind `upload_bf16` buffers directly; no host F32
expansion is needed. `Session::bind_inputs` can retain these frozen buffers while
F32 trainable parameters update. The converted Tensor is not a trainable input
leaf: requesting its gradient is rejected, and differentiation stops at the
BF16 storage boundary. Gradients of other F32 parameters can still depend on
the converted weights. This is not an implicit straight-through estimator or
BF16 arithmetic. Backend conversion fusion/materialization and peak device
memory savings are not guaranteed. Native tests cover matmul values, first and
second F32 derivatives, wrong input dtype, scalar/empty conversion, and resident
F32 updates with unchanged BF16 bindings.

`StateGraph::parameter_bf16_as_f32(shape)` adds the same frozen conversion with
parameter identity, so existing `Linear`, `Embedding`, and normalization modules
can consume it without numeric binding indices. `bind_parameters` validates BF16
storage; `StateProgram::parameter_type` reports BF16, while `Parameter::tensor()`
is F32. Clones alias one input. Failed wrong-dtype or duplicate bindings preserve
the previous binding. These inputs are not trainable state slots.
`Parameter::storage_dtype()` exposes the runtime buffer type without requiring
a compiled plan. Module checkpoint loaders honor it: F32 parameters retain
F32/F16/BF16-to-F32 conversion, while BF16 parameters require BF16 files and
upload exact bits. Full mapping, alias, shape and dtype preflight still precedes
payload reads. `load_module_for_program` additionally validates retained parameter
identities and compiled storage types. Mixed frozen BF16/trainable F32 modules
load through `load_module_parts` or `load_module_for_program`, with one upload
per shared parameter; module export preserves each buffer's storage dtype.
The TinyLlama example opts into this path with `--bf16-weights`, accepts only
BF16 checkpoint tensors in that mode, and records storage/compute dtype and
uploaded bytes in its report. The default remains F32 storage.

TinyLlama's restricted model definition now lives in
`crates/rxla-weights/examples/tinyllama/model_graph.rs`, shared by scalar decode and the
`tinyllama_prefill` diagnostic. Static token chunks use per-row runtime RoPE,
causal masks and head-layout transposes, with a capacity/finite-logit commit guard.
The private per-model/client WeightStore shares Rc buffers across graph variants;
it is not a cross-client/checkpoint cache. `tinyllama_prefill MODEL REPORT` checks
8-token chunks against scalar execution across the full 22-layer model, including
all prompt logits and final KV state. It uses BF16 weight storage with F32
arithmetic. `--prompt-file PATH` supplies UTF-8 prompt text exactly as written;
chat formatting remains the caller's responsibility. Whole eight-token blocks
use the chunk plan; the remaining zero to seven tokens use the scalar plan after
resident state handoff, without padding or a third tail-shape compilation. Prompts
shorter than eight tokens construct and compile only the scalar plan, sharing its
read-only crates/rxla-weights/executable across independently initialized sessions.
`--max-new-tokens N` sets the generation budget (default 24), and `--capacity N`
sets the static KV/attention capacity (default 128, up to the validated model's
2048-position configuration). Prompt plus generation budget must fit before any
plugin loading or weight upload. Capacity changes the compiled graph and state
shapes; this does not grow an existing session or extrapolate RoPE. Reports state
the actual capacity, budget and EOS/length stop reason. A 177-token prompt at
capacity 256 with 64 generated tokens passes the independent Transformers check;
one additional capacity-2048 repeated-text stress case uses a 2014-token prompt
and matches all 34 generated tokens against Transformers, ending at position 2047.
This is near-limit position coverage, not a representative long-context suite.
Larger capacity
uses more KV storage and attention work; no speedup is implied.
This is not yet a serving scheduler. The diagnostic
now moves chunk-prefilled resident KV/position into the scalar plan using
`switch_program_parameters`, then checks greedy continuation against the scalar
reference. Handoff itself performs no state download/upload or compilation;
the diagnostic separately downloads state/logits for correctness checks.

`cargo run -p rxla-train --example train_frozen_bf16` demonstrates generic
low-rank adaptation using the existing modules: a frozen BF16 linear map plus
two F32 trainable linear projections. Ordinary `sgd_step` discovers the two
resident parameters and leaves the input-backed base frozen. The example checks
losses, both gradients and both updated weights against an independent dense F64
reference at every one of 250 steps, verifies convergence and unchanged base bits,
and requires one compilation. The up projection starts at zero, so the initial
down gradient is checked to be zero while the up gradient is nonzero. This is a
tiny general training integration example, not LLM training, a training benchmark,
BF16 optimizer arithmetic, or evidence of large-model adaptation quality.

cargo build -p rxla-core
cargo test -p rxla-core
PJRT_PLUGIN_PATH=/path/to/libzml_cpu.so cargo run -p rxla-core --example matmul
PJRT_PLUGIN_PATH=/path/to/libzml_cpu.so cargo test -p rxla-core -- --ignored
cargo xtask generate
cargo xtask check
```

`xtask check` regenerates PJRT bindings and protobuf types in a disposable
temporary directory and compares all seven outputs byte-for-byte with the
committed files. It exits nonzero on missing, changed or unexpected files and
does not write to the source tree. Cargo may still populate its build cache.
Use this in maintainer/CI checks, not `build.rs`: it requires the generation
toolchain (libclang and rustfmt, plus the vendored protoc dependency), while
ordinary tensor library builds continue to use only pregenerated Rust sources.
Run with the checked-in Cargo.lock and a compatible host generation toolchain;
this is a reproducibility check, not a cross-platform ABI verification.

`generate` also stages both generators in a temporary directory before updating
source files, so a generation failure does not leave half-regenerated sources.
The final file writes are not a multi-file filesystem transaction. Unexpected
old generated files are reported for explicit review rather than silently deleted.

Plugin loading is unsafe because the caller chooses trusted native code. Buffers
and executables retain their client and plugin; synchronous operations await
completion, and inputs are explicitly marked non-donatable.

Before plugin initialization, the loader checks that error destruction/reporting,
event await/destruction, and client/buffer/executable destruction slots exist.
This prevents discovering missing cleanup or synchronization functions only after
acquiring native resources or starting a transfer into borrowed Rust memory.
Optional operations such as artifact serialization are still checked on demand.

Successfully opened plugin libraries remain loaded until process exit, including
when later ABI validation or initialization fails. PJRT client destruction is not
a process-wide plugin shutdown/join guarantee: background threads and TLS hooks
may still refer to plugin code. Client, buffer and executable destructors still
run normally; only the dynamic-library loader reference is intentionally retained.
There is no hot-unload or bounded cache of distinct plugin libraries. Use process
isolation if unloading/replacing plugins or reclaiming their process-global
runtime allocations is required. A Linux subprocess test verifies the real plugin
remains mapped after dropping the last client/buffer and can be reopened.

Verified initial CPU plugin: ZML release `202609101243.20.1.7ca6884ea2cb`,
`https://mirror.zml.ai/plugins/202609101243.20.1.7ca6884ea2cb/zml-cpu-linux-amd64.tar.zst`.
Archive SHA256: `f1670d21b92102b8c7c7f9d0f00d6a7a1a5d90a83d7e28ba5fbeb1f9ce4c170c`.
The plugin is not committed, installed globally, or silently downloaded by Cargo.
Other plugin/platform versions require independent validation.

The scalar `tinyllama` example also accepts `--prompt-file PATH`. It reads exact
UTF-8 without trimming or adding a chat template; empty/invalid UTF-8/missing
files are rejected. Include model-specific chat delimiters yourself when desired.
The existing bounds check requires valid token IDs and room for 24 new tokens
within capacity 128, before loading the PJRT plugin. The report stores the actual
prompt and token IDs, so the independent checker validates the supplied prompt.
`benchmarks/prompts/tensor-question.txt` provides a complete chat-format fixture.
For example, from the standalone repository root with a trusted configured plugin:

```sh
cargo --config fast-load.toml run --offline -p rxla-weights --example tinyllama -- \
  target/tinyllama-chat /tmp/new-tensor-chat-report.json --bf16-weights \
  --prompt-file benchmarks/prompts/tensor-question.txt --top-k 40 \
  --seed 1311768467463790320 --benchmark-runs 1
```

The question fixture was validated on RTX 5080 with 40 prompt tokens and 24
sampled tokens: independent Transformers/Threefry reference choices all matched,
maximum checked logit error `1.138448715209961e-5`, and warmed replay matched.
Reports are `/tmp/xla-tinyllama-tensor-chat.json` and
`/tmp/xla-tinyllama-tensor-chat-validation.json`. This proves numerical/decoding
agreement, not answer quality: this small model's answer incorrectly restricted
tensors to three dimensions. Generation stopped at the 24-token limit.

The scalar `tinyllama` example accepts optional `--top-k K` (1..=32000),
defaulting to unchanged greedy decoding. Top-k uses temperature 1 and initial
counter zero. Optional `--seed U64` (decimal, default 0, requires `--top-k`)
maps directly to raw Threefry key words `[low32, high32]`, without hashing. The
key initializes runtime state, not graph constants. Every accepted model step,
including prompt steps whose sampled token is discarded, consumes K blocks.
KV, position and RNG updates share the acceptance guard. `--benchmark-runs N`
restores the selected key, resets other state, reuses weights, and checks both generated IDs and final RNG
words against the diagnostic run. It uses a separately compiled two-output
executable; no per-token compilation is performed.

On the RTX 5080 with BF16 weight storage, `--top-k 40 --benchmark-runs 1`
generated 24 tokens beginning “Rust is a language built by a community”. The
independent Transformers FP32 CPU checker reproduced all 24 choices using its
own scalar Threefry and F64 Gumbel computation; the first eight full-vocabulary
logit vectors had maximum absolute error `1.2874603271484375e-5`. Warmup and
replay matched, with final counter 2520 for 63 model steps and two compiled
executables total. Reports: `/tmp/xla-tinyllama-topk40-final.json` and
`/tmp/xla-tinyllama-topk40-validation.json`. This is a fixed-key, one-prompt
correctness check, not general model-quality or statistically robust performance
evidence. The completion stops at the 24-token limit, not EOS. No configurable
temperature or top-p is implemented by this runner yet.

The nonzero seed `1311768467463790320` (`0x123456789abcdef0`, key words
`[0x9abcdef0,0x12345678]`) was also checked with top-k=40 and BF16 weights on
the RTX 5080. All 24 tokens matched the independent reference, maximum checked
logit error was `1.430511474609375e-5`, and warmup/replay restored both key words
and the final counter 2520 correctly. Reports are `/tmp/xla-tinyllama-seeded.json`
and `/tmp/xla-tinyllama-seeded-validation.json`. This exercises nonzero high bits
and signed-I32 storage of the low word; it is not a statistical RNG-quality test.

The complete TinyLlama Chat FP32 example generates through EOS on CPU and CUDA,
with all 21 greedy tokens matching Transformers for one fixed prompt. The CUDA
RTX 5080 run has first-eight-step full-vocabulary maximum error 1.24e-5 after
explicitly requesting HIGHEST matmul operand precision. See the
historical reproduction protocol and limitations recorded at the time.
This validates the model path, not general LLM performance or model compatibility.
For repeated model runs, the opt-in `--config fast-load.toml` profile and a
trusted executable cache are validated on CUDA too: one observed run reduced
loading/building from 10.36 s to 1.26 s with identical recorded outputs. This
optimizes only `rxla-weights` (including examples), not the full dev workspace;
see the linked report for rebuild tradeoffs and timing limitations.

The restricted real OCR graph validation
also executes all 242 nodes of the local PP-OCRv6 detector through Rust/PJRT.
Two synthetic 64×64 input cases match ONNX Runtime CPU tensor outputs with a
strict per-element tolerance; this does not validate OCR text accuracy or speed.
Larger book/table image inputs execute, but fail that same strict ORT agreement
gate on 13/14 pixels respectively. Independent FP64 diagnostics favor Rust on
12/13 and 14/14 rejected pixels. Complete-map FP64 comparisons also show lower
maximum, mean absolute and RMS errors for Rust on both inputs; the strict ORT
gate remains failed, and these results do not establish OCR text quality.

`Client::info()` returns owned API-version, platform/version, process-index and
addressable-device metadata, identifying the currently selected device. It does
not expose raw device handles or add multi-device execution. Metadata strings
are diagnostic labels, not complete binary/device fingerprints for disk caches.

```sh
PJRT_PLUGIN_PATH=/path/to/libzml_cpu.so cargo run -p rxla-core --example doctor
```

This loads trusted native code, prints metadata, then compiles and executes a
small HLO calculation. It is not a sandbox or a comprehensive plugin compatibility
test. State-management API constraints are documented in [STATE-DESIGN.md](STATE-DESIGN.md);
tracked state-slot/session threading is implemented; automatic module derivation
and the broader transformation rules are not.

## Composite model operations

`x.erf()` lowers to the native HLO `erf` opcode, without substituting a tanh
approximation. The pinned CPU plugin passes a 241-point sweep on [-6,6] against
independent F64 numerical integration (observed maximum absolute error 1.54e-7),
plus infinity/NaN/zero cases. This is bounded numerical validation, not a
correct-rounding guarantee over all F32 bit patterns or all plugins.

`minimum` is an equal-shape binary operation, like `maximum`. `clamp(low, high)`
supports ordered, non-NaN scalar bounds (including infinities).
`hard_sigmoid(alpha, beta)` explicitly computes `clamp(alpha*x + beta, 0, 1)`
with finite coefficients. Do not assume all exporters use the same slope: the
local PP-OCRv6 detection graph uses both 0.2 and approximately 1/6. Both variants
have live CPU comparisons against an F64 reference.

`x.linear(&weight, bias)` applies `x @ weight.T + bias` on the last axis,
preserving all leading dimensions. Weight layout is `[out, in]` and optional
bias is exactly `[out]`, matching the checkpoint convention used by TinyLlama.
The lower-level `matmul` keeps its ordinary mathematical layout convention.

`x.layer_norm(&normalized_shape, weight, bias, epsilon)` normalizes over matching
trailing axes using population variance. Scale and bias are independently optional
and must exactly match `normalized_shape`. Epsilon must be finite and positive.
Statistics use F32 centered two-pass variance, not subtraction of squared means;
this is still subject to F32 range and precision limits. RMSNorm remains separate.

`x.batch_norm_inference(axis, mean, variance, weight, bias, epsilon)` uses supplied
running statistics and explicit channel-axis broadcasting. Each parameter must
have exact shape `[channels]` and belong to the input graph. The F32 expression
is `(x - mean) * rsqrt(variance + epsilon) * weight + bias`; epsilon is finite
and positive. Statistic values are runtime data and must be valid (in particular,
nonnegative variance); there is no data-dependent validation or statistics update.
This is inference-only, not BatchNorm training or automatic state management.
NHWC convolution outputs normally use axis 3. Runtime statistics/weights can be
bound to a session and changed without recompiling the fixed-shape graph.

These methods compose existing Rust tensor operations; they do not introduce
opaque custom calls, native kernel dependencies, or additional compilation units.
For example, a user-defined block can be built as:

```rust,ignore
let hidden = x.layer_norm(&[hidden_size], Some(&scale), Some(&offset), 1e-5)?;
let hidden = hidden.linear(&up_weight, Some(&up_bias))?.silu()?;
let output = x.add(&hidden.linear(&down_weight, Some(&down_bias))?)?;
```

Compile the final output once so XLA receives the complete block. Live CPU tests
cover vector/batched Linear and compare LayerNorm against independent F64
centered statistics, including constant and large-offset inputs.

`x.max_pool2d(Pool2dOptions { window, strides, padding })` pools spatial H/W
axes of NHWC data, preserving batch and channel axes. Default options are 2×2,
stride 2, no padding. Padding is negative infinity; output sizes use floor
division, with zero spatial output when the padded input is smaller than the
window. Windows containing only padding return negative infinity. The lowering
uses HLO `reduce-window` with a maximum reducer, not a custom native kernel.
There is no ceil mode, window dilation or returned argmax indices. Live tests
compare against a host window reference for multiple batches/channels, overlapping
windows, asymmetric padding, entirely padded windows and empty outputs.

`x.upsample_nearest2d([height_scale, width_scale])` repeats NHWC pixels by
positive integer factors using reshape/broadcast/reshape. Output pixel `(h,w)`
reads input `(h/height_scale,w/width_scale)` with integer division. It is not a
general arbitrary-size resize API and has no half-pixel/align-corners convention,
antialiasing or bilinear interpolation. Shape arithmetic is overflow-checked.
CPU tests include asymmetric factors, multiple batches/channels and empty inputs,
plus a single compiled convolution → ReLU → pool → upsample → channel-concatenate
graph compared with an independent host reference. This is a small multi-scale
vision block, not validation of a complete YOLO model.

`x.conv_transpose2d(&kernel, ConvTranspose2dOptions { .. })` uses NHWC inputs
and **HWOI** weights `[Kh,Kw,Cout,Cin]` (distinct from forward convolution's
HWIO). It supports spatial strides, kernel dilation, nonnegative explicit
cropping/padding, and high-side output padding below the corresponding stride.
Spatial inputs/outputs must be positive. Groups and explicit output-shape
overrides are not supported. Bias is a separate broadcast/add.
The HLO lowering uses convolution with input dilation and reversed windows;
live CPU tests compare it with an independent F64 scatter-add reference,
including asymmetric cropping and output-padding cases.

## Tracked state and sessions

The `module` namespace separates object-safe `Parameterized::visit_parameters`
from `Module::forward(&Tensor)`. Every Module is Parameterized, but components
with integer/multiple inputs need only implement Parameterized and expose their
own typed forward method. Collection and checkpoint loading accept
`&dyn Parameterized`; no float conversion or dummy forward implementation is
needed. Available components include `Embedding`, `Linear`, `GatedMlp`, `Conv2d`,
`RmsNorm`, `LayerNorm`,
`Activation` (Relu/Silu/Gelu/GeluTanh) and nested `Sequential` containers:

```rust,ignore
use rxla_train::module::{Activation, Linear, Module, Sequential};
let linear = Linear::new(&mut graph, 128, 64, true)?;
let model = Sequential::new(vec![Box::new(linear), Box::new(Activation::Silu)]);
let entries = rxla_train::module::parameters(&model)?;
let output = model.forward(&input)?;
// Compile output through StateGraph, load each entry, then bind its parameter.
```

Modules build the same graph; virtual dispatch happens during graph construction,
not per device kernel, and does not introduce separate compilation boundaries.
Cloning Linear shares parameter identities. Construction registers parameters but
does not initialize/upload weights. `module::parameters(&model)` recursively
collects unique identities in first-visit order. Each entry contains a primary
`name`, subsequent `aliases`, and a `parameter` handle for session binding. For
example a tied layer at paths `0` and `1.1` produces one `0.weight` entry with
alias `1.1.weight`, not two uploads. Duplicate paths and empty path segments are
errors; equal shapes or indices from different graphs never imply shared identity.
Sequential paths are structural indices, not guaranteed checkpoint names; editing
the topology can change them. A loader must explicitly map checkpoint keys and
decide how to handle aliases (including conflicting duplicate checkpoint values).
Custom components must implement `Parameterized::visit_parameters` and expose all parameters;
parameter-free modules explicitly use a no-op. Collection does not call forward
or mutate the graph. Derive macros and stateful Module transforms remain
unimplemented; experimental scalar-loss gradients can be built explicitly for
modules whose operations are supported. Custom modules implement `Module` with ordinary
tensor operations; arbitrary Rust side effects are not captured. Empty Sequential
is identity; incompatible layer shapes are rejected when recording `forward`.

For concrete custom model types, `impl_parameterized!` generates the visitation
implementation without a build script, procedural macro crate or new dependency:

```rust,ignore
rxla_tensor::impl_parameterized!(Model {
    parameter scale => "norm.scale",
    optional_parameter bias => "norm.bias",
    module head => "head",
    optional_module adapter => "adapter",
    modules layers => "layers",
});
```

Entries run in declaration order. Child names are prefixed; a `modules` list uses
paths such as `layers.0.weight`. Boxed module trait objects work through normal
method dispatch. Shared identities and all aliases remain visible to the existing
collector, which still rejects duplicate/invalid paths. `Model {}` explicitly
declares a parameter-free type. This is an explicit field list, not reflection:
omitted parameters are not discovered. RNG, optimizer and normalization state
still need their own state registration/checkpoint schema. The macro generates
neither forward nor initialization and does not accept generic impl/where-clause
declarations; manual trait implementations remain available. The BatchNorm
training example and independent downstream consumer use the macro.

To optimize only part of an existing resident model, use a stable parameter set:

```rust,ignore
let head = module::ParameterSet::select(&model, |entry| entry.name.starts_with("head."))?;
let optimizer = optim::MomentumSgd::new(&mut graph, &head, 0.5)?;
let accumulator = optim::GradientAccumulator::new(&mut graph, &head, 2)?;
```

Selection validates the full model before invoking its predicate once per unique
identity, preserving first-visit order and all aliases. A tied parameter is
selected as a whole; choosing an alias also trains its other uses. Inspect
`entry.aliases` when selection depends on non-primary paths. The returned set
implements `Parameterized` and is a snapshot of discovered handles, not a live
filter: rebuild it if model structure changes. Empty selections are permitted,
but optimizers require at least one resident parameter. Input-backed parameters
remain frozen even when selected. This is optimizer scope, not a global
requires-grad flag, detach operation or protection against other state writes.
Excluded resident parameters still need full-model initialization/checkpoint
state. A native head-only training test retains a linear backbone bit-for-bit,
checks forty microbatches/twenty updates against an F64 recurrence and compiles once.

`Linear::new_trainable(graph, input, output, bias)` and
`Conv2d::new_trainable(graph, input, output, kernel, options, bias)` register
resident trainable parameters directly. They share shape/configuration checks
with `new`, whose input-backed inference/binding behavior is unchanged. No
constructor initializes values or allocates device buffers: initialize the slots
returned through `weight()`, `bias()` or `module::parameters` before creating a
session. Clones still share identities; optimizers discover each tied parameter
once. `from_parameters` remains available for mixed frozen/trainable components
and existing checkpoint layouts. The classifier and both CNN training/accumulation
paths use these constructors; their native training checks remain in the gate.

`RmsNorm`, `LayerNorm` and `GatedMlp` likewise provide `new_trainable` with the
same arguments as their input-backed `new` constructors. LayerNorm with
`affine=false` still registers no parameters. Scale/bias initialization remains
explicit (typically ones/zeros); no random initialization policy is inferred for
MLP weights. Native tests run each trainable component for ten updates, verify
every parameter group changes and compare each SGD update against an F64 host
recurrence using downloaded gradients. Each component compiles once. These are
component training checks, not full-model convergence or independent validation
of gradient formulas; the existing operator autodiff tests cover the latter.

`Embedding::new_trainable(graph, vocabulary, width)` registers a resident table
with the same integer-input and clamped-index semantics as `new`. It does not
introduce sparse updates, a padding index or automatic initialization. Tying an
output projection with `Linear::from_parameters(embedding.weight().clone(), None)`
retains one parameter identity. A native five-step test compares loss, the sum
of lookup/projection gradients and momentum updates with an independent F64
reference, including repeated and out-of-range indices. Only one table and one
velocity slot are allocated, and varying indices reuse one compilation. This is
a tied-component correctness check, not LLM training support or a performance test.

`init::initialize_trainable(&graph, &model, &client, callback)` builds the model's
initial `(StateSlot, Buffer)` list. The callback receives a `ParameterEntry`
(primary name, aliases and parameter shape) and returns `Result<Vec<f32>>` for
one unique resident parameter. Frozen input-backed parameters are skipped; tied
weights invoke the callback once. Use explicit per-name `Initializer` seeds/streams
or constants to select your policy rather than relying on an inferred layer type.
All paths, selected graph owners and shapes are checked before callbacks/uploads;
each callback result must have exactly the required element count. Nonfinite
values are allowed, leaving that validation policy to the caller. Only one host
tensor's returned data is retained at a time. On failure, newly created buffers
are dropped, but callback side effects/RNG progress are not rolled back. The
helper does not mutate an existing session or initialize optimizer/statistic/RNG
slots: append those components' initial state explicitly. It can run after graph
recording/compilation, and an all-frozen model returns an empty list. The classifier
uses this helper; native tests verify alias handling, independent sessions,
preflight-before-callback and retry after malformed callback output.

After dropping the builder, use
`init::initialize_trainable_for_program(&program, &model, callback)` instead.
It selects the compiled plan's client and validates every selected parameter
against that plan before invoking callbacks; slots added after compilation and
foreign graph slots are rejected even when their shapes match. Initializing only
the model still requires appending optimizer/statistics/RNG state before session
creation. `StateProgram::state_type(&slot)` exposes validated single-slot dtype
and shape for this purpose; `state_layout` keeps its stricter complete-set contract.
The CNN accumulation comparison initializes its reusable compiled models through
this API, and a native test drops StateGraph before creating a second independent
session without recompilation.

`GatedMlp::new(&mut graph, features, hidden, Activation::Silu)` constructs a
bias-free SwiGLU-style block: `down(silu(gate(x)) * up(x))`. `from_layers` accepts
existing Linear modules (including optional biases/tied parameters) and validates
their widths and graph ownership. Parameter paths use `gate_proj`, `up_proj` and
`down_proj` prefixes. Normalization and residual addition are deliberately outside
the block. TinyLlama now builds its MLP with this module and uses Embedding,
RmsNorm and Linear for its other supported components. Attention uses the GQA
tensor API; RoPE, mask construction and resident cache management remain explicit
graph code. This does not introduce separate
kernel/compiler boundaries or claim a new fused kernel implementation.

`module::Conv2d::new(&mut graph, in_channels, out_channels, [kh, kw], options,
use_bias)` accepts NHWC activations and registers OIHW checkpoint weights
`[out_channels, in_channels/groups, kh, kw]`, plus optional `[out_channels]` bias.
`from_parameters` reuses existing identities. The OIHW-to-HWIO transpose is part
of the recorded graph, so checkpoint loading needs no host-side layout rewrite.
Whether XLA eliminates/prepacks that transpose is backend-dependent and has not
been benchmarked here. Stride, dilation, groups/depthwise and explicit nonnegative
padding follow the existing tensor convolution semantics; output is NHWC, not
NCHW. Native tests compare ordinary/grouped/depthwise cases with a direct F64
OIHW cross-correlation reference using asymmetric padding, stride and dilation.

`Embedding::new(&mut graph, vocabulary, width)` registers one F32 table with
layout `[vocabulary, width]`. Its inherent `forward(&Index)` returns shape
`indices.shape + [width]`, including scalar and empty index tensors. It implements
Parameterized, not the float-input Module, so it does not fit inside Sequential.
Compose it explicitly with subsequent Modules. `Embedding::from_parameter` and
`Linear::from_parameters` can share the same table identity for a tied output
projection. The mapped-loader regression test verifies one upload and two runtime
index batches with exact expected projection results. Indices inherit `take`
clamping to `[0, vocabulary-1]`; there is no padding-index override, gradient
masking, negative wrapping or bounds-error mode.

For integer checkpoint payloads, `Checkpoint::read_i32(name)` returns
`HostI32 { shape, values }` and `upload_i32(&client, name)` upload exact I32
buffers. Both reject floating-point and other integer dtypes rather than cast.
This preserves counters/index values such as 16,777,217 and I32 extrema without
passing through F32. Only the selected payload is read; scalar and empty tensors
are supported, and successful reads/uploads contribute to the existing load
statistics. Host tests check byte-selective I/O and dtype rejection; native tests
verify uploaded values exactly. This enables loading integer state buffers, not
automatic state discovery or a full training checkpoint schema.

`rxla_weights::write_buffers(&mut writer, &[("weights", weight_buffer),
("optimizer/m", moment_buffer), ("step", integer_buffer)])` streams named F32/I32
buffers into one safetensors file, without requiring the `training` feature.
Names are sorted and must be unique, nonempty and not `__metadata__`; shapes,
dtypes and the complete header are checked before writing/downloading. Each tensor
is downloaded individually and payload writes are at most 64 KiB. Host memory
includes the largest tensor, header and caller buffering, not all tensor payloads.
The existing `write_module` uses the same encoder while retaining its module-only
parameter discovery and alias policy.

For session state, explicitly select `session.state(&slot)?` buffers with stable
names; their shared borrows keep that session from running during capture. Distinct
names referring to the same buffer produce separate entries. The exporter does
not discover missing state or attach optimizer configuration/executable code.
It does not open, flush, fsync or atomically publish files; failure may leave partial
bytes that must be discarded. Tests cover mixed/scalar/empty buffers, exact I32
values and F32 signed zero, deterministic naming order, pre-write rejection and
partial writer failures with unchanged source buffers. A complete named-state
schema and training-resume workflow still need to be supplied by the caller.

`write_buffers_with_metadata(writer, buffers, &HashMap<String, String>)` includes
caller-defined string metadata in the same bounded safetensors header. Store a
versioned model/state schema, optimizer description and input cursor explicitly;
JSON may be stored as a string, but no serde schema or configuration inference is
built into this API. `Checkpoint::metadata()` returns the optional map.
Before loading state, `checkpoint.require_metadata(&[("schema", "my-model/v1"),
("optimizer", expected_optimizer_description)])` checks exact string matches.
Missing/mismatched keys and duplicate expectations fail without payload I/O or
uploads; unrequested metadata keys are allowed. Checking is opt-in, not part of
`load_state`. Metadata does not authenticate a file or prove crates/rxla-core/configuration
consistency, and map serialization order is not a canonical hash representation.
Host tests cover Unicode/JSON strings, absent metadata and exact-match failures;
the disk training-resume test saves/checks schema, AdamW settings and microbatch
cursor before loading any state tensors.

`save_buffers_new(path, buffers, metadata_or_none)` is a filesystem convenience
for versioned checkpoints: it writes to a `tempfile` in the existing target
directory, synchronizes file contents, and publishes without replacing an existing
target. Existing files, directories and symlinks are rejected. The no-clobber
operation handles competing writers too; ordinary validation/write/sync failures
clean up the temporary file. Use a trusted parent directory. This follows
`tempfile`'s platform/filesystem-dependent publication behavior, not a universal
atomicity guarantee; crash/cleanup failures can leave a temporary hard link.
The parent directory is not fsynced, so power-loss durability is not promised.
There is no overwrite mode, latest-pointer update or retention policy. Host tests
race two writers, preserve existing files/dangling symlinks and check missing
parents; native tests check temporary cleanup on invalid mappings and exact I32
payloads. The disk training-resume test uses this entry point for its checkpoint.

With the `training` feature, `checkpoint.load_state(&program, &mapping)` restores
a complete named state set, where mapping is `&[("weight", weight_slot),
("optimizer/m", moment_slot), ("step", count_slot), ...]`. Every program slot
and every selected name must occur exactly once; mapping order is irrelevant and
unselected checkpoint tensors are ignored. The loader validates the entire slot
set and exact F32/I32 metadata before any payload reads/uploads, uses the program's
client, and returns owned buffers for `program.session(...)` or
`session.replace_state(...)`. It never mutates a live session while loading.
Partial I/O/upload failure drops temporary buffers; completed load statistics
remain counted. Fixed inputs and optimizer/configuration choices are not restored.

`StateProgram::state_layout(&slots)` exposes this complete-set preflight without
allocating device buffers, returning dtype/shape pairs in caller order; slot
indices are not exposed as persistent IDs. Names remain the caller's semantic
contract: validation cannot detect a same-shaped weight/moment name swap or an
incompatible model with the same layout. Native tests reject incomplete/duplicate/
foreign slots and missing/wrong-type/wrong-shape names with zero payload reads,
then compare continued mixed-state execution after restore with uninterrupted
execution, including an I32 counter above the exact F32 integer range.

For the corresponding complete-state save, `write_state(writer, &session,
&mapping, metadata_or_none)` streams to a caller-owned writer, and
`save_state_new(path, &session, &mapping, metadata_or_none)` uses no-clobber file
publication. Both require every slot in `session.program()` exactly once, reject
duplicate/foreign slots and invalid names before any output, and hold shared
session borrows during capture. This catches omitted optimizer moments or
accumulation counters that raw `write_buffers` cannot detect. Completeness covers
resident slots only: fixed input weights, model configuration, data-loader/RNG
state outside the graph remain the caller's responsibility. The full-state save
cannot detect semantically swapped same-shaped names. Native tests verify that
incomplete/invalid mappings leave writers and target directories unchanged,
preserve existing checkpoints, and round-trip mixed state plus metadata. The
fresh-process training-resume regression now saves with `save_state_new`.

The native `training_resume` test exercises an actual temporary safetensors file
with AdamW mid-accumulation: after three accepted microbatches (one optimizer step
plus a partial gradient), it writes seven named state tensors covering weight,
moments, beta powers, accumulated gradient and I32 count. Targets come from a
seeded host normal stream; the same file's application-owned metadata stores
the RNG snapshot (version, expanded seed, stream, word position), sampling
policy/dependency versions and microbatch cursor. The u128 position is encoded
as a decimal string, avoiding numeric narrowing in JSON consumers. A freshly rebuilt graph
rejects the old slot identities and restores the file using its own named slots.
All losses and state values match uninterrupted training exactly over the next
nine microbatches on the tested CPU backend. Rebuilding uses the code cache (one
backend compilation total), independently of mutable-state restoration.
An additional variant applies device-sampled dropout to those random targets.
It checkpoints eleven slots: the original seven plus the two Threefry key and
two counter words. The application metadata pins the device augmentation policy,
and the sequence acceptance predicate gates accumulation/AdamW updates. The
checkpoint precedes low-word counter carry; resumed execution includes both kept
and dropped targets. The host-only baseline remains a separate test.
The test also launches a fresh subprocess with the checkpoint path, validates
schema/optimizer/RNG-policy metadata, restores the host stream and saved input cursor, independently loads the
CPU plugin and compiles the graph once, then runs those nine microbatches. Its
reported PID differs from the parent and its exported random targets, dropout
masks, losses, final state and final RNG snapshot match
the parent's uninterrupted execution exactly. Both variants also export all resident state bits after
each resumed microbatch as I32 payloads; all nine steps match bitwise, including
partial accumulation, optimizer moments/powers, and device RNG when enabled.
The child does not inherit graph
handles, device buffers or the parent's in-memory compiler cache. This verifies
same-host/same-version CPU process restart, not cross-device or cross-version
portability. Model construction remains explicit, rather than reconstructed from
arbitrary checkpoint code. The test owns/removes its temporary files and
explicitly synchronizes them, but this does not add atomic checkpoint publication,
automatic configuration serialization, automatic RNG discovery/data-loader resume, or distributed
snapshot coordination to the public API.

`RmsNorm::new(&mut graph, features, epsilon)` registers last-axis scale weights.
`LayerNorm::new(&mut graph, normalized_shape, epsilon, affine)` normalizes explicit
trailing axes with centered F32 variance; affine=true registers scale and bias,
false registers neither. Existing/tied parameters can be supplied with
`RmsNorm::from_parameter` and `LayerNorm::from_parameters`; the latter also allows
scale-only or bias-only configurations. Parameter visitation exposes `weight`
and/or `bias` and therefore works with the same mapped checkpoint loader.
Neither constructor initializes parameters. Epsilon is finite/positive static
configuration, whereas affine values are runtime data and can be rebound without
recompilation. Shapes/ownership are checked before applying the underlying tensor
operations. These are not running-statistics or training-mode BatchNorm layers.
Tests compare a composed RMSNorm/LayerNorm graph with an independent F64 formula
and verify parameter rebinding with one backend compilation.

Enable `rxla-weights`' optional `training` feature to load a collected module from
a safetensors checkpoint:

```rust,ignore
// HashMap<String, String>: every primary module path and alias -> checkpoint key.
let bindings = checkpoint.load_module(&client, &model, &mapping)?;
session.bind_parameters(bindings)?;
```

Mapping is exact: missing/extra module paths are rejected. Every alias of a shared
parameter must select the same checkpoint key; choosing different keys is an
error even if their tensor contents might match. Unselected checkpoint keys are
ignored. All selected names, shapes and F32/F16/BF16 dtypes are validated before
any payload reads or uploads. Each unique parameter is uploaded once as F32;
distinct parameters may explicitly select the same source and get separate buffers.
I/O/native failures can still happen partway through loading: temporary buffers
are dropped, completed work remains in loading statistics, and no session has
been modified. Binding the returned complete set is a separate validated action.
The optional feature keeps tensor-module integration out of the default reader.
Unified checks exercise it alongside `disk-cache`, including a real mapped,
tied-weight F16/BF16 load and execution test.

For identity-based F32 weight binding, use `graph.parameter(shape)` and build
operations with `parameter.tensor()`. Cloning a `Parameter` preserves its input
identity, including tied weights. `session.bind_parameters(vec![(parameter,
Arc::new(buffer))])` replaces the complete fixed-input set without numeric offsets.
It rejects foreign/duplicate handles and validates buffer client, shape and dtype
before commit. Empty bindings clear the set; unbound inputs keep registration
order. Values remain runtime arguments and changing weights does not recompile.
This is not parameter autodiff or a named checkpoint schema. TinyLlama binds all
weights this way, without assuming that weights begin at input index two.

`KvCache::update_at_checked` is the bounds-checked alternative to clamping:
all runtime I32 starts must be nonnegative and leave room for the entire update
on every axis. It returns `KvCacheUpdate { keys, values, accepted }` after recording
the paired update. `accepted` is scalar F32 1/0; invalid starts select both old
caches unchanged, including when rejected payloads contain NaN/Inf. Gate any
separate position/length counter explicitly with this result. In-bounds overwrites
are allowed; this is not an append-only policy, finite-value guard or cache-validity
tracker. Selection is not lazy execution. Static validation errors leave both
symbolic versions unchanged. CPU/CUDA tests cover multirow chunks, negative and
too-large starts, invalid non-sequence-axis starts, valid overwrites, paired state
preservation and zero update gradients on rejection with one compilation.

`cargo run -p rxla-core --example chunked_attention` composes checked KV writes,
resident I32 position, runtime causal positions and SDPA into one fixed-shape
executable. It consumes three two-row chunks into a six-row cache, checks every
attention output against an independent full-history F64 reference and advances
position only when the paired write is accepted. Two calls after exhaustion
(including NaN payloads) return an explicit rejection/zero output while preserving
both caches and position. Startup initializes unused cache entries to zero; causal
masking is not a general guarantee that arbitrary NaNs in unused V slots are safe.
Only the new Q/K/V chunk is supplied to each call. State inspection in the example
is diagnostic, not required for scheduling. CPU and selected CUDA gates run it.
This is fixed-size chunk attention, not a full-model prefill implementation,
variable final-chunk handling, a fused kernel or a throughput benchmark.

For paired inference state, `KvCache::new(&mut graph, shape)` registers K/V
slots with a caller-chosen layout. Model graph construction can use:

```rust,ignore
let (keys, values) = cache.update_at(&mut graph, &k, &v, &starts)?;
```

This records both updates together and returns full cache tensors for attention.
Use `key_slot()`/`value_slot()` to initialize or inspect each session's buffers.
It is fixed-capacity F32 state, not automatic append or a paged cache: positions
are explicit, out-of-range starts clamp, and the caller constructs the mask.
TinyLlama and the masked-decode test use this facade; general Module derivation
is not implemented.

`StateGraph::state_i32(shape)` registers resident integer state. `read`, `write`
and `write_many` operate on ordinary dtype-carrying tensors and validate mixed
F32/I32 updates atomically. Slot shape/dtype cannot
change. Initialization, replacement and execution results all check the recorded
dtype before session commit. The `stateful_cache` example now maintains both
cache and position internally: each call supplies only the new F32 data. It is
still fixed-capacity; wrapping integer arithmetic and slice clamping do not
automatically detect exhaustion or implement a ring buffer.

TinyLlama likewise registers position as I32 state and supplies only the token
buffer on each call. The same old-position value drives graph-computed RoPE angles, attention
mask and every layer's K/V update; the increment is a hidden state output committed
with those updates. Its host loop prechecks the total context budget. A single
diagnostic position download after generation checks successful execution count;
there is no per-token position upload/download. The reference verifier also
checks this count against consumed tokens (the last predicted token is not fed
back). This is not a throughput claim or an asynchronous/donating execution path.

The example keeps only 32 static inverse frequencies as host constants. It casts
the resident position to F32, multiplies by those frequencies, and computes sine
and cosine in the graph instead of embedding capacity-sized lookup tables. The
attention mask uses native I32 iota coordinates. This removes host-generated
position tables, not the fixed-capacity KV cache or its context bounds. F32
position conversion can round integers beyond 2^24; this example does not claim
arbitrary-long-context RoPE accuracy or implement model-specific RoPE scaling.

`StateGraph::compile` returns mixed visible F32/I32 buffers in requested order,
while final state roots stay hidden. Every visible and hidden result is type-checked before commit;
even a malformed visible integer result leaves old state intact. The cache
example returns selected F32 data plus the updated I32 position, with no cast.

`Tracer::compile_many` and `Compiler::compile_many` accept tensor roots of any
supported runtime dtype. Use `execute` and the matching typed buffer downloads
on returned buffers; the existing host convenience methods remain F32-only.
Tests verify exact mixed output restoration, cache reuse, independent sessions,
integer counters, mixed-state replacement and rejected-result atomicity. The
disk-cache suite also launches a fresh child process: typed hidden state code
restores with zero backend compilations while new position/data/weight bindings
remain independent from the parent. This is code caching, not session persistence.

`StateGraph::state` registers an F32 slot. `read` returns its current symbolic
value; `write` changes that symbolic version, not live buffers. `compile` accepts
only visible tensor results and adds every final state version as a hidden output
root. An empty visible-result list is valid when slots exist.

For single-slot transformations, `graph.update(&slot, |old| old.add(&delta))`
reads the current symbolic value, validates and records the returned value, and
returns that new version. `update_index` provides the I32 equivalent. Closures
execute once during graph construction: tensor operations are recorded, ordinary
Rust mutations are not captured or replayed by a session. Invalid slot identity
or dtype rejects before invoking the closure; closure errors or invalid returned
values preserve the current slot version, but do not undo graph nodes or host
side effects. Sequential calls observe earlier writes, not simultaneous updates.
Native tests exercise changed runtime inputs, integer counters, preserved state
after recording errors, and a host counter that runs only during construction.

`graph.update_if(&mask, &slot, |old| old.add(&delta))` and its I32 counterpart
`update_index_if` record a conditional single-slot transformation. They return
the selected version: scalar F32 zero (including negative zero) retains the old
value, while nonzero (including NaN) takes the proposal. Later updates read this
selected value. Mask ownership/shape and slot ownership/type are checked before
the closure runs. Invalid proposals leave the current symbolic version unchanged.
These helpers use the same selection lowering as `write_many_if`, not lazy
branches: even a constant false mask still invokes the closure during graph
construction. Use the batch API for jointly validated multi-slot updates.
Native tests check repeated session calls, selected-value gradients, mixed
F32/I32 state, failed proposals, and that the closure runs only once per graph.

Use `write_many(&[(&k_slot, &next_k), (&v_slot, &next_v)])` for coupled symbolic
updates. All slot identities, shapes and graph ownership are checked before any
version changes. Duplicate slots (including cloned handles) are rejected; an
empty batch is a no-op and omitted slots are preserved. Values are constructed
before the call, so swapping previously read state values is simultaneous.
Errors leave slot versions unchanged, but do not undo tensor nodes the caller
already constructed. This is graph-building atomicity, separate from runtime
session commit semantics.

`StateProgram::session` binds initial buffers by slot identity and creates an
independent state owner without recompiling. `Session::run(&mut self, inputs)`
passes only visible inputs; state buffers are bound internally and committed after
successful execution and output validation. `state(&slot)` borrows a resident
buffer for inspection. Slot shape is fixed; this layer currently supports F32/I32
state and F32/I32 visible inputs. There is no native-buffer donation, async API,
automatic RNG handling or tracing of arbitrary Rust mutations.

`StateGraph::compile_pruned` opts into removing unused input parameters from the compiled ABI.
Reachability includes all visible outputs AND final state versions, so inputs
used only by hidden updates remain live. The original graph is not changed;
ordinary `compile` still preserves all declared inputs.

`program.input_indices()` lists required original visible-input registration
numbers in call order, before fixed binding. For example, `[0, 2]` means input 1
was removed, not that original input 2 has been renamed to 1. Numeric fixed
bindings and `Parameter` handles keep their original identities. Binding a
pruned input fails before changing existing bindings. Clearing fixed bindings
restores only the retained inputs; inherited session bindings keep this mapping.
`session.input_count()` reports the remaining dynamic buffer count.

All state slots remain in the schema, even when a state's old value is unused
because it is overwritten. Initialization/checkpoint completeness and ownership
transfer remain unchanged between pruned and unpruned plans of the same schema.
Pruning changes the executable ABI/cache key as needed, not parameter values or
state identities. Native tests cover interleaved inputs/state, hidden-only input
dependencies, original-index fixed binding, cached compilation, independent
sessions, transfer to an unpruned program, and zero-input constant programs.

The disk-cache native regression also launches a fresh process whose rebuilt
source graph contains additional unused F32/I32 inputs, changing original input
numbers from `[1, 2, 3]` to `[2, 3, 5]`. Both graphs compact to the same executable
ABI. The parent compiles once; the child records one disk hit and zero backend
compilations, then one memory-cache hit. It binds different weights and initial
state, verifies numerical updates and exact I32 counters above 2^24, and checks
independent sessions plus rejected-argument state preservation. Numeric bindings
use the child's original input numbers, not the parent's. Only executable code
is persisted: graph-local identities, the original-input mapping, weights and
state are rebuilt/rebound by the caller. This is a same-plugin/same-host CPU
compatibility test, not cross-version or cross-device cache portability.

`session.bind_inputs(Vec<(usize, Buffer)>)` fixes selected visible inputs,
such as inference weights, once. Indices are the original visible input order,
excluding hidden state parameters. Subsequent `run` calls supply only unbound
inputs in their original relative order; `input_count()` reports that count.
For example, TinyLlama binds 201 weight buffers and then calls
`session.run(&[&token])` at every step, with position in an I32 state slot. `Arc` lets independent sessions
share read-only weights without copying while owning separate mutable KV states.

`session.new_session(initial_state)` creates another session with the same
compiled program and current fixed bindings, without rebinding each weight.
Supply every state slot exactly once, in any order, with its own owned buffer:

```rust,ignore
let mut request = template.new_session(vec![
    (cache.key_slot().clone(), fresh_keys),
    (cache.value_slot().clone(), fresh_values),
    (position.clone(), zero_position),
])?;
request.run(&[&token])?;
```

This inherits both identity-based and numeric fixed bindings and preserves the
dynamic input order. It shares only read-only fixed buffers and code, not the
template's mutable state. Later rebinding either session is independent, and
dropping the template does not invalidate the new session. Initial state is
explicit: this is not cloning a prompt's populated KV cache or an automatic
zero initializer. Invalid initialization consumes supplied buffers but leaves
the template intact. There is no tensor payload copy, execution or compilation
in this method, and it does not introduce cross-thread or asynchronous execution.
The native KV-cache regression interleaves two sessions with resident positions,
checks distinct key/value contents after weight rebinding, and continues the
child after dropping the parent; the compiler records one compilation.

For a schema whose entire state should start at zero, use the explicit helper:

```rust,ignore
let initial = template.program().zero_state(&[
    cache.key_slot().clone(), cache.value_slot().clone(), position.clone(),
])?;
let mut request = template.new_session(initial)?;
```

`zero_state` validates the complete, unique slot set before uploading anything,
then creates fresh typed buffers using host staging. Scalar and empty shapes are
supported. It does not compile, execute, copy the template's state, or initialize
fixed input bindings. **It zeros resident parameters too**: use checkpoint or
explicit buffer initialization for schemas with weights or nonzero initial state.
Invalid missing/duplicate/foreign/later-registered slots return errors. Native
CPU/CUDA tests cover mixed F32/I32 state and independent session execution.

Each `bind_inputs` call replaces the entire binding set; an empty vector clears
it. Duplicate/out-of-range indices, foreign clients and mismatched shapes/dtypes
are rejected before commit, preserving previous bindings and state. Rebinding
does not recompile or embed weight values in HLO/cache keys. `replace_state`
does not change input bindings. Execution still validates the full parameter list;
this is an ownership/API improvement, not a claim of reduced runtime validation
cost or automatic cross-client sharing.

`Session::into_state(self)` consumes a session and returns all owned state buffers
with their slot identities, without needing replacement buffers. Resume with
`program.session(saved_state)` or transfer into another compatible session with
`replace_state`. This moves ownership, not tensor data: the buffers remain device
resident, not serialized or offloaded. Fixed inputs/weights are not included;
retain shared weights and explicitly rebind them after creating the new session.
The original session cannot be used after transfer, enforced by Rust ownership.

For an independent copy, including across CPU/CUDA clients, use:

```rust,ignore
// Compile the same StateGraph separately for the destination client first.
let copied = session.copy_state_to_client_via_host_with_limits(
    &destination_client,
    64 * 1024 * 1024,  // Maximum payload for any individual tensor.
    256 * 1024 * 1024, // Maximum sum of all resident-slot payloads.
)?;
let mut fork = destination_program.session(copied)?;
// Rebind destination-compatible fixed crates/rxla-weights/inputs explicitly, if required.
```

This copies every resident state slot, including resident parameters, and keeps
the source session usable. Slot identities are preserved; this does not map
independently constructed schemas. Fixed input bindings are excluded. Copying is
synchronous via host memory, even for the same client. All native download sizes
are checked before any payload transfer, with checked addition for the total.
Aliases count once per slot because every slot is copied separately. These limits
bound per-tensor payload and total transfer volume, not peak host memory, device
allocation or concurrent sessions. `copy_state_to_client_via_host` remains the
convenience variant with only a per-tensor limit. `Buffer::host_payload_bytes`
exposes the same native size query without payload allocation or transfer.
An error drops partial destination copies without replacing any source state.
Successful copies are independent and survive the source session's destruction.
This is not live migration, zero-copy transport, or cross-thread native ownership.

Conditional multi-slot updates can be recorded directly:

```rust,ignore
graph.write_many_if(&accepted, &[
    (&cache_slot, next_cache.into()),
    (&position_slot, next_position.into()),
])?;
```

`accepted` is a graph-local scalar F32 mask: zero preserves all listed slots;
nonzero (including NaN) selects their proposed values. Prefer a comparison-made
0/1 mask for validation conditions. Mixed slot shapes/types are supported, and
all updates are checked before any symbolic slot advances. Proposed expressions
still compute; visible outputs are not automatically guarded. Read the slots
after recording the update to return their conditionally selected versions.

```sh
PJRT_PLUGIN_PATH=/path/to/libzml_cpu.so cargo run -p rxla-core --example stateful_cache
```

## Experimental scalar-loss differentiation

`StateGraph::trainable_parameter(shape)` registers an F32 state-backed Parameter
usable by existing module `from_parameters` constructors and parameter discovery:

```rust,ignore
let weight = graph.trainable_parameter(&[out_features, in_features])?;
let model = Linear::from_parameters(weight.clone(), None)?;
let prediction = model.forward(&input)?;
let gradients = loss.grad(&[weight.tensor().clone()])?;
graph.write(weight.state_slot().unwrap(), &updated_weight)?;
```

Initialize `state_slot()` in `program.session(...)`, not `bind_parameters`.

`init::Initializer` supplies explicit seeded host F32 initialization, ready for
`Client::buffer_f32`, without embedding initial values in HLO constants:

```rust,ignore
use rxla_train::init::Initializer;

// Assign a stable stream ID per unique parameter, independent of traversal order.
let mut init = Initializer::new(42, 0);
let values = init.xavier_uniform(&[64, 128], 128, 64, 1.0)?;
let buffer = client.buffer_f32(&[64, 128], &values)?;
// Bind as a fixed inference weight, or initialize its resident training slot.
```

`uniform(shape, lower, upper)`, `xavier_uniform(shape, fan_in, fan_out, gain)`
and `kaiming_uniform(shape, fan_in, gain)` return row-major `Vec<f32>`. Xavier
uses bounds ±gain*sqrt(6/(fan_in+fan_out)); Kaiming uses
±gain*sqrt(3/fan_in), with gain sqrt(2) for ReLU. Fans are explicit, positive
counts: for OIHW convolution weights fan-in is input-channels-per-group times
kernel area. Fan-out conventions, transposed layouts and nonlinearity gains
remain caller policy, not inferred from arbitrary tensor shapes. Biases can be
initialized to zero with `uniform(shape, 0., 0.)` without consuming randomness.

`normal(shape, mean, stddev)` samples an untruncated normal distribution using
`rand_distr::StandardNormal` in F64, scales in F64, then rounds to F32. For
example, `Initializer::new(42, 0).normal(&[vocab, width], 0., 0.02)?` produces a
host embedding table; the appropriate standard deviation remains model policy.
Mean/stddev must be finite and stddev nonnegative. Zero stddev fills the mean
without consuming randomness. Samples that overflow F32 return an error, not
clipped values or an implicitly truncated distribution. Sampling uses a local
copy of the RNG state and commits it only on success, so even sample-overflow
errors leave the stream unchanged; identical retries repeat the failure.
Normal sampling consumes a variable number of random words and its exact
sequence depends on the distribution implementation as well as the generator.

`bernoulli(shape, probability)` returns F32 values exactly equal to 0 or 1,
ready for upload as a dropout keep mask. It uses the upstream Bernoulli
distribution's 64-bit threshold instead of thresholding rounded F32 uniforms;
extremely small probabilities inherit that threshold's resolution. Probability
must be finite in [0, 1]. Endpoints fill without consuming random draws, and
empty outputs/invalid arguments/allocation failures also preserve RNG state.
The method does not apply inverted-dropout scaling or upload/generate masks on
the device. Its random sequence differs from manual F32 uniform thresholding;
checkpoint replay must retain the sampling policy as well as the seed/state.
Host tests compare the upstream sequence, empirical frequency, endpoint/error
behavior, chunked replay and subsequent normal draws after snapshot restoration.

This uses the `rand_chacha` library's ChaCha8 generator, seeded without OS entropy
or global state. Each nonconstant uniform element consumes one word's high 24 bits,
interpolated in F64 then rounded to F32 (rounding can include the upper bound).
Equal seeds, stream IDs and draw sequences reproduce values for this
implementation; do not assume a cross-version checkpoint format. Explicit
per-parameter streams avoid traversal-order dependence; tied parameters should
be initialized once, not independently per alias. Invalid arguments and failed
allocation do not advance the stream. Empty tensors and constant fills consume
no draws. This is host initialization, not a JIT random operator, dropout,
automatic module initialization, or automatic training RNG checkpointing.

Host tests check reproducibility, stream separation, chunked draws, extreme
finite bounds and variance scaling. The real CPU initialization test trains
three seeded resident weights against an independent F64 SGD reference for 40
steps each, using one compiled program. Different initial values are runtime
state and do not trigger recompilation.

`initializer.snapshot()` returns explicit `InitializerState` fields: format
version, expanded 32-byte ChaCha seed, stream ID and a u128 word position.
`Initializer::from_snapshot(&state)` restores an independent stream without
replaying the prefix. Partial generator blocks and variable-length normal draws
are accounted for; unsupported versions and positions outside 0..2^68 are
rejected rather than narrowed/wrapped. Ordinary generator advancement wraps at
its 2^68-word period. Snapshotting itself does not advance the stream.

This is a host RNG state representation, not an automatic file format or part
of `Session::into_state`. Persist it explicitly alongside model/optimizer state,
data cursor and compatible distribution/config metadata. Exact distribution
replay still depends on compatible sampling code/dependencies and math; the
snapshot is not a cross-version guarantee. Restoring the same snapshot twice
intentionally produces duplicate random streams, not independent substreams.
The seed reveals future random values; do not log snapshots if those values
must be kept private. Host tests check mixed normal/uniform replay across partial
blocks, large valid word offsets and rejected version/position fields.
Normal initialization additionally checks the upstream sampling/scaling sequence,
empirical moments and tails, chunked reproducibility and failed-draw rollback.
A real CPU test uploads three seeded normal embedding tables into the same
compiled program, checks exact lookup values and repeated-index squared-loss
gradients against a host reference, and verifies the resident table stays intact.

Input and resident parameter identities remain distinct even when their numeric
indices coincide; aliases of one resident parameter are still deduplicated.
Existing `Linear::new` and `StateGraph::parameter` keep their input-parameter
semantics; training does not silently change inference bindings. Binding a
resident parameter as a fixed input fails before replacing any bindings.

The parameter tensor is a start-of-step symbolic value, not a live reference to
the latest recorded StateGraph write. Module forward expressions built after an
optimizer write still use that start-of-step value; explicitly read the state
slot for post-update computations within the same graph. On the next Session
call it receives the updated resident buffer. Parameter discovery may mix fixed
and resident handles: callers must classify them by `state_slot()` when loading
and initializing, not blindly pass all of them to `bind_parameters`.
The optional checkpoint loader's `load_module_parts` performs this split into
owned resident `states` and fixed `inputs`, retaining its full preflight and
shared-parameter mapping rules. Add non-parameter state explicitly before session
creation; loading does not construct or mutate a session.
For an already compiled plan, prefer
`checkpoint.load_module_for_program(&program, &model, &mapping)`. This uses the
plan's client and additionally validates every model parameter's retained identity
and type before reading payloads or uploading buffers. Pruned inputs, foreign
graphs and parameters registered after compilation are errors, not silently
skipped: pass a retained submodule explicitly when appropriate. It returns the
same `ModuleBuffers` split, preserving alias rules and loading each unique
parameter once. `StateProgram::parameter_type` exposes the corresponding metadata
check for both input-backed and resident handles. Native tests verify zero reads/
uploads for incompatible plans and successful mixed fixed/resident execution
after dropping the builder. Existing loaders remain available before compilation.
`Session::parameter(&handle)` borrows the current runtime buffer uniformly for
resident parameters and fixed-bound input parameters. Unlike `Parameter::tensor`,
it reflects completed training steps, state replacement and input rebinding.
The lookup performs no native work or data transfer; an explicit `to_vec_f32()`
downloads data for inspection. Unbound dynamic parameters cannot be inspected,
even after a call: Session does not retain dynamic call arguments. Foreign and
post-compilation handles are rejected. The returned borrow is not a snapshot
and prevents mutable Session operations while it is in use.
The optional `rxla_weights::serialize_module(&session, &model)` uses this lookup
to export current parameters as F32/BF16 safetensors bytes, preserving buffer
storage dtype. Unique primary paths are
saved once; aliases must map back to those keys on reload. It neither writes
files nor includes optimizer/other session state. The returned result resides in
host memory. `write_module` instead streams to a caller-owned writer, downloading
one whole parameter at a time and encoding in 64 KiB chunks; peak extra memory
still includes the largest parameter and the writer's own buffers. Publication,
flush/durability and discarding partial output on errors remain caller policies.
`train_linear` now uses a real Linear module with resident weight and bias;
native tests cover collection, training, binding rejection and this version rule.

`loss.grad(&[x.clone(), weight.tensor().clone()])` appends a reverse-mode F32
gradient graph and returns gradients in the requested order. Loss must be scalar
and differentiation targets must be graph input/parameter leaves (including
initial reads of state slots), not intermediate expressions. Disconnected targets
get zeros; duplicate targets return the same gradient. There is no native work
during differentiation: compile the loss and gradients together, or compose an
explicit optimizer update into `StateGraph` before compiling.

For plain module-level SGD, use:

```rust,ignore
let learning_rate = graph.input(&[])?;
rxla_train::optim::sgd_step(&mut graph, &model, &loss, &learning_rate)?;
let program = graph.compile(&mut compiler, &[loss])?;
// Initialize a session once; each run updates its resident weights.
```

The helper discovers and deduplicates module parameters, differentiates the scalar
loss and records simultaneous updates. Only parameters registered with
`trainable_parameter` are selected; input-backed parameters stay frozen. Learning
rate values can change between executions without recompilation. There must be
at least one trainable parameter, and none may already have been written in the
same graph: the helper rejects stale start-of-step parameter expressions instead
of silently composing multiple optimizer steps. Repeated session executions are
supported. Errors preserve symbolic slot versions, but do not roll back tensor
nodes. This is plain F32 SGD without momentum, weight decay, finite guards or
implicit initialization; use explicit grad/state operations for custom updates.
Native tests cover tied identities, frozen inputs, disconnected parameters,
runtime rates and independent sessions. `train_linear` demonstrates its use.

`optim::clip_grad_norm(&gradients, max_norm)` returns clipped gradients and the
original scalar global L2 norm. This is a graph-building utility for explicit
gradient/update composition, not an implicit option on optimizer `backward_step`:

```rust,ignore
let gradients = loss.grad(&parameter_tensors)?;
let (clipped, original_norm) = rxla_train::optim::clip_grad_norm(&gradients, 1.0)?;
// Construct updates using `clipped`, then record them with graph.write_many(...).
```

The nonempty tensor list must share one graph; empty tensors are allowed. Repeated
entries count repeatedly, so deduplicate parameter aliases before differentiation.
The bound is a finite positive construction-time F32 value. The calculation
normalizes by the largest absolute element before squaring and constructs clipped
values from normalized gradients, avoiding a tiny scale that could underflow before
multiplication. The reported norm can still overflow when the true norm exceeds
F32 range, without making finite clipped outputs overflow. Zero inputs stay zero;
NaN/Inf inputs are not sanitized or rejected. Backend rounding/subnormal flushing
still applies. There is no implicit detach, state mutation or host transfer, and
no guarantee of well-defined higher derivatives at zero or the clipping boundary.
CPU tests compare mixed-shape/repeated/empty tensor inputs with F64 references,
including `f32::MAX`, `1e30`, `1e-30` and a `1e-30` bound. A 12-step resident SGD
test checks both clipped and unclipped phases with one backend compilation.
Normalization uses two factors near `sqrt(max_abs)`: directly dividing by
`max_abs` passed isolated clipping tests but lost the gradient contribution for
`f32::MAX` inputs when composed with Adam on the tested CPU backend. The split
form with explicit optimization barriers passes the isolated clipping and combined
crates/rxla-weights/moments/powers F64 regressions; splitting without barriers failed the
isolated test after compilation. `Tensor::optimization_barrier()` lowers to XLA's
`opt-barrier` and has an identity derivative, unlike `detach`. It is not a device
copy or synchronization. Native tests check its values and first/second derivatives.
The opcode spelling follows the [OpenXLA opcode list](https://github.com/openxla/xla/blob/main/xla/hlo/ir/hlo_opcode.h).
This is not a
guarantee against every backend's algebraic reassociation or subnormal behavior.

`optim::MomentumSgd` adds resident velocity state without host-side per-step
parameter traversal:

```rust,ignore
let optimizer = rxla_train::optim::MomentumSgd::new(&mut graph, &model, 0.9)?;
optimizer.backward_step(&mut graph, &loss, &learning_rate)?;
let program = graph.compile(&mut compiler, &[loss])?;
let mut initial = optimizer.zero_state(&client)?;
initial.extend(model_state_buffers); // Every model/other state slot is required.
let mut session = program.session(initial)?;
```

The rule is `v = momentum * v + grad; weight = weight - learning_rate * v`.
Momentum is a finite construction-time value in `[0, 1)`; learning rate remains a
scalar tensor. There is no dampening, Nesterov, weight decay or implicit finite
guard. Zero learning rate still advances velocity. Disconnected parameters receive
zero gradients, but their previous velocity decays and can still move weights.
Weights and velocities are committed together using ordinary state outputs.
Each session owns independent buffers; the optimizer only retains symbolic
identities. `zero_state` explicitly allocates/uploads one host zero array at a
time, and device velocity storage adds one F32 tensor per unique trainable
parameter. Use `optimizer.velocity(&parameter)` to inspect a slot or supply
restored buffers; `Session::replace_state` requires a complete state set.
This is not an optimizer checkpoint file format. CPU tests compare weights,
velocities and loss with independent F64 recurrences, including nonzero restored
momentum, tied/frozen/disconnected parameters, runtime rate changes and session
isolation. The same stale-parameter/error rules as plain SGD apply.

Momentum also exposes `parameters()`, `gradients(&graph, &loss)`,
`apply_gradients` and `apply_gradients_if`, matching Adam's explicit gradient
pipeline below. Parameters are deduplicated in discovery order and frozen inputs
are excluded. Application validates count, exact shapes, owner and scalar
rate/condition before state writes; callers must preserve ordering because
same-shaped swaps cannot be detected. No implicit autodiff or detach is performed
when applying supplied gradients. Existing `backward_step` methods use this same
application path. Native tests compare clipped runtime gradients with independent
F64 momentum recurrences, including restored nonzero velocity, rejected NaN
inputs, zero learning rate, zero gradients and `f32::MAX` inputs. This is a pair
of matching concrete APIs, not a new generic optimizer trait or distributed reducer.

`optim::Adam::new(&mut graph, &model, AdamOptions::default())` provides
bias-corrected Adam. Record with `backward_step` / `backward_step_if`, then supply
`initial_state(&client)` alongside model buffers when constructing the session.
Defaults are beta1=0.9, beta2=0.999 and epsilon=1e-8. The update is
`m=beta1*m+(1-beta1)*grad`, `v=beta2*v+(1-beta2)*grad²`, then
`weight -= rate*(m/(1-beta1^t))/(sqrt(v/(1-beta2^t))+epsilon)`.
Epsilon is outside the square root. Betas must be finite in `[0,1)` and epsilon
finite and positive; there is no AMSGrad, weight decay, clipping or finite guard.
Learning rate is a scalar tensor, so schedules can change it without compiling.

For AdamW, use `Adam::new_with_weight_decay(&mut graph, &model, options, decay)`.
The construction-time decay must be finite and nonnegative. It changes only the
weight update to `(1 - rate*decay)*weight - rate*bias_corrected_update`; it is not
added to the gradient and does not enter either moment. Decay zero uses the exact
ordinary Adam graph path. Every selected resident parameter decays, including
biases and disconnected parameters; frozen inputs are excluded. There are no
implicit bias/norm exclusions or per-parameter groups. A rejected conditional step
preserves decay, weights, moments and powers together; zero learning rate still
advances moments/powers but does not decay weights. The existing explicit-gradient,
inspection and initialization methods also apply. CPU tests check independent F64
recurrences for zero/nonzero gradients, runtime zero/nonzero rates, rejected NaN
gradients and tied parameters, with one backend compilation. This adds no new
optimizer state buffers; a future checkpoint must still retain the construction
configuration (including decay) as well as the runtime state.

Adam stores two F32 moments per unique trainable parameter plus two shared F32
beta powers, initialized to one and advanced with each accepted step. Zero rate
still advances moments and powers. A rejected conditional step preserves weights,
both moments and both powers together. Input-backed parameters are frozen;
disconnected trainable parameters get zero gradients but retain decaying momentum.
`moments(&parameter)` and `beta_powers()` expose slots for inspection/restoration;
a consistent resume must restore these with weights from the same step. This is
not a checkpoint file format. All arithmetic, including beta-power accumulation,
is F32 and follows backend rounding/underflow; no long-horizon accuracy guarantee
is implied. Native tests compare 24-step recurrences to independent F64 values
for default, zero and alternative betas, runtime targets/rates, tied/frozen and
disconnected parameters, and rejected NaN batches. The `train_binary` native
tests also train with Adam for 100 steps: loss 0.005381027, all four labels correct,
one backend compilation. This toy example is not an optimizer performance ranking.

For clipping or other explicit gradient processing with Adam:

```rust,ignore
let gradients = optimizer.gradients(&graph, &loss)?;
let (clipped, norm) = rxla_train::optim::clip_grad_norm(&gradients, 1.0)?;
optimizer.apply_gradients_if(&mut graph, &clipped, &learning_rate, &accepted)?;
```

`parameters()` exposes the unique trainable parameters in gradient order, excluding
frozen inputs. `apply_gradients` / `apply_gradients_if` do not invoke autodiff and
also accept runtime gradient inputs. They validate count, exact shape and graph
ownership before state writes, and reject stale parameters like `backward_step`.
Same-shaped reordered gradients cannot be detected; callers must preserve order.
No automatic detach or finite check is added. Native tests combine clipping and
Adam with runtime gradients, zero and extreme inputs, rejected NaN updates and
invalid calls followed by a valid update, checking all Adam state against F64.
Existing `backward_step` methods reuse this application path without clipping.

`MomentumSgd::prepare_gradients(&graph, gradients, rate)` returns a non-Clone
`OptimizerUpdate` containing proposed weights AND velocities without recording
state writes. Its scalar `finite_mask()` checks every proposed output element
(empty tensors are vacuously finite). Combine this mask with loss/RNG/other
guards before `commit_if(&mut graph, &accepted)`. Commit consumes the proposal
and rejects changed source-state versions, foreign graphs or invalid predicates
before any slot is written. `commit()` is unguarded; existing momentum application
methods use this path without silently adding finite checks. This checks actual
update results, not positivity of the rate, loss validity or general numerical
accuracy. `Adam::prepare_gradients` exposes the same interface for Adam/AdamW,
including weights, first/second moments and both beta powers in one proposal.
Its existing application methods also remain unguarded by default. Native tests
check repeated accepted steps against an F64 recurrence for both optimizers,
zero-rate moment advancement, finite-gradient second-moment overflow, weight
overflow and nonfinite beta powers. Rejected steps preserve every state bit.

Multiple optimizer groups can share a single commit:

```rust,ignore
let update = first.prepare_gradients(&graph, &first_gradients, &first_rate)?
    .merge(second.prepare_gradients(&graph, &second_gradients, &second_rate)?)?;
let accepted = loss.is_finite_mask()?.mul(&update.finite_mask()?)?;
update.commit_if(&mut graph, &accepted)?;
```

`merge` consumes both proposals, rejecting overlapping parameter identities
(including tied weights) and foreign graphs without recording writes. The merged
finite check and commit cover both groups' weights, moments and counters/powers;
there is no implicit finite guard until the caller opts in as above. Parameter
order is concatenated, first then second. A merged proposal used with an
accumulator must match its complete parameter order, or joint commit rejects it.
Native tests combine AdamW and Momentum with different rates, verify that either
group's overflow preserves all seven state slots, and check accepted updates
against independent F64 recurrences. Stale sources in a later group are checked
before any group is written. This is not implicit scheduling of optimizers.

Native tests cover finite-gradient weight overflow, velocity overflow, NaN
gradients, infinite rates, zero-rate momentum advancement and stale/error paths.
A joint RNG test retries two finite-gradient/overflowed-weight attempts without
advancing either optimizer or random state, then accepts a valid gradient using
the same random draw. Device dropout training now combines proposed-update
finiteness with loss and counter-availability guards, retaining its convergence,
rejected-loss and counter-exhaustion checks. Existing optimizer behavior remains
unchanged unless the caller explicitly uses the proposal's finite mask.

For unequal microbatch sample/token counts, use
`GradientAccumulator::new_weighted(&mut graph, &model, microbatches)` and
`prepare_weighted(&graph, &mean_gradients, &weight)`. The scalar F32 weight is
normally the count underlying that batch's mean loss. It computes
`sum(weight * mean_gradient) / sum(weight)`, not the mean of batch means. All
parameters share that weight. Count still determines when a window completes;
an extra scalar slot, exposed by `weight_sum_slot()`, stores total weight and is
included by `initial_state()`. Preserve it in mid-window checkpoints.

Weights must be finite and strictly positive; zero-weight batches reject rather
than consume a window position. Weighted-sum or total overflow rejects the whole
proposal. Empty restored windows require total zero; nonempty windows require a
finite positive total. Full windows and explicit partial flushes clear sums,
count and total together. Existing optimizer-proposal coordination and stale
version checks include the total slot. `prepare`/`accumulate_if` deliberately
reject weighted mode so missing weights cannot silently change semantics.
F32 arithmetic still applies: this is not an exact integer-count accumulator,
and gradients must be caller-supplied means with the stated normalization.
Runtime tests verify weighted means, atomic SGD commits, flush/reset, rejected
crates/rxla-weights/overflow and preservation of partial state on CPU and CUDA.
The normalized CNN comparison described below also exercises unequal batches.

The `rxla-weights` native `weighted_resume` test saves a partial weighted window
to disk and restores it into a rebuilt graph whose optimizer/accumulator slots
were registered in a different order. Explicit names restore parameter,
momentum, gradient sum, I32 count and F32 total weight. Further accepted and
rejected batches reproduce the uninterrupted session's complete state exactly,
including a second completed window and another partial window. Omitting the
total-weight slot rejects save/load before publication/payload reads; the caller
also checks an explicit sample-weighting/window policy in checkpoint metadata.
CPU and selected CUDA gates run this test. This is same-environment rebuilt-graph
restoration, not process-crash durability, automatic policy discovery or a
distributed checkpoint protocol.

`optim::GradientAccumulator::new(&mut graph, &model, microbatches)` registers
one F32 sum per unique trainable parameter and a scalar I32 count. It uses the
same model-discovery order as Adam/Momentum; frozen inputs are excluded.
`initial_state(client)` uploads zeros; `sum_slot(parameter)` and `count_slot()`
expose checkpoint identities. Construction fixes a positive window size up to
i32::MAX; averaging divides by that size converted to F32.

```rust,ignore
let gradients = optimizer.gradients(&graph, &loss)?;
let batch = accumulator.accumulate_if(&mut graph, &gradients, &loss.is_finite_mask()?)?;
optimizer.apply_gradients_if(&mut graph, &batch.gradients, &rate, &batch.ready)?;
```

`accepted` means this microbatch was recorded; `ready` means a full window was
accepted and its sums/count reset. Pass `ready`, not `accepted`, to the optimizer.
The accumulator additionally rejects ANY nonfinite proposed sum and invalid
restored counts outside `[0, window)`, preserving every accumulator slot. It does
not repair corrupt checkpoint state. Empty tensors are vacuously finite. Gradient
count/shape/ownership and scalar-condition validation precede all state writes.
Returned means may be nonfinite on rejected attempts and are only complete-window
means when ready. There is no implicit detach or optimizer-result validation.
For joint acceptance with RNG, use `prepare(&graph, gradients)` instead of the
one-call helper. Its non-Clone `AccumulationProposal` exposes `acceptable()`
(finite sums and valid count) without recording writes. Combine it with loss
checks before committing the RNG, then commit the proposal with the RNG's returned
acceptance predicate:

```rust,ignore
let proposal = accumulator.prepare(&graph, &gradients)?;
let finite = loss.is_finite_mask()?.mul(proposal.acceptable())?;
let accepted = sequence.commit_if(&mut graph, &finite)?;
let batch = proposal.commit_if(&mut graph, &accepted)?;
optimizer.apply_gradients_if(&mut graph, &batch.gradients, &rate, &batch.ready)?;
```

Preparing/dropping a proposal leaves resident state unchanged. Commit consumes
it and checks graph ownership and current parameter/sum/count symbolic versions;
stale or invalid commits fail without modifying accumulator state. A failed
later graph-building call does not roll back earlier calls: this is not a
universal builder transaction. RNG is accepted on each valid microbatch, while
the optimizer advances only on a full accepted window. The stochastic AdamW
disk-resume test uses this composition. Native joint tests verify NaN/overflow
rejection does not advance RNG, RNG exhaustion does not clear a partial gradient
window, and a rejected draw is reused on retry.

To also guard optimizer overflow before clearing accumulated gradients, use
`proposal.gradients()` to prepare the optimizer before committing either group.
`proposal.would_step()` is 1 only for an acceptable full window or valid flush:

```rust,ignore
let proposal = accumulator.prepare(&graph, &gradients)?;
let update = optimizer.prepare_gradients(&graph, proposal.gradients(), &rate)?;
let safe = proposal.would_step()?.select(
    &update.finite_mask()?, &graph.constant(&[], &[1.])?,
)?;
let accept = loss.is_finite_mask()?.mul(proposal.acceptable())?.mul(&safe)?;
// With RNG: let accept = sequence.commit_if(&mut graph, &accept)?;
let batch = proposal.commit_if(&mut graph, &accept)?;
update.commit_if(&mut graph, &batch.ready)?;
```

Check optimizer finiteness only when `would_step()` is true: a finite partial
sum whose square overflows can still cancel against a later microbatch.
An overflowed full-window update retains the previously accepted sums/count
and all optimizer state; the rejected incoming microbatch is not stored.
The same composition protects partial-window flushes. `train_epochs` uses
this guard for training and flush plans. Native tests verify repeated rejects,
subsequent cancellation/retry and empty flush behavior. This remains explicit
runtime acceptance, not rollback across independent graph-building calls.

Without additional RNG/statistics groups, the shorter joint-commit API performs
the optimizer finite guard and records both groups in one validated write:

```rust,ignore
let proposal = accumulator.prepare(&graph, &gradients)?;
let update = optimizer.prepare_gradients(&graph, proposal.gradients(), &rate)?;
let batch = proposal.commit_with_optimizer_if(&mut graph, update, &condition)?;
```

Prepare the optimizer from this proposal's means. Joint commit checks the exact
ordered parameter identities, rejecting another model, a reordered set or a
partial optimizer even when all shapes match. Tied aliases are deduplicated by
both components and remain supported. Arbitrary gradient-expression provenance
is not inferred. Both proposals are consumed.
`accepted` reports microbatch acceptance; `ready` reports an optimizer update.
Scalar nonzero requests (including NaN) still require the finite/count guards.
Invalid conditions, stale sources or incompatible writes leave both groups'
symbolic versions unchanged. Other builder calls and other state groups are not
part of this transaction. `train_epochs` uses this API for both full and tail
updates. Native tests retain overflow/cancellation/retry coverage; builder tests
verify a valid proposal survives failed joint commits without partial writes.

For coordination with additional state, `proposal.with_optimizer(update)?`
returns a non-Clone `AccumulatedOptimizerUpdate`. Its `acceptable()` includes
sum/count validity and the full-window/flush optimizer guard. Combine this with
loss/statistics checks before advancing RNG, then call its `commit_if` using
the RNG's returned acceptance flag. Commit validates and writes both owned
groups together; preparing or dropping the compound proposal writes neither.
`commit_with_optimizer_if` is the one-call shorthand for this composition.
The ten-slot training acceptance and AdamW fresh-process restore tests use the
compound interface. It does not make earlier RNG/statistics builder writes
transactional; those remain separate explicit commits.
Native accumulator tests also exercise two differently shaped parameters,
NaN/Inf rejection, overflow of individually finite sums, rejected requests, full
window reset, bad restored counts, and one compilation across all runtime cases.

For accumulation across multiple optimizer groups, use identity-based routing:

```rust,ignore
let proposal = accumulator.prepare(&graph, &gradients)?;
let a = first.prepare_gradients(&graph, &proposal.gradients_for(&first_group)?, &first_rate)?;
let b = second.prepare_gradients(&graph, &proposal.gradients_for(&second_group)?, &second_rate)?;
let update = proposal.with_optimizer_groups(vec![a, b])?;
let batch = update.commit_if(&mut graph, &condition)?;
```

`gradients_for` follows the group's deduplicated trainable-parameter order,
skipping frozen inputs and rejecting unknown/foreign identities. It clones graph
handles, not device data. Group composition accepts reordered/noncontiguous
groups but requires exact collective coverage of the accumulator, rejecting empty,
overlapping or incomplete groups before writes. It does not infer arbitrary
gradient-expression provenance; prepare each optimizer from its routed means.
The existing single-proposal `with_optimizer` retains its strict order check.
Native tests compare reversed, noncontiguous groups with different momentum/rates
against an F64 recurrence, including partial acceptance, full-window optimizer
overflow rejection and cancellation/retry, using one compilation.

`cargo run -p rxla-train --example train_parameter_groups` is a complete
multi-optimizer classifier example: AdamW updates weights, Momentum updates bias,
and an accumulator combines two equal three-sample microbatches. Groups are
deliberately supplied in reverse model order. It checks all loss/gradient-sum/
parameter/moment/power states against an independent F64 softmax-and-optimizer
reference for 200 accepted microbatches / 100 joint updates. Two infinite-rate
attempts with finite loss preserve every bit of its ten state slots mid-window,
then normal training resumes. Final microbatch loss is approximately 0.00230;
the read-only inference plan predicts all six synthetic labels and preserves
state after buffer-ownership handoff. Training plus inference compile twice.
This example uses state downloads for verification, not throughput measurement,
and is included in the real-plugin regression gate.

`accumulator.prepare_flush(&graph)` explicitly proposes an end-of-window flush
without adding another gradient. It divides stored sums by the actual count
(converted to F32), accepting only finite sums and counts in `[1, window)`.
Commit it with `proposal.commit_if(&mut graph, &condition)`, then gate the
optimizer with returned `ready`. An accepted flush resets sums/count; an empty
window, nonfinite sum, corrupt count or rejected request preserves them and does
not request an optimizer step. Repeated empty flushes therefore do not decay
momentum. There is no RNG draw, implicit epoch detection, or sample-count weighting.
A flush proposal has the same stale-version/ownership checks as ordinary proposals.

Native tests compare one- and two-microbatch tails against actual-count means and
Momentum updates, repeat flush calls, and preserve rejected NaN/Inf state bitwise.
They also accumulate two real microbatches in a window of three, move the resident
buffers (with explicit slot-identity remapping) to a separately compiled flush
program, and verify the final mean/update/reset. Both programs compile once.
Session/plan selection and state remapping remain explicit; there is no automatic
trainer that swaps these programs at an epoch boundary.

`cargo run -p rxla-train --example train_accumulate` demonstrates resident gradient
accumulation using this interface (requires trusted `PJRT_PLUGIN_PATH`). Two
equal-sized microbatches contribute mean gradients to an F32 accumulator plus an
I32 count. Only the second accepted microbatch applies their average to Adam;
weights, moments and powers remain unchanged after the first. Accumulators reset
after that update. The example explicitly gates loss and accumulated-gradient
finiteness: a rejected microbatch preserves the partial sum, count and all Adam
state. Visible losses/proposals still compute; this is not lazy execution or an
automatic guarantee that every possible optimizer proposal is finite.

The native test performs 20 Adam steps from 40 accepted microbatches with 20
interleaved NaN batches rejected, checking every update against an independent
F64 merged-batch recurrence and requiring one backend compilation. Test assertions
download values; accumulation/update arithmetic itself stays on the device.
This example has no batch-dependent model state or stochastic layers, so it does
not claim that accumulated microbatches reproduce merged-batch BatchNorm/Dropout.
Unequal microbatch sizes need sample-weighted sums rather than this fixed average;
partial-tail flushing, generalized accumulation policies, distributed reduction
and a high-level trainer are not implemented by this example. The accumulator and
count must also be included in any mid-accumulation state handoff/checkpoint.

`session.switch_program_parameters(&destination, &mapping, bindings)` accepts
destination Parameter identities rather than numeric fixed-input indices. It
shares switch_program's validation-before-mutation and state-buffer movement
contract. Parameter ownership, storage dtype, duplication and pruned bindings are
checked before changing the old session. The original numeric-input API remains
available. This avoids exposing registration offsets during prefill/decode or
training/inference handoffs; state identity mapping is still explicit.

`session.switch_program(&destination, &mapping, fixed_inputs)` switches a live
session to another compiled plan. `mapping` is a complete list of
`(source_slot, destination_slot)` pairs; order is irrelevant, and every slot on
both sides must appear exactly once with matching dtype/shape. This supports
different registration orders and pruned input ABIs. Resident buffers transfer
ownership without graph execution, recompilation, or tensor payload copies.
Fixed input bindings are explicitly replaced using destination registration
numbers and cloneable `Buffer` handles; an empty set clears them. Remaining retained
inputs become dynamic in destination order. There is no implicit semantic name
matching, state-count migration, or cross-client tensor transfer.

All mappings, destination buffer compatibility and fixed bindings are validated
before switching. Errors leave the source plan, state and bindings intact, so
the original session remains runnable. Native tests switch reordered mixed
F32/I32 state forward/backward, replace/clear fixed bindings, reject incomplete,
duplicate, foreign and type-incompatible mappings, reject pruned/invalid fixed
inputs and foreign clients, and continue executing after failures. The gradient
tail test uses this API to switch from accumulation to flush without unpacking
the session. Callers still choose when to switch and which identities correspond.

For named schemas, `source_program.map_state_by_name(&source_names,
&destination_program, &destination_names)` creates that mapping once for reuse.
Each list contains `(&str, StateSlot)` pairs and must cover its entire program
exactly once. Names must be unique, nonempty and form the same exact set on both
sides; matched dtype/shape must agree. Order may differ, and output follows source
list order. This replaces hand-written name searches without executing a graph,
querying buffers or changing session state. `switch_program` still validates
actual buffer/client compatibility and replacement fixed inputs. Names are
application-owned semantic contracts, not automatically inferred parameter names;
equal names/shapes do not prove a model-version migration is correct. The epoch
example uses this helper in both directions, and native switching tests cover
invalid names, missing/duplicate/foreign slots and reordered mixed-dtype schemas.

`cargo run -p rxla-train --example train_epochs` combines accumulation, tail flush
and program switching in a complete deterministic epoch loop. Each of ten epochs
has four equal-sized microbatches and a window of three: one full-window Adam
step and one actual-count tail step. It switches the same session to a no-input
flush plan and back to the two-input training plan using explicit named state
mapping, preserving all seven parameter/optimizer/accumulator slots. It creates
initial buffers only for training; the flush phase receives the live buffers.
The example interleaves forty rejected NaN batches, repeats empty flushes, and
checks every loss and state transition against an independent F64 recurrence.
All twenty Adam steps, including ten tails, use two backend compilations total.
The native test is included in `scripts/check.sh --with-plugin`. This remains
a scalar synthetic correctness example, not a throughput or model-quality result.

To hand off a running Adam session without resetting its optimizer state:

```rust,ignore
let state = session.into_state(); // Consumes the old session, moves all buffers.
let mut resumed = program.session(state)?;
resumed.bind_parameters(fixed_bindings)?; // Fixed inputs must be bound again.
resumed.run(&dynamic_inputs)?;
```

The same program accepts the same slot identities; this is an in-process ownership
transfer, not a serialized checkpoint or cross-graph/cross-client migration.
No state buffers are copied/downloaded by the transfer. Fixed input bindings are
not resident state and do not survive `into_state`; dynamic inputs must still be
provided on each execution. Native Adam tests start from nonzero moments/powers,
transfer at step five, and compare all subsequent state values with uninterrupted
training and an independent F64 recurrence. A separate idle session stays unchanged.
The tests include a disconnected parameter whose restored momentum keeps moving
its weight: restoring only weights with fresh Adam moments would instead freeze
that parameter. A zero-rate step still advances its moments and beta powers.
This verification does not provide disk checkpoint/resume support.

For explicit momentum batch rejection, use
`optimizer.backward_step_if(&mut graph, &loss, &learning_rate, &condition)`.
The condition must be a graph-local scalar F32 mask. Zero (including negative
zero) retains every selected weight and velocity; nonzero (including NaN)
commits them all. Use a predicate such as `loss.is_finite_mask()`, not the raw
loss. Proposed updates still compute and visible outputs remain unguarded;
there is no automatic check that finite loss implies finite gradients/updates.
RNG, counters and other state are outside this optimizer's commit group.
`train_classifier` now uses a Linear module and momentum SGD, rejecting invalid
integer labels both before training and after nonzero velocities have developed.
It verifies that all weight/velocity buffers remain unchanged and training then
continues with the same executable. Separate native tests cover signed-zero
rejection of NaN/infinite proposals and nonzero/NaN condition semantics.

For nonscalar outputs, `output.vjp(&inputs, &cotangent)` seeds the same reverse
engine with a same-shape, graph-local F32 cotangent. It returns the vector-Jacobian
product without constructing a dense Jacobian or requiring a synthetic scalar
loss. Cotangents may be runtime inputs, so their values can change without
recompilation. Input-leaf, unsupported-operation and disconnected-input rules
match `grad`. The seed's own construction is not differentiated as an extra loss
factor during this sweep: for `y=x*x` with seed `x`, the result is `2*x*x`, not
the derivative of `sum(x*x*x)`. A later differentiation of that result sees its
explicit seed dependencies unless the seed was detached. Native tests cover
matrix outputs, dynamic seeds, scalar/empty outputs and this dependency rule.
This is a graph-building primitive, not a distributed backward scheduler.

Supported primitives: add/sub/mul/div, neg/exp/expm1/log/log1p/sqrt/rsqrt/tanh/erf,
abs, elementwise maximum/minimum, softplus, ReLU (gradient zero at either signed zero),
matmul (including vector promotion and batch broadcasting), reshape, transpose,
broadcast, sum/max reductions, static/dynamic slice and dynamic update,
concatenate/constant padding, average/max pooling,
conv2d/conv_transpose2d input/kernel gradients and gather data gradients.
Composite operations such as SiLU and mean can therefore be
differentiated. Any unsupported operation on a backward path from loss is rejected
before gradient recording, even if independent of the requested targets. In particular,
direct comparison gradients, backward through max-pool-gradient nodes,
and arbitrary state/control-flow transformations are not supported yet.
Mathematical derivative expressions do not regularize singularities or nonfinite
values. This is not general training support or a promise of higher-order support
for every future gradient rule; a polynomial second derivative is tested.

`take` and `take_along_axis` propagate gradients to their F32 data operand using
native scatter-add into zeros. Repeated indices accumulate rather than overwrite.
Indices are explicitly clamped to the same range as forward gather, since native
scatter's out-of-range behavior otherwise differs. Integer indices and their
construction paths are nondifferentiable; no index gradient is inferred. Native
tests cover all axes, scalar/multidimensional/empty indices, repeated and extreme
out-of-range indices, batched selection, a nonlinear embedding loss, and indexed
log-probability loss gradients. The gather adjoint's scatter-add reverse rule
gathers its cotangent using the original indices and batching mode; index paths
remain nondifferentiable. Native cubic-loss tests check first through fourth
derivatives on every axis for repeated/clamped runtime indices, multidimensional
and batched selections, and empty selections. This enables higher-order gather
composition, not differentiation of arbitrary scatter operations. Parallel floating-point
accumulation order and cross-device bitwise reproducibility are not guaranteed.
An indexed log loss can now be composed from log-softmax and take-along-axis;
validate class IDs separately because low-level gather still clamps them.

`dynamic_slice` gradients insert the cotangent into zeros using the same clamped
runtime indices. `dynamic_update_slice` zeros the overwritten region of its base
gradient and extracts that region for the update gradient. Both reuse existing
native primitives, so their reverse rules compose for higher derivatives. Integer
start indices and their construction paths are not differentiated. CPU cubic-loss
tests check first through fourth data derivatives with changed runtime starts,
negative/extreme clamped indices, empty regions and overwritten base values;
additional tests cover scalar arrays and starts produced by nondifferentiable
argmax. This does not differentiate arbitrary control flow or infer gradients
across separate Session executions.

Slice gradients restore the input shape with edge/interior zero padding, including
strided slices; empty selections produce zeros. Concatenation splits the incoming
gradient, and repeated operands accumulate their contributions. The public pad
operator's fill is constant: its input gradient crops the padded region. Native
tests cover strides/nondivisible limits, zero-length split pieces, repeated
concatenation operands, nonzero fill, scalar/empty arrays, and both RoPE layouts
with independently checked input/cosine/sine gradients. SliceGradient's reverse
rule extracts the original slice from its incoming cotangent, enabling further
differentiation without new native operations. A native cubic-loss regression
checks first through fourth derivatives for strided and empty selections, with
changed runtime inputs. This does not guarantee arbitrary high-order numerical
stability or support for unrelated unsupported operations in the same graph.

Average-pooling gradients reuse a sum-pooling adjoint: channels are reshaped into
independent batches and a transposed convolution with an all-ones spatial kernel
accumulates overlapping windows. This avoids a dense channel-by-channel identity
kernel. Output padding restores the original input extent when forward stride
division leaves a remainder; cropped/padded positions do not become input
gradients. Both full-window and valid-element divisors are handled by the existing
F32 forward composition. Empty input/output cases produce zero input gradients.
Native CPU tests compare forward values, VJPs and gradients through squared input
against independent scalar window/scatter loops, including multiple batches and
channels, overlapping/sparse windows, asymmetric/entirely padded windows (with
count_include_pad), stride remainders and empty dimensions. Entirely padded
exclude-pad windows still have NaN forward values and undefined gradients.
The adjoint's transposed convolution now has a reverse rule, enabling selected
second-order compositions; a squared-input pooling loss is independently checked.
The internal convolution adjoints also have reverse rules for further composition.
No GPU validation or performance improvement is implied.

Max-pooling backward lowers to HLO `select-and-scatter`: a scalar F32 `>=`
selection computation picks one maximum in each window and an F32 add computation
accumulates the incoming values at selected positions. Unlike max reduction's
equal-split rule, tied max-pooling entries receive a single-winner subgradient;
the tied index depends on backend/reduction schedule and is not guaranteed to be
the first index. NaN-containing windows have undefined gradients. Source windows
entirely outside the input are trimmed and padding adjusted before lowering,
because native select-and-scatter leaves empty-window selection unspecified.
They contribute zero, as do empty input/output cases. CPU references cover
unique maxima, overlap, multiple batches/channels, asymmetric/large padding,
stride remainders/gaps and empty shapes; separate tests verify a single winner
for tied finite values and equal infinities. The generated select-and-scatter
primitive has no reverse rule yet, so general higher-order gradients fail.

Conv2d reverse-mode supports both NHWC inputs and HWIO kernels for the existing
groups/stride/dilation/asymmetric-padding options. Input gradients lower to a
feature-grouped convolution with base dilation, reversed spatial windows and
group-aware kernel channel transposition. Kernel gradients use a convolution
whose batch/features/spatial dimensions are reassigned, with batch grouping to
pair each input-channel group with its output-channel group. Neither path expands
one Rust graph branch per group. High-side padding is derived from the requested
gradient shape to retain stride remainders. Empty input/output cases return zero
gradients. CPU tests independently accumulate forward values, input gradients and
kernel gradients in scalar loops for ordinary/grouped/depthwise-multiplier cases,
including multiple batches, changed runtime inputs/weights, dilation/stride
combinations and large/asymmetric padding. Internal input/kernel adjoint nodes
also have reverse rules, reusing convolution and the complementary adjoint.
This is primitive correctness evidence, not full YOLO training, GPU validation
or a performance comparison.

Ungrouped `conv_transpose2d` input/kernel gradients use the adjoint identity
`<transpose_conv(x,k),dy> = <x,conv(dy,k)>`: input gradients are an ordinary
convolution, while kernel gradients reuse the internal Conv2d kernel-adjoint
node with swapped input/cotangent roles. Output padding is accounted for through
the actual spatial shapes and derived high-side convolution padding. Native CPU
tests compare both gradients and forward values against independent scalar scatter
loops for multiple batches/channels, stride/dilation/cropping/output-padding
combinations, changed runtime crates/rxla-weights/inputs and empty batches. Backward through
the internal kernel-adjoint node is supported through the complementary input
adjoint and ordinary convolution. A native grouped 1×1 squared-loss regression
checks the joint input/weight Hessian-vector product against independent analytic
F64 formulas for two directions. A scalar convolution polynomial checks mixed
derivatives through fourth order and their zero fifth derivative. These establish
specific higher-order composition cases, not numerical stability of arbitrary
high-order networks, all configurations or nonfinite inputs. Other unsupported
operations on a backward path (such as max-pool-gradient nodes) still fail.

The `train_cnn` example composes trainable OIHW Conv2d (1→8, 3×3), SiLU,
average/max pooling and a Linear head (8→2) with sparse cross entropy and resident
momentum SGD. A custom `Parameterized` implementation exposes both submodules to
the optimizer, so the example also exercises module weight layout conversion,
parameter discovery and joint state updates. Each pooling variant compiles once
and runs 250 updates over eight synthetic 4×4 horizontal/vertical-line images.
Convolution and head weights use seeded Xavier uniform initialization with
explicit OIHW/linear fans and separate stream IDs; biases start at zero.
The executable uses seed 42. Each native test runs seeds 0, 42 and 999 with fresh
parameter/optimizer state while reusing the same compiled pooling variant.
Tests require all eight predictions correct, loss below 0.1 and below 20% of its
initial value, changed convolution/head weights and finite resident velocities.
Observed final pre-update CPU losses for seeds 0/42/999 were
0.0028524303/0.0025968754/0.0030188847 (average) and
0.0008493098/0.0015132787/0.0008576269 (max). All six runs classified all eight
samples correctly. Both tests are included in `check-xla.sh --with-plugin`.

The additional `real_cnn_accumulation_matches_whole_batch_updates` native test
compares an eight-image batch with two four-image microbatches using
`GradientAccumulator`. It checks all four parameter tensors and four momentum
tensors after each of 250 optimizer updates, preserves all parameter/momentum
state after the first microbatch, preserves partial sums/count on invalid-label
rejection, and verifies every sum is cleared at a completed window. Each pooling
variant compiles two programs total and reuses them across seeds 0/42/999.

Average pooling's independently evolving whole/microbatch sessions agree within
`5e-5 * (1 + abs(reference))` for state and `5e-5` for mean loss. Max pooling's
independent trajectories do NOT maintain that state tolerance: an initial test
failed at seed 0, step 173. The retained independent baseline observes maximum
state differences of about 6.98e-4/8.95e-5/2.38e-7 across the three seeds. A separate
whole-batch reference starts each max-pool step from the microbatch session's
current parameters/momentum; all single-step comparisons pass the original
tolerances. This separates local update agreement from accumulated trajectory
drift, without claiming the exact source of that drift has been proven.
The comparison additionally runs a host F64 OIHW-convolution/SiLU/pool-selection
reference on each trajectory's current weights, without adding outputs to the
compiled graphs (which could perturb fusion). It retains ALL tied maxima as a
bit set and measures the gap to the best losing candidate. Seed 0 first differs
at step 101, image 1/channel 3: bottom-row versus top-row winner sets, with gaps
about 9.89e-7 and 5.23e-6. Seed 42 first differs at step 189, image 1/channel 4,
with opposite row selections and gaps about 2.91e-5 and 3.89e-5. Seed 999 has no
reference selection-set difference in 250 steps. A host unit test checks unique,
pair-tied and all-tied selections and their gap calculation.
This is evidence that the independently evolving weights cross different pool
selection boundaries, consistent with amplification of small numerical drift.
It does NOT identify the first rounding difference or directly observe native
select-and-scatter's tied-index choice; neither tolerance nor pooling semantics
was changed. The measured training losses/state deltas remain unchanged.
All independent whole-batch and accumulated runs classify all eight samples
correctly. Final accumulated losses are approximately .00285243/.00259688/.00301889
(average) and .000849295/.001513279/.000857627 (max). Reference state alignment
uses host downloads/uploads in the test and is not a production training step
or a performance comparison. The existing full-batch tests remain unchanged.
Run the seed-42 comparison directly with
`cargo run -p rxla-train --example train_cnn -- --compare-accumulation`.

These seeded initializations are regression cases, not a general initializer
recommendation or convergence guarantee for arbitrary seeds. Before the seeded
initializer integration, a preliminary 4-channel max-pooling variant stalled
near loss 0.17347; an independent same-initialization PyTorch run reproduced that
plateau after 1000 updates. The final example uses eight channels for both pooling
variants without changing the data or lowering its acceptance thresholds. This
demonstrates composed CNN training on CPU, not generalization, full YOLO training,
GPU correctness or comparative backend performance.

```sh
PJRT_PLUGIN_PATH=/path/to/trusted/libzml_cpu.so \
  cargo run -p rxla-train --example train_cnn
```

Attention backward composition is independently checked on CPU: two batches,
four Q heads, and one/two/four KV heads exercise MQA/GQA/MHA. A scalar F64
attention reference with central differences checks every Q/K/V element and
finite mask-bias element, including per-head masks shared across batches and
causal negative-infinity entries. Masked entries have zero incoming score
gradient; every row in this test has a finite unmasked entry. The same executable
is reused for dense and causal masks. This validates shared-K/V and broadcast
gradient accumulation, not all-masked behavior, full transformer training,
FlashAttention performance, or GPU correctness.

F32 `select` differentiates its two value branches by routing the incoming
gradient to the selected branch. The numeric mask and its construction path are
nondifferentiable, including comparison-made masks; its gradient is zero. Both
branches still compute: an unselected branch with singular derivative arithmetic
can still introduce NaNs during backward computation. This is not lazy control
flow or a guarantee of numerical safety for arbitrary piecewise expressions.

`x.detach()` is an explicit reverse-mode boundary: forward values/dependencies
are unchanged, but differentiation stops at that use of `x`. Other undetached
uses still contribute gradients. This neither snapshots/copies device values nor
blocks compiler optimizations. Forward-only unsupported operations behind this
boundary are allowed; an undetached unsupported path still errors.
Softmax and log-softmax detach their maximum-based centering offset internally.
Their derivative with respect to a common shift is zero, so treating that offset
as constant gives the correct smooth derivatives without choosing a max tie rule.
Native tests compare weighted softmax/log-softmax gradients with independent F64
formulas on both matrix axes, including tied and widely separated finite logits.
They do not establish gradients for all-masked rows or nonfinite logits.

```sh
PJRT_PLUGIN_PATH=/path/to/trusted/libzml_cpu.so \
  cargo run -p rxla-train --example train_linear
```

The example learns `y = 2*x + 1` using explicit SGD and resident weight/bias
state, with 100 calls to one compiled forward/backward/update graph. Each call
returns the pre-update loss; no automatic optimizer, RNG or training mode is
introduced. Native tests verify MLP gradients against independent F64 central
differences, all supported unary rules against analytic references, layout and
broadcast accumulation, shared/disconnected inputs, vector/batched matmul, empty
broadcasts, and a converging resident-state training loop. They do not establish
full-model or distributed training readiness. No new binding generator or native
build dependency is required.

For classification, `logits.cross_entropy_with_probs(&targets, class_axis)`
computes `-sum(targets * log_softmax(logits), class_axis)`. It removes the class
axis without implicitly averaging batches; use `.mean(...)` or `.sum(...)`
explicitly. Targets are dense F32 with exactly the logits shape. Runtime target
nonnegativity and unit sums are caller responsibilities, not silently corrected:
arbitrary coefficients compute a weighted log loss. Both logits and targets are
differentiable; use `targets.detach()` to explicitly stop target gradients.
Finite logits/targets are expected; zero times an infinite log probability is
not masked away. Sparse integer labels, ignore-index and label smoothing are
not implicit policies. Native references check both derivatives and losses
against independent F64 formulas, including large finite logits and empty batches.

```sh
PJRT_PLUGIN_PATH=/path/to/trusted/libzml_cpu.so \
  cargo run -p rxla-train --example train_classifier
```

`logits.cross_entropy_with_indices(&class_ids, class_axis)` accepts I32 targets
with the class axis omitted. It uses indexed log probabilities, not a dense
one-hot label tensor. Negative or out-of-range IDs produce NaN per-example loss,
not a clamped valid class and not a runtime Rust error. Invalid-label gradients
are not meaningful: validate labels or explicitly guard state updates on finite
loss. Shape/owner/axis errors still fail during graph construction. Batch
reduction is explicit, and no ignore-index or smoothing policy is implied.

`Tracer::iota_i32(shape, axis)` and `StateGraph::iota_i32(shape, axis)`
generate I32 coordinates along a static axis with native HLO iota, without a
host-built literal array or an additional input/state slot. Coordinates repeat
over the other dimensions. Scalars have no valid axis; empty tensors are valid.
The axis length is limited to `i32::MAX + 1` so coordinates fit I32. A resident
scalar offset can be broadcast and added with `wrapping_add` to form successive
position blocks, then converted to F32 for trigonometric features. Overflow policy
remains explicit (wrapping is not bounds validation). Native tests check each axis,
empty shapes and three resident-offset position blocks with no dynamic input
buffers and one compilation. The backend may constant-fold iota; this is not a
promise of a particular device kernel or a general dynamic-shape arange API.

`Index::to_f32()` explicitly converts I32 values numerically to F32 in the graph.
It is not a bitcast and does not differentiate integer construction paths, even
when indices originate from argmax over floating-point inputs. Other floating
operands in a downstream expression still differentiate. Keep persistent counters
and checkpoints in I32: integers in `[-2^24, 2^24]` are exactly representable, but
larger integers may round on conversion. Native tests cover extrema, the 2^24
boundary, scalar/empty arrays, argmax-origin zero gradients, and a resident I32
position counter driving sine features and frequency gradients over six session
executions with one backend compilation.

`Tensor::sin()` and `cos()` operate elementwise on F32 radians through native HLO
sine/cosine. Their reverse rules are cosine and negative sine, respectively,
and compose for higher derivatives. This permits positions and learnable frequency
tensors to generate trigonometric features inside the compiled graph rather than
requiring host-precomputed tables. Native tests compare values and first through
fourth derivatives with F64 references over tested angles up to magnitude 1000,
including zero/small values and scalar/empty arrays. A broadcasted position-times-
frequency example verifies gradients to both runtime inputs. Argument reduction
and accuracy for arbitrarily large angles remain backend-dependent; this is not
a new automatic positional-encoding/RoPE policy or a mixed-precision guarantee.

`x.rotary_embedding_angles(&angles, layout)` computes cosine/sine in the graph
then applies the existing rotation, using the same half-width angle broadcasting
as `rotary_embedding`. For example, positions `[S,1]` and frequencies `[D/2]`
can be explicitly broadcast to `[S,D/2]` and multiplied to supply angles for
`[B,S,D]` activations. This leaves frequency schedules, scaling and axis policy
explicit; it does not silently replace precomputed tables in existing LLMs.
Native tests compare both SplitHalf and Interleaved layouts against independent
F64 rotation/gradient formulas, including input, position and frequency gradients
with batch accumulation and changed runtime positions/frequencies. Existing
cos/sin-table APIs remain available when precomputation is preferable.

`input.dropout_with_mask(&keep_mask, keep_probability)` implements inverted
dropout as graph select followed by division. The mask must be F32 with the
same shape/graph; use explicit broadcasting for shared-axis masks. Zero drops
an element, nonzero (including NaN) keeps it, consistent with `Tensor::select`.
Ordinary dropout requires the caller to provide Bernoulli(keep_probability)
masks. `Initializer::bernoulli(shape, keep_probability)` generates such masks
before upload; this incurs host generation/upload, not device RNG.
Keep probability is a finite construction-time value in (0, 1]. Changing mask
values is a runtime operation and does not recompile. Even at probability 1,
the explicit mask is honored; bypass dropout in inference rather than supplying
an arbitrary mask. No training/evaluation flag or hidden random state exists.

Value gradients use the same mask/scaling; mask derivatives are zero. The
dropped branch is selected to zero before scaling, so dropped NaN/Inf inputs
do not contaminate the output. Kept nonfinite values and genuine F32 overflow
remain possible. Recompute/checkpoint workflows must reuse the same mask;
generating a new mask during recomputation changes the function. Native tests
check changing/all-kept/all-dropped masks, first/second derivatives of squared
loss, zero mask derivatives, dropped nonfinite inputs, scalars, empty tensors
and nonbinary select semantics. This is not automatic random generation or a
complete stochastic training/recomputation framework.

The `train_dropout` example connects seeded host mask generation, input dropout
(keep probability 0.5), resident Linear parameters and Momentum SGD for synthetic
binary classification. Each step uploads a new mask; host RNG state is explicitly
separate from session state and is not automatically checkpointed. At step 150,
the example captures `random.snapshot()` plus `session.into_state()` and restores
both into new objects. Every subsequent Bernoulli mask matches an uninterrupted
reference stream exactly. This is an in-memory completed-step checkpoint handoff,
not file publication, process-restart recovery or data-cursor persistence. Training loss,
both parameters and both velocities are compared with an independent F64
logistic-regression update for the exact same masks over 300 steps. There is no
monotonic noisy-loss assumption. Parameter initialization and masks use separate
random streams.

The example compiles a dropout-free inference snapshot with `compile_pruned`
before optimizer writes,
then moves the trained device buffers into it with `into_state`. Repeated
inference has identical outputs and leaves every resident slot unchanged. The
unused mask input is removed from the inference ABI: only data and labels (for
the reported loss) are supplied, and an extra argument is rejected. The executable runs
seed 42; the native gate runs seeds 0/42/999, sharing just two compilations total
(one training and one inference). Observed CPU inference losses were
0.04441974/0.045791704/0.0448303 after adopting the Bernoulli sampler, with all four
examples correctly classified. These differ from the former uniform-threshold
sampler because the mask sequence changed, not because of a new optimizer or
backend optimization.
This verifies stochastic training/state handoff, not generalization, an automatic
RNG/recompute transform, or GPU performance.

```sh
PJRT_PLUGIN_PATH=/path/to/trusted/libzml_cpu.so \
  cargo run -p rxla-train --example train_dropout
```

`train_device_dropout` demonstrates the graph-side alternative: four resident
I32 slots hold two Threefry key words and low/high counter words. Each step
reserves four random blocks, converts one word per block to a uniform keep mask,
and trains a Linear classifier with Momentum SGD. No host mask/key/counter is
uploaded per step: the call supplies only activation data and labels. A shared
predicate gates optimizer and counter writes on finite loss, finite proposed
crates/rxla-weights/velocities and no counter wrap.
The example intentionally retains the RNG position after a rejected attempt;
other applications may choose a different advancement policy.
Training returns an explicit scalar accepted flag along with loss and mask:
a finite loss alone does not imply an update was committed. Exhaustion cases
start at 2^64-4, 2^64-3 and 2^64-1, run twice with finite inputs, and verify
accepted=0, repeatable masks/loss and all eight slots unchanged bit-for-bit.
This conservative policy rejects even the final four-block batch ending exactly
at wrap. It does not silently reseed; callers must stop or explicitly supply a
new key/counter policy instead of retrying forever. Valid steps report accepted=1,
and the NaN-label rejection below also reports accepted=0.

Tests run seeds 0/42/999 for 300 valid updates each, compare loss/weight/bias/both
velocities against F64 calculations for the downloaded masks, check low-word
counter carry, and hand off all device state at step 150. A NaN-label attempt
after 50 steps preserves all eight state slots bit-for-bit; the subsequent valid
step receives the same mask. Dropout-free inference after state transfer leaves
all four RNG words unchanged and correctly classifies the four synthetic inputs.
Observed inference losses are 0.045604967/0.04453469/0.04537983. The three seeds
share two compiled programs (training/inference). Masks and state are downloaded
for correctness checks, so this example is not a throughput benchmark or proof
that device RNG is faster. It does not add automatic RNG discovery, sharding,
recompute transforms or file checkpoint orchestration. Its native test is in
`check-xla.sh --with-plugin`.

```sh
PJRT_PLUGIN_PATH=/path/to/trusted/libzml_cpu.so \
  cargo run -p rxla-train --example train_device_dropout
```

`prediction.huber_loss(&targets, delta)` returns an unreduced, same-shape F32
regression loss: `0.5*r²` for `abs(r) <= delta`, otherwise
`delta*(abs(r)-0.5*delta)`, with `r=prediction-targets`. Delta must be finite and
positive; target broadcasting and detach are explicit. Both prediction and target
differentiate. At zero residual the prediction curvature is one. At `+/-delta`
the first derivative is continuous but the second derivative is undefined;
the implementation chooses the quadratic-side value (one). Inactive quadratic
residuals are masked before multiplication to prevent large residuals overflowing
that branch and contaminating gradients. Nonfinite residuals are not repaired,
and a genuinely out-of-range loss may overflow F32. Native tests check values,
both input gradients and prediction/target cross-curvature against independent
formulas, including zero/boundaries, scalar/empty tensors, `1e30` and F32 extrema.
There is no reduction or implicit conversion to SmoothL1 (which scales by delta).

`logits.binary_cross_entropy_with_logits(&targets)` returns a same-shape F32
elementwise binary loss using `y*softplus(-x) + (1-y)*softplus(x)`. It avoids
probability/log underflow and cancellation from subtracting large nearly equal
loss terms. Use mean/sum explicitly. Logits and soft labels are differentiable;
targets must match shape/graph, and their values in [0,1] are the caller's
responsibility (no runtime validation/clamping). Nonfinite inputs are not repaired;
zero times infinity can produce NaN. Class weighting, smoothing and ignore masks
remain explicit. CPU tests check losses, both input gradients and logits curvature
against independent F64 formulas, including hard/soft labels, large logits,
signed zeros, scalar/empty inputs and detached targets. The `train_binary` example
uses a Linear module with resident momentum SGD for 100 iterations of a four-point
binary classification problem, compiling once. Its native test runs in
`check-xla.sh --with-plugin`; this is not a full YOLO or generalization benchmark.

The deterministic `train_classifier` example combines integer-label cross entropy,
autodiff, resident F32 weight/bias updates and I32 argmax diagnostics. It compiles
one forward/backward/momentum-SGD program, runs 100 updates and checks all six
training predictions plus decreasing loss. A finite-loss guard retains weights,
biases and both velocities on invalid labels injected before training and after
20 valid updates; training then resumes with valid labels.
The example's native test is included in
`check-xla.sh --with-plugin`; host mode keeps it ignored. This is a small training
smoke, not a generalization, full-model accuracy, GPU or performance benchmark.

ReLU retains a dedicated semantic operation with forward HLO `maximum(x, 0)`;
its backward rule selects the incoming gradient only where `x > 0`. This avoids
implicitly inheriting a generic maximum tie rule. Finite-value boundary tests,
forward NaN/infinity behavior, and scalar/empty gradient cases run on the CPU
plugin. Elementwise maximum/minimum split the incoming gradient equally at ties
(including signed zeros and equal infinities), route it to the selected operand
at ordered unequal values, and produce NaN gradients for both operands if either
operand is NaN. Shared operands accumulate both contributions. Broadcasting must
still be explicit. Max-pooling uses a separate single-winner rule described above.

`max(axes, keepdims)` reductions distribute the incoming gradient equally across
every tied maximum in each reduced slice, including equal infinities and signed
zeros. The winner mask and its F32-reduced multiplicity are nondifferentiable;
there is no derivative of the discrete choice. A slice containing any NaN input
gets NaN gradients throughout, explicitly detected from the inputs: native forward
max reductions may ignore NaNs depending on backend/reduction order. Forward max
behavior is unchanged. Empty input gradients remain empty; empty axes are identity.
CPU tests cover all individual axes, multi-axis/all-axis reductions, keepdims,
dynamic winner patterns, nonfinite values, empty dimensions and a second derivative
of squared max under the chosen tie convention. This is not a classical derivative
at a tie or a promise of exact tie counts beyond F32 integer precision.

`min(axes, keepdims)` composes `-max(-x)` and inherits the same backward tie/NaN
policy. Empty reductions return positive infinity; empty axes are identity.
`logsumexp(axes, keepdims)` computes the log of summed exponentials using a
detached maximum shift. Finite slices have softmax gradients without differentiating
the centering choice. A nonfinite maximum is replaced by a zero shift to avoid
`inf-inf`: all-negative-infinity and empty reductions return negative infinity,
positive-infinity inputs return positive infinity unless a NaN contaminates the
slice, and NaNs propagate. Nonfinite-result gradients are undefined; finite slices
with negative-infinity masks give zero gradients at masked entries. Empty axes
return the input unchanged. CPU tests compare finite values/VJPs with independent
F64 formulas across axes/keepdims, large logits, ties and runtime inputs, plus a
Hessian-vector check at equal logits. No new native opcode or dependency is needed.

`moments(axes, keepdims, correction)` returns mean and two-pass variance with
matching shapes; `variance(...)` returns just the latter. Variance divides the
sum of squared centered values by `count - correction`: use 0 for population or
1 for sample variance. Correction is explicit, and count must exceed it. Empty
reduced dimensions are rejected; empty non-reduced dimensions are valid. Empty
axes describe one observation per input element (population variance zero for
finite inputs). The F32 two-pass formula avoids the `mean(x*x)-mean(x)^2`
cancellation pattern, but is not Welford, compensated summation or a promise of
accuracy for arbitrary offsets/counts. NaNs/infinities are not ignored or repaired.
Mean and variance remain differentiable graph values. CPU tests check axes,
keepdims, corrections, large offsets, constant/scalar/empty inputs and a joint
mean-plus-variance loss gradient against independent F64 formulas. Standard
deviation is an explicit `variance(...).sqrt()`; epsilon regularization is also
explicit because sqrt has a singular derivative at zero variance.

`batch_norm_training(axis, &weight, &bias, epsilon)` returns `BatchNormTraining`
with `output`, `mean` and `variance` graph values. It normalizes over all dimensions
except the channel axis using two-pass population variance; statistics are
`[channels]` and variance excludes epsilon. Affine tensors must be `[channels]`
in the same graph, with finite positive epsilon. One observation is allowed
(population variance zero); empty observation sets are rejected. Native CPU tests
compare output/statistics and all input/scale/bias gradients against independent
F64 formulas for different channel axes, constant/nonconstant inputs and a single
observation. This is the differentiable training computation, not a stateful
BatchNorm module: running averages, counters, unbiased running-variance correction
and commit guards remain explicit. Returned statistics retain gradient paths;
detach them when recording nondifferentiable running-stat updates. The existing
`batch_norm_inference` consumes supplied running statistics without updating them.

`BatchNormState::new(&mut graph, channels)` registers two non-parameter resident
slots. `initial_state(&client)` explicitly uploads mean-zero/variance-one buffers
for session initialization; no affine parameters or optimizer slots are included.
`update(&mut graph, &batch, rate)` records `(1-rate)*old + rate*batch` using
detached batch statistics. Rate is a finite construction-time number in [0,1]
weighting the **new** batch; population variance is used without automatic
unbiased correction. Rate 0 preserves old values and rate 1 directly replaces
them, avoiding endpoint `0*NaN` contamination. Other runtime values are not checked.
`update_if(..., &condition)` jointly gates mean and variance with the same scalar
numeric-mask semantics as state writes. It does not gate visible outputs or
include affine parameters, optimizer state, counters or RNG in that commit group.
`read(&graph)` returns current symbolic versions, including preceding writes;
slot accessors support runtime inspection and complete-state replacement.
Clones share identities while independent sessions own independent buffers.
CPU tests cover EMA values, rejection/retry, detached gradients, rate endpoints,
NaN replacement and session isolation. This is explicit state management, not
transparent capture of arbitrary Rust mutation or a complete BatchNorm module.

`BatchNormState` implements `state_tree::StateTree` with local names `mean` and
`variance`; nest it with `tree normalization => "normalization"` in a model state
tree. This includes running statistics only: affine parameters, epsilon and EMA
policy must be managed separately. Use `initial_state`, not generic zero-filled
initialization, when the initial variance should be one. The named-tree checkpoint
test accepts a training EMA update, rejects a NaN batch, saves the statistics and
loads them into a separate inference graph. Repeated inference matches an
independent F64 reference and leaves both statistics unchanged. CPU/CUDA coverage
uses the same client in one process; it does not establish deployment portability.
The same test target also nests Adam and BatchNorm in one macro-derived training
state tree. It rejects optimizer-only checkpointing as incomplete, restores all
seven slots into a graph with different registration order, and checks bit-exact
continuation against uninterrupted training. A shared proposal-finiteness mask
keeps weights, moments, beta powers and statistics unchanged on NaN batches;
accepted batches update both weights and statistics. This is explicit shared
gating, not a device-failure rollback guarantee.

`BatchNormState::prepare(&graph, &batch, rate)` returns a non-Clone
`BatchNormUpdate` with no recorded state writes. `finite_mask()` checks all
PROPOSED means/variances and returns scalar F32 0/1, allowing callers to combine
it with loss, gradient or RNG guards before any commit. `commit_if` consumes the
proposal and jointly updates both statistics; stale statistic versions, foreign
graphs and invalid predicates fail without updating either slot. `commit` is
unguarded, and existing `update`/`update_if` delegate through the same preparation
path without adding automatic finite checks. Rate 0 checks retained values, not
unused nonfinite batch statistics; rate 1 checks replacements. Finiteness does
not imply nonnegative variance or validate unrelated optimizer state. Native
tests exercise rates 0/0.5/1, NaN/Inf proposals and a shared resident counter guard,
alongside the existing detach/EMA/isolation tests. The training example explicitly
combines proposed-statistic finiteness with its loss/accumulation acceptance.

These proposals can share one explicit microbatch acceptance boundary:

```rust,ignore
let stats = statistics.prepare(&graph, &batch_statistics, 0.1)?;
let accumulation = accumulator.prepare(&graph, &gradients)?;
let update = optimizer.prepare_gradients(&graph, accumulation.gradients(), &rate)?;
let update = accumulation.with_optimizer(update)?;
let finite = loss.is_finite_mask()?
    .mul(&stats.finite_mask()?)?
    .mul(update.acceptable())?;
let accepted = sequence.commit_if(&mut graph, &finite)?;
let batch = update.commit_if(&mut graph, &accepted)?;
stats.commit_if(&mut graph, &accepted)?;
```

The native `training_acceptance` integration test composes all four components
in one compiled program with ten resident slots. Controlled statistic/gradient
inputs plus device uniform draws exercise valid partial/full windows, nonfinite
mean/variance/gradient, finite-input sum overflow, finite-sum optimizer weight
overflow with repeated retries, explicit rejection, 32-bit
counter carry and 64-bit RNG exhaustion. Every rejection preserves all state
bits and repeats the uncommitted random draw. Accepted statistics and momentum
updates are checked against a host recurrence, while sums/counters are checked
exactly. All cases share one compilation. This is a state-coordination fixture,
not a trained model. The AdamW fresh-process checkpoint fixture also gates RNG
and accumulation with candidate-update finiteness before committing; its
application-owned schema is version 4 to identify the changed acceptance policy.

`train_batchnorm --accumulate` additionally exercises two-microbatch gradient
windows with BatchNorm's distinct update cadence. It alternates original and
feature-shifted six-example batches for 300 accepted microbatches / 150 momentum
updates. Running mean/variance advance on accumulator `accepted`, while weights
and velocities advance only on `ready`. Every statistic update is checked against
an independent F64 EMA; all eight parameter/velocity tensors remain bitwise
unchanged on the first microbatch of each window. Both the ordinary and accumulated examples
also explicitly guard optimizer proposals with a runtime learning rate. Two
infinite-rate attempts with finite loss preserve all ten/fifteen state slots,
including statistics and the partial window, before normal training resumes.
Accumulated mode uses `commit_with_optimizer_if`; the inference ABI still needs
only data, not labels or the training learning rate. Invalid labels inserted into a
partial window preserve all fifteen slots, including statistics and the I32
accumulation count. Final loss is 0.005479957, and data-only inference predicts all
six labels correctly while preserving every state bit. Training/inference compile
twice total. The original non-accumulating mode remains separately tested.
This specifies per-microbatch BatchNorm statistics, not equivalence to BatchNorm
on a concatenated large batch. Run with `cargo run -p rxla-core --example
train_batchnorm -- --accumulate`; both modes are in the native regression gate.

`train_batchnorm` demonstrates the full training-to-inference state handoff.
It registers affine/head parameters, momentum velocities and running statistics
(ten slots), then compiles a read-only inference plan before recording training
writes. Both plans retain the same graph-local slot identities. Inference uses
`compile_pruned` to omit labels and the training rate from its input ABI;
training requires data, labels and a scalar learning rate. Training combines
finite loss, finite optimizer candidates and finite proposed statistics into one predicate for both optimizer and statistic
updates; after 50 steps an invalid-label batch verifies that all ten state values
remain unchanged. After 150 valid updates, `session.into_state()` transfers the
existing device buffers into an inference session without uploading replacement
model/statistic buffers. Inference is checked to preserve every state value and
classify all six synthetic samples correctly across repeated data-only calls.
An extra label argument is rejected before changing any state. Observed final
training loss was 0.005479957; there
are two backend compilations total (one per plan), not one per iteration.
The test downloads values for verification; the ownership handoff itself does
not download/copy them. This is not a zero-allocation execution, dynamic-shape
inference, checkpoint serialization or generalization benchmark. Its native test
is included in `check-xla.sh --with-plugin`.

```sh
PJRT_PLUGIN_PATH=/path/to/trusted/libzml_cpu.so \
  cargo run -p rxla-train --example train_batchnorm
```

`abs` uses the sign of the input, with zero gradient at either signed zero and
NaN at NaN. `clamp(low, high)` composes elementwise max/min: for distinct finite
bounds its gradient is 1 inside, 0 outside and 0.5 at the boundaries. Equal bounds
explicitly stop gradients (otherwise composing two tie rules would give 0.25);
the original forward NaN behavior is retained. These are explicit subgradient
conventions, not classical derivatives at kinks. Unselected upstream arithmetic
can still produce NaNs; these rules are not general numerical-recovery guards.

Softplus now has its own semantic node and sigmoid backward rule while lowering
to the same stable `max(x,0) + log1p(exp(-abs(x)))` forward computation. This avoids
incorrectly combining ReLU/abs zero-point conventions into a zero derivative for
the smooth softplus function: its gradient is exactly 0.5 at either signed zero.
CPU tests check broadcast/tied operands, nonfinite conventions, clamp boundaries,
wide-range softplus against F64 formulas and a moderate-range second derivative.
Higher derivatives at extreme values are not guaranteed numerically stable.

Sigmoid also retains a semantic node: forward uses `z = exp(-abs(x))` and selects
`1/(1+z)` or `z/(1+z)` by input sign, avoiding intermediate exponential overflow.
Backward uses `sigmoid(x)*sigmoid(-x)` instead of `y*(1-y)`, retaining positive-tail
derivatives when `y` rounds to one. This fixes a reproduced NaN gradient at finite
input -1000 in the former `1/(1+exp(-x))` composition. CPU tests independently check
sigmoid, SiLU and softplus values plus first/second derivatives against F64 formulas
at points spanning [-1000,1000], including signed zeros and small normal tails.
Subnormal results may flush to zero. Scalar/empty sigmoid and NaN/infinity cases
are checked separately. These results do not guarantee numerical stability for
arbitrary higher derivatives, nonfinite SiLU inputs, or all composite expressions.

```sh
PJRT_PLUGIN_PATH=/path/to/trusted/libzml_cpu.so \
  cargo run -p rxla-train --example train_xor
```

The nonlinear XOR smoke uses a 2→4→2 ReLU network, integer-label cross entropy,
and explicit SGD on all four resident parameter slots. A deterministic symmetric
hidden initialization and zero output layer make this a reproducible correctness
fixture, not a recommended initialization policy. After 200 updates it checks
all four XOR predictions, loss below 0.03, changed hidden-layer weights, and one
backend compilation. It is included in plugin-mode checks alongside the linear
classifier, without claiming generalization, robustness to arbitrary starts, or
full-model training readiness.

## Explicit compilation cache

Training programs use the same code cache. A native child-process regression
compiles an autodiff + momentum-SGD + finite-state-guard graph in the parent,
then restores it from disk with zero backend compilations in a fresh child.
Each process uses different initial weights, momentum, step counter and targets;
learning rate also changes at runtime. Losses/gradients and state updates match
an independent F64 recurrence. A NaN input retains all optimizer/parameter state,
and another session remains independent. The test checks one memory hit as well
as the child's disk hit, without timing or claiming a startup speedup.
This caches executable code, not optimizer checkpoints: restore or initialize
training buffers explicitly, and keep the existing trusted-plugin/namespace
compatibility contract. It does not remove Rust builds or graph construction.

`Compiler::new(client, CacheLimits::default())` creates a client-local LRU cache;
`compile(&graph, &output)` and `compile_many` return shared `Arc<Executable>` handles.
The defaults retain at most 32 entries and 16 MiB of serialized HLO keys. Key bytes
do **not** include backend executable memory or temporary compilation allocations.
Zero entry capacity disables retention. Oversized graphs compile but bypass caching.
Eviction/`clear` only release cache references; callers may keep executables alive.

Keys include the full HLO module: shapes, dtypes, constants and output order.
Runtime input values are not included; keep model weights as input buffers to
avoid embedding them in the graph and cache keys. Cache hits skip PJRT compilation,
but currently still lower/serialize the graph. This is byte-identity caching, not
semantic equivalence matching: unreachable computations are removed, but arbitrary
algebraically equivalent graphs are not canonicalized. Adding parameters changes
the input ABI and key even when those parameters are unused.
Compile options are fixed for a Compiler's immutable client/device selection;
any further configurable options must participate in cache identity. Caches are
neither global nor shared across compiler instances. Persistent cache namespaces
must distinguish selected device placement as well as plugin/environment
compatibility. The selected-device compile-option revision has a new disk key
domain: previous default-placement cache files remain on disk but do not hit.

With the `disk-cache` feature, unsafe
`DiskCache::new_for_client(directory, compatibility_key, &client, max_entry_bytes)`
derives a placement-specific namespace from the base key and client metadata:
platform/version, PJRT API version, process index and selected device ID, kind
and description. Use it to avoid manually appending device IDs. The base key
must still identify the exact trusted plugin build and relevant environment;
metadata is not a complete binary/driver/topology fingerprint. Attach the result
only to a compatible client: the cache does not retain or enforce client ownership.
Inspection, invalidation and trimming use the derived namespace. The two-device
`cpu_placement` example with `--features disk-cache` shares one directory/base
key, verifies independent fresh-client hits, then trims device one's cache and
checks that device zero still hits. This does not impose an aggregate
multi-device quota.

`stats()` reports hits, backend misses/failures, automatic evictions, bypasses,
retained entries and key bytes. Graph validation failures do not count as misses.
`clear()` empties retained entries without resetting cumulative counters.

For repeated cache lookups of a fixed graph, explicitly prepare a snapshot:

```rust,ignore
let lowered = graph.prepare(&output)?; // Lower + encode once, no plugin needed.
let executable = compiler.compile_lowered(&lowered)?;
```

`Tracer::prepare_many` accepts ordered mixed-dtype outputs. A `LoweredProgram` owns
a StableHLO payload and encoded key, not the source
graph or native handles, and is Send + Sync. It preserves all inputs declared
at preparation time; later source graph mutations do not alter its ABI. It can
be shared across workers, each with its own Compiler/client. Ordinary and
lowered-program compilation share memory/disk
cache keys. This trades host memory for skipped lowering/encoding; lookup still
hashes key bytes and updates LRU state. `LoweredProgram::input_count`,
`input_spec(index)` and `output_count` inspect its immutable runtime signature
without a plugin or allocation; input dtype is the storage dtype, including
BF16 inputs converted to F32 inside the graph. Pruned snapshots report compact
input order and out-of-range indices return None.
`output_spec(index)` returns an `OutputSpec` with static result shape/dtype,
also without compilation or allocation. For BF16 inputs widened in the graph,
the input spec is BF16 while the resulting output spec is F32. State snapshots
offer the same query for visible results only; hidden updates remain available
through the state schema queries. These signatures do not describe backend
physical layouts, memory placement or allocation sizes.
Keep and execute an executable directly
when no cache lookup is needed. Native compilation and execution are not sped up
by preparing a graph, and this is not automatic graph-mutation memoization.

Use `tracer.prepare_pruned(&outputs)` to explicitly remove unreachable
inputs. It returns `(prepared, input_indices)`, where each index refers to the
original parameter registration order, listed in the compact executable order.
Supply only those buffers in that order. Output order (including duplicates)
is preserved; constant-only outputs need no input buffers. Preparing a forward
snapshot does not remove derivative dependencies from the source graph.
This API snapshots value outputs, not StateProgram updates or state slot schemas;
use StateGraph's state-aware preparation or compilation for resident state.

`StateGraph::prepare` and `prepare_pruned` return a
`PreparedStateGraph` containing hidden final state outputs, slot identities,
types and argument mappings along with the lowered graph. Call
`prepared.compile(&mut compiler)` to obtain a normal `StateProgram`, then
initialize/bind its Session as usual. Empty visible outputs are valid when state
updates exist. Pruning keeps every state slot in the schema, even if its original
buffer is no longer a compiled input. Later source writes and slot registrations
do not alter the snapshot. The object is Send + Sync with no native handles;
share it via Arc if needed, but create native programs and buffers on their owner
threads. It is a transition snapshot, not a checkpoint of current state values,
fixed weight buffers or an automatically migrated Session. CPU/CUDA tests cover
hidden F32/I32 updates, input pruning, source mutation, cache-key reuse and
independent sessions initialized from the same prepared transition.
Before loading any plugin, `PreparedStateGraph::input_indices`, `state_type`
and `state_layout` expose its frozen visible-input mapping and state metadata.
The layout query requires the complete unique slot set and returns dtype/shape
in caller order, using the same validation as compiled StateProgram layouts.
Foreign/later slots, duplicates and incomplete sets fail; scalar and empty state
shapes are valid. These are host metadata queries, not reads of runtime state.
`PreparedStateGraph::parameter_type` additionally validates model parameter
identities and reports storage dtype/shape before compilation, for both fixed
inputs and resident trainable parameters. BF16 storage is distinguished from
the F32 symbolic conversion; pruned, foreign and later parameters are rejected.
An additional CPU/CUDA test binds different BF16 weights to two sessions from
one prepared pruned transition. Wrong-dtype, pruned and foreign parameters are
rejected without changing prior bindings/state; rebinding one session changes
its subsequent outputs without recompilation or changing the other session.
A CPU cross-process disk-cache test now publishes an ordinary state compilation
in the parent and restores it through a freshly prepared transition in the child.
The child records a disk hit and zero compilation time, initializes different
state/weight values, and checks interleaved F32 state and I32 indexing over
multiple updates. Same-shaped slots from an unrelated schema are rejected;
native cache reuse does not import slot identities or session buffers.

With `rxla-weights`' `training` feature,
`checkpoint.preflight_module_for_prepared(&prepared, &module, &mapping)` checks
parameter identities, checkpoint headers, alias consistency, shapes and storage
dtype rules before loading a plugin or compiling. It uses the same header
validation as module loading, reads no tensor payloads and performs no uploads.
After successful preflight, compile the transition and call
`load_module_for_program` normally; loading deliberately validates again.
This is not checkpoint authentication, a reservation against later changes, or
validation of application-owned schema/version metadata.
The mixed BF16/F32 module round-trip test now follows this order on CPU and
CUDA: prepare and preflight before Client::load, compile, load/bind, execute,
then export and restore. It verifies zero payload reads/uploads during preflight,
rejects conflicting aliases again during loading, and uploads the tied BF16
weight only once (two total uploads and 12 bytes for both unique parameters).

For complete resident state, use
`checkpoint.preflight_state_for_prepared(&prepared, &name_slot_pairs)` before
compilation. It shares `load_state`'s header checks: every schema slot exactly
once, unique nonempty names and exact F32/I32 shapes/dtypes, with no casts.
Scalar/empty state is supported; extra checkpoint tensors are ignored. No tensor
payload is read or uploaded. Load with `load_state` after compiling, and validate
application schema/optimizer/RNG metadata separately: compatible shapes do not
prove that two same-shaped states have been mapped to the correct names.
CPU/CUDA continuation tests now preflight before compiling the restored
transition: one restores an Adam tree with tied weights, moments and power
accumulators, and another restores F32/I32 state with reversed slot registration
order. Preflight leaves payload-read/upload and compilation counters unchanged;
subsequent state updates still match their uninterrupted sessions exactly in
these tests. Application metadata is explicitly checked before restoration.

When checkpoint names already follow the module/tree's first-visit canonical
names, use `preflight_named_module_for_prepared(&prepared, &module)` or
`preflight_state_tree_for_prepared(&prepared, &tree)` instead of constructing
name mappings yourself. Tied aliases resolve to one canonical checkpoint key;
there is no shape-based name guessing. Renamed schemas still require explicit
mappings. Both methods retain the no-payload-read/no-upload preflight contract.
Each call visits the caller's discovery implementation once, then validates
captured entries; host-only tests count these visits for both module and state
trees, including aliases with no separate checkpoint key. Names/selection must
still remain semantically stable across separate preflight and load calls.

`stats().compile_time` is a cumulative `std::time::Duration` for cache-miss
compilation attempts, including failed attempts and executable metadata setup.
It excludes graph lowering, HLO key encoding, cache I/O and execution; memory
and disk hits do not increase it. This is wall time, not backend CPU time or
total request latency, and clearing the cache does not reset it. Logging need
not be enabled to inspect it. CPU tests cover memory hits, clearing and fresh
process disk hits; a CUDA memory-cache test also checks the timing invariant.

```sh
PJRT_PLUGIN_PATH=/path/to/libzml_cpu.so cargo run -p rxla-core --example compile_cache
```

The example reports first-compile and repeated-lookup timings for one tiny graph,
including frontend lowering in lookup time. It is not a model performance benchmark.
It separately prints cumulative cache-miss compilation time, asserts that 1000
memory hits preserve it, and checks four changed input/weight pairs against host
F64 matmul/SiLU calculations using the same executable. CPU and CUDA checks run
this example; no speedup threshold is asserted, and debug-build timings should
not be interpreted as release performance.
This is an in-process cache, not an automatic disk cache; a fresh `Compiler` still
misses after a restart. Explicit native artifact export/load is available below.

## Application-controlled tracing

The crates/rxla-core/runtime crates emit `tracing` spans but never install a subscriber or
write logs by themselves. Applications choose filtering, formatting and exporters.
No input/weight values, HLO contents or native addresses are recorded.

- DEBUG `xla.compile`: frontend lowering/serialization and cache lookup, with
  output count, serialized key size and cache-hit status. Backend compilation is
  nested as DEBUG `pjrt.compile`, including HLO/options byte counts.
- DEBUG `xla.session.run`: synchronous input binding, validation, backend execution
  and state commit. DEBUG `pjrt.execute` covers the lower-level call through its
  completion wait, with input/output counts.
- TRACE `pjrt.host_to_device` and `pjrt.device_to_host`: synchronous transfer scopes;
  metadata only, not buffer contents. Host convenience execution therefore shows
  transfers separately from execution.

```sh
PJRT_PLUGIN_PATH=/path/to/libzml_cpu.so cargo run -p rxla-core --example trace
```

This example installs a thread-local formatting subscriber and reports span-close
durations. A real CPU test verifies that two identical compilation requests emit
two cache lookup spans but only one backend compile span, then one upload,
execution and download. Times are host wall-clock scopes, not per-kernel device
profiling; tracing overhead is not benchmarked. The subscriber is a dev-dependency
only; consumers receive the lightweight `tracing` instrumentation API.

## Optimized compiler IR inspection

`Executable::optimized_hlo_proto()` returns an owned `HloModuleProto` from the
compiled executable, rather than the frontend's pre-optimization module. It
handles both PJRT `hlo` and `hlo_with_config` encodings explicitly; the latter's
module configuration is not returned by this convenience method. Unsupported
formats or unavailable plugin APIs produce errors, not guessed decoding.
The lower-level `rxla_pjrt::Executable::optimized_program(format)` returns owned
raw bytes and the actual format label when the full wrapper is needed.

```sh
PJRT_PLUGIN_PATH=/path/to/libzml_cpu.so cargo run -p rxla-core --example inspect_hlo
```

The example inventories instructions before and after XLA optimization. On the
pinned CPU plugin, `(x + 0) * 2` loses the addition and its entry computation
contains a parameter and fusion. Inspection leaves execution usable; returned
IR remains valid after the executable is dropped. This enables checking graph
rewrites, fusion bodies and layout metadata, but is neither a device profiler
nor native assembly inspection. It does not control optimization passes. These
IR bytes are also not the native artifacts used by the serialization API below.

## Explicit native executable artifacts

`Executable::input_count()`, `input_spec(index)` and `output_count()` expose the
calling convention without plugin calls or allocations. `InputSpec` borrows an
immutable shape slice and carries the F32/I32 dtype; out-of-range indices return
None. Input order is graph parameter registration order, including declared
unused parameters, so a generic runner can allocate correctly typed buffers.
These queries remain available after `deserialize_with_metadata` in a new process.
Output shapes are still queried on returned buffers; this is not state-schema
introspection. Tests cover mixed F32/I32 restoration, scalars, empty/unused
parameters and programs with no inputs.

### Optional persistent compiler cache

Enable `rxla-core`'s `disk-cache` feature to attach a trusted directory:

```rust,ignore
let disk = unsafe {
    DiskCache::new(private_directory, compatibility_key, 256 * 1024 * 1024)?
};
let mut compiler = Compiler::new(client, CacheLimits::default()).with_disk_cache(disk);
let executable = compiler.compile(&graph, &output)?;
```

`compatibility_key` is caller-provided and **must** identify the exact plugin
build, target hardware/platform and other compilation-affecting environment
details. The exact raw `XLA_FLAGS` value is now captured at cache construction
and included automatically, distinguishing unset from empty. Set it before
plugin initialization and never change it during the process: the backend may
parse it once, so this snapshot is not an effective-options query. Other
environment variables and contents of files referenced by flags are not hashed;
their compatibility remains the caller's responsibility. The directory and
its ancestors must remain protected against untrusted modification. Constructor
unsafety covers this obligation for every compiler to which the cache is attached.

Lookup order is memory LRU, disk, then HLO compilation. A disk hit restores the
tensor executable and enters the memory LRU. Keys include the full lowered HLO,
namespace, captured flags and internal format/fixed-option revision; runtime weights and input
values are not embedded in keys. Blake3 filenames are backed by full-key equality
and artifact checksum checks before native loading. Checksums detect accidental
corruption, not malicious native code.

`DiskCache::inspect()` performs an explicit, synchronous, read-only directory scan.

`DiskCache::trim_compatible_to_bytes(budget)` explicitly removes validated
entries for the current namespace/captured flags, oldest modification time first
(filename breaks ties). This is publication/modification age, not access-time
LRU. Zero clears this validated subset; automatic trimming is disabled by default.
Corrupt, oversized, incompatible and unrelated files, directories and symlinks
are excluded by the scan, so the budget is not a total directory quota. It scans
fully before deleting, with bounded per-entry reads and no native loading.
`DiskCacheTrim` reports removed files/logical bytes, already-absent selected paths
and remaining observed compatible bytes. A removal error may leave partial
progress. Coordinate with concurrent writers for deterministic behavior; a
publisher can replace a selected path or recreate an evicted entry. This is not
a transactional quota or an allocated-block/free-space measurement. Already
loaded executables and the Compiler memory cache remain usable; clear memory
separately when a subsequent compilation must miss both caches. Host tests cover
age/budget/no-op behavior and exclusions; a real-plugin test verifies live and
memory-cached execution survive, followed by recompilation after memory clear.

To apply that policy after cache publication, configure the cache before attaching it:

```rust,ignore
let disk = unsafe { DiskCache::new_for_client(directory, compatibility_key, &client, max_entry_bytes)? }
    .with_auto_trim_to_bytes(2 * 1024 * 1024 * 1024);
let mut compiler = Compiler::new(client, CacheLimits::default()).with_disk_cache(disk);
```

Configuration alone and cache hits do not trigger cleanup. Successful publication
attempts (including an already-existing filename) synchronously scan and trim.
This adds miss-path I/O proportional to the directory scan, so opt in deliberately.
The newest file can itself be evicted when it exceeds the retention budget;
live executables and memory-cache entries remain usable. Maintenance failures
increment `disk_write_errors` without failing compilation, even if publication
already succeeded. This is best-effort namespace retention, not a hard quota
across concurrent writers, incompatible files or the complete directory.
Native CPU/CUDA tests cover an exact two-entry byte budget, a one-byte reduction
evicting the older entry, and duplicate-key publication preserving existing file
contents while triggering maintenance. A fresh compiler restores the survivor
with a disk hit and zero backend compile attempts; evicted live handles still run.

The read-only `DiskCacheInspection` reports candidate entry count and logical encoded bytes,
plus compatible, incompatible, corrupt, oversized and ignored counts. It examines
only immediate regular files with the cache's 64-lowercase-hex `.pb` naming scheme;
directories, symlinks and unrelated names are ignored. Reads are bounded by the
configured per-entry limit, with oversized files left undecoded. Ordinary cache
lookups do not trigger a scan.

Compatible entries have the current namespace/flags/version, matching filename key
and artifact checksum; this does not deserialize or validate native executability.
Incompatible envelopes have a valid artifact checksum but a different configuration
or version, and their original filename key is not verified. I/O errors return an
error rather than an incomplete successful inventory. Counts are observations,
not an atomic snapshot during concurrent cache writes, and bytes are logical file
sizes rather than allocated filesystem blocks. Inspection never deletes entries
and is not an eviction policy or an untrusted-cache security audit.

`disk.invalidate(&lowered_program)` explicitly removes just the entry for that
exact prepared StableHLO program, namespace and captured flags, returning false
if it is already absent. Obtain it with `Tracer::prepare` or
`Tracer::prepare_pruned`; the latter also returns the compact input mapping.
It deletes valid or invalid entries alike; no recursive removal, native loading
or automatic repair occurs. Later graph/state changes can produce a different key.
Other namespaces and keys are unaffected. Already loaded executables and compiler
memory entries remain usable: call `compiler.clear()` or create another compiler
if you want the next call to recompile and republish. Concurrent writers can recreate
an entry, so guaranteed invalidation requires caller coordination.

`compiler.disk_cache()` borrows the optional attached cache, so inspection and
invalidation use its original configuration without constructing a second cache.
`compiler.invalidate_memory(&lowered_program)` removes only the matching memory entry,
preserving other entries and their LRU order. It returns false if absent, updates
entry/key-byte gauges, and does not count as automatic eviction. Previously returned
executables remain valid. To invalidate both tiers explicitly:

```rust,ignore
let program = graph.prepare(&output)?;
if let Some(disk) = compiler.disk_cache() {
    disk.invalidate(&program)?;
}
compiler.invalidate_memory(&program);
```

Disk failure propagates before memory removal in this example. This is not an
atomic transaction with concurrent publishers. Removing memory alone can still
lead to a disk hit; removing disk alone can still lead to a memory hit. Native
CPU/CUDA tests exercise retained handles, unaffected entries, exact key-byte
accounting and subsequent LRU eviction after selective invalidation.

This provides an explicit escape hatch for corrupted files which the default
no-clobber store deliberately leaves untouched. CPU/CUDA recovery tests corrupt
only a test-owned entry, confirm fallback preserves it, invalidate it, verify the
memory hit remains, then clear/recompile and obtain a healthy disk hit. Host tests
also verify namespace isolation and refusal to recursively remove a directory.

The disk envelope/key format is version 2. Existing version-1 cache files are
not reused or deleted; models compile once into the new key space. Host tests
verify isolated-process capture, distinct paths for different flag values, and
envelope rejection even if an entry is placed under the wrong key.
A native CPU test runs four isolated processes with unset/empty/unset/empty
flags: the first two compile into separate entries and the last two each restore
their own entry with zero backend compilations. This verifies isolation, not an
optimization-dependent numerical or performance difference between those values.

New entries are written to same-directory temporary files and synced before
no-clobber publication; an existing entry wins. Publication does not overwrite
cache files. Explicit invalidation or opt-in trimming can delete entries;
automatic trimming excludes corrupt entries. Read/restore/write errors are
counted and logged at debug level; compilation remains usable on cache failure.
A corrupt entry therefore keeps causing fallback until the operator removes it.
`max_entry_bytes` bounds a file, not total disk usage or temporary serialization
memory. Disk trimming can be explicit or opt-in after publication as described
above; there is no strict concurrent disk quota or crash-durable directory-sync guarantee yet.
`clear()` only clears the memory LRU. Stats separate `disk_hits`,
`disk_read_errors`, `disk_write_errors` from memory hits and backend compile misses.

Fresh-process CPU tests verify a disk hit with zero backend compile attempts,
memory promotion, reload after memory clear, graph/namespace invalidation,
corruption fallback and oversized-entry bypass. Default builds do not enable
this feature or compile its optional `blake3`/`tempfile` dependencies.

A separate two-process publication test compiles the same graph in both children,
then uses a pipe barrier to permit concurrent publication only after both are
ready. Both stores succeed, exactly one complete cache file remains with no
temporary-file leftovers, and the parent restores it and executes new inputs.
This checks the no-clobber winner/loser path on the local filesystem. It does not
establish distributed-filesystem semantics, crash recovery, or single-flight
compilation: competing processes can still spend time compiling the same graph.

A separate fresh-process stateful test rebuilds a `StateGraph` schema and loads
its native code from disk with zero backend compile attempts. It interleaves
F32 state, F32 visible inputs and an I32 position input; updates paired K/V-like
buffers at different positions; and checks every state value against a host
reference. The load process binds different weights and initial state, proving
those runtime values were not embedded in the cached artifact. Two sessions
share fixed weight ownership while retaining independent mutable state. Invalid
input shapes leave both state slots unchanged, and execution remains valid after
caller-owned graph/program/compiler/client/weight handles are dropped.

The separate historical TinyLlama cache validation
also exercises complete-model restoration with zero backend compile attempts
and unchanged generated tokens/logits. A cache hit still requires graph construction and lowering
to obtain the key; it skips backend compilation, not all frontend work. State
schemas are rebuilt, and session contents are initialized separately rather than
restored from the compilation cache.

### Explicit export and restoration

`Executable::serialize()` exports backend-specific native bytes. The unsafe
`Client::deserialize_executable()` loads them and returns the low-level PJRT
executable; its buffer-oriented interface does not reconstruct tensor signatures,
state schemas or session bindings. Callers must preserve that metadata separately.

For tensor-level reuse, `Executable::serialize_with_metadata()` exports a
versioned protobuf envelope containing native code, F32/I32 input signatures and
output count. `unsafe Executable::deserialize_with_metadata(&client, &bytes)`
restores the tensor wrapper without graph construction or HLO compilation; its
`run`, `run_many` and `execute` interfaces retain input validation. These bytes
are a different format from raw `serialize()` output and must not be passed to
the raw PJRT deserializer. The envelope is experimental, not a portable model.

Host tests reject malformed/version-incompatible metadata and invalid shapes or
types before native loading. A fresh-process CPU test restores mixed F32/I32
inputs and multiple outputs, rejects bad input signatures, and checks lifetime
ownership without invoking compilation in the load process. A separate test
checks the restored F32 host convenience API. No new dependency or build-time
generation was introduced.

The **entire envelope must be trusted and unchanged**, and its native code must
match the exact plugin build/platform/device environment. Metadata checks do not
authenticate code or verify backend compatibility. This API does not yet persist
state schemas or bind weights. The optional `DiskCache` integration above adds
compiler lookup; the caller still supplies the backend compatibility key.

```sh
PJRT_PLUGIN_PATH=/path/to/libzml_cpu.so cargo run -p rxla-core --example artifact -- save /trusted/path/matmul.bin
PJRT_PLUGIN_PATH=/path/to/libzml_cpu.so cargo run -p rxla-core --example artifact -- load /trusted/path/matmul.bin
```

The save example refuses to overwrite existing files. The load branch never calls
HLO compilation. Tested on the pinned CPU plugin in two separate processes:
exported a 7,585-byte matmul/reduction artifact and restored both outputs correctly.
That size is for this tiny example only; this is not a startup/performance benchmark.

Artifacts may contain native code: never load untrusted or modified files, and
use the same plugin/library version and compatible execution platform. PJRT makes
no portable or stable serialization guarantee. The standalone raw-file example
does not implement fingerprints, authentication, cache publishing, eviction or
fallback; the optional compiler cache above supplies only its documented subset.
Plugin support on other backends is unverified.

## Source pin / regeneration

Files under `vendor/xla` are from OpenXLA commit
`6c5e7717f9af43bb5256d667c9e910be450345ae` (upstream Apache-2.0 notices retained).
Roots: `xla/service/hlo.proto`, `xla/pjrt/proto/compile_options.proto`, and
`xla/pjrt/c/pjrt_c_api.h`; imported XLA protos are vendored transitively.
Google well-known protos come from xtask's pinned Cargo protoc dependency.
Generation needs libclang only in xtask, and obtains protoc through
`protoc-bin-vendored`. prost 0.14 cannot generate Editions schemas; xtask narrowly
normalizes the descriptor of the enum-only `backends.proto` to proto3, refusing
unexpected message/service additions. Original upstream sources stay unchanged.

## Weight-loader measurements

`rxla_weights::Checkpoint::stats()` returns cumulative successful read/upload counts,
source payload bytes, F32 upload bytes, and host durations for read, decode and
upload. Reads include seeking and byte-buffer allocation; decoding includes F32
output allocation; upload timing starts after decoding and includes synchronous
PJRT upload. Header parsing is excluded. Repeated reads count again, failed
operations do not increment the failed stage, and successful decoding remains
counted even if a later upload fails. These are host wall times, not device kernel
measurements or storage throughput guarantees. TinyLlama reports these fields in
`weight_loading`. The counters require no new dependencies or global subscriber.

For repeated model runs, opt into the supplied Cargo configuration from
the standalone repository root:

```sh
cargo --config fast-load.toml run -p rxla-weights --features disk-cache \
  --example tinyllama -- target/tinyllama-chat target/your-new-report.json
```

This enables opt-level 2 only for the `rxla-weights` package (including its
examples), leaving other crates' dev optimization settings unchanged. Debug
assertions/information remain enabled. It is not a default profile change, and
downstream users must opt in through their own workspace Cargo configuration.
On the measured TinyLlama run, loading/building fell from 10.42 s to 2.98 s and
weight decoding from 8.92 s to 1.10 s, with identical saved outputs. The optimized
build took 10.43 s once; an unchanged subsequent build took 0.05 s. These are
single local observations, not clean-build or incremental-edit benchmarks.
Editing this package can incur the optimization cost again; omit the configuration
when prioritizing its edit/build cycle. Details are in the TinyLlama results.

## Explicit singleton axes and stacking

`tensor.unsqueeze(axis)` inserts one size-one dimension; `tensor.squeeze(axis)`
removes only the selected singleton dimension and errors for a non-singleton or
absent axis. This avoids silently removing a batch axis when other dimensions
happen to equal one. Insertion axes range from 0 through rank, inclusive.

`Tensor::stack(&[a, b], axis)` joins equal-shaped tensors along a new axis, unlike
`concatenate`, which joins along an existing axis. Inputs must be nonempty and
belong to one graph. Scalar and zero-sized tensors are supported. Implementation
uses reshape plus concatenate, keeping the operations visible to XLA without a
new primitive or dependency. CPU tests check every insertion axis for scalar,
vector, matrix and empty tensors, and a single-input stack/squeeze round trip.

## Circular shifts

`tensor.group_norm(groups, weight, bias, epsilon)` normalizes `[N,C,...]` per
sample and channel group, with optional `[C]` affine tensors. Groups must divide
the positive channel count; spatial dimensions must be nonempty, while an empty
batch is valid. Epsilon must be finite and positive. It reshapes into groups and
reuses two-pass LayerNorm population statistics, with no running state or
training/evaluation switch. All operands remain differentiable. This is a
composite F32 graph, not a dedicated fused kernel or mixed-precision API.
CPU/CUDA correctness tests cover groups 1/2/4, rank 2/3/4 and empty batches,
comparing forward values and input/affine gradients to an independent f64
coordinate reference, plus a Hessian-vector product against directional finite
differences of the reference gradient. Finite-input F32 rounding/overflow still
apply.

`module::GroupNorm::new(&mut graph, groups, channels, epsilon, affine)` registers
input-backed affine parameters; `new_trainable` registers resident trainable
parameters instead. `from_parameters` supports shared or independently optional
scale/bias handles. Initialization is explicit (normally ones for scale, zeros
for bias), with no implicit device allocation. It implements object-safe `Module`
and `Parameterized`, visits `weight` then `bias`, and cloned modules retain tied
parameter identities. A non-affine module still enforces its configured channel
count. Tests check aliases, storage modes, invalid/cross-graph parameters and ten
SGD steps through a worker-free resident session, comparing each parameter update
to host arithmetic, requiring changed parameters and loss no greater than its
initial value (within the test tolerance).

The synthetic CNN example also accepts `--group-norm`: Conv2d NHWC output is
transposed to NCHW, normalized with two channel groups, then transposed back
before SiLU/pooling and the classifier. These layout operations and all momentum
SGD updates belong to one compiled graph; there is no host feature transfer or
separate normalization executable. Scale starts at one and bias at zero.
The native `real_group_norm_cnn_training` test exercises average/max pooling,
three seeds and 250 steps per seed, requiring all eight synthetic line images
classified correctly, final loss below 0.1 and below 20% of its initial value,
and changes to convolution/head weights and both normalization parameters.
Each pooling configuration compiles once and reuses that executable across seeds.
This is a small integration test, not OCR/YOLO accuracy or a throughput benchmark.

```sh
cargo run --offline --manifest-path Cargo.toml \
  -p rxla-train --example train_cnn -- --group-norm
```

Set a trusted compatible `PJRT_PLUGIN_PATH` first. The CPU standard gate and
selected CUDA gate include the multi-seed normalized CNN training test.

`train_cnn --compare-group-norm-accumulation` additionally compares batch 8 with
two equal batch-4 microbatches for the average-pooling normalized CNN. GroupNorm
statistics are per sample, so this does not change the normalization population
as BatchNorm microbatching would. Three-seed native tests compare all model
parameters (including normalization scale/bias), momentum buffers and loss at
each of 250 updates, retaining the existing `5e-5*(1+abs(reference))` state and
`5e-5` loss tolerances. No per-step realignment occurs for this smooth pooling
path. The first microbatch leaves parameters/momentum untouched; rejected
invalid-label batches preserve partial sums/count, and a completed window resets
accumulators. Both shapes compile once each and are reused across seeds.
This particular comparison does not test unequal microbatch sizes or normalized max-pool accumulation;
the existing independent max-pool tie diagnostic models only the unnormalized
CNN. Both CPU and selected CUDA gates include the new comparison.

`train_cnn --compare-unequal-accumulation` compares a weighted whole batch of 8
against batches of 3 and 5 on the same average-pooling GroupNorm CNN. Three
static shapes compile once each, then are reused for three seeds and 250 updates.
Between microbatches, `Session::switch_program` moves resident model, momentum,
gradient-sum, count and total-weight buffers using an explicit name-checked
mapping; no host state reconstruction or per-update reference alignment is used.
The test retains the same per-step state/loss tolerances as the equal-batch test,
checks all eight final labels, and checks that an invalid second microbatch
preserves the already accumulated weight 3, sums, count and optimizer state.
Successful windows reset all accumulation slots. Diagnostic comparisons do
download state; this is correctness evidence, not an allocation/performance
benchmark or distributed reduction implementation. CPU and selected CUDA gates
include the three-seed unequal-batch test.

`tensor.repeat_interleave(repeats, axis)` repeats each element consecutively
along one existing axis: `[a,b]` repeated twice becomes `[a,a,b,b]`, not tiled
`[a,b,a,b]`. Counts are nonnegative construction-time constants, zero gives an
empty axis, and invalid axes/negative counts/shape overflow are rejected. This
uses reshape/broadcast with no new native kernel or codegen dependency; it does
not promise a physical copy or a performance improvement. Gradients sum repeated
contributions. Native tests cover every axis of rank-1/2/3 tensors, empty axes,
counts 0/1/2/3, weighted quadratic gradients and second derivatives. The CUDA
correctness gate includes these tests. This explicit expansion is available to
model authors; the existing grouped-query attention implementation retains its
grouped formulation rather than automatically expanding compact KV heads.

`tensor.roll(shift, axis)` circularly shifts one explicit axis, with positive
shifts moving values toward higher indices: `[a,b,c]` rolled by one becomes
`[c,a,b]`. The construction-time I64 shift is reduced modulo the axis length,
including `i64::MIN` and `i64::MAX` without negation overflow. Valid empty axes
and multiples of the length are no-ops; invalid axes still fail for zero shifts.
Compose calls for multiple axes. Dynamic shifts and implicit flattening are not
part of this API. Implementation uses narrow plus concatenate, so normal XLA
optimization and existing differentiation rules apply, without another native
kernel or dependency. Native tests compare an independent coordinate mapping,
weighted gradients and inverse shifts for ranks 1–3, singleton and empty axes.

## Tensor padding

`tensor.flip(&[axis, ...])` reverses element order along distinct, in-range
axes without changing shape. An empty axis list is a no-op; axis order is
canonicalized before constructing native HLO `reverse`. Its gradient reverses
the cotangent along the same axes. Tests check weighted gradients and that
applying the same reversal twice restores the input.

`tensor.pad_reflect(&[[low, high], ...])` mirrors the input without repeating
the edge element: `[a,b,c]` with widths `[1,2]` becomes `[b,a,b,c,b,a]`.
Each nonzero width must be smaller than the original dimension; repeated
reflection beyond this limit is rejected. Unpadded empty/singleton axes and
scalar/no-op cases remain valid. Shape and width validation precedes graph
construction. The implementation composes narrow, flip and concatenate,
without host index arrays or an external kernel. CPU/CUDA regression tests use
an independent reflected-coordinate reference to check values and first/second
derivatives across ranks 1–3, asymmetric widths and empty output.

`tensor.pad_replicate(&[[low, high], ...])` extends each edge using its nearest
input value, including multidimensional corners. Widths are nonnegative and may
exceed the original dimension. Any axis receiving padding must be nonempty;
other empty axes and scalar/no-op cases are supported. All shape/width checks
run before construction. The implementation composes existing narrow,
broadcast and concatenate operations; no native kernel or host index table is
introduced. Existing differentiation rules accumulate the repeated edge copies.
CPU/CUDA tests independently map output coordinates back to clamped input
coordinates and verify values plus first/second derivatives for ranks 1–3,
singleton edges and empty output. This is not reflection or interior padding.

`tensor.pad_with_scalar(&[[low, high], ...], &fill)` accepts a same-graph
rank-zero F32 tensor, including a runtime input or trainable state value. It
lowers to the existing HLO Pad opcode; no new native binding or compiler build
is required. Both operands are differentiable: the input gradient extracts the
interior cotangent, while the fill gradient sums only border cotangents using
a selection mask. It does not subtract two large total sums. Runtime fill
changes do not change the graph. Zero-width padding validates the fill then
returns the input, so the fill gradient is zero. Cropping/interior padding and
per-channel/vector fill values are not part of this API. CPU/CUDA regression
tests cover changing fills, both first derivatives, fill second derivatives,
large interior cotangents, empty inputs and no-op padding.

`tensor.pad(&[[low, high], ...], value)` adds nonnegative edge padding to every
axis without changing axis order. For NHWC spatial padding, pass
`[[0,0], [top,bottom], [left,right], [0,0]]`. The scalar F32 fill value is a graph
constant; nonfinite fills are permitted and do not replace unpadded elements.
All-zero widths are a no-op, including scalar tensors. Empty dimensions can be
padded to produce filled output, and outputs may remain empty.

Rank/width errors, negative widths and dimension/element-count overflow are
rejected before node creation. Use `slice`/`narrow` for cropping; interior
padding is not implemented. Constant padding lowers to native HLO
`pad`, not host-side allocation of a padded tensor. CPU tests compare asymmetric
padding across ranks 1–3 with a row-major reference for different runtime inputs,
including empty inputs/outputs and NaN/infinite fills.

## Average pooling

`tensor.avg_pool2d(options, count_include_pad)` uses NHWC layout and the same
`Pool2dOptions` as max pooling. Padding values are zero, output sizes use floor
division, and windows/strides must be positive with nonnegative explicit padding.
When `count_include_pad=true`, the divisor is the full window area. When false,
the divisor counts only valid spatial input elements; entirely padded windows
produce NaN (0/0), rather than silently assigning a denominator of one.

The numerator is an HLO sum reduce-window. Excluding padding uses a broadcasted
spatial ones tensor and another reduce-window to compute counts, shared across
batch/channels. All nodes remain visible to XLA; no native kernel or dependency
is added. Counts and arithmetic are F32, so very large counts can round. Ceil
mode, dilation and divisor overrides are not implemented.
CPU tests compare both denominator conventions against independent F64 window
enumeration, including asymmetric padding, entirely padded windows, strides,
multiple batches/channels, identity windows and empty outputs. Existing max
pooling tests also pass after sharing shape validation and window lowering.

## Stable elementary functions and activations

`log1p()` and `expm1()` are also available as native HLO operations, retaining
small values near zero instead of composing `log(1+x)` or `exp(x)-1`.
`abs()` is native, and the semantic `softplus()` node lowers to
`max(x,0) + log1p(exp(-abs(x)))` at beta=1 without a threshold approximation.
Large positive finite inputs avoid exponential overflow; tiny negative tails
remain representable until normal F32 underflow. `log1p(-1)` is negative infinity
and values below -1 produce NaN. CPU tests compare small inputs down to magnitude
1e-12 with F64 `ln_1p`/`exp_m1`, and softplus over [-1000,1000], including
infinity/NaN behavior. These additions do not add automatic differentiation.

`tensor.gelu()` uses `0.5*x*(1 + erf(x/sqrt(2)))`;
`tensor.gelu_tanh()` uses
`0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))`.
Choose the variant specified by the model rather than treating the approximation
as an interchangeable performance option. Both preserve shape and use existing
HLO operations, so no new opaque kernel, compiler boundary or dependency is added.
The erf formula is not exact arithmetic: F32 cancellation may round small
negative tails to zero, and nonfinite inputs retain ordinary formula behavior.

CPU tests compare 401 points in [-10,10] with independent F64 Gaussian integration
and the F64 tanh formula. Observed maximum absolute errors were 5.644e-7 and
5.917e-7 respectively (test gate 2e-6). Tests also check scalar/empty execution,
shape preservation and distinct erf/tanh lowering. This adds activation forward
computation, not automatic differentiation or complete model coverage.

## Stable log probabilities

For logits `[B,S,V]` and runtime I32 token IDs `[B,S,1]`:

```rust,ignore
let token_log_probs = logits.log_softmax(2)?.take_along_axis(&token_ids, 2)?;
// Result: [B,S,1]. No one-hot tensor or host probability download required.
```

Index tensors also support `unsqueeze(axis)` (insert a singleton axis),
`flatten(start, end)` (merge an inclusive axis range),
`squeeze(axis)` (remove exactly one singleton axis),
`reshape` (same element count) and `transpose`
(a permutation of all axes). For IDs supplied as `[S,B]`, use
`ids.transpose(&[1, 0])?.unsqueeze(2)?` before selection. These are graph
operations on I32 data, not downloads or F32 conversions; layout changes do not
change the original runtime parameter shape. CPU and CUDA tests compare sequence-major
IDs selecting batched log probabilities with an independent F64 reference across
two runtime ID sets, and cover scalar/empty index layouts and invalid shapes.
Singleton-axis round trips preserve exact I32 values, including integer extremes
and values not exactly representable as F32, on every insertion axis tested.

Both Tensor and Index expose `flatten(start, end)`: for example, a tensor with
shape `[B,C,H,W]` becomes `[B,C*H*W]` with `flatten(1, 3)`, without manually
computing the feature dimension. Row-major order and dtype are preserved via
reshape. Scalars support `flatten(0, 0)` to `[1]`; invalid/reversed ranges and
flattened dimensions exceeding i64 are errors. Empty dimensions remain empty.
CPU/CUDA tests cover all contiguous ranges of a rank-three example, scalar/empty
inputs, exact I32 payloads and reshape gradients. No host transfer is introduced.

`move_axis(source, destination)` moves one axis to its final position and keeps
the relative order of other axes: `[B,C,H,W].move_axis(1, 3)` gives `[B,H,W,C]`.
`swap_axes(first, second)` exchanges just two axes, so swapping 1 and 3 instead
gives `[B,W,H,C]`. Both are available on Tensor and Index, record a transpose,
preserve dtype and Tensor gradients, and do not transfer values to the host.
Axes are zero-based and must be within rank even when equal; scalars have no
valid axis. Empty tensors are supported. These APIs do not promise a zero-copy
device layout or avoid whatever layout conversion the backend chooses.
CPU/CUDA tests cover every source/destination pair on rank-three normal and
empty tensors, rank-one identities, exact I32 values and nonuniform VJPs.
The shared TinyLlama decode/prefill builder uses `unflatten` + `move_axis` for
Q/K/V heads and `move_axis` + `flatten` to merge attention outputs. A host-only
test compares exact HLO protos against the former reshape/transpose expressions
for one- and eight-token chunks with four and thirty-two heads. This checks
layout-expression equivalence, not a fresh full-model inference benchmark.

`unflatten(axis, &sizes)` performs the inverse axis-level operation on Tensor
and Index: for `[B,S,H*D]`, `unflatten(2, &[H,D])` yields `[B,S,H,D]`.
Sizes must be explicit, nonempty and nonnegative, with product equal to the
selected dimension even when some other axis makes the whole tensor empty.
No `-1` inference or scalar axis is accepted. Other axes, dtype, element order
and Tensor gradients are preserved through reshape.

`box_cxcywh_to_xyxy()` and `box_xyxy_to_cxcywh()` convert center/size and corner
representations entirely within the graph, preserving `[...,4]` shape and
gradients. They do not reorder or clamp signed widths/heights, normalize by image
size, or add pixel-inclusive offsets. Constrain predicted sizes explicitly when
feeding GIoU. CPU/CUDA tests cover batched/empty shapes, signed sizes and weighted
linear VJPs; round-trip exactness is checked only on exactly representable dyadic
fixtures, not promised for arbitrary floating-point coordinates.

For explicit host postprocessing, `rxla_tensor::vision::nms(&boxes, &scores,
threshold, max_output)` accepts `&[[f32;4]]` xyxy boxes and returns original
indices in descending score order. Equal numeric scores, including signed zero,
break ties by original index. Suppression uses strict `IoU > threshold`, with
zero-area boxes assigned zero IoU. Inputs must be finite and corners ordered;
validation still applies when max_output is zero. This class-agnostic utility
does not filter confidence scores, group classes, download tensors or run a GPU
kernel. Transfer and candidate selection remain explicit. It uses F64 geometry
and O(N log N + N*K) work, with no N*N matrix. Host tests cover strict thresholds,
ties, limits, degenerate/huge coordinates and greedy suppression chains.
`vision::nms_by_class(&boxes, &scores, &classes, threshold, max_output)` suppresses
only equal i32 class IDs. The cap applies globally across classes, and returned
indices remain globally score-sorted with original-index ties. Negative class
IDs are opaque labels, not invalid sentinels. This uses direct class comparisons,
not coordinate offsets, so even extreme finite coordinates do not compromise
class separation. Class count and all ordinary NMS inputs are validated before
returning, including when the requested output cap is zero.
The `detection_postprocess` example connects this to PJRT: top-k selects four
of eight synthetic scores, gathers the associated boxes before cxcywh-to-xyxy
conversion, then explicitly downloads selected boxes/scores/I32 source IDs and
I32 classes (112 payload bytes). Host confidence filtering and both NMS modes
preserve original indices; extreme class labels remain exact without F32 casts.
CPU/CUDA checks cover ordinary scores, stable ties, and an empty filtered result
over three executions with one compilation. Top-k truncation can differ from
NMS over all candidates; this is a small postprocessing pipeline, not YOLO model
execution, GPU NMS or a performance benchmark.

For detection geometry, `boxes.box_area()` accepts `[...,4]` xyxy coordinates
and returns `[...]`. `predictions.box_iou(&targets)` compares corresponding
boxes with broadcast-compatible prefixes: `[N,4]` against `[N,4]` yields `[N]`,
and `[4]` can broadcast across predictions. This avoids constructing an `[N,N]`
matrix just to take its diagonal; a host HLO test checks bounded intermediate
sizes for 128 matched boxes. `a.pairwise_box_iou(&b)` accepts `[...,N,4]` and `[...,M,4]`,
broadcasts their batch prefixes and returns `[...,N,M]`. Widths/heights clamp at
zero; inverted/degenerate boxes have zero area and zero union produces zero IoU
using a safe denominator. There is no inclusive-pixel `+1` or epsilon bias on
valid unions. Coordinates and intermediate areas must remain finite. These are
ordinary differentiable graph compositions (piecewise min/max/ReLU gradients),
not host postprocessing, NMS or a full detector. Empty sets are supported.
CPU/CUDA checks compare against independent F64 geometry, batched broadcasting
and finite-difference gradients away from coincident edges.
`cargo run -p rxla-train --example train_boxes` fits two initially overlapping
boxes with resident guarded SGD. CPU/CUDA runs check every loss against F64 over
600 steps, unchanged state on rejected steps, and one compilation even when
recreating the session at step 300. A disjoint-box control remains unchanged:
ordinary IoU has no useful movement gradient there. This synthetic example is
not a complete detection loss, detector training, or performance measurement.

`predictions.box_giou(&targets)` adds the smallest axis-aligned enclosing-box
penalty to aligned/broadcast IoU: `IoU - (enclosing_area - union)/enclosing_area`,
following the [GIoU definition](https://giou.stanford.edu/). A training loss can
use `giou.mul_scalar(-1.)?.add_scalar(1.)?`. Supply ordered finite xyxy coordinates
with finite areas; this API does not reorder inverted boxes. Zero-area union and
enclosure terms use safe denominators. The enclosure term can provide gradients
when boxes do not intersect, but min/max boundary derivatives remain piecewise.
CPU/CUDA F64 and finite-difference checks cover overlapping/disjoint/contained
boxes, identical/zero-area cases, empty sets and a disjoint gradient-ascent step.
This does not promise convergence for arbitrary parameterizations or a full
detector training recipe.
The `train_boxes` example also fits two initially disjoint boxes with GIoU,
using 2,000 guarded steps and slower learning-rate decay. Each step's loss is
checked against an independent F64 formula, with session recreation halfway
through and one compilation per loss variant. The initial 600-step IoU schedule
only reduced CPU GIoU loss from 1.066875 to about 0.677; the longer schedule
reaches about 0.00171. This is a tuned synthetic fixture, not a claim that the
same optimizer schedule generalizes to detector training.

`take_along_axis` selects separately at each non-axis position, unlike `take`,
which inserts a shared index tensor's shape into the result. Data and indices
must have equal rank and equal non-axis dimensions; broadcast explicitly when
needed. Output shape equals index shape. Negative/out-of-range indices clamp to
the selected axis bounds, not Python-style wrapping or runtime errors. Callers
handling token IDs must validate vocabulary membership when clamping is unwanted.
The selected data axis must be nonempty; empty batch or index axes are supported.
Lowering uses HLO gather batching dimensions with no one-hot expansion. The
dimension model follows the [gather specification](https://openxla.org/stablehlo/spec#gather);
real CPU tests check all three axes, changing runtime indices, clamping, empty
outputs and composition with log-softmax against host reference calculations.

`logits.log_softmax(axis)` preserves shape and computes log probabilities directly
from centered logits. It does not compute `softmax().log()`: for logits
`[1000, 0, -1000]`, its result is `[0, -1000, -2000]` even when ordinary
probabilities would underflow to zero. This is useful for token log-probabilities
and future loss/RL APIs; it does not add autodiff or a training loop.

The chosen axis must exist and be nonempty. Other empty dimensions are allowed.
Negative-infinity masks produce negative-infinity log probabilities when there
is a finite maximum in the slice. All-masked slices, positive infinities and NaNs
have no finite-result guarantee; extreme finite differences may overflow F32.
CPU tests compare every axis of a rank-three tensor against an independent F64
reference, including large common offsets, and exercise extreme separation,
masking, singleton axes and empty output. The implementation is ordinary HLO
max/subtract/exp/sum/log composition, with no opaque kernel or new dependency.

## Migration is not complete

Current graph supports batched matmul with batch broadcasting and vector
promotion, multiple outputs, equal-shape elementwise arithmetic,
transcendentals, reshape/transpose, static strided slicing, narrow/split/concatenate,
runtime-indexed dynamic slice/update,
explicit singleton broadcasting, sum/max/mean
reductions, stable softmax/log-softmax, ReLU/SiLU/sigmoid, RMSNorm and grouped conv2d, with F32 constants and
parameters. Real-plugin tests validate an MLP composition, large-logit softmax,
reductions, layouts, RMSNorm, multihead attention against a scalar f64 reference,
multi-output ordering/duplicates, empty slices, RoPE rotate-half composition and
resource lifetimes. A real-plugin cache test compiles once and repeatedly updates
different positions, validating clamping and preservation of the old input buffer.
Resident execution checks input counts, shapes and dtypes before dispatch.
Convolution tests compare standard/grouped/depthwise operations, including stride,
dilation and asymmetric padding, against independent scalar f64 cross-correlation.
A small vision graph combines convolution, SiLU, strided depthwise convolution,
global average pooling and a classification head and checks its final probabilities.
Remaining work includes broader crates/rxla-core/NN
operations, fuller execution validation, automatic disk-cache quotas and automatic
compatibility fingerprints, plugin packaging/doctor, asynchronous and concurrent execution contracts,
custom ops, and model validation. The IREE frontend's useful NN/shape designs
will be ported without retaining its compiler/runtime dependency. This milestone
is a real protobuf execution path, not yet a complete MLX-style library.

Static indexing intentionally rejects negative bounds, out-of-range bounds and
non-positive strides instead of applying Python-style normalization. `split`
accepts explicit sizes that must sum to the axis length; zero-sized pieces are
supported. These operations are useful for projection splitting and RoPE.

`Tracer::input_i32_scalar` creates a scalar I32 parameter (in declaration order alongside
F32 parameters); `index_constant` supplies fixed axes. Pass these to
`dynamic_slice` or `dynamic_update_slice`. Unlike static slicing, dynamic indices
are clamped to `[0, dimension - slice_size]` by XLA, not rejected or wrapped.
Updates return a new value and do not mutate/donate the input buffer. This enables
fixed-capacity KV storage without recompiling for each position, but is not yet a
complete decoder or an in-place cache implementation.

Use `Client::buffer_i32` and `Executable::execute` for mixed F32/I32 inputs. The
host convenience `run`/`run_many` methods only accept all-F32 signatures. `Index`
is a distinct I32 indexing tensor, not yet a general integer arithmetic API.
PJRT buffers support F32/I32 transfers with checked typed readback; broader dtypes,
general integer arithmetic, public boolean tensors and cache donation remain future work.

`Tracer::input_i32(dims)` and `constant_i32(dims, values)` create integer
tensors; `index_input`/`index_constant` remain scalar conveniences. Dynamic slice
starts still require scalars. `table.take(&ids, axis)` replaces the chosen axis
with the index tensor's shape, e.g. `[vocab, hidden]` and `[batch, sequence]` IDs
produce `[batch, sequence, hidden]` embeddings. The axis must be nonempty. Indices
clamp to the valid range: negative indices select the first element, not the last.
Validate token IDs separately if invalid IDs should be errors. Tests cover runtime
and constant indices, empty/duplicate/multidimensional indices, and an embedding
→ projection → softmax graph reused with different token IDs on the CPU plugin.

`Index::broadcast_to` supports integer singleton expansion. `Index::le_mask`
performs a signed elementwise comparison and returns an F32 0/1 tensor;
`mask.log()` converts it to a 0/-infinity additive attention mask. Comparison
requires equal shapes (broadcast explicitly). This is not a general boolean
tensor API. A row with no valid attention positions still has undefined softmax
results; callers must validate decode positions and ensure at least one valid key.

`mask.select(&on_true, &on_false)` selects F32 values elementwise using an F32
mask: zero (including negative zero) selects false; nonzero values, including
NaN, select true. All three tensors must have equal shapes and graph ownership;
broadcast explicitly. Lowering uses HLO compare/select, not multiplication, so
NaN/Inf in an unselected value does not contaminate the result. Both branches
remain graph operands: this does not provide lazy control flow or avoid their
computation. `x.is_finite_mask()` produces a shape-preserving F32 0/1 mask using
native HLO is-finite; for example, `x.is_finite_mask()?.select(&x, &fallback)?`
replaces nonfinite values with an explicitly chosen same-shape fallback. This
is an application policy, not an automatic numerical-error recovery mechanism.
Native tests cover changing masks without recompilation, nonfinite operands,
nonbinary masks, signed zeros, and scalar/empty outputs.

F32 tensors provide `eq_mask`, `ne_mask`, `lt_mask`, `le_mask`, `gt_mask`, and
`ge_mask` for elementwise comparisons returning F32 0/1 masks. Operand shapes
and graph ownership must match; scalar thresholds must be explicitly broadcast.
These use HLO FLOAT comparisons: NaN makes equality and ordered comparisons
false, inequality true; positive and negative zero compare equal. This is not
bitwise equality or a total floating-point ordering. Compose thresholds directly
with selection, for example `x.gt_mask(&threshold)?.select(&x, &fallback)?`.
Native tests compare all six operators against Rust over pairwise NaN/infinity/
zero and finite-extreme inputs, changing runtime values without recompiling,
and cover composition with select plus scalar/empty outputs.

`mask.select_index(&on_true, &on_false)` applies the same numeric-mask rules to
I32 `Index` operands, without converting the selected values through F32.
This supports exact conditional integer state updates, for example:

```rust,ignore
let position = graph.read_index(&position_slot)?;
let advanced = position.wrapping_add_scalar(1)?;
let next = active_mask.select_index(&advanced, &position)?;
graph.write_index(&position_slot, &next)?;
```

Here a same-shape runtime mask controls each position independently. Mask changes
do not require recompilation. This only selects the new position: it does not
automatically mask KV writes or skip finished-sequence computation, and callers
must still enforce context bounds. Native tests cover exact values beyond F32
integer precision, I32 extremes/wraparound, zero/NaN/nonbinary masks, scalar and
empty arrays, and repeated masked session updates using one compiled program.

`x.rotary_embedding(&cos, &sin, RotaryLayout::SplitHalf)` rotates pairs of last-axis
channels; `Interleaved` selects adjacent pairs instead. Width must be positive and
even. Cos/sin broadcast to the input shape with last dimension halved: one value
per channel pair, not duplicated full-width tables. Frequency generation, position
lookup and scaling remain caller choices, and tables may be runtime inputs.
Partial rotation can be composed with narrow/concatenate. Native F64 tests cover
both layouts, batch broadcasting and changed runtime angles without recompiling.
TinyLlama uses SplitHalf and half-width host-generated position tables.

`tensor.argmax(axis, keep_dims)` returns I32 Index results. Equal maxima select
the lowest index; a row containing NaNs selects its first NaN. All-negative-infinity
and signed-zero ties therefore return index zero unless an earlier NaN rule
applies. The reduced axis must be nonempty and no longer than i32::MAX; other
empty axes are supported. Return the indices through `compile_many` or consume
them in gather/state updates without converting through F32. Native tests compare
every axis with a direct Rust reference, including ties, NaNs and infinities.
Lowering uses maximum plus integer minimum reductions and comparisons/iota,
not an unrolled scan or a guaranteed fused argmax kernel. This API alone does not
yet change the TinyLlama runner's logits download policy.

I32 `Index` tensors also support `bitwise_and`, `bitwise_or`, `bitwise_xor` and
`bitwise_not` on their exact 32-bit representations. Binary operands require the
same graph and shape; broadcasting remains explicit. `shift_left(bits)`,
`shift_right_logical(bits)` and `shift_right_arithmetic(bits)` take a construction-
time count in 0..32 and reject 32 or larger. Left shifts discard high bits,
logical right shifts fill with zero, and arithmetic right shifts extend the sign
bit. Results remain I32, without a floating-point roundtrip. They lower to native
[XLA integer operations](https://openxla.org/xla/operation_semantics#shiftrightlogical),
not C++ custom calls.

Native tests compare all 32 valid shift counts plus bitwise operations against
Rust I32/u32 references, including extrema, alternating bit patterns, values
above 2^24, scalar and empty shapes. A resident three-lane bitwise transition
runs 20 steps with no dynamic inputs and one compilation; its full I32 state
matches the host reference each step. An argmax→bitwise→F32 path has zero input
gradient, preserving integer nondifferentiability. These primitives support
graph-side RNG composition but are not by themselves a random generator, statistical-quality
validation or a security guarantee. Shift counts are not yet runtime tensors.

`random::threefry2x32([&key0, &key1], [&counter0, &counter1])` implements
[Random123 Threefry2x32-20](https://github.com/DEShawResearch/random123/blob/main/include/Random123/threefry.h)
using graph I32 wrapping additions, shifts and XOR. It returns two same-shaped
I32 tensors containing all 32 output bits. The four input words must have the
same graph/shape; each element is one block invocation, and scalar keys can be
explicitly broadcast. No host random draw, floating-point conversion, native
custom call or new build dependency is introduced. The 20 rounds are recorded
while building the graph, then executed through the normal HLO/PJRT path.

This is a stateless block function, not an implicit global stream. Reusing the
same key/counter gives the same output. Counter advancement, stream assignment,
seed entropy and output-word/batch ordering are caller policy; I32 values are
bit patterns, not signed random-number bounds. It does not implement JAX's
seed/split API or claim cryptographic security. Native tests match all three
official 20-round [known-answer vectors](https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors)
exactly for scalar and vector calls and cover empty tensors. Changing runtime
keys/counters reuses the executable. A separate four-lane resident test uses two
I32 state slots as a low/high 64-bit counter, checks carry and wrap against a
host u32/u64 reference for 20 steps, resumes via `into_state` midstream and
verifies an idle session is unchanged. It has no dynamic inputs and compiles
once. Counter wrap is deliberately observable in the test: applications must
prevent exhausting/reusing counter space when distinct blocks are required.
`random::threefry2x32_blocks([&key0, &key1], [&low, &high], shape)` accepts
same-graph scalar words and reserves consecutive 64-bit counters in row-major
element order. It returns `ThreefryBlocks`: two I32 tensors of `shape`, a proposed
scalar `[low, high]` next counter, and scalar F32 `counter_wrapped` (0/1). Each
element consumes one block (two random words), even when only one word is used.
Scalars consume one block; empty shapes consume none. Per-call block count is
limited to i32::MAX. Unsigned carry is implemented with I32 bit operations and
integer comparisons, not lossy counter-to-F32 conversion.

No state write is recorded by the helper. After using `draw.bits` to compute
the model/loss, callers can include `draw.next_counter` in an explicit
`write_many_if` alongside model/optimizer writes. Rejected writes therefore
need not advance the random stream. Multiple draws from an unchanged counter
reuse the same blocks: explicitly chain proposed counters to allocate disjoint
ranges. Arithmetic wraps modulo 2^64; `counter_wrapped` reports crossing the
period boundary but does not automatically reject a draw or reseed. Applications
must handle exhaustion/reuse according to their policy.

`random::ThreefryState` groups four resident scalar I32 slots in key0, key1,
counter-low, counter-high order. `initial_state(client, [key0, key1], counter)`
uploads explicit raw words; `slots()` exposes identities for checkpoint mapping.
Cloning the descriptor shares identities, not buffers or independent streams.
For multiple draws within a step, its non-Clone proposed sequence chains counters:

```rust,ignore
let random = ThreefryState::new(&mut graph)?;
let mut sequence = random.begin(&graph)?;
let first = sequence.blocks(&[batch, hidden])?;
let second = sequence.blocks(&[batch, hidden])?;
// Build model/loss using these bits, or convert via uniform_f32_from_bits.
let accept = sequence.commit_if(&mut graph, &loss.is_finite_mask()?)?;
optimizer.backward_step_if(&mut graph, &loss, &rate, &accept)?;
```

Draws do not record resident writes until commit. Dropping a proposal leaves
state unchanged, and failed draw construction does not advance its local cursor.
Commit consumes the proposal, rejects stale key/counter symbolic versions or a
foreign graph, and jointly gates both counter words. Its returned scalar 0/1
predicate combines the caller's request with a sticky no-wrap condition: ANY
64-bit wrap rejects the whole sequence, including an exact end-period boundary.
Use that returned predicate for optimizer and other state updates as well.
This guards against duplicate commits, not arbitrary independent sessions using
the same key/counter; it does not assign streams, reseed, or suppress computation
of rejected outputs. Reuse the same generated values during recomputation.
These are graph-construction methods, not an automatically captured Rust RNG.
The `random_state` tests cover chained/empty draws, failed construction, stale
and foreign proposals, accepted/rejected execution, carry, and sticky wrap. The
device dropout training example uses this API and retains its per-step F64
reference checks and shared RNG/optimizer rejection tests.

For distributions, `sequence.uniform_f32(shape)` consumes one block per element
and converts word0's high 24 bits to the discrete F32 grid in [0, 1), discarding
word1. `sequence.bernoulli(shape, probability)` returns unscaled F32 0/1 masks;
interior probabilities use `uniform < probability`, so their effective sampling
probability is `ceil(probability * 2^24) / 2^24`. This is deliberately not the
host initializer's 64-bit sampling policy. Probability 0 (including -0) produces
positive zeros, and probability 1 produces ones, without consuming blocks.
Empty shapes consume none; invalid probabilities/shapes do not advance the
proposed cursor. Values still require sequence commit before RNG state advances.
Masks can be obtained directly with `sequence.bernoulli(&[4, 1], 0.5)`.
Native tests compare distributions to raw-word conversion across multiple
probabilities, steps and counter carry, and verify endpoint-only sampling remains
accepted at the final 64-bit counter with all RNG slots unchanged.

For training, `sequence.dropout(&input, keep_probability)` combines sampling
and inverted dropout and returns `DropoutSample { output, keep_mask }`.
Probability must be finite in (0, 1]; this is KEEP probability, not drop rate.
At probability 1, output is the original input and the mask is all ones, with
no RNG consumption. Other probabilities use the sequence's high-24-bit Bernoulli
policy and divide retained values by the requested probability; on probabilities
off that grid, expected scaling is not exactly one. Invalid probabilities or
cross-graph inputs leave the proposed cursor unchanged. Inference explicitly
uses the input directly, without calling this training-only method.
Autodiff uses the existing mask; explicit reconstruction can call
`input.dropout_with_mask(&sample.keep_mask, keep_probability)` without drawing
again. This is not automatic activation-checkpoint orchestration. The device
dropout training example now uses this combined API. Native tests compare its
mask to raw Threefry bits, forward and reconstructed outputs, first/second
derivatives, identity and empty cases, rejected-step replay, and the exact
counter increment across carry: differentiation does not consume extra blocks.

Native batch tests compare every output word and proposed/committed counter
against a host reference across 32-bit carry and 64-bit wrap, for scalar, 2×3
and empty shapes. They interleave accepted/rejected updates and move state into
new sessions after each step; each shape compiles once for all starting counters.

`Index::take_along_axis(&indices, axis)` performs the same per-position gather
as its Tensor counterpart, but preserves I32 data without any F32 conversion.
Ranks and non-axis dimensions must match; the source axis must be nonempty.
Negative/oversized indices clamp to its endpoints. Tests exercise all axes,
empty output axes, I32 extrema and integers above F32's exact range.

This permits top-k sampling entirely on device: obtain `(values, candidates)`
from `logits.topk(k, axis)`, sample local indices from `values`, insert the removed
category dimension with `reshape`, then gather from `candidates` using the I32
local indices. The result contains original token IDs, not candidate ranks.
Validate the **original** logits before filtering if NaN/+infinity are errors:
top-k can discard invalid entries, so the candidate sampler's validity alone
does not validate the original distribution. The composition test uses finite
logits and verifies both candidate choices map to the correct original IDs.

`tensor.cumsum(axis)` returns inclusive F32 prefix sums with the same shape.
Empty tensors and length-one axes are identity; invalid axes and overflowing
nonempty padded extents are rejected. It lowers to one prefix `reduce-window`
with addition, not a Rust-unrolled chain or a dense triangular matrix. Reverse
mode is a suffix sum, supporting higher derivatives through composed ops.
Tests cover all axes, empty/singleton shapes, length 1024, nonfinite prefixes,
and first/second derivatives against independent scalar loops. Backend summation
order may differ from a sequential F32 sum; native scan performance is not yet
benchmarked. This general tensor primitive does not itself select a sampling policy.

`tensor.argsort(axis, descending)` returns same-shaped I32 ordering indices:
false means ascending, true descending. Both directions put NaNs last and
preserve original order among equal values (including signed zeros) and among
NaNs. Empty axes work; invalid or non-I32-sized axes are rejected. Combine with
F32/I32 `take_along_axis` to reorder payloads. The integer choice is
nondifferentiable, while gathering F32 values scatters gradients back to the
selected original entries. It shares stable HLO sort lowering with top-k;
tests also combine both directions, top-k and argmax in the same executable.

`tensor.topk(k, axis)` returns `(values, indices)` with that axis replaced by k,
in descending value order. It requires `0 <= k <= axis_size` and an I32-sized
axis; empty non-axis dimensions are supported. Equal values, including signed
zeros, preserve original index order. NaNs come after all non-NaN values and
preserve their own input order. Values retain the selected input bit patterns;
the API does not specify NaN payload preservation. For k=0, it returns empty
values/indices without emitting a sort, including when the source axis is empty.
Tests check zero first/second derivatives even with nonfinite source values.
The categorical sampling APIs still reject k=0: an empty candidate set cannot
define a categorical distribution. Gradients scatter through
selected entries (including deterministic tie selections), not through indices.

For positive k, HLO lowering uses one stable sort of values/indices, slices the
indices, and gathers the values. Graph instruction count does not grow with k. XLA controls
whether it optimizes away a full sort; this is not a measured fast GPU top-k
kernel or a top-p sampling implementation. Native tests compare axis variants,
ties, signed zeros, infinities, NaNs, empty batches, and first/second derivatives.

`random::categorical_from_bits(&logits, &bits, axis)` draws device-side
categorical indices using Gumbel-max, returning `indices` and a per-row F32
`valid` mask with the category axis removed. Supply same-shape I32 uniform words
from Threefry and commit the proposed RNG counter explicitly; this helper has no
hidden state or host sampling. Finite logits and negative-infinity category masks
are accepted. NaN, positive infinity, and rows without any finite category return
`valid=0` and placeholder index zero. Callers must handle validity before accepting
a token. Scale logits explicitly if temperature sampling is desired.

The sampler centers logits before adding noise and uses a 23-bit uniform midpoint
grid strictly inside `(0, 1)`. It is a finite-precision distribution approximation,
not an exact sampler for arbitrarily rare events, and floating-point near-ties
may differ across backends. Tests cover both category-axis positions, empty
batches, masked/invalid rows, maximal equal F32 logits, deterministic replay,
an independent F64 Gumbel reference, and a fixed-seed Threefry frequency smoke.
No top-k/top-p decoding policy is implied.

`random::top_p_categorical_from_bits(logits, bits, p, axis)` implements nucleus
sampling with finite p in `(0,1]`. It stably sorts descending, retains the smallest
prefix whose probability reaches p (including the crossing category), samples
with Gumbel noise, and returns original IDs plus per-row validity. p=1 bypasses
filtering. It validates every original logit, including discarded NaNs; invalid
rows return valid=0 and placeholder ID zero. Bits match the **full** logits shape
and correspond to sorted rank, not original ID. F32 softmax/prefix-sum rounding
can change cutoff decisions near p; no exact real-arithmetic guarantee is made.
Preceding mass is computed by shifting cumulative sums, not subtracting the
current probability. The first category is explicitly retained so a positive
subnormal threshold flushed to zero cannot remove every candidate; a CPU
regression test reproduces that boundary. Original validity still rejects
invalid distributions. No temperature or combined top-k policy is inferred.

`ThreefrySequence::top_p_categorical(logits, p, axis)` reserves one block per
original logit even when most candidates are discarded. Construction errors
preserve the proposed cursor; reduce validity and commit explicitly, using the
returned acceptance for other state writes. Boundary/crossing tests cover both
axes, invalid source distributions, and state rejection/advancement. This API
has not yet been connected to the TinyLlama CLI. A separate
An historical 32,000-class sampling diagnostic compares
top-k and top-p on the same RTX 5080: medians were about 0.140 and 0.185 ms,
respectively, including scalar downloads but excluding RNG/model computation.
A longer ordered-sample run found roughly 6 ms outliers in both modes. Phase
timing localized most observed stalls to the first scalar retrieval after execute
returned; swapping token/validity download order moved those stalls accordingly.
The plugin/driver/scheduler root cause is still unknown. This is not a
serving-latency guarantee.

For explicit top-k decoding, `random::topk_categorical_from_bits(logits, bits,
k, axis)` composes stable top-k, Gumbel sampling, and integer ID mapping. It
returns original category IDs and validates every **original** logit, including
discarded NaNs; invalid rows return valid=0 and placeholder ID zero. The bits
shape replaces the logits' category dimension with k. Words correspond to
descending candidate rank, not original category position. Consequently k equal
to vocabulary size samples the same distribution approximately, but does not
promise the same draw as `categorical_from_bits` given the same words.

`ThreefrySequence::topk_categorical(logits, k, axis)` allocates those words using
one block per candidate, including k=1. It preserves the proposed cursor on
construction errors; callers explicitly reduce validity and commit the sequence.
Original NaN/+infinity or all-masked rows remain invalid even if filtering would
have hidden them. Tests compare k=1/2/full and both axes with an independent F64
reference, verify stable cutoff ties, and check two chained draws reserve exactly
the candidate count while an invalid row can reject the whole batch. Neither
API adds temperature scaling, top-p, or a full LLM decoding loop.

`ThreefrySequence::categorical(&logits, axis)` reserves one block per logit and
uses word0, so callers need not manually allocate the words. Like other sequence
methods, failed construction leaves its proposed cursor unchanged. It does not
automatically reject invalid distributions: reduce the returned per-row `valid`
mask to a scalar, combine it with your acceptance predicate, and pass that to
`sequence.commit_if`. Use the **returned** acceptance to guard token/KV or other
state writes too, since counter exhaustion also rejects the sequence. An accepted
batch advances the whole sequence, not a separate counter for each valid row.
Categorical integration tests exercise rejected calls, invalid rows, two chained
draws, low-word carry, full-counter wrap, and host-value snapshot restoration
into independent device buffers with one compiled program. This is RNG/session
restoration coverage, not an end-to-end LLM checkpoint or performance benchmark.

`random::uniform_f32_from_bits(&bits)` maps the high 24 bits of each I32 bit
pattern to an exactly representable F32 value divided by 2^24. Uniform input
bits therefore produce a discrete uniform grid in [0, 1): zero is possible,
one is excluded, and the low eight bits are discarded. This avoids rounding
the full 32-bit integer to F32 and accidentally reaching 1. The conversion does
not add entropy or advance state. It can consume either Threefry output word.

```rust,ignore
let words = rxla_tensor::random::threefry2x32([&k0, &k1], [&c0, &c1])?;
let uniform = rxla_tensor::random::uniform_f32_from_bits(&words[0])?;
let half = graph.constant(&[], &[0.5])?.broadcast_to(uniform.shape())?;
let mask = uniform.lt_mask(&half)?;
let output = input.dropout_with_mask(&mask, 0.5)?;
```

Thresholding this grid quantizes Bernoulli probabilities in steps of 2^-24;
it is not the host Bernoulli sampler's 64-bit threshold policy. Counter/key
management and matching the mask probability to dropout scaling remain explicit.
Native tests verify exact bit-to-float mapping at zero/all-one/sign boundaries,
scalars and empty tensors. An 8192-element Threefry→uniform→mask→dropout graph
uses runtime scalar keys/base counter plus activation data, with no uploaded
mask. Both tested counter ranges match host bit-conversion, mask, output and
gradient calculations elementwise, and share one compilation. Basic mean and
keep-frequency checks are smoke tests, not a statistical-quality certification.

These tests establish CPU algorithm/state correctness, not a full statistical
test suite, GPU performance or an automatic RNG/distribution manager.

I32 Index tensors support `wrapping_add`, `wrapping_sub`, `wrapping_mul` and
`wrapping_add_scalar`. Binary shapes must match; use `broadcast_to` explicitly
for scalar/per-axis expansion. Arithmetic stays I32 in HLO and wraps modulo 2^32,
without F32 conversion, saturation or runtime overflow errors. These operations
can compute `base + offsets * stride` inside the graph for chunk positions and
gather inputs. Gather/dynamic-slice index clamping remains a separate operation;
overflow can produce negative positions, so callers must choose safe ranges when
that is not intended. Native tests check exact signed results at I32 limits and
beyond F32's exact-integer range, plus strided gathers with changing runtime inputs.
These operations can advance an I32 state slot via explicit recorded updates;
no global or implicit position counter is introduced.

`query_positions.causal_attention_mask(&key_positions)` accepts rank-one I32
Index tensors `[Q]` and `[K]`, producing an additive `[Q,K]` mask. Signed integer
comparison keeps extreme positions and adjacent positions above 2^24 exact;
there is no floating-point conversion or potentially overflowing subtraction.
Positions may be runtime inputs or derived from resident state, so fixed-shape
executables can be reused at different positions. Negative, duplicate and unsorted
positions are compared as supplied, not recognized as padding. Position gradients
are stopped; Q/K/V differentiation remains available. All-masked rows retain
ordinary softmax behavior, not an implicit zero result. Native tests cover changing
runtime positions with one compilation, value gradients and extreme coordinates.
TinyLlama uses its resident position reshaped to `[1]` with fixed key coordinates.

`graph.causal_attention_mask(queries, keys, query_offset)` (also on StateGraph)
builds a static additive `[queries, keys]` mask: entry `(i,j)` is zero when
`j <= query_offset+i`, otherwise negative infinity. Offset zero is ordinary
prefill; a later chunk or one-row decode supplies its absolute start position.
Keys are indexed from zero. Counts/offset are nonnegative and I32-coordinate
overflow is rejected. Empty masks are allowed. Mask construction uses graph
integer coordinates rather than a host literal and explicitly stops gradients
through the static positions. This neither tracks a dynamic position nor
updates a KV cache. Extra padding/window restrictions remain explicit.
Native CPU/CUDA tests compare full, 2+3 chunked and single-row queries against
each other and a dense F64 reference; future V positions have zero gradient.
This validates attention/mask semantics, not a full-model batched prefill engine
or a FlashAttention kernel.

`q.scaled_dot_product_attention(&key, &value, mask, scale)` records
`softmax(Q K^T * scale + mask) V`. Q/K/V have rank >=2 and shapes
`[..., queries, depth]`, `[..., keys, depth]`, `[..., keys, value_depth]`.
Leading dimensions broadcast through matmul; depth and key count must be positive.
Scale defaults to `1/sqrt(depth)`; explicit finite scales include zero/negative
values. Optional additive F32 masks broadcast to scores (0 keeps, -infinity masks).
There is no implicit causal mask, boolean conversion, dropout, head repetition or
KV update. Rows need a finite maximum; all-masked rows are not silently zeroed.
The implementation is ordinary graph composition, not a guaranteed fused or
memory-efficient FlashAttention kernel. Native tests cover multihead F64 reference
agreement, broadcast masks and explicit scales. Attention probabilities can still
be built explicitly with the lower-level matmul/softmax API for diagnostics.

`q.grouped_query_attention(&key, &value, mask, scale)` adds explicit head grouping:
Q is `[..., query_heads, queries, depth]`; K/V are
`[..., kv_heads, keys, depth/value_depth]`. Ranks and batch prefixes must match
exactly and query_heads must be a positive multiple of kv_heads. Consecutive query
heads share one KV head. The query heads/rows are reshaped into grouped matmul
rows and restored afterward, without inserting repeated K/V heads. Mask is first
broadcast to `[..., query_heads, queries, keys]`, then grouped consistently, so
multi-token per-head/per-query biases retain their order. Mask expansion may
materialize on a backend; this is not a memory-efficiency guarantee. Native F64
reference tests cover two batches, three query tokens, distinct per-head masks,
different value depth, and MQA/GQA/MHA (1/2/4 KV heads for four query heads).

The stateful masked-decode test combines resident K/V slots, runtime updates,
integer-position masks and two-head attention. Four steps reuse one executable
and match an independent f64 valid-prefix reference, including probabilities.
Unused cache entries are intentionally large/nonzero to verify exclusion. This
is an attention-step correctness test, not a full LLM decoder, tokenizer, model
loader, FlashAttention kernel or performance benchmark.

`Tensor::conv2d` uses NHWC input and HWIO kernels, with explicit nonnegative
padding in `Conv2dOptions`. It implements cross-correlation (no kernel reversal),
as typical neural-network convolutions do. Grouped kernels have shape
`[Kh, Kw, Cin/groups, Cout]`; depthwise uses `groups=Cin`, with contiguous output
channels per group. Use transpose for NCHW/OIHW model data. Bias and activation
are separate tensor operations; no inference-only fused API is required.
Current validation is CPU correctness, not a convolution performance benchmark
or proof of complete YOLO/model-import support.

`CustomCall::typed_ffi(name)` is the explicit backend-extension boundary. It
records a single-result `stablehlo.custom_call` with XLA typed-FFI API version 4;
opaque backend config, side-effect declaration and legacy API-version override
remain explicit builder choices. `call` is unsafe because RXLA can verify the
IR contract but cannot prove that an externally registered CPU/CUDA handler
obeys the declared buffer shapes, dtypes and ABI. All operands must share one
lazy trace, and output shape/dtype are part of the recorded semantic IR, so
semantic snapshots and structured regions preserve the call. This is the route
for kernels such as FlashAttention and specialized KV-cache operations; handler
registration and plugin packaging deliberately remain outside `rxla-core`.

Structured conditionals are native multi-result SSA operations. `Tensor::cond`
is the single-result convenience API; `Tensor::cond_many` preserves heterogeneous
result shapes and dtypes in one `stablehlo.if`, with semantic snapshots using
result aliases rather than duplicating the operation. `Cx::cond` additionally
threads every existing resident-state version as a hidden region result: both
branches start from the same snapshot and only the selected branch's versions
become visible afterward. State declaration and RNG stream advancement inside a
conditional remain rejected because they require separate schema and RNG-effect
merge semantics. This multi-result foundation is also required for loop-carried
index, user carry and output buffers in a future native `while`/`scan`.

Tensor rank and axis selection remain compile-time metadata, while dimension
sizes can be requested as runtime SSA values. `Tensor::static_dim(axis)` performs
an optional metadata query; `Tensor::dim(axis)` emits
`stablehlo.get_dimension_size` and returns a scalar I32 Tensor, and
`shape_tensor()` concatenates those values into a rank-one I32 Tensor. Such
values can feed structured `cond` and later `while` operations.
`static_numel()` is the metadata-only optional query; `numel()` multiplies
runtime dimension SSA values and therefore returns a scalar I32 Tensor instead
of panicking when a dimension is dynamic.

Bounded dynamic signatures use `Dim::Bounded { upper }`; raw public `-1`
dimensions remain invalid. The upper bounds are part of `TensorType`, the Pliron
ranked-tensor type, semantic snapshots, cache/artifact metadata and emitted
`#stablehlo.bounds`. `Tensor::dim_bound` and executable input/output specs expose
them without confusing an upper bound with a runtime extent. Same-shape
elementwise operations preserve bounds, and `broadcast_as` retains the target's
bounded shape.

This is currently an IR and compilation feature, not a promise that every PJRT
plugin can construct and execute bounded-dynamic buffers. The development ZML
CPU plugin compiles an identity signature, but ordinary elementwise compilation
requires XLA's internal `PadToStatic` custom-call target, which that plugin does
not register; a normal `[3]` PJRT buffer also does not carry the physical bound
and dynamic-size metadata required by an executable expecting `f32[<=8]`.
Consequently tests prove typed propagation and real CPU compilation separately,
but do not claim dynamic-buffer execution until the PJRT upload abstraction can
represent that metadata and the backend registers the required runtime targets.
