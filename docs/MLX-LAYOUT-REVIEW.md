# MLX reference review: layout, views and execution

Reviewed 2026-09-13 against upstream `main` source (moving references, not a
pinned MLX release). This is a source/design review, not a benchmark or an
implementation of strided PJRT buffers. It complements the accepted
[execution planning requirement](EXECUTION-PLAN-DESIGN.md).

## What the implementation actually does

- MLX's descriptor combines graph links with shape, strides, shared storage,
  offset, contiguity flags and execution status/events. It separately checks
  descriptor and data ownership for donation eligibility. This demonstrates
  that lazy graph construction and layout metadata can coexist; laziness is
  not a reason to omit a storage model. [array.h](https://github.com/ml-explore/mlx/blob/main/mlx/array.h)
- Descriptor initialization computes default row-major strides even for an
  unevaluated array. Storage setters can replace those strides. Shared-buffer
  assignment preserves the allocation owner and adjusts the offset. Thus
  descriptor defaults are not evidence that every operation materializes a
  row-major result. [array.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/array.cpp)
- CPU reshape asks `prepare_reshape` whether copying is necessary; otherwise it
  uses a shared-buffer reshape. CPU `Contiguous` can also choose a copy based on
  allocation size, not just contiguity. A view retaining a large allocation is
  not always the desirable outcome. [CPU primitives](https://github.com/ml-explore/mlx/blob/main/mlx/backend/cpu/primitives.cpp)
- MLX distinguishes unscheduled, evaluated-but-not-necessarily-complete and
  available data. Availability checks interact with completion events rather
  than treating allocation or scheduling as success. [array.h](https://github.com/ml-explore/mlx/blob/main/mlx/array.h),
  [array.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/array.cpp)

These are observations about the inspected implementation, not guarantees for
all MLX backends, all view operations, arbitrary overlapping writes or zero-copy
external imports. We have not audited every transpose/slice backend path.

## Mapping to our code

| Area | Current Rust/XLA implementation | Decision |
| --- | --- | --- |
| Semantic values | `crates/rxla-core/src/pliron_ir/`: typed SSA values, transpose permutation and slice steps | Keep storage-independent math; slice steps are not byte strides |
| StableHLO boundary | Pliron conversion exports logical tensor types; PJRT chooses physical layouts | Audit optimized layouts and boundary constraints; do not confuse logical element strides with buffer byte strides |
| Runtime storage | `crates/rxla-pjrt/src/runtime.rs::BufferInner` owns an opaque native buffer and client via Rc | Keep thread-affine native ownership; add owned physical-layout inspection before advertising views |
| Upload/download | Upload accepts dense typed slices with exact logical element count; neither path sets explicit layout fields | Non-contiguous host input is a missing adapter capability; downloaded host order must not be confused with device order |
| Completion | `PendingExecution` retains inputs/executable/outputs and waits on an event | Preserve this foundation; ready can mean failed, so completion status must still be checked |
| Layout ABI | Generated PJRT types include tiled/strided layouts and GetMemoryLayout | Generated fields alone do not prove plugin capability, zero-copy import or arbitrary device-view support |

## Adopted constraints

1. Separate semantic Tensor type, requested layout constraints and actual storage
   layout. An unresolved layout must remain unresolved, not be reported as dense.
   Physical layout needs strided, tiled and opaque/unsupported cases; ordinary
   strides cannot describe every XLA tiled allocation.
2. For conventional views, explicitly model allocation ownership, logical shape,
   byte offset and signed byte strides. Use units in field/API names. Distinguish
   logical payload bytes, reachable address span and retained allocation bytes.
   Do not infer physical allocation size from `numel * itemsize`.
3. Validate shape/stride rank, checked address arithmetic, alignment and reachable
   bounds before exposing safe host views. Treat empty and singleton dimensions
   explicitly. Zero strides imply possible aliasing; negative strides need
   bounds checks in both directions. No unrestricted mutable overlapping view.
   Unsupported layouts must be rejected or explicitly packed, never silently
   misinterpreted as dense.
4. Preserve transpose/slice/reshape as graph operations until a backend-supported
   storage realization is selected. A host-side descriptor edit cannot create a
   native PJRT view. No universal zero-copy promise for these operations.
5. Donation requires backend support plus alias/liveness and in-flight consumer
   checks. Rust reference-count uniqueness alone cannot authorize native reuse.
   Keep donation disabled until its execution and failure contract is tested.
6. Stage interfaces must describe layout requirements and explicit conversions.
   Account for packing, transfers and retained allocations in planning costs.
   Prefer owned metadata and simple concrete types; do not add stride/rank
   generics throughout the Tensor API.
7. Logical metadata inspection must not compile, execute or synchronize. Physical
   metadata queries may call the plugin but must not silently evaluate a graph.
   Distinguish unsupported queries from dense layouts and from query failures.

## First validation checkpoint (2026-09-13)

`Buffer::memory_layout()` now returns owned `BufferMemoryLayout::{Tiled, Strided}`
metadata using the pinned legacy API. It does not create views, download data,
wait on an execution event or infer dense layouts on failure. The decoder bounds
metadata lengths, validates rank/permutations, and preserves tile groups and
signed byte strides. Native pointers still require a trusted plugin.

The initial implementation incorrectly required initialized output structure
headers: CPU passed, CUDA returned small/invalid `struct_size` values. Inspection
of upstream [ConvertToBufferMemoryLayoutData](https://github.com/openxla/xla/blob/main/xla/crates/rxla-pjrt/c/pjrt_c_api_helpers.cc)
shows that the legacy producer leaves output headers uninitialized. The decoder
now reads only initialized active payload fields via raw field pointers, without
copying whole union members or reading output headers. Outer API slot/argument
size checks remain. A regression constructs output with uninitialized headers.
This is legacy output handling, not a relaxation of input ABI validation.

The `layout_probe` example passes on the existing CPU plugin and RTX 5080 CUDA
plugin documented in [CUDA validation](CUDA-validation.md). Both report:

| Value | Logical shape | Reported minor-to-major | Tiles |
| --- | --- | --- | --- |
| Dense upload | `[2,3]` | `[1,0]` | none |
| Transpose output | `[3,2]` | `[1,0]` | none |
| Transpose then reshape | `[6]` | `[0]` | none |
| Scalar upload | `[]` | `[]` | none |
| Empty upload | `[0,3]` | `[1,0]` | none |
| Singleton axes | `[1,3,1]` | `[2,1,0]` | none |

Output values match `[0,3,1,4,2,5]` exactly; uploads round-trip, and copied
metadata remains usable after buffer destruction. The transpose result has a
dense row-major physical order, not a claimed shared strided input view. These
results do not establish intermediate layouts, copies, aliasing or speedup.
Native strided/tiled-nonempty variants remain untested; decoder unit tests cover
them. All 14 PJRT library tests (including five new layout tests) and PJRT
all-target/example Clippy pass. No full CPU/CUDA model regression was rerun.

Reproduce from the standalone repository root with the usual trusted plugin env:

```sh
cargo test --offline -p rxla-pjrt --lib
cargo run --offline -p rxla-core --example layout_probe
```

## Host layout foundation and const-generic storage

`Shape` and `ByteStrides` now own private
`SmallVec<i64, 5>` storage, using exactly `smallvec = 2.0.0-alpha.13` (an alpha,
not an RC). Public accessors return slices; capacity is not a Tensor generic or
a rank limit. Tests verify inline rank five and heap fallback at rank six.
The PJRT strided-query variant now uses the unit-specific `ByteStrides` type;
`ByteStrides::from_elements` provides checked conversion at adapter boundaries.
Existing semantic graph shape vectors have not been mechanically migrated.

`StridedLayout` validates concrete shape, matching rank, nonzero element byte
size, payload overflow and the complete reachable byte interval. It carries
unsigned byte offset and signed byte strides; conversion from element strides
checks multiplication overflow. Negative and zero strides are valid
descriptions, not evidence that PJRT accepts them as device views. Tiled native
layouts remain separate. Unknown/query errors never become dense layouts.

`HostView` borrows read-only raw byte storage and checks the interval against its
length. It provides checked element reads and explicit row-major packing into
caller-provided storage. No allocation, typed-reference alignment promise,
mutable alias, DLPack pointer or native upload occurs during packing. The backing
slice length is not represented as the original allocation's capacity. Empty
views access no bytes but still require their offset to be within the slice;
canonical empty row-major layouts use zero strides. Singleton strides do not
affect row-contiguity. Byte-strided raw reads need not be element-aligned; a
future typed-view adapter must separately prove alignment.

Five host tests cover units/inline capacity, transpose/reversal/broadcast/steps,
multi-byte values, scalar/empty/singleton shapes and invalid bounds/overflow.
An exhaustive test within those five checks 6,272 small two-dimensional
shape/stride/offset combinations against independently enumerated addresses.
All 19 PJRT library tests pass. DLPack and device views remain out of scope.

## Remaining implementation gates

1. Migrate legacy layout inspection to the layout extension when its bindings and
   plugin capabilities are validated; add native non-default layout fixtures.
2. Integrate the tested host descriptor at explicit upload boundaries. Only enable
   direct strided upload after plugin support and host-memory lifetime semantics
   are verified; otherwise use explicit packing. No implicit typed reinterpretation.
3. Audit transpose/reshape chains through optimized HLO and host round trips.
   Compare initial versus final layouts and measure conversions without asserting
   that an HLO optimization proves runtime zero-copy behavior.
4. Add layout constraints/conversions to execution plans, then investigate native
   interop and donation separately. Require alias, pending-consumer, failure and
   retained-allocation tests before exposing writable views or memory reuse.

SmallVec is the only new direct dependency. No backend switch, normal-build code
generation, unsafe Send/Sync, native Tensor view API or zero-copy interop was
introduced by this work.

## Lazy execution follow-up (2026-09-17)

This follow-up inspected the local MLX checkout at commit `1f8e74e3f`, in
particular `mlx/array.{h,cpp}`, `mlx/transforms.cpp`, `mlx/compile.cpp`,
`mlx/stream.cpp`, and `python/mlx/nn/layers/base.py`. The commit is recorded so
later reviews can distinguish source changes from design disagreements.

MLX does not maintain a separate eager value type. An `array` descriptor is both
a lazy graph value and, after evaluation, an owner of storage. It has three
explicit states: unscheduled, evaluated but possibly incomplete, and available.
`eval(outputs)` returns immediately for an empty list, waits when every output
has already been scheduled, and otherwise schedules the union of the requested
roots. Successful non-tracer values are detached from their primitive and
inputs after scheduling. Multi-result primitives track sibling outputs so the
primitive is scheduled once and all results transition together.

Those observations support the following RXLA decisions:

1. Keep one public `Tensor` type rather than adding an eager tensor. A materialized
   Tensor remains usable as a leaf of later lazy expressions.
2. Keep `eval_many` as the primitive operation. It must ignore already
   materialized roots, accept an empty collection, compile shared ancestry once,
   and atomically publish every requested result. Single-value `eval` is only a
   convenience wrapper.
3. Cutting an evaluated node to a bound input is the RXLA analogue of MLX graph
   detachment. RXLA rewrites the shared SSA node because later expressions may
   already refer to it; it must not merely set a buffer flag while retaining the
   whole executed ancestry.
4. Do not attach an MLX-style stream to every RXLA semantic operation. MLX uses
   streams and fences because it schedules backend primitives itself. RXLA hands
   a whole StableHLO program to XLA, whose scheduler owns graph-internal ordering.
   Placement belongs in execution planning; PJRT events belong at submission and
   materialization boundaries.
5. Add asynchronous Tensor materialization only when readiness and failure are
   represented explicitly. `PendingExecution` is the low-level foundation, but
   a Tensor must not look materialized merely because a PJRT buffer was allocated
   or submitted. MLX's evaluated-versus-available distinction is the relevant
   model.
6. Keep reusable `Program` optional. MLX is lazy without requiring users to call
   `compile`; its compile transform traces on the first concrete signature,
   simplifies/fuses, caches by inputs, and substitutes later inputs. RXLA may
   compile internally during `eval`, while explicit `Program` remains useful for
   stable ABI inspection and repeated serving calls. A user-visible `jit`
   decorator is not required for the core experience.

Two MLX choices are intentionally not copied. Its `Device` is a closed CPU/GPU
pair because those are the in-tree backends; RXLA retains backend names plus
ordinals because PJRT implementations form an open set. MLX constructors infer
dtypes from C++/Python values, whereas RXLA keeps the explicit dtype argument at
the raw host-storage boundary and verifies it against `TensorElement`.

MLX's Python `Module` recursively discovers arrays stored in object containers,
marks paths frozen through a side set, and replaces parameter leaves during
optimizer updates. That is useful evidence for tree filtering and partial
updates, not a reason to adopt constructor-owned Rust modules. RXLA's parameter
effects keep shape dependencies at the use site and should expose equivalent
named selection over the resulting schema without moving parameter creation
back into constructors.
