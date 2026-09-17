# Tensor descriptor and managed storage

Status: first implementation, 2026-09-13. F32/I32/BF16 share Tensor and managed input storage.
This is not a complete MLX-style auto-evaluating array API.

## Ownership model

```text
Tensor (one Arc pointer)
  -> immutable TensorDescriptor
       -> optional private tracing identity { session, node ID }
       -> SmallVec<i64, 5> logical shape + explicit dtype
       -> optional input binding
            host:   Storage owner + checked StridedLayout
            native: Storage owner -> Arc<Buffer> -> PJRT allocation + client

PendingExecution -> native input/output/executable owners until completion
```

Tensor clones share one descriptor and do not copy shape or increment a second
shape allocation's refcount. Attaching storage returns a new descriptor; other
clones retain their previous binding. Distinct descriptors can share a storage
owner while describing distinct host views. Storage identity is not a universal
alias detector: independently wrapping the same external allocation may create
distinct owners. No writable aliases, donation or copy-on-write are provided.

The descriptor is immutable and exposed only as an opaque Deref target; its
fields are crate-private and no DerefMut exists. Arc is deliberate: std already
provides a one-allocation control block and a one-pointer handle, without custom
unsafe refcount logic. A custom intrusive pointer remains optional future work
requiring allocation/clone benchmarks, overflow/drop tests and a safety audit;
there is no demonstrated performance gain from replacing Arc at this point.

Tensor and Storage are `Send + Sync`. Pliron 0.18 makes context auxiliary data
`Send`, and a compile-time assertion verifies `ProgramIr: Send`; no unsafe trait
implementation is added for the IR. Each graph retains a weak link to its lazy
session, so moving a Tensor to another thread does not silently start a different
graph. Host owners and raw-allocation deleters must be `Send + Sync`. PJRT buffers
already use Arc-backed thread-safe ownership; in-flight submit/wait handles keep
their separate runtime contract. Integer values use the same Tensor descriptor
with `DType::I32`.

## Managed ownership and execution

`Storage::host(dtype, owner)` retains an owned `AsRef<[u8]>` provider, such as Vec,
Box, Arc or an mmap wrapper. It does not copy the payload. Its final release
drops the provider; retaining a view/Tensor retains that owner. Providers must
keep logical contents stable while used as input. Every requested HostView
revalidates its reachable interval against the currently supplied slice. Only
read-only byte access is exposed, never aligned typed references. Storage erases
the Rust element type and records dtype explicitly; it does not use a per-dtype
Vec enum. Tensor descriptors also record their logical dtype, propagated from
the IR when reconstructing autodiff values. `Tracer::input_dtype` selects
F32/I32/BF16. Shape transforms and gathers preserve
dtype; integer arithmetic stays exact. Unsupported BF16 arithmetic and non-F32
autodiff are rejected. to_f32 is explicit conversion, not a bitcast.

`Storage::device(Arc<Buffer>)` retains an existing native owner without copying.
`with_host_storage` validates shape, dtype, element size and view bounds;
`with_device_storage` validates native shape/dtype against the descriptor. Host
binding also rejects same-width dtype mismatches (I32 versus F32). Both reject computed
graph values: externally attaching data is input binding, not asserting that
an arbitrary computation has completed. This avoids a fake materialization cache.

Operators still construct graph values. Their results do not automatically
capture input bindings, and a graph/snapshot never owns resident data. Keep the
bound inputs and pass them explicitly to `Compiler::execute_bound(output, inputs)`
in parameter order. The method rejects wrong graphs/order, missing storage and
count mismatches. It accepts F32/I32/BF16 bound inputs. It returns ordinary native Buffers.

`to_buffer(client)` explicitly packs/uploads host data; repeated calls on a host
binding upload repeatedly. `to_device(client)` returns a resident bound Tensor
for reuse. Its subsequent to_buffer calls clone the same native wrapper, without
another upload. Foreign-client resident data is rejected, never implicitly
transferred. `Storage::upload(layout, client)` supports F32, I32 and BF16 raw host
storage through the unified graph binding API. It packs directly into one
typed, aligned staging vector at the native boundary; there is no intermediate
packed byte vector, and I32/BF16 never round through F32. This is still an explicit
copy/upload, not zero-copy. Byte interpretation is native endian; byte counts,
layout widths, bounds and overflow are validated before native calls.

The dtype checkpoint adds same-width mismatch rejection and unaligned byte-view
packing tests. CPU/CUDA native uploads preserve I32 MIN/MAX and 16,777,217 exactly,
and BF16 raw bits including negative zero and a NaN payload. These are storage
round trips, not low-precision arithmetic.

Readiness remains owned by PendingExecution, not an optimistic descriptor flag.
Dropping a Tensor during execution does not release resources retained by that
pending task. Its completion errors must still be observed through wait.

## Evidence and remaining work

Host tests verify pointer-sized Tensor handles, shared descriptors, inline shape
storage/spill, unchanged original bindings and HLO, gradient identity, bad binding
rejection and a custom host owner dropped exactly once after the last view owner.
`managed_tensor` exercises a transposed host view, explicit upload, owner release,
three resident executions and pending completion after dropping Tensor/compiler
owners. The example passes on the configured CPU and CUDA plugins.

The full CPU gate (`scripts/check-xla.sh --offline --with-plugin`) also passes,
including default/cache tests, compile-time Send/Sync assertions, state and
worker tests, Clippy and the independent prepared-state consumer. Log:
`/tmp/xla-managed-tensor-full-cpu.log`. This is not a full-model speed comparison.

The selected CUDA gate also passes (`/tmp/xla-managed-tensor-selected-cuda.log`),
including the independent prepared-state consumer. The managed example was
additionally run directly on both backends; its ignored native test passed on
CPU. It is now included in the CPU example and selected CUDA scripts for future
runs. No full-model, multi-GPU or speed-parity claim follows from these checks.

Runtime results are managed Tensor handles with PJRT storage and owned
physical-layout metadata. They can be used as inputs to another Program or as
materialized leaves in further lazy Tensor expressions. Lazy graph identity is
private and is never part of the public descriptor API.

`Tensor::from_slice(shape, dtype, values)` copies explicitly typed values, while
`Tensor::builder(shape, dtype)?.from_vec(values)` retains an owned allocation
without copying its payload. The builder also exposes an unsafe borrowed-slice
constructor with an explicit lifetime contract. These constructors and the
lower-level `from_host_storage` construct materialized leaves in a private lazy
session. Ordinary operations
append expressions without dispatching; `Tensor::eval` and
`Runtime::eval_many` compile, cache, and materialize requested roots. Explicit
Tracer/Program construction remains available for repeated execution.

`TensorBuilder::from_raw_parts` makes external allocation ownership explicit:
its deleter is retained by `Storage` and invoked exactly once after the final
owner is dropped, including validation failures. DLPack 0.9 import builds on
that ownership model through an RAII `Managed` owner. Compact CPU tensors,
including dlpark's zero-copy `ImageBuffer` producer, preserve the original data
pointer and producer deleter. Non-compact and device DLPack imports remain
rejected until their allocation spans and PJRT external-buffer contracts can be
represented without pretending a device pointer is host memory.

Still pending: layout-aware native views,
external device import, donation, and typed host-storage upload optimization.
No DLPack or intrusive pointer implementation is included.

## References and what was adopted

[MLX array](https://github.com/ml-explore/mlx/blob/main/mlx/array.h) separates
shared descriptors from shared data owners and tracks layout and execution
information. We adopt descriptor/storage separation, not its direct backend
buffer-view capabilities.

[PyTorch TensorImpl](https://github.com/pytorch/pytorch/blob/main/c10/core/TensorImpl.h)
separates tensor metadata from storage and documents shared version counters for
views and saved values. We retain immutable bindings for now; future mutation
must include version/alias semantics rather than adding writable pointers alone.
