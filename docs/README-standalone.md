# rxla

Standalone Rust tensor workspace backed by XLA HLO protobufs and PJRT.

This repository is the extracted XLA workspace from the former development
tree. It has no dependency on an IREE source checkout, CMake build, MLIR, or
LLVM.

## Crates

- `rxla-xla-proto`: checked-in HLO protobuf types.
- `rxla-pjrt`: dynamically loaded PJRT client/runtime bindings.
- `rxla-core`: graph construction, dtype-erased Tensor handles, storage, AD,
  state, and HLO lowering.
- `rxla-safetensors`: optional SafeTensors/model utilities.
- `xtask`: explicit protobuf/binding generation tools. From this repository,
  use `cargo xtask check` or `cargo xtask generate`.

## Quick start

New applications should use `Tensor` with `Tracer`/`Program` and `Runtime`.
The tracer's graph belongs to one tracing session and is discarded after an
immutable program snapshot is produced; the executor compiles, caches, and
dispatches that snapshot in eager or lazy mode. The public `Graph`/`Compiler`
surface is retained for compatibility with existing low-level callers.

```sh
cargo test --offline -p rxla-core --lib
cargo test --offline -p rxla-core --test unified_tensor
```

Native execution requires a compatible PJRT plugin:

```sh
PJRT_PLUGIN_PATH=/path/to/libzml_cpu.so \
  cargo test --offline -p rxla-core --test unified_tensor -- --include-ignored
```

All runtime dtypes use `Tensor`; the removed `Index` compatibility type is no
longer part of the API. The active design and limitations are documented in `SCOPE.md` and
`TENSOR-STORAGE-DESIGN.md`.
