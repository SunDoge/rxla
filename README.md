# RXLA

[![CI](https://github.com/SunDoge/rxla/actions/workflows/ci.yml/badge.svg)](https://github.com/SunDoge/rxla/actions/workflows/ci.yml)

RXLA is an experimental Rust tensor and compiler stack built around Pliron,
StableHLO, XLA, and PJRT. It aims for an MLX-like lazy tensor experience while
keeping tracing, compilation, placement, state, and execution explicit enough
for compiler transformations and future distributed workloads.

The project currently prioritizes inference. Training support intentionally
starts small, with reverse-mode autodiff and functional SGD/Adam building
blocks rather than a large module framework.

> RXLA is early alpha software. APIs and crate boundaries may change.

## Quick start

Create lazy tensors, compose operations, then materialize them through a PJRT
runtime:

```rust,ignore
use rxla::{DType, Runtime, Tensor};

let x = Tensor::from_slice([2], DType::F32, [1.0, 2.0])?;
let y = Tensor::from_slice([2], DType::F32, [3.0, 4.0])?;
let z = (&x + &y)?.exp()?;

let mut runtime = unsafe { Runtime::load("/path/to/libpjrt_plugin.so")? };
z.eval(&mut runtime)?;
```

Run the facade example with a trusted PJRT plugin:

```sh
cargo run -p rxla --example basic -- --plugin /path/to/libpjrt_plugin.so
```

`PJRT_PLUGIN_PATH` is also accepted by examples that expose a plugin argument.
Loading a PJRT shared library is unsafe because its ABI and provenance must be
trusted by the application.

For reusable computations, use `Tracer` to construct a `Program`; use
`Runtime::eval` for ordinary lazy expressions. `Tensor` is the single value
handle across supported dtypes, including I32 index tensors. Indexing is
expressed through operations such as `take`, `take_along_axis`, gather lowering,
and dynamic slicing rather than a separate `Index` type.

Models are ordinary Rust `apply` functions. Typed input extractors keep shapes
and dtypes outside the mathematical body, while parameter shapes are inferred
where the corresponding tensors are already available:

```rust
use rxla::{
    DType, Tensor,
    nn::{Cx, Model, ModelInput, Result},
};

fn apply(cx: &mut Cx, image: Tensor, label: Tensor) -> Result<(Tensor, Tensor)> {
    let logits = cx.layer("head")?.linear(10).apply(&image)?;
    let loss = logits.cross_entropy_with_indices(&label, 1)?.mean(&[0], false)?;
    Ok((loss, logits))
}

let model = Model::new(apply)
    .inputs((
        ModelInput::new([32, 768]),
        ModelInput::new([32]).with_dtype(DType::I32),
    ))
    .trace()?;

assert_eq!(model.schema().parameters()[0].shape(), [10, 768]);
assert_eq!(model.outputs().len(), 2);
```

The same handler mechanism supports one input, two through sixteen independent
arguments, arrays, vectors, and application-defined structs. `compile()` keeps
that model ABI attached to the executable, so `CompiledModel::run` validates
runtime inputs and named parameters and reconstructs typed outputs. See the
[parameter-effect design](docs/PARAMETER-EFFECT-DESIGN.md) for the effect and
ABI invariants.

## Architecture

RXLA separates the frontend from execution without maintaining two competing
semantic graph representations:

```text
Tensor API / Tracer
        ↓
Pliron SSA IR
        ↓
StableHLO
        ↓
XLA compile options and PJRT
        ↓
CPU, CUDA, or another PJRT backend
```

Pliron is the semantic IR and transformation boundary. StableHLO is the typed
backend interchange accepted by XLA. PJRT owns device discovery, compilation,
buffers, and execution. Backend selection is explicit, so one process can own
multiple clients for heterogeneous execution.

Parameters and mutable state are modeled through explicit contexts and effects.
This lets parameter shapes depend on tensors available at the use site without
requiring PyTorch-style constructor plumbing.

## Workspace

The foundational crates are:

- `rxla`: public facade.
- `rxla-core`: tensor API, tracing, autodiff, state, compilation, and runtime.
- `rxla-ir`: Pliron dialect, verification, analysis, and StableHLO lowering.
- `rxla-pjrt`: low-level PJRT client and buffer interface.
- `rxla-xla-proto`: typed XLA configuration and artifact protobufs.
- `rxla-cache`: filesystem compilation cache primitives.
- `rxla-nn`: typed model handlers, parameter/state effects, schemas, and
  neural-network operations.

Experimental integration crates include `rxla-onnx`, `rxla-safetensors`,
`rxla-train`, and `rxla-models`. They are not all part of the initial public
release set.

## Development

Run the host-only development checks without loading a native plugin:

```sh
sh scripts/check.sh --offline --quick
```

The complete check script covers formatting, workspace targets, tests, and
Clippy. Native CPU/CUDA checks require an explicitly trusted PJRT plugin; see
[CUDA validation](docs/CUDA-validation.md).

Run the same real CPU execution smoke tests used by CI with:

```sh
PJRT_CPU_PLUGIN_PATH=/path/to/libzml_cpu.so sh scripts/check-cpu.sh --offline
```

Python reference and benchmark scripts use the locked uv environment:

```sh
uv run --frozen python scripts/jax_oracle.py broadcast --emit stablehlo
```

Commits follow Conventional Commits and are checked by Cocogitto:

```sh
cog commit feat "add an operation" core
cog check
```

Release-plz remains responsible for workspace versions, tags, releases, and
crates.io publishing.

## Documentation

- [Product vision and roadmap](docs/VISION-ROADMAP.md)
- [Scope and current limitations](docs/SCOPE.md)
- [Implementation and API notes](docs/IMPLEMENTATION-NOTES.md)
- [Execution-plan design](docs/EXECUTION-PLAN-DESIGN.md)
- [Parameter-effect design](docs/PARAMETER-EFFECT-DESIGN.md)
- [State design](docs/STATE-DESIGN.md)
- [Tensor storage design](docs/TENSOR-STORAGE-DESIGN.md)
- [Tensor image augmentation experiment](docs/IMAGE-AUGMENTATION-BENCHMARK.md)
- [Distributed PJRT and NCCL](docs/PJRT-DISTRIBUTED.md)
- [MLX layout review](docs/MLX-LAYOUT-REVIEW.md)
- [Deployment notes](docs/DEPLOYMENT.md)
- [Profiling guide](docs/PROFILING.md)
- [Release process](docs/RELEASING.md)

Performance results are workload- and backend-specific. See the benchmark
documents under [`docs/`](docs/) rather than treating individual measurements
as general performance guarantees.

## License

Apache-2.0.
