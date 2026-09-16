# Parameter declaration as a tracing effect

Status: public tracing API and resident optimizer effects implemented
(2026-09-16).

RXLA's primary model-construction API will not require an `nn.Module`-style
construction phase that predeclares every parameter shape. A parameter is
declared where it is used while tracing a program, through an explicit context
effect such as `Cx::param`. The existing `module` API remains supported
compatibility infrastructure while this direction is implemented; it is not the
target model authoring abstraction.

```rust
fn head(cx: &mut Cx, x: Tensor, classes: usize) -> Result<Tensor> {
    let width = x.shape()[x.shape().len() - 1];
    let weight = cx.param("weight", &[width, classes])?;
    x.matmul(&weight)
}
```

The shape dependency is explicit (`x.shape() -> weight.shape`), while parameter
registration, stable identity, initialization, checkpoint naming and runtime
binding are handled by `Cx`. Model configuration should express user choices
(number of classes, depth, feature policy), not duplicate dimensions already
known at a tensor use site.

The model body is written once. It is never allowed to infer whether it is the
"first" invocation by consulting global state or a mutable variable registry.
Instead, an explicit interpreter chooses the meaning of the same effect:

| Context | `cx.param("weight", shape, dtype)` |
| --- | --- |
| schema/init trace | `DeclareParam`: validate and record schema, initializer and canonical path |
| apply/compile trace | `ReadParam`: look up the already-frozen schema and emit its parameter SSA input |
| execution | bind/read the corresponding resident or supplied buffer; no declaration is possible |

Conceptually, callers capture a model body in `Model` once. `Model::trace`
interprets it to discover the schema and then produces the applied trace. The
lower-level `init` and `apply` methods remain available when checkpoint tooling
needs to inspect or restore a schema between those phases. This is an API
boundary, not a second model implementation: users never write a separate init
function, repeat tracing closures, or manually plumb a parameter tree.

The first implemented API exposes this directly:

```rust
use rxla::{Tensor, nn::{Cx, Model, ModelInput, Result}};

fn apply(cx: &mut Cx, x: Tensor) -> Result<Tensor> {
    let x = cx.named("hidden")?.linear(8).apply(&x)?.relu()?;
    cx.named("head")?.linear(3).apply(&x)
}

let (schema, applied) = Model::new(apply)
    .inputs(ModelInput::new([2, 4]))
    .trace()?;
assert_eq!(schema.parameters()[0].path(), "hidden.weight");
assert_eq!(applied.outputs()[0].shape(), [2, 3]);
# Ok::<(), rxla::nn::Error>(())
```

Run the facade example with `cargo run -p rxla --example model`.

Input structure is described separately from the model function, so `apply`
receives ordinary lazy tensors rather than creating placeholder-like values in
its body. `ModelInputs` is a typed extractor and `ModelHandler` adapts ordinary
Rust function arities: tuple specifications become separate function arguments,
while arrays, vectors and application-defined structs remain structured values.

```rust
# use rxla::{Tensor, nn::{Cx, Model, ModelInput, Result}};
fn apply(cx: &mut Cx, image: Tensor, timestep: Tensor) -> Result<Tensor> {
    let features = cx.named("image")?.linear(32).apply(&image)?;
    Ok(features.add(&timestep)?)
}

let model = Model::new(apply).inputs((
    ModelInput::new([4, 32]),
    ModelInput::new([4, 32]),
));
let (_, applied) = model.trace()?;
assert_eq!(applied.schema().inputs().len(), 2);
# Ok::<(), rxla::nn::Error>(())
```

Applications can implement `ModelInputs` for a domain struct such as
`DiffusionBatch` and return a matching `DiffusionTensors` struct. The blanket
`ModelHandler` implementation passes that value as one argument; implementations
for two through sixteen extractors spread them across the corresponding function
arguments. This is the same marker-trait technique used by Rust web-framework
handlers, without their async request machinery. It keeps input naming and
grouping in Rust's type system without arity-specific model classes or a string
map. Calling `cx.input` inside a zero-input `Model` remains available for
low-level or dynamically assembled definitions.

Trainability belongs to a particular transformation, not permanently to a
parameter declaration. The immutable schema provides typed parameter identities
and deterministic, scope-based selections instead:

```rust
# use rxla::{Tensor, nn::{Cx, Result, init}};
# fn apply(cx: &mut Cx) -> Result<Tensor> {
#     let x = cx.input(&[2, 4])?;
#     cx.named("head")?.linear(3).apply(&x)
# }
let (schema, _) = init(apply)?;
let trainable = schema.select_under("head");
for (id, parameter) in trainable.parameters() {
    println!("{id:?}: {}", parameter.path());
}
# Ok::<(), rxla::nn::Error>(())
```

`AppliedModel::parameter_tensors(&selection)` maps those stable identities to
the corresponding SSA tensor handles in selection order. Passing that slice to
`Tensor::grad` already performs partial autodiff; optimizer integration can use
the same ordering. The same model can therefore train everything, freeze a
backbone, or update an adapter without changing its forward definition. Mutable
non-parameter data belongs to a separate state effect; immutable assets belong
to constants.

The deliberately small initial training surface provides fused SGD and Adam:

```rust
# use rxla::{Tensor, nn::{Cx, Model, Result}};
# use rxla_train::prepare_model_sgd;
# fn linear_loss(cx: &mut Cx) -> Result<Tensor> {
#     let x = cx.input(&[2, 3])?;
#     let y = cx.named("linear")?.linear(1).bias(false).apply(&x)?;
#     Ok(y.mul(&y)?.sum(&[0, 1], false)?)
# }
let (schema, model) = Model::new(linear_loss).trace()?;
let trainable = schema.select_under("linear");
let step = prepare_model_sgd(&model, &trainable, &model.outputs()[0], 0.01)?;
let stablehlo = step.prepare(&model)?;
assert_eq!(stablehlo.output_count(), trainable.len());
# Ok::<(), Box<dyn std::error::Error>>(())
```

The lowered program contains forward, loss, reverse-mode autodiff and
`parameter - learning_rate * gradient` together. A step therefore executes as
one XLA program and returns device-resident replacement parameter buffers.

Adam uses the same transformation boundary, but its moments and step counter
are named resident effects appended to the applied model. They are hidden state
roots rather than extra public graph inputs or outputs. A stateful session
therefore commits model state, RNG and optimizer state in the same execution;
only selected replacement parameters cross the visible output ABI. This is not
a global optimizer registry: the transform explicitly declares stable paths
under `__optimizer.adam`, and a session owns the resulting buffers.

`Cx` intentionally exposes only effect primitives and `named`. The layer
vocabulary lives on the temporary named namespace, so adding layers does not
turn `Cx` into an ever-growing god object. `Named` also dereferences to `Cx`,
which permits closure-free structural scopes:

```rust
# use rxla::{Tensor, model::{Cx, Result}};
fn block(cx: &mut Cx, x: &Tensor) -> Result<Tensor> {
    let mut block = cx.named("block")?;
    let x = block.named("input")?.linear(32).apply(x)?.relu()?;
    block.named("output")?.linear(32).apply(&x)
}
```

`AppliedModel::prepare` creates an ordinary immutable `LoweredProgram` for a
stateless application. Once model or transform state exists, callers use
`prepare_stateful`/`compile_stateful`; final state versions become hidden roots
while parameters remain normal named ABI inputs. Both paths preserve the frozen
parameter-effect ABI and lower through the same Pliron program.

At execution, `AppliedModel::bind` accepts positional model-input buffers and
path-addressed parameter buffers, validates their schema types, then produces
the ABI-ordered slice for `Executable::execute`. This keeps checkpoints keyed
by stable semantic names rather than incidental PJRT input positions.

## Invariants

1. A parameter identity is its lexical context path plus its local name. Nested
   scopes produce stable names such as `unet.down.0.attention.q.weight` without
   threading parameter structs through every function.
2. An init trace freezes the full effect ABI: ordered inputs (shape and dtype)
   and parameter declarations (path, dtype and shape). Every apply trace
   resolves against that schema; a missing, extra or reordered effect, missing
   path, duplicate incompatible declaration or changed property fails with a
   structural-schema error. Initializer policy may become schema metadata;
   trainability remains a transformation-time parameter selection.
3. Shape inference is allowed to depend on static or specialization-known tensor
   dimensions. A parameter cannot be sized from arbitrary runtime data. Dynamic
   dimensions require an explicit static constraint or separate specialization.
4. Parameter declaration is an IR effect, not a host allocation. Init tracing
   creates schema entries; apply tracing creates parameter SSA inputs;
   `Session`/checkpoint code owns initialization, binding and mutation.
5. The parameter table is deterministic in declaration order and preserves both
   canonical checkpoint paths and intentional aliases. It is part of the
   immutable program artifact, alongside input/output ABI and state effects.
6. Transforms operate on a frozen schema and preserve parameter effects.
   `vmap`, `scan`, `grad`, rematerialization and partitioning never multiply
   declarations merely because they duplicate or replay computation. They
   choose explicit sharing/batching rules for already-known parameter inputs.
   Compilation does not turn a declaration into a hidden host-side allocation.

## Consequences for the IR

Pliron regions are the source of lexical scope. Region-capable control flow is
therefore a prerequisite: branch-local declarations, captures and resulting
schema must have defined semantics before `Cx` becomes public. The IR must also
support zero/multiple operation results and explicit terminators; StableHLO
`if`, `case` and `while` have precisely these properties.

The implementation order is:

1. Complete region, terminator and multi-result support in semantic Pliron and
   StableHLO lowering.
2. Add an internal scoped parameter-schema collector during tracing, including
   consistency checks and canonical naming.
3. Expose a small Rust-first `Cx` API and tensor shape query surface.
4. Port model examples incrementally. Do not add new primary APIs that require
   `Module::__init__`-style parameter plumbing.

This deliberately differs from PyTorch `LazyLinear`: laziness is not a special
case attached to selected layers. It also deliberately differs from TensorFlow
1 variable scopes: no global graph, `reuse=True`, `AUTO_REUSE`, or hidden
call-order state determines parameter identity. Parameter shape dependency is a
general, verified tracing primitive interpreted explicitly by init and apply.
