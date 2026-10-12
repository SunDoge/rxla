# Explicit layer paths without a visible context

`rxla_nn::model` (also `rxla::model`) is an optional spelling of ordinary
effect-based model construction, not a different execution mode.

```rust
use rxla_core::Tensor;
use rxla_nn::{model, Linear, Model, ModelInput, Result, TensorApply};

#[model]
fn classifier(x: Tensor) -> Result<Tensor> {
    #[layer("head")]
    let head = Linear::new(3);
    x.apply(&head)
}

let definition = Model::new(classifier).inputs(ModelInput::new([2, 4]));
let applied = definition.trace()?;
// head.weight: [3, 4], head.bias: [3]
```

`#[model]` injects a hidden first argument of type `Cx` into the expanded
function. `Model::new` supplies it normally. `#[layer]` is a helper consumed
by that outer macro, not an independent procedural attribute on a statement.
The expansion uses `cx.clone().at("head")?.layer(Linear::new(3))`.
There is no global/TLS context, implicit naming or change to compilation.

Mixed path segments work in ordinary loops and blocks:

```rust
for index in 0usize..4 {
    #[layer("blocks", index, "conv")]
    let conv = Conv2d::new(channels, [3, 3]).options(
        Conv2dOptions { padding: [[1, 1], [1, 1]], ..Default::default() },
    );
    x = x.apply(&conv)?;
}
```

Paths are relative to the function's injected context. Each segment uses
`ModelPathSegment`: names and `usize` indices may be mixed. Literals must be
nonempty single names without `.` or `/`; dynamic names are checked at trace
time. Binding expressions and path expressions are each evaluated once.
Equal paths reuse compatible parameters; distinct paths declare independent
parameters, irrespective of Rust variable names. Parameters are declared only
when a layer is applied, with input-derived dimensions. An unused bound layer
does not add unused weights to the schema.

## Deliberate limits

- Synchronous non-const free functions only. Return a compatible `Result`
  because generated path validation propagates errors with `?`.
- Layer declarations use an untyped single-variable `let`. Destructuring,
  explicit layer type annotations, let-else and repeated attributes are rejected.
- Layer helpers inside closures or nested items are rejected rather than
  implicitly capturing a context with surprising ownership/lifetime behavior.
- Ordinary function calls are not rewritten. The expanded model function
  still takes `Cx` first; it cannot be called as a context-free eager function.
  Nested-model path binding is not yet provided by this syntax. Existing
  explicit `Cx` functions and `path!` remain available for composition.
- Literal checks do not verify dynamic shapes or a whole parameter tree at
  Rust compile time. Existing model/IR verification remains responsible.

The proc-macro implementation is a separate `rxla-nn-macros` crate re-exported
by `rxla-nn`. Dependency lookup supports renamed `rxla-nn` and facade `rxla`
dependencies; generated code does not require importing `Cx` in model source.
Host-only tracing tests exercise paths, sharing, unused bindings and hygiene
without creating PJRT clients or compiling GPU programs.

Run the convolution → normalization → pooled linear example without a plugin:

```sh
cargo run -p rxla --example model_layers
```

It prints the parameter tree and verifies/lowers its IR, without loading weights
or running inference. Compiler UI tests cover empty paths, invalid literal
paths and accidental closure capture, with source-local diagnostics.
