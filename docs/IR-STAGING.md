# Explicit IR staging

`AppliedModel::compile_staged` partitions one model's immutable semantic SSA at
named tensor taps. Model code is written once: there is no second handwritten
preparation implementation or StableHLO string rewriting.

```rust,ignore
let compiled = model.compile_staged(
    &mut compiler,
    &["schedule_modulation", "schedule_output_scale"],
)?;
assert_eq!(compiled.preparation_inputs(), [1]);
assert_eq!(compiled.execution_inputs(), [0, 2, 3]);
let bound = compiled.bind_parameters(weights.bindings())?;
let prepared = bound.prepare(&timestep)?;
let next: Buffer = bound.run((&sample, &step_size, &context), &prepared)?;
```

The preparation stage returns the selected values. Their execution-stage
producers become typed parameters; ordinary dependency pruning removes unused
inputs and producers from each stage. Cut order determines preparation output
order. Independent and dependent cuts are supported. Original model input
indices remain visible through the two input-index lists, while parameter
binding is still by name, including interleaved input/parameter declarations.
Both programs go through the normal verified lowering and executable cache.

This is explicit staging, **not automatic invariant detection**. The caller
chooses cuts and verifies which inputs preparation consumes. Qwen's example
checks that preparation depends on timestep alone—not sample, step size or
prompt. Its small modulation and output-scale vectors are cached for the
schedule; token-sized broadcasts remain in execution.

## Lifetime and restrictions

- Prepared values belong to a particular bound model and weight binding.
  Another binding cannot consume them, even if its shapes match.
- Changing preparation inputs or weights requires preparation again. There is
  no hidden global cache, content hashing or automatic invalidation.
- The prototype accepts pure straight-line programs only. State effects,
  conditionals, multi-result operations and custom calls are rejected
  conservatively. Structured loops are not supported by semantic capture.
- Empty, duplicate, out-of-range and parameter-only cuts are rejected.
- This API does not introduce donation, offload or shared mutable caches.

## Validation

IR-only tests cover ABI/dependency pruning, source immutability, cut ordering,
dependent cuts, malformed cuts and effect rejection. They require no PJRT
runtime. A CPU Qwen test checks batch 1/2, F32/BF16 projections and four chained
Euler steps against the original program, with absolute tolerance `1e-5`.
Wrong input counts and prepared values from another binding are also rejected.

```sh
cargo test -p rxla-ir --lib
PJRT_CPU_PLUGIN_PATH=/path/to/trusted/libzml_cpu.so \
  cargo test -p rxla-models prepared_schedule_matches_original_steps \
  --lib -- --ignored
```

The Qwen serving experiment is opt-in with `--prepare-schedule`; it cannot be
combined with the example's device loop or transformer offload. Preparation
cost is reported separately and included in pipeline initialization, not in
steady-state request latency. See [measurements](QWEN-IMAGE-2.1-BENCHMARK.md).
