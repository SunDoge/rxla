# Whole graph versus split execution (2026-09-13)

`crates/rxla-core/examples/pipeline_bench.rs` compares `Y = (X W) W`, with F32
X `[8,64]` and W `[64,64]`, within one client/device per process:

- `fused`: one compiled graph containing both matrix multiplications. The name
  describes the graph boundary, not a claim that XLA emits one fused kernel.
- `split_sync`: execute the same single-matmul executable twice, waiting after
  each execution, with the intermediate staying on the device.
- `split_submit`: submit twice, passing the first pending output to the second;
  wait on the second and then the first to observe both statuses.

Every path downloads the final 512-element result. Timing includes Rust/native
dispatch, completion, buffer cleanup inside the call, and final host download;
it excludes compilation, initial uploads, checking, CSV formatting and plugin
startup. Inputs and weights are resident runtime parameters, not embedded graph
constants. All three paths are warmed ten times. Each of 1000 rounds runs the
balanced order `fused, sync, submit, submit, sync, fused`: 2000 samples per mode.
The fixed dyadic inputs are checked against an independent two-stage F64 host
reference after every call (absolute tolerance 1e-5, nonfinite outputs rejected).
There are no transfers of intermediate results to host in either split mode.

## Observations

Two sequential fresh processes per backend; the CPU and CUDA experiments were
not run concurrently. Optimized Rust release build (`debug_assertions=false`),
Intel Core Ultra 9 285K CPU and RTX 5080 CUDA, pinned ZML PJRT build
`202609101243.20.1.7ca6884ea2cb`. CUDA dependencies are recorded in
[CUDA validation](CUDA-validation.md). No core affinity or fixed clock controls
were applied. These are within-device comparisons, not a CPU-versus-GPU ranking.

| Backend / process | Whole graph median / p95 µs | Split sync median / p95 µs | Split submit median / p95 µs |
| --- | ---: | ---: | ---: |
| CPU A | 15.578 / 30.200 | 20.936 / 38.094 | 17.838 / 34.939 |
| CPU B | 16.442 / 25.829 | 21.791 / 33.883 | 17.891 / 32.094 |
| CUDA A | 79.294 / 231.961 | 104.333 / 260.835 | 92.762 / 244.971 |
| CUDA B | 85.495 / 236.876 | 113.338 / 250.986 | 101.538 / 250.814 |

All 24,000 measured outputs passed, plus warmups. CSVs (6000 rows each) and
stderr metadata/summaries remain at `/tmp/xla-pipeline-{cpu,cuda}-{a,b}.{csv,log}`.
These task-local files are not packaged repository fixtures.

On this small workload, split submission reduced median latency relative to
split synchronous execution, but the whole graph remained fastest in both runs
on each backend. Large CUDA p95 tails remain. These observations do not identify
the source of latency, prove kernel overlap, or predict full OCR/LLM throughput.
Use submission to connect independently required stages; prefer whole-graph
compilation when possible, preserving XLA's opportunity for global optimization.
Larger shapes, model workloads and event-driven scheduling remain unmeasured here.

## Reproduce

From the standalone repository root, provision a trusted plugin and its native
dependencies, then run each backend separately:

```sh
cargo build --offline --release -p rxla-core --example pipeline_bench
# Set PJRT_PLUGIN_PATH (and CUDA native environment where applicable).
# Use unused report paths; shell noclobber prevents accidental replacement.
set -C
target/release/examples/pipeline_bench 1000 > /tmp/new-pipeline.csv 2> /tmp/new-pipeline.log
```

The ordinary CPU gate runs the reference/validation unit test, not the timed
benchmark. `cargo clippy -p rxla-core --example pipeline_bench -- -D warnings`
also passed for this change. No performance threshold is used as a CI gate.
