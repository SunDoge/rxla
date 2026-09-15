# Cumulative log-sum-exp: local synthetic measurements

Measured 2026-09-13 after `2e6769cf6e`, using the new
[benchmark example](crates/rxla-core/examples/logcumsumexp_bench.rs). This is not a
full-model benchmark or a CPU-versus-GPU recommendation.

## Method

Release Rust build, PJRT 0.115, CPU selected device 0 (the plugin exposed four
logical CPU devices), and NVIDIA RTX 5080 CUDA selected device 0. CUDA dependencies
and flags were those in [CUDA validation](CUDA-validation.md), including
`--xla_gpu_cuda_data_dir=/usr/local/cuda-13.3`. CPU and CUDA benchmark processes
ran separately. No device-clock pinning or system-wide load isolation was used.

Inputs have shape `[4, N]`, N = 64, 256, 1024, 4096. Modes rotate order each
round. Three warmup executions per mode are excluded, followed by 30 samples in
the first process and 100 samples in a fresh repeat process. Resident inputs
exclude upload from timing. Execution timing includes synchronous execution,
output allocation and host download; it is not kernel-only timing. Validation
against an independent F64 sequential log-domain prefix reference is outside
timing and runs for every result.

The stable scan is measured on both bounded values in `[-4, 4]` and a ramp from
`-1000` to `1000`. The direct `exp -> cumsum -> log` comparator is tested **only
on bounded values**; it is not a valid general replacement on wide inputs.
Each shape compiles one scan executable, reused for both input distributions,
and one direct executable. Compile timings exclude Rust compilation, graph
construction and input upload but include the graph compile call and PJRT work.

## Observations

For N = 4096, execution milliseconds:

| Backend / samples | Stable bounded median / p90 | Direct bounded median / p90 | Stable wide median / p90 |
| --- | --- | --- | --- |
| CPU / 30 | 1.027 / 1.141 | 0.275 / 0.491 | 1.000 / 1.167 |
| CPU / 100 | 0.858 / 0.949 | 0.108 / 0.154 | 0.863 / 0.907 |
| CUDA / 30 | 0.289 / 0.677 | 0.224 / 0.503 | 0.229 / 0.766 |
| CUDA / 100 | 0.106 / 0.356 | 0.091 / 0.454 | 0.106 / 0.417 |

Within-backend stable/direct median ratios were approximately 3.7–7.9 on CPU
and 1.16–1.29 on CUDA. The large changes between runs and high CUDA p90s rule out
a precise throughput claim. Small workloads include substantial host/runtime
overhead; do not infer kernel speed from these numbers.

At N = 4096 the scan compile calls took 68.5/68.9 ms on CPU and 129.2/123.1 ms
on CUDA; the direct comparator took 24.6/32.5 and 67.6/56.9 ms respectively.
These are two observations, not a compile-time distribution.

Every tested shape and mode passed the absolute-error threshold 0.002. Across
the two runs, the largest observed scan error was about 1.42e-6 on bounded data
and 5.54e-5 on wide data; direct bounded error was below 6.34e-7. This measures
forward outputs only, not long-sequence gradient precision.

Conclusion: the doubling scan is a numerically useful baseline, with significant
CPU cost at long prefixes. A native associative scan or backend-specific fusion
is worth investigating before recommending it for hot paths. A compact Rust
graph does not establish a single kernel or optimal total work.

## Work-efficient tree follow-up

An additional same-process comparison (100 samples per mode, otherwise the same
method) keeps both public algorithms: `logcumsumexp` is the original doubling
scan, and `logcumsumexp_tree` is the opt-in tree scan. The latter recursively
scans adjacent pairs and reconstructs interleaved prefixes, for O(N) total
elementwise work instead of O(N log N). At N = 4096 on bounded inputs:

| Backend | Tree compile / doubling compile (ms) | Tree median / doubling median (ms) | Tree p90 / doubling p90 (ms) |
| --- | --- | --- | --- |
| CPU | 169.184 / 66.996 | 0.389 / 0.848 | 0.489 / 0.937 |
| CUDA | 525.726 / 122.317 | 0.166 / 0.145 | 0.329 / 0.351 |

CPU execution improved about 2.2x, but compilation cost increased about 2.5x.
CUDA compilation increased about 4.3x and execution did not improve. Therefore
the doubling implementation remains the default and tree scanning is opt-in.
These are single additional process observations, not a universal crossover
policy. The tree's maximum observed bounded error at N = 4096 was 2.46e-6 and
wide-input error 5.54e-5; all shapes passed the same 0.002 forward tolerance.

The current benchmark prints five modes (`tree_bounded`, `direct_bounded`,
`tree_wide`, `doubling_bounded`, `doubling_wide`) and compiles three executables
per shape. Earlier three-mode observations above describe the previous benchmark
revision and must not be confused with the tree timings.

## Reproduction

From the Rust workspace, with a trusted plugin and its dependencies configured:

```sh
PJRT_PLUGIN_PATH=/trusted/plugin.so cargo run --release --offline \
  -p rxla-core --example logcumsumexp_bench -- 100
```

Stdout is CSV with every tested shape, mode, compile time, median, p90 and maximum
error. Device metadata and build mode go to stderr. Repeat in a fresh process;
do not run beside another GPU benchmark or gate. Performance execution remains
opt-in; the regular CPU gate only runs the small host-reference test.
