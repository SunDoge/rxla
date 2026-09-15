# Explicit CPU/GPU placement: host-to-host latency

Measured locally on 2026-09-13 using
[`heterogeneous_bench`](crates/rxla-core/examples/heterogeneous_bench.rs), release mode,
the pinned ZML CPU/CUDA PJRT plugins already used for correctness validation.
This is a synthetic placement experiment, not a model benchmark or automatic
partitioning policy.

Host: Intel Core Ultra 9 285K (24 logical CPUs); NVIDIA GeForce RTX 5080,
driver 615.71.09. CPU and CUDA clients remain live in the same process.

## Work and measurement boundary

All modes compute `sum((x * 0.5 + 1) @ w, axis=1)` with F32 tensors and
resident weights. CPU-whole and GPU-whole compile one complete graph on that
backend. Split execution compiles CPU preprocessing, GPU matmul and CPU reduction
separately, with synchronous host-staged copies at both boundaries.

Timing includes initial input upload, execution, intermediate copies and final
host download. It excludes compilation, initial weight uploads and correctness
checking. The modes therefore start and finish at the same host boundary.
Different compiler fusion opportunities are part of the placement tradeoff;
this does not isolate memcpy cost alone.

Each of two separate processes warmed every mode ten times per shape, then ran
500 rounds in order CPU/GPU/split/split/GPU/CPU: 1,000 measured requests per
mode/shape/process. CSV printing occurs after measurements. All 18,000 measured
outputs, plus warmups, pass an independent F64 reference with absolute tolerance
1e-5. Inputs are deterministic dyadic values; this is not broad numerical testing.

## Results

Median latency in microseconds, reported separately for runs A and B:

| Input / square weight width | Whole CPU A / B | Whole GPU A / B | CPU→GPU→CPU A / B |
| --- | ---: | ---: | ---: |
| `[1,64]` / 64 | 16.602 / 16.448 | 94.501 / 111.123 | 126.801 / 139.024 |
| `[8,256]` / 256 | 26.417 / 32.829 | 90.659 / 117.516 | 134.264 / 184.628 |
| `[32,512]` / 512 | 229.161 / 168.865 | 105.009 / 112.092 | 180.790 / 241.376 |

Tail latency and inter-run variation are substantial: GPU whole-graph p95 ranges
from 264.890 to 503.346 us; split p95 from 377.338 to 722.182 us. These are
non-isolated workstation runs, not confidence intervals or service-level latency
guarantees. GPU compilation also emitted register-spill warnings; compilation is
outside timing, and this experiment does not attribute runtime costs to them.

The ranking is stable for whole CPU versus whole GPU: CPU wins the two small
cases, GPU wins the larger case. Split is slower than whole GPU in every run,
by about 1.25–2.15x in median latency. On the largest case, split versus CPU
changes ordering across runs, so there is no stable split-over-CPU conclusion.

## Consequence for the library

Keep whole-subgraph placement explicit and profile representative workloads.
Do not automatically route cheap elementwise/reduction operators to CPU merely
because they are small: host staging and lost graph fusion can outweigh savings.
Heterogeneous placement remains useful when a stage genuinely requires another
backend or data already resides there; neither condition is established by this
synthetic all-supported workload. A bounded multi-request scheduler, direct
device copies and real OCR/YOLO workloads require separate measurements.

Raw local evidence (not portable repository assets):
`/tmp/xla-heterogeneous-bench-{a,b}.csv` and corresponding `.log` files.
Each CSV has 9,000 samples plus its header. To reproduce, configure the two
trusted plugin paths and CUDA dependency environment as for the heterogeneous
gate, then run from the Rust workspace:

```sh
cargo run --release --offline -p rxla-core --example heterogeneous_bench -- 500
```
