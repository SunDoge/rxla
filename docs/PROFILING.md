# Native CUDA profiling

## Command-buffer control experiment

After identifying graph parameter updates, four unprofiled processes ran in
order default/direct/default/direct, all pinned to logical CPU 0 with the same
TinyLlama binary, FP32 model, batch 1, capacity 128 and prompt. Each process did
one replay warmup and five measured replays. Direct mode adds
`--xla_gpu_enable_command_buffer=` (empty list) to `XLA_FLAGS`; the default mode
omits this override. Disk executable caching was disabled in **all** processes
to avoid restoring a module compiled with different flags. Each process compiled
the correctness and scalar-replay executables once, outside timing.

| Mode/process | Prompt median, 40 steps | Decode median, 20 steps | Decode mean per token from median trial |
| --- | ---: | ---: | ---: |
| Default, first | 237.687 ms | 120.245 ms | 6.012 ms |
| Direct, first | 228.911 ms | 114.961 ms | 5.748 ms |
| Default, second | 239.730 ms | 120.297 ms | 6.015 ms |
| Direct, second | 229.456 ms | 114.992 ms | 5.750 ms |

For this workload, direct mode reduces measured decode time by about 4.4% in
both pairs (roughly 174 versus 166 tokens/s). This is a modest, reproducible
local result, not a universal reason to disable CUDA Graphs. Earlier unpinned
8 ms/token runs are not the control group. Compilation on the pinned core took
about 7.6 seconds for the correctness executable and is excluded from replay;
the earlier unpinned compilation timings are not equivalent either.

All four reports' generated token IDs, eight saved full-vocabulary logit vectors
and final positions exactly match the independently validated scalar-replay
reference. Every replay also checks all generated tokens and final position.
A separate Nsight process using direct mode contains **no CUDA Graph API rows**
and does contain direct kernel launches, confirming the intended execution path.
That trace includes compilation/autotuning and only 180 token steps, unlike the
earlier cached 420-step capture; their API counts/durations must not be compared
as performance deltas. [Samples, flags and trace summary](../benchmarks/rust-xla-command-buffer-ab.json).

Library defaults are unchanged. Reproduction: keep the CUDA environment below,
use `taskset -c 0` before the TinyLlama executable and `--benchmark-runs 5`,
alternate the override above, and use new report filenames. Unset
`XLA_CACHE_DIR` and `XLA_CACHE_NAMESPACE` for this comparison. If using disk
caching instead, explicitly separate namespaces by effective compiler options;
the current frontend does not fingerprint all process-level XLA flags.

## TinyLlama: first measured bottlenecks

Nsight Systems 2026.5.1 CLI successfully captured CUDA Graph **nodes** on RTX
5080 without root or CPU sampling. No library profiling dependency, global
installation or permission change was needed. The official CLI Debian package
was extracted under `/tmp/xla-nsys.6UIQbi`; its SHA256 is
`61829db6392e5c293ada1319df86a97c3356bc810335a1975cb551fcfe3eca08`.
This locally recorded hash identifies the artifact, not an independent vendor
signature. Obtain trusted tools from [NVIDIA](https://developer.nvidia.com/nsight-systems/get-started).

The capture contains the entire process: weight loading, one correctness pass,
one replay warmup and five measured replays, totaling 420 token steps. Both
executables were restored from cache (zero native compilations). This is not
a decode-only range. Captured replay latencies were visibly higher than the
unprofiled run; do not use profiler wall times as serving throughput.

| Kernel | GPU kernel time share | Calls | Mean duration | HLO interpretation |
| --- | ---: | ---: | ---: | --- |
| `input_reduce_fusion_59` | 44.8% | 9240 | 102.549 us | MLP gate/up projections, two `[5632,2048]` matrices times the activation vector |
| `input_reduce_fusion_60` | 22.7% | 9240 | 51.897 us | MLP down projection, `[2048,5632]` matrix times the activation vector |

Mapping comes from separately dumped optimized HLO for both correctness and
replay modules. `fused_reduce.59` contains broadcast/multiply/two reductions;
`fused_reduce.60` contains broadcast/multiply/reduction and a Triton backend
configuration. The original HLO has highest-precision dot operations for these
projections. Names alone do not imply these are normalization or softmax
kernels. Each kernel count is consistent with 22 layers times 420 steps.

On the host, `cuGraphExecKernelNodeSetParams_v2` accounts for 171380 calls and
621.734 ms, 45.6% of recorded CUDA **API** time. `cuGraphLaunch` accounts for
420 calls and 51.591 ms. These are not GPU kernel durations and cannot be added
to the kernel percentages as fractions of end-to-end time. Profiling overhead
also affects API durations. Persistent input/state addresses or a different
command-buffer strategy are hypotheses to investigate, not proven fixes.

Recorded GPU transfer durations were 171.225 ms H2D, 0.235 ms D2H and 0.083 ms
D2D. H2D includes initial 4.4 GB weight upload and state initialization, so it
must not be attributed entirely to decode. No bandwidth/occupancy counters were
collected; the trace does not prove whether the projection kernels are limited
by memory bandwidth, instructions or occupancy.

The immediate investigation priorities are the large FP32 matrix-vector
projections and per-call CUDA Graph node updates, not full-logit downloads or
FlashAttention for this short-context workload. Longer contexts can differ.
[Recorded summary](../benchmarks/rust-xla-tinyllama-nsys.json).

## Reproduction

Build the example first with `fast-load.toml`, then retain the trusted CUDA
plugin/library environment from [CUDA validation](CUDA-validation.md). Set
`NSYS_BIN` to the extracted CLI and `PROFILE_PREFIX` to a new absolute output
prefix. Run from the repository root; the model checkpoint must already exist.

```sh
"$NSYS_BIN" profile --trace=cuda --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --force-overwrite=false --output="$PROFILE_PREFIX" \
  target/debug/examples/tinyllama \
  target/tinyllama-chat \
  "$PROFILE_PREFIX-model.json" --benchmark-runs 5
"$NSYS_BIN" stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  --format json --output="$PROFILE_PREFIX-summary" "$PROFILE_PREFIX.nsys-rep"
```

The original capture is `/tmp/xla-nsys.6UIQbi/tinyllama.nsys-rep`, with its
SQLite export next to it. Nsight CLI options are documented in the
[official user guide](https://docs.nvidia.com/nsight-systems/UserGuide/).

For HLO mapping, run a **separate** process with executable disk caching disabled
(`unset XLA_CACHE_DIR XLA_CACHE_NAMESPACE`) and add
`--xla_dump_hlo_as_text --xla_dump_to=NEW_DIRECTORY` to the existing `XLA_FLAGS`.
Preserve the CUDA toolkit flag. This intentionally recompiles to obtain dumps;
restored executables do not necessarily emit them. Original dump directory:
`/tmp/xla-hlo-profile.7RDwaQ`, modules 0000 (correctness) and 0232 (replay).
See [OpenXLA HLO dumping](https://openxla.org/xla/hlo_dumps). Native timing and
HLO dump files may contain model structure or paths; review before sharing.
