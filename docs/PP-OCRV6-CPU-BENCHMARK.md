# PP-OCRv6 CPU work

Status: host-side foundation plus a complete detector import and network-only
benchmark, not yet an end-to-end accuracy claim.

RXLA keeps the PP-OCRv6 host pipeline explicit. The detector and recognizer
networks will execute through XLA CPU, while image resize/normalization, crop
batching, DB box extraction and CTC text decoding remain ordinary Rust code.
This avoids compiling data-dependent contour processing into the tensor graph
and gives each stage an independent latency budget.

Implemented so far:

- detector page resize to a multiple of 32 and fused RGB-to-planar-F32
  normalization;
- parallel crop resize, normalization, zero padding and contiguous
  `[N, 3, 48, 320]` packing for the medium recognizer;
- parallel greedy CTC decode with configurable blank token;
- a standalone benchmark with no PJRT or model-download requirement.

Run it with:

```console
cargo run --release -p rxla-models \
  --example pp_ocr_v6_cpu_bench -- --runs 100 --crops 32
```

## 2026-09-15 local result

Machine: Intel Core Ultra 9 285K, 24 logical CPUs, release profile. The fixture
is a synthetic 1920×1080 RGB page, 32 variable-width text crops, and a
`[32, 40, 18000]` probability tensor. Times include allocation of each stage's
result. They exclude image decoding, DB contour extraction and all neural
network execution.

| Host stage | Mean | p95 |
| --- | ---: | ---: |
| Detector resize + normalization | 17.499 ms | 20.007 ms |
| 32-crop recognition preparation | 2.421 ms | 3.687 ms |
| 32-sequence greedy CTC decode | 3.138 ms | 4.637 ms |

The first serial CTC implementation measured 20.270 ms mean on the same
fixture. Parallelizing independent sequences reduced that measured mean by
6.46×. An attempted channel-parallel normalization made detector preparation
slower (18.021 ms versus the earlier serial layout conversion), so it was not
kept.

These numbers are a regression baseline, not a comparison with PaddleOCR or
OpenVINO. The official detector is now imported below; the recognizer and a
complete decode-to-text comparison remain future work.

## ONNX Runtime Medium baseline

The sibling `/home/me/code/python/ppocrv5-onnx` checkout now contains a
stage-level runner at `benchmarks/ppocrv6_cpu.py`. It uses that project's uv
environment and automatically downloaded PP-OCRv6 Medium detector and
recognizer ONNX files. Run the recorded configuration with:

```console
uv run --group benchmark benchmarks/ppocrv6_cpu.py img/demo.png \
  --threads 12 --batch 1 --warmup 1 --iterations 3
```

The input is the repository's real 5100×3300 demo page. It produces 128 text
boxes. Crops are sorted by width/height ratio before batching so a single wide
crop does not pad unrelated recognition inputs. Software is ONNX Runtime
1.22.1 CPUExecutionProvider and OpenCV 4.11.0.86.

| ORT stage | Mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| Detector preprocessing | 2.278 ms | 2.269 ms | 2.393 ms |
| Detector inference | 208.057 ms | 207.216 ms | 213.780 ms |
| DB postprocess + crop | 53.406 ms | 53.534 ms | 53.607 ms |
| Recognizer preprocessing | 26.036 ms | 26.316 ms | 27.302 ms |
| Recognizer inference | 3846.582 ms | 3903.221 ms | 3908.554 ms |
| CTC decode | 112.297 ms | 112.117 ms | 119.633 ms |
| Total | 4248.656 ms | 4304.217 ms | 4314.443 ms |

The output score checksum was `125.247022748` in all recorded trials. Session
creation, model download, image decoding and the warmup trial are excluded.
The measurements show that recognition inference is 90.5% of end-to-end time;
optimizing Rust image conversion alone cannot close the gap. RXLA's meaningful
CPU comparison must therefore prioritize compiled PPLCNetV4/LightSVTR/CTC
kernels, fixed-width buckets and thread affinity. Host work remains separately
measured so it cannot be hidden inside an inference number.

### Pure ONNX graph baseline

`benchmarks/onnx_cpu.py` in the sibling ONNX repository removes preprocessing,
postprocessing and image decoding entirely. It passes a preallocated zero F32
tensor to `InferenceSession.run`; timing includes synchronous graph execution
and output allocation. Ten measured trials follow two warmups.

| Threads | Detector `[1,3,640,960]` | Recognizer `[1,3,48,320]` |
| ---: | ---: | ---: |
| 1 | 995.895 ms | 85.769 ms |
| 4 | 330.798 ms | 24.370 ms |
| 8 | 249.748 ms | 19.675 ms |
| 12 | 209.478 ms | 17.491 ms |
| 16 | **178.309 ms** | **11.876 ms** |
| 24 | 303.609 ms | 26.063 ms |

ORT 1.22.1 peaks at 16 threads on the hybrid-core Core Ultra 9 285K for both
fixed shapes. Using all 24 logical CPUs is substantially slower, so RXLA must
compare at 16 threads and report thread affinity rather than comparing against
an untuned ORT default. The detector output is 2.344 MiB; recognizer output is
2.855 MiB. Checksums were identical across thread counts.

This is now the network-only target: RXLA needs the same model, shapes, F32
inputs and output materialization before any performance advantage can be
claimed. The current RXLA PP-OCRv6 code does not yet import these complete ONNX
graphs.

### RXLA detector comparison

A temporary typed exporter in the sibling ONNX repository mapped the official
276-node Medium detector to the restricted RXLA graph runner. This is a bridge
used to measure the backend, not the proposed general `rxla-onnx` crate. The
input and output are the same F32 tensors as the ORT detector benchmark:
`[1,3,640,960]` and `[1,1,640,960]`.

| Backend | Network-only mean | Relative |
| --- | ---: | ---: |
| ONNX Runtime 1.22.1, 16 threads | **178.309 ms** | 1.00× |
| RXLA ONNX → IR → StableHLO, resident buffers | 450.418 ms | 2.53× slower |
| Legacy typed bridge, resident buffers | 477.759 ms | 2.68× slower |
| RXLA CPU PJRT, upload + execute + download | 466.381 ms | 2.62× slower |

RXLA compilation took 244 ms. Ten resident and ten roundtrip trials followed
one warmup per mode. The unexpectedly lower roundtrip mean is ordinary run
variance and must not be interpreted as negative transfer cost. Explicitly
setting `XLA_FLAGS='--xla_cpu_multi_thread_eigen=true
intra_op_parallelism_threads=16'` did not improve the result: five resident
trials averaged 486.687 ms.

The maximum absolute difference from ORT was `2.4400651e-6`. Only 2 of 614,400
values exceeded the legacy runner's `2e-7 + 1e-4*abs(reference)` threshold, so
the runner reported failure even though the graph mapping is close enough for
performance diagnosis. This is not yet an accepted correctness result.

The first-class `rxla-onnx` path now imports every operator in the 276-node
detector into `ProgramIr`, verifies the generated StableHLO, and executes it
through PJRT. It deduplicates ONNX tensor identities into 226 ABI parameters:
222 initializer-backed values and four graph constants, with the image input
separate. Three independent groups of 20 measured executions produced 449.883,
453.696, and 447.675 ms means, for an equally weighted mean of **450.418 ms**.
Compilation took 190.523, 197.240, and 189.121 ms. Output materialization is
outside the execution timer. The maximum absolute error against the saved ORT
reference was `2.4400651e-6`.

This removes the temporary Python exporter from the architecture and improves
resident execution by 5.7% over the legacy bridge, but RXLA still has no
detector performance advantage: it is 2.53× slower than tuned ORT. The next
optimization is layout propagation that keeps internal activations in NHWC and
only transposes at graph boundaries. Inference Conv+BatchNorm is also a natural
ONNX/IR rewrite: fold scale and offset into constant convolution weights and
bias before StableHLO lowering. This detector has no standalone BatchNorm
nodes because its exporter already performed that fold; the recognizer has
three and will exercise the generic pass.

The first layout-propagation implementation subsequently moved NCHW↔NHWC
conversion from every convolution and pool to graph boundaries, while mapping
concat/reduction axes and broadcast constants into the physical layout. Three
groups of 20 executions measured 440.486, 440.593, and 446.308 ms, averaging
**442.462 ms**. This is another 1.8% reduction from the first-class importer,
7.4% from the legacy bridge, and remains 2.48× slower than ORT. Maximum absolute
error was `2.6524067e-6`. The modest improvement confirms that XLA had already
canonicalized many adjacent transpose pairs; layout propagation is still the
right IR invariant, but convolution tiling and dispatch overhead remain the
dominant targets.

### JAX CPU versus ZML CPU plugin

To distinguish an RXLA/ZML packaging problem from XLA CPU behavior, the RXLA
benchmark can export its exact StableHLO module and ABI arguments:

```console
PJRT_PLUGIN_PATH=/path/to/libzml_cpu.so cargo run --release -p rxla-onnx \
  --example ppocr_detector -- MODEL.onnx BUNDLE \
  --runs 1 --export-dir /tmp/rxla-ppocr-jax
uv run python benchmarks/ppocr_stablehlo_jax.py \
  /tmp/rxla-ppocr-jax --warmup 2 --iterations 20
```

The Python runner bypasses JAX tracing and passes that textual StableHLO
directly to the CPU client's `compile_and_load`. It uploads the same image and
226 parameters in the same ABI order and blocks on every execution. JAX 0.11.1
and jaxlib 0.11.1 produced three 20-run means of 439.008, 445.522, and 443.455
ms: an aggregate **442.662 ms**. RXLA/ZML's corresponding aggregate was
442.462 ms. The 0.05% difference is ordinary measurement noise, and both paths
reported the identical `2.65240669e-6` maximum error.

This A/B test rules out the ZML PJRT shared object as the source of the 2.48×
gap to ONNX Runtime. Binary inspection also shows that both the JAX wheel and
ZML plugin contain the current YNN/Slinky and oneDNN CPU paths. For this graph,
the performance limitation is shared XLA CPU lowering/runtime behavior rather
than RXLA's PJRT wrapper or a stripped-down ZML build.

### CPU profile investigation

XLA's optimized-HLO and buffer-assignment dumps were captured with
`--xla_dump_hlo_as_text`, `--xla_dump_hlo_as_long_text`, and
`--xla_dump_buffer_assignment_analysis`. The optimized entry computation has:

- 127 YNN custom fusions;
- 155 `transpose_copy_fusion` definitions;
- 225 loop fusions;
- a 208.40 MiB preallocated temporary buffer and 276.85 MiB total default-space
  buffer assignment.

This proves that XLA selected its YNN convolution path, but also retained
substantial layout work. It does not by itself assign wall time to individual
fusions. An offline OIHW→HWIO weight-packing experiment removed the explicit
frontend weight transpose pattern but made resident execution substantially
slower, from 477.759 ms to 719.951 ms. The experiment was reverted. The result
indicates that YNN's matcher/layout selection benefits from the original
pattern; counting transpose instructions alone is not a runtime profile.

`--xla_hlo_profile=true` produced no per-HLO report through this CPU PJRT
execution path. Direct inspection of the loaded plugin's extension chain found
types 24, 19, 9, 5, 6, and 4, but no PJRT profiler extension (type 1). This
means that XLA cannot currently give RXLA a PJRT-native per-HLO profile with
this plugin.

After setting `kernel.perf_event_paranoid=1`, Nsight Systems 2026.3.2 captured
315,668 CPU samples with LBR call stacks over 21.538 seconds. The workload
contains compilation, one correctness execution, one warmup and 20 measured
executions for each benchmark mode. The 42 repeated benchmark executions
dominate the capture; their profiled means were 488.398 ms resident and
482.253 ms roundtrip. Leaf samples were grouped by symbol as follows. These
percentages are CPU sample shares across worker threads, not fractions of wall
latency, and the category names are a post-processing classification rather
than XLA annotations.

| Leaf-symbol category | Samples | Share |
| --- | ---: | ---: |
| YNN dot microkernels | 116,687 | 36.97% |
| Slinky interpreter/administration | 67,325 | 21.33% |
| YNN dot scheduling/runtime | 48,951 | 15.51% |
| libc or unresolved symbols | 44,478 | 14.09% |
| generated anonymous JIT code | 22,365 | 7.08% |
| Slinky copy paths | 13,751 | 4.36% |
| Other | 2,111 | 0.67% |

The largest individual leaf was YNN's
`dot_fp32_1x8x1_1x4x1_sse2` kernel at 66,400 samples (21.03%). It exceeded the
main `dot_fp32_6x16x1_1x8x1_fma3` kernel at 50,283 samples (15.93%). The narrow
SSE2 tail kernel being more expensive than the wider FMA3 kernel is stronger
evidence for inefficient convolution tiling, channel tails or layout-induced
shapes than the HLO transpose count alone. Slinky evaluation and YNN scheduling
together account for another 36.84% of leaf samples, so dispatch granularity is
the second concrete target. Explicit Slinky copy paths are only 4.36%; removing
copies in isolation cannot plausibly close the 2.68× gap to ORT.

The profile used all 24 available CPUs, with the main 24 worker threads each
collecting roughly 12.5k–13.5k samples. Restricting process affinity was a
regression: five resident trials averaged 569.692 ms on CPUs 0–15 and 885.942
ms on CPUs 0–7. ORT's optimum at 16 intra-op threads therefore does not imply
that YNN should use the same setting. Until PJRT or the CPU plugin exposes a
real execution-thread-pool option, RXLA should not emulate one with affinity.

The capture can be reproduced and inspected without changing RXLA:

```console
sudo sysctl kernel.perf_event_paranoid=1
PJRT_PLUGIN_PATH=/path/to/libzml_cpu.so nsys profile \
  --force-overwrite=true --output=/tmp/rxla-ocr-cpu-sampled \
  --trace=osrt --sample=process-tree --backtrace=lbr \
  --cpuctxsw=process-tree \
  target/release/examples/ocr_graph /path/to/rxla-bundle report.json \
  --benchmark-runs 20
nsys export --type sqlite --output=/tmp/rxla-ocr-cpu-sampled.sqlite \
  /tmp/rxla-ocr-cpu-sampled.nsys-rep
sqlite3 /tmp/rxla-ocr-cpu-sampled.sqlite '.tables'
```

The next useful optimization is therefore IR/lowering work that gives YNN
better-packed convolution dimensions and fewer narrow tail tiles, followed by
coarser fusion/dispatch. A first-class profiler extension in the plugin would
make it possible to map those samples back to individual StableHLO operations;
RXLA cannot add that visibility solely in its Rust wrapper.
