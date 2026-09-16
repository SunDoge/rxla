# Stable Diffusion benchmark

This document records reproducible Stable Diffusion measurements separately
from the benchmark scripts and checkpoint fixtures. Measurements are local
observations, not general backend performance claims.

## Functional IR pipeline smoke

On September 15, 2026 the complete tiny Stable Diffusion example was migrated
from parameter-owning modules to the effect-based `init`/`apply` API. CLIP, UNet
and VAE parameters are declared at use sites, loaded by `ModelSchema` path, and
compiled from Pliron-derived StableHLO. PNDM history remains in device buffers
between executions. No legacy model module or stateful parameter session is
constructed.

The checkpoint is `bumblebee-testing/tiny-stable-diffusion`. The run used two
denoising steps, guidance `7.5`, no warmup, one measured iteration, prompt
`a red cube`, and the runner's deterministic seed-zero latent. The image is
`128×128`.

| PJRT backend | compile | total | CLIP | denoise | VAE | download |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ZML CPU | 778.571 ms | 769.208 ms | 45.406 ms | 625.548 ms | 98.233 ms | 0.145 ms |
| JAX CUDA 13 plugin, RTX 5080 | 15.150 s | 42.304 ms | 3.493 ms | 31.483 ms | 7.317 ms | 1.104 ms |

The CUDA run loaded the PJRT plugin installed by the project environment:

```text
.venv/lib/python3.14/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so
```

The plugin's wheel-provided RPATH selected the same environment's CUDA 13.0 and
cuDNN 9.24 libraries, so no manually assembled `LD_LIBRARY_PATH` was required.
The selected device was an NVIDIA GeForce RTX 5080 with compute capability
12.0a. CPU and CUDA checksums were `24979.269657820` and `24979.552041411`.

These un-warmed one-iteration measurements are an execution and integration
gate after the API migration. They are not a CPU/GPU comparison and do not
replace a warmed 20-step benchmark. In particular, the reported compilation
time includes three independently compiled stages and cold CUDA code generation.

## Functional RXLA versus PyTorch

A same-input comparison was run on the RTX 5080 with 20 denoising steps,
guidance `7.5`, two warmups and ten measured iterations. RXLA generated and
saved the initial latent; the PyTorch oracle consumed that exact file. Both
runners synchronize CUDA at each measured stage boundary.

| Runner | mean | p50 | p95 | CLIP mean | denoise mean | VAE mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Functional RXLA/PJRT/XLA | 86.777 ms | 87.139 ms | 89.293 ms | 0.753 ms | 84.026 ms | 1.544 ms |
| PyTorch/cuDNN | 91.270 ms | 91.290 ms | 91.950 ms | 0.983 ms | 89.004 ms | 1.246 ms |

On this snapshot RXLA is `4.9%` faster end to end. Its CLIP and denoising stages
are faster, while PyTorch's VAE stage is faster. RXLA compilation took
`15.573 s` and is excluded from steady execution; model loading is excluded for
both runners. The final image maximum and mean absolute differences were
`0.0715807` and `0.00870391`. This 20-step error includes repeated numerical
drift through the scheduler; isolated component comparisons remain the stronger
diagnostic for operator correctness.

The RXLA result uses reusable `BoundParameters`: all 304 UNet parameter paths,
shapes and dtypes are validated and ordered once, outside the denoising loop.
Each step then validates only its changing inputs and assembles the executable
ABI linearly. Before this change, the same model redundantly rebuilt a 304-entry
path map on every step and measured `93.053 ms` mean in the preceding five-run
snapshot. Removing that host work preserved the output checksum exactly and
reduced observed end-to-end time by `6.7%`.

## CUDA profile

Nsight Systems 2026.5.1 captured a six-second steady-state window after native
compilation on the same RTX 5080. Profiling overhead makes the captured wall
time unsuitable as a throughput measurement; the kernel shares and call shapes
are used only to identify work inside the already-benchmarked executable.

| GPU kernel | kernel time | calls | mean | interpretation |
| --- | ---: | ---: | ---: | --- |
| `triton_softmax_105` | 27.3% | 4,852 | 157.941 us | UNet spatial self-attention, `[2, 8, 1024, 1024]` probabilities |
| `gemm_fusion_dot_10` | 9.4% | 4,852 | 54.658 us | self-attention `Q K^T`, `[16, 1024, 8] × [16, 8, 1024]` |
| cuDNN TF32 NHWC convolution | 8.2% | 20,594 | 11.261 us | convolution |
| `gemm_fusion_dot_11` | 7.0% | 4,851 | 40.810 us | self-attention probabilities times `V` |
| cuDNN `convertTensor_kernel` | 5.3% | 43,108 | 3.477 us | convolution layout/type conversion |

The mapping is confirmed by a separately dumped optimized HLO module. In
particular, the dominant softmax is not a generic normalization: it sits
between the two matrix products of the 1,024-token, eight-head UNet
self-attention. The current StableHLO is the ordinary materializing
`softmax(Q K^T) V` decomposition, so a backend-recognizable fused or
memory-efficient attention operation is now the highest-value optimization
target. It should be represented as an attention semantic op in RXLA rather
than hidden behind a Stable-Diffusion-specific layer. Lowering can retain the
portable decomposition when a target has no fused implementation.

The CUDA custom-call route cannot accelerate this exact graph yet. The cuDNN
fused-attention lowering used by JAX accepts F16/BF16 inputs, whereas this
checkpoint currently executes attention in F32. Consequently the required
order is: preserve attention as a semantic IR operation, add explicit mixed
precision at the model boundary, then select the cuDNN custom call only when
its dtype, head-dimension and device constraints are satisfied. Adding the
custom call to today's F32-only model would produce an API that cannot run.

An isolated JAX 0.11.1 experiment verified that this is worth pursuing on the
same CUDA stack. It used the profiled self-attention shape, expressed in JAX's
BTNH convention as `[2, 1024, 8, 8]`, ten warmups and 100 synchronized calls:

| dtype / implementation | mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| F16 ordinary XLA decomposition | 0.247 ms | 0.212 ms | 0.434 ms |
| F16 cuDNN fused attention | 0.158 ms | 0.110 ms | 0.361 ms |
| BF16 ordinary XLA decomposition | 0.274 ms | 0.224 ms | 0.433 ms |
| BF16 cuDNN fused attention | 0.220 ms | 0.149 ms | 0.451 ms |

F16 fused attention reduced the isolated median by 48.0%; BF16 reduced it by
33.4%. F16 fused-versus-decomposed output differences had maximum `4.88e-4`
and mean `1.15e-5`; BF16 had maximum `1.95e-3` and mean `9.10e-5`. These are
isolated kernel-path results, not projected pipeline speedups.

The experiment also identified a compiler boundary that must be fixed before
adding the custom call. RXLA currently serializes portable StableHLO while
constructing `Program`, before `Runtime::compile` supplies a PJRT target. A
cuDNN lowering selected there would make the same program unusable on CPU.
Target-dependent lowering therefore belongs in compilation, while
`rxla.attention` remains target-independent. This avoids exposing a
backend-named attention method in the tensor API and lets CPU, CUDA and future
backends choose different implementations from the same IR.

Transfers are not the bottleneck in this window: GPU memory-operation time was
1.088 ms of memset and 0.155 ms of host-to-device copies. The cuDNN conversion
share makes convolution layout propagation a secondary target, but changing
the public tensor layout without an isolated convolution experiment would be
premature.

CUDA Graph parameter updates were also visible on the host:
`cuGraphExecKernelNodeSetParams_v2` made 179,523 calls and occupied 914.029 ms
of profiled CUDA API time. This is profiler-inflated host API time, not GPU
kernel time. An alternating unprofiled control test therefore compared the
default with `--xla_gpu_enable_command_buffer=`:

| Mode/process | mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| Default, first | 86.777 ms | 87.139 ms | 89.293 ms |
| Direct, first | 86.197 ms | 86.258 ms | 90.575 ms |
| Default, second | 88.274 ms | 88.915 ms | 90.718 ms |
| Direct, second | 86.769 ms | 87.364 ms | 90.523 ms |

The two-process averages are 87.526 ms default and 86.483 ms direct, a 1.2%
local difference with no p95 improvement. That is not strong enough to change
the library default. Persistent bindings may eventually make graph updates
cheaper, but attention fusion has much clearer GPU evidence and a larger upper
bound.

## F16 cuDNN attention result

RXLA now preserves attention until target-aware compilation. The Stable
Diffusion example explicitly enables `Compiler::with_f16_attention(true)`.
On CUDA, only unmasked rank-four self-attention with equal query/key lengths of
at least 512 and a supported head dimension is converted to F16 at the op
boundary and lowered to `__cudnn$fmhaSoftmax`; short and cross attention remain
F32 portable StableHLO. The compiler default is still strict F32, and CPU
always uses the portable lowering.

Two independent 20-step runs with two warmups and ten measured iterations were:

| Run | mean | p50 | p95 | denoise mean |
| --- | ---: | ---: | ---: | ---: |
| F16 FMHA, first | 58.400 ms | 58.999 ms | 61.285 ms | 55.591 ms |
| F16 FMHA, second | 64.261 ms | 64.207 ms | 70.911 ms | 61.174 ms |

Against the earlier 86.777 ms F32 snapshot, the observed end-to-end reduction
is 25.9–32.7%. A new PyTorch comparison measured 93.456 ms mean; the RXLA
result's maximum and mean absolute differences were `0.0713725` and
`0.00870616`, essentially unchanged from the F32 comparison's `0.0715807` and
`0.00870391`. The output checksum can vary slightly between independent CUDA
compilations/autotuning runs, so image error against the oracle is the
correctness measure rather than bit identity with the F32 pipeline.

## Reproduction

CPU:

```sh
cargo run -p rxla-safetensors --example stable_diffusion --features model,disk-cache -- \
  "$PJRT_CPU_PLUGIN_PATH" "$TINY_SD_DIRECTORY" "a red cube" 2 7.5 0 1
```

CUDA using the plugin managed by the project `uv` environment:

```sh
cargo run -p rxla-safetensors --example stable_diffusion --features model -- \
  .venv/lib/python3.14/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so \
  "$TINY_SD_DIRECTORY" "a red cube" 2 7.5 0 1
```

PyTorch using the same project environment and RXLA-generated latent:

```sh
uv run --no-sync python benchmarks/stable_diffusion_torch.py \
  "$TINY_SD_DIRECTORY" rxla-sd-latent.f32 \
  --steps 20 --guidance 7.5 --warmup 2 --iterations 10 \
  --compare rxla-sd.f32
```

The runner prints compilation, synchronized stage execution, output download,
and checksum separately. Use nonzero warmup and multiple iterations before
drawing steady-state performance conclusions.

## Full Stable Diffusion 1.5 image

The example also accepts the standard Diffusers SD 1.x architecture and writes
PNG directly. Only the tokenizer, text encoder, UNet, VAE, and scheduler files
from `stable-diffusion-v1-5/stable-diffusion-v1-5` are required; the safety
checker and monolithic checkpoints are not loaded by RXLA.

```sh
cargo run -p rxla-safetensors --example stable_diffusion --features model -- \
  .venv/lib/python3.14/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so \
  "$SD15_DIRECTORY" \
  "a cinematic photograph of an astronaut riding a horse, detailed, natural lighting" \
  20 7.5 1 3 rxla-sd15.png --preset sd-v1 \
  --cache-dir "$HOME/.cache/rxla/executables/sd15"
```

On the RTX 5080, the first validated run measured:

| cache | resolution | steps | build/compile | inference | CLIP | denoise | VAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cold | 512×512 | 2 | 105.819 s | 312.525 ms | 12.323 ms | 222.835 ms | 76.181 ms |
| 3/3 disk hits | 512×512 | 20 | 8.186 s | 1.343 s | 5.073 ms | 1.271 s | 64.766 ms |

The three serialized native executables occupy about 5 MiB. The cached time
still includes rebuilding schemas, deserializing executables, and loading roughly
4 GiB of F32 weights; it is not backend compilation. Cache directories contain
native code and must be private and trusted.

The generated image was visually checked:
it contains a coherent astronaut on a horse rather than the noise-like output
produced before correcting the legacy Diffusers `attention_head_dim` semantics.
In SD 1.x that legacy value `8` denotes eight attention heads, not a per-head
width of eight.

### PyTorch precision comparison

The local PyTorch oracle now accepts `--precision f32|f16`. PyTorch 2.14 uses
its SDPA attention path for this Diffusers model. Five measured FP16 iterations
on the same RTX 5080 gave:

```sh
uv run --no-sync python benchmarks/stable_diffusion_torch.py \
  "$SD15_DIRECTORY" rxla-sd-latent.f32 --steps 20 --guidance 7.5 \
  --warmup 1 --iterations 5 --precision f16
```

| implementation | precision | 20-step inference | denoise | VAE |
| --- | --- | ---: | ---: | ---: |
| RXLA | F32 with eligible attention in F16 | 1.343 s | 1.271 s | 64.766 ms |
| PyTorch SDPA | F16 | 621.693 ms | 575.934 ms | 43.654 ms |
| PyTorch SDPA | F32 | 1.678 s | 1.594 s | 80.202 ms |

RXLA is 20.0% faster than the measured F32 baseline but 2.16× slower than the
FP16 baseline. The next material inference optimization is whole-model mixed
precision; attention-only F16 does not exercise the RTX 5080 tensor cores for
most UNet convolution and projection work.

### Whole-model mixed precision

`Compiler::with_compute_dtype` performs a typed Pliron/StableHLO transform:
floating-point SSA results and constants use the selected compute dtype, while
a generated entry wrapper preserves the model's F32 PJRT ABI. This keeps
checkpoint schemas and scheduler buffers independent of compilation precision.
Long eligible attention remains an F16 cuDNN FMHA custom call.

Direct whole-model F16 is exposed for experimentation, but SD 1.5 produces
non-finite values because reductions such as GroupNorm accumulate in F16. BF16
has F32's exponent range and produced a coherent image:

```sh
cargo run -p rxla-safetensors --example stable_diffusion \
  --features model,disk-cache -- \
  .venv/lib/python3.14/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so \
  "$SD15_DIRECTORY" \
  "a cinematic photograph of an astronaut riding a horse, detailed, natural lighting" \
  20 7.5 1 5 rxla-sd15-bf16.png --preset sd-v1 --precision bf16 \
  --cache-dir "$HOME/.cache/rxla/executables/sd15"
```

Five measured iterations after one warmup were:

| implementation | compute | mean | p50 | p95 | denoise | VAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RXLA | BF16 + F16 FMHA | 588.873 ms | 588.986 ms | 592.288 ms | 555.168 ms | 28.413 ms |
| PyTorch SDPA | BF16 | 623.102 ms | 623.153 ms | 623.255 ms | 577.298 ms | 43.701 ms |
| PyTorch SDPA | F16 | 621.693 ms | 621.749 ms | 621.784 ms | 575.934 ms | 43.654 ms |

The typed mixed-precision path is 2.28× faster than RXLA's attention-only F16
run and 5.5% faster than the same-machine PyTorch BF16 baseline. Its cold
compile was 39.636 s; subsequent processes use a precision-specific persistent
artifact through the same cache directory.
