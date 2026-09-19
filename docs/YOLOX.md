# YOLOX training and inference

RXLA implements the official YOLOX-tiny topology, checkpoint-compatible model
paths, image preprocessing, decode/NMS, SimOTA matching, reverse-mode training,
and resident grouped momentum-SGD updates. CPU and CUDA use the same model and training code;
only the PJRT plugin path changes.

## Prepare official weights

Download the official `yolox_tiny.pth`, then convert it once with the locked uv
environment:

```sh
uv run python benchmarks/convert_yolox.py \
  yolox_tiny.pth yolox_tiny.safetensors
```

The converter records the `rxla-yolox-v1` format marker and maps PyTorch
BatchNorm state names to RXLA's model schema.

The deterministic training oracle checks the converted checkpoint, model
topology, SimOTA matcher, losses, reverse-mode head gradients, selected
full-model parameter gradients, BatchNorm state, and one complete Nesterov SGD
update against the official-compatible PyTorch implementation without image
resize or decoding differences:

```sh
uv run --no-sync python benchmarks/yolox_oracle.py \
  /path/to/xla_cuda_plugin.so \
  yolox_tiny.pth yolox_tiny.safetensors .cache/yolox/oracle
```

At the default 64×64 oracle size, all five CSPDarknet stages, three PAFPN
features, and three raw heads are checked. It then compares dense SimOTA box,
objectness, and class assignments plus total, IoU, objectness, class, and L1
losses. On the CUDA 13 PJRT plugin, the largest observed raw-head absolute error
was `2.10e-5`, the assignments selected the same two foreground anchors, and
the largest loss error was `4.30e-6`. The largest absolute gradient error over
all three raw heads was `2.54e-7`. A second graph runs the complete model in
training mode and checks matcher outputs, loss, selected parameter gradients,
BatchNorm state, and the resulting parameter updates. Gradients and updates are
sampled at the stem, immediately before and after the SPP max-pool branches,
and at the first class predictor. Their largest observed relative L2 errors
were both `7.59e-4` (`rtol=2e-3`); the training loss differed by `1.44e-5`.
The optimizer comparison uses the training configuration's `0.9` momentum,
Nesterov update, and `5e-4` decay for non-BatchNorm weights. The stem running
mean and variance differed by at most `9.54e-7` and `2.45e-4`, respectively.
RXLA applies the same Bessel correction as PyTorch when updating running
variance while retaining population variance for the current batch's
normalization.

## Inference

```sh
cargo run -p rxla-safetensors --release --features model \
  --example yolox_infer -- \
  /path/to/xla_cuda_plugin.so \
  yolox_tiny.safetensors input.jpg detections.png
```

The example letterboxes to 416×416, executes the three YOLOX heads, performs
decode and class-aware NMS, and writes an annotated image.

## Training on COCO

```sh
cargo run -p rxla-safetensors --release --features model \
  --example yolox_coco_train_step -- \
  /path/to/xla_cuda_plugin.so \
  yolox_tiny.safetensors \
  --size 416 --batch-size 8 --steps 100 \
  --output yolox_tiny_trained.safetensors \
  coco \
  /data/datasets/coco2017/annotations/instances_train2017.json \
  /data/datasets/coco2017/train2017 \
  391895
```

The model is compiled once. One stateful XLA program executes the model forward,
device-side SimOTA assignment, loss, backward pass, BatchNorm updates, and SGD
update. SimOTA receives at most 120 padded targets per image, matching the
official training transform, and lowers center filtering, pairwise IoU,
classification cost, dynamic-k selection, and conflict resolution to StableHLO.
Assignments are detached before entering the differentiable loss. The output
checkpoint excludes optimizer moments but includes parameters and model
BatchNorm state, so it can be passed directly to `yolox_infer`.

The default total learning rate follows the official trainer and scales
`basic_lr_per_img = 0.01 / 64` by the local batch size. For example, batch 8
uses `0.00125`. An explicit `--learning-rate` overrides the resulting total
optimizer rate rather than the per-image constant.

Freshly initialized models use the official YOLOX policies: convolution weights
use PyTorch-compatible Kaiming-uniform initialization, BatchNorm uses
`epsilon=1e-3` and `momentum=0.03`, and objectness/classification predictor
biases start at `logit(0.01)`.

The example accepts a static `--batch-size`; variable numbers of targets are
padded and masked before transfer. JPEG decode and augmentation run across the
batch with Rayon. Dataset shuffling, augmentation, and prefetch remain in the
data pipeline rather than the YOLO model definition.

CUDA training can use convolution-only automatic mixed precision with
`--compute-dtype bf16`. Convolution inputs and F32 master weights are cast for
the cuDNN operation and its adjoints, then converted back at the layer boundary.
BatchNorm, SimOTA, losses, model state, momentum, and parameter updates remain
F32. This is deliberately narrower than whole-program dtype demotion.

Training defaults to `--convolution-precision default`, allowing XLA/cuDNN to
use accelerator-native reduced-precision contraction algorithms such as TF32
for F32 convolutions. Use `--convolution-precision highest` for the strict
StableHLO precision contract used by the numerical oracle. Precision is part of
`Conv2dOptions` and propagates through both input- and weight-gradient
convolutions rather than being a process-global flag.

For sequential training, COCO can instead be stored as standard TFRecord
framing with Apache Fory Standard Row payloads. The JPEG remains compressed;
Fory provides typed, borrowed views of its bytes and the box/label arrays.

```sh
uv run --no-sync python benchmarks/fory_tfrecord.py \
  /data/datasets/coco2017 \
  /data/datasets/coco2017-fory-4096.tfrecord \
  --samples 4096

cargo run -p rxla-safetensors --release --features model \
  --example yolox_coco_train_step -- \
  /path/to/xla_cuda_plugin.so yolox_tiny.safetensors \
  --compute-dtype bf16 \
  --size 416 --batch-size 8 --prefetch 16 --augmentation-workers 16 \
  --shuffle-buffer 1024 --seed 7 \
  --steps 100 \
  tfrecord /data/datasets/coco2017-fory-4096.tfrecord
```

The training reader requires the adjacent `.tfrecord.json` manifest and checks
the container, Fory version, and schema hash before reading any records. Each
TFRecord length and payload CRC32C is then verified while streaming.
The reader remains sequential; a deterministic bounded pool above the storage
layer supplies randomized samples without seeking. This is also the sampling
boundary for compound augmentations: each training Mosaic consumes four samples
from the pool without replacement, applies the official random-center layout
and YOLOX-tiny's `0.5..1.5` affine scale range. Tiny disables MixUp by default;
when enabled explicitly, it draws another nonempty sample from the same pool
with `0.5..1.5` jitter, optional flip, crop, and a 50/50 blend. HSV
perturbation and a 0.5-probability horizontal flip then run on the composite.
Post-affine boxes follow the current YOLOX 0.3 training path: affine itself
retains clipped boxes, then `TrainTransform` removes boxes whose scaled minimum
side is at most one pixel. If that would remove every label, it falls back to
the pre-HSV/pre-flip image and labels. The stricter width/area/aspect candidate
filter belonged to YOLOX 0.1's older `random_perspective` path and is not mixed
into the current recipe.
The final no-augmentation phase consumes one image, disables Mosaic/MixUp, and
enables the official raw-box L1 loss. Its weight is a runtime scalar, so the
phase transition reuses the same compiled forward/backward/SGD program. Decoded
RGB images are converted to the BGR channel order used by the official OpenCV
input pipeline before entering the model.

Both COCO-file and TFRecord sources roll over at end of input, so `--steps` can
span any number of epochs instead of stopping after one physical pass. A COCO
`image_id` cursor applies only to the first partial epoch; later epochs restart
at the beginning. TFRecord input reopens the same validated file. A genuinely
empty source still terminates with an error rather than spinning forever, and
the bounded shuffle pool remains continuous across storage rollovers.

Long runs can write and resume the complete training transformation, rather
than only inference weights. `--training-output state.safetensors` includes
parameters, BatchNorm state, SGD momentum, packed model EMA, and the EMA update
counter. `--resume state.safetensors` validates the global schedule, model and
data-pipeline configuration before restoring exact dtypes and shapes. Use
`--run-steps N` to stop after `N` updates while retaining `--steps` as the
global learning-rate/no-augmentation schedule:

```sh
# First invocation performs 1,000 updates of a 90,000-step schedule.
yolox_coco_train_step PLUGIN WEIGHTS \
  --steps 90000 --run-steps 1000 \
  --training-output step-1000.safetensors \
  tfrecord train.tfrecord

# Continue the same schedule and deterministic sample stream.
yolox_coco_train_step PLUGIN --resume step-1000.safetensors \
  --steps 90000 \
  --training-output step-90000.safetensors \
  tfrecord train.tfrecord
```

To reconstruct the shuffle pool and variable Mosaic/MixUp consumption exactly,
resume replays the sampling plan from the beginning without decoding or
augmenting images. The source and its ordering must therefore remain unchanged.
This costs sequential compressed-record reads but does not repeat device work.
In a CUDA check, a one-step snapshot resumed for step two produced all 630
parameter, BatchNorm, optimizer, and EMA tensors bit-identically to an
uninterrupted two-step run; replaying the first TFRecord step took 1.7 ms.

## Verified baseline

On an RTX 5080 with JAX's CUDA 13 PJRT plugin, device SimOTA selects exactly the
same oracle assignments as the official-compatible PyTorch implementation and
the retained Rust host reference. At 416×416 and batch size 8, a 20-step run
over the Fory TFRecord with default Mosaic, no MixUp, and two final
no-augmentation/L1 steps processed 160 output images in 869.0 ms, including
first-step execution warmup: 184.1 images/s. The unified
forward/SimOTA/loss/backward/SGD graph took 36.4–37.2 ms after warmup. A bounded
three-batch producer hid the 39–51 ms Mosaic preparation time; measured consumer
wait remained 0.0 ms. Compilation took 12.4 seconds. Timings vary with image
contents, target counts, host CPU, and PJRT/XLA versions.

For comparison, the preceding host-SimOTA design reached 124.9 images/s on the
same hardware. It required an additional approximately 8 ms inference pass,
9–11 ms of host matching, a raw-head download, and dense-target uploads per
step. Moving matching into XLA improved measured end-to-end throughput by about
47% while preserving oracle assignments and losses.

With Mosaic, affine, MixUp, HSV, and flip all enabled, the corresponding
twenty-step run took 1.482 seconds for 160 output images (107.9 images/s),
including first-step device warmup and two final no-augmentation/L1 steps.
Augmentation prepared a typical batch in 63–66 ms, but a three-batch bounded
prefetch kept measured consumer wait at 0.0 ms throughout the run.

A dedicated synthetic benchmark now runs the same U8 input pattern, 12 targets
per image, device SimOTA, loss, BatchNorm updates, Nesterov SGD, warmup count,
and per-step synchronization in RXLA and the official-compatible PyTorch model.
At 416×416, batch 8, 10 warmups, and 100 measured F32 steps, RXLA with default
convolution precision measured `16.871 ms` (`474.2 images/s`), while
`torch.compile` measured `21.453 ms` (`372.9 images/s`). A shorter 20-step BF16
run measured `12.901 ms` (`620.1 images/s`) for RXLA, `16.918 ms` (`472.9
images/s`) for `torch.compile`, and `24.189 ms` for eager PyTorch. The Python
loss still has a graph break at its per-image `Tensor.item()` SimOTA loop, while
RXLA compiles matching and optimization into the same program.

The same sparse-target graph completed a seeded 100-step BF16 run over the
Fory COCO stream at `502.1 images/s`. Consumer wait remained `0 ms`; late
steady device observations were `10.8–12.4 ms`. This is within the observed
run-to-run range of the earlier `513.6 images/s` host-pipeline result rather
than evidence of a separate end-to-end gain.

Host augmentation now parallelizes affine warp rows, HSV pixels, MixUp blend
rows, and letterbox conversion in addition to batch and Mosaic JPEG work. In a
paired 200-step BF16 run at 416×416, batch 8, three augmentation workers, and
the same executable-cache hit, the preceding scalar-pixel implementation
reached `388.1 images/s`; the parallel implementation reached `444.9
images/s`, a 14.6% end-to-end improvement. Typical Mosaic preparation fell
from `51–59 ms` to `26–39 ms` per batch, while the ordered bounded pipeline
retained deterministic sample order.

The real trainer now enables the official YOLOX parameter EMA by default,
including its `0.9998 * (1 - exp(-updates / 2000))` decay ramp. SGD and EMA are
one stateful device program, with no parameter download between them; the
export path writes EMA parameters under their canonical model names. The EMA
values are packed into one logical F32 vector for the hot update and split only
when a checkpoint is requested. On the same seeded 100-step BF16 workload, the
initial per-parameter implementation reached `447.6 images/s`; packing
parameters raised that to `470.6 images/s`. After folding floating model state
into the same packed operation, a subsequent run reached `490.8 images/s`,
versus `492.6 images/s` with `--disable-model-ema`. Late observed device time
was `10.9 ms` with packed EMA and `11.1 ms` in that no-EMA run, so the remaining
total-throughput difference is below the noise of these short end-to-end runs.
BatchNorm running statistics are initialized from their exact pre-forward
values, share the packed EMA update, and are exported under canonical state
paths; non-floating counters and RNG state are not averaged.

Metric observation is decoupled from device execution: the real trainer only
downloads and validates its six scalar metrics on the first, final, and
`--log-every` steps. Intermediate PJRT submissions retain their device-side
state dependencies without forcing a host synchronization. With
`--log-every 100`, the same seeded BF16/EMA workload reached `503.1 images/s`
instead of the preceding fully synchronized `490.8 images/s`. A matching
no-EMA asynchronous run reached `502.2 images/s`, so packed parameter and
BatchNorm EMA had no measurable throughput cost in this pair of short runs.

Device SimOTA normally retains the official per-image padding capacity of 120.
For datasets or augmentation policies with a known lower object-count ceiling,
`--target-capacity` reduces that static matching dimension; excess targets are
explicitly capped in the same way as the official 120-entry preprocessing
limit. On the RTX 5080 BF16 synthetic workload at 416×416, batch 8, and 12 valid
targets per image, capacity 120 measured `12.466 ms` (`641.8 images/s`), while
capacity 64 measured `11.779 ms` (`679.2 images/s`), a 5.8% throughput gain.
Capacity 32 measured `11.880 ms`, so smaller shapes are not monotonically faster
under XLA code generation. The default remains 120 for recipe compatibility;
this knob is fixed per compiled variant and does not create an automatic bucket
cross-product with multi-scale image sizes.

`--random-resize` enables YOLOX's ten-step multi-scale schedule. It selects a
square 32-pixel multiple in the inclusive `--random-size-min` to
`--random-size-max` range (official defaults `10..20`), precompiles only the
unique static sizes that this run will use, and switches at ten-step
boundaries. Parameter, BatchNorm, momentum, EMA, and counter buffers migrate
between the size-specialized programs on the same PJRT client without a host
tensor round-trip. A real CUDA/COCO check trained ten steps at 64×64, migrated
to 96×96, and completed the eleventh update with finite loss. That step also
verified the official empty-match behavior: after augmentation removed every
box, SimOTA produced zero foreground and the normalized objectness-only loss
continued instead of treating the batch as an error. Multi-scale remains
opt-in because compiling several static XLA programs is expensive; two tiny
size variants took 43.4 seconds in this check. Building the example with
`--features model,disk-cache` and passing a trusted `--compile-cache DIR`
persists PJRT-native executables per exact StableHLO/shape. In a fresh-process
two-size test, the first run compiled both programs in `43.762 s`; the second
reported two disk hits, zero backend misses, and restored them in `0.664 s`.
The two entries occupied 6.7 MiB. Deserialized CUDA executables still paid lazy
first-execution setup, but compile-plus-run startup fell from roughly 43.9 s to
1.1 s.

Run the comparable RXLA side with:

```sh
cargo run -p rxla-safetensors --release --features model \
  --example yolox_train_benchmark -- \
  /path/to/xla_cuda_plugin.so yolox_tiny.safetensors \
  --size 416 --batch-size 8 --warmup 10 --iterations 100 \
  --targets-per-image 12 --compute-dtype f32
```

The input boundary preserves decoded images as U8 through host batching and
PCIe transfer, then performs the exact U8-to-F32 conversion in the compiled
graph. Target classes likewise cross the boundary as compact I32 indices rather
than host one-hot vectors. For batch 8, 120 padded targets, and 80 COCO classes,
this reduces their per-step upload from 307,200 to 3,840 bytes. SimOTA gathers
the selected class cost directly and only materializes one-hot classes for the
final matched anchors; this also removes the class-selection `dot_general` that
caused unnecessary GEMM autotuning. Mosaic exposes its four independent JPEG
decode/resize jobs to the same Rayon pool as the batch, rather than serializing
them inside each output image. Before the native max-pool adjoint, this let a 100-step real-data run
sustain 295.6 images/s (2.707 seconds for 800 images), 37.9% above the U8-only
result. Steady device steps were then 22.6–24.8 ms, observed Mosaic preparation
was 22.6–28.3 ms, and consumer wait remained 0.0 ms.

The first Nsight profile appeared to expose a convolution problem: RXLA launched
about 1,154 kernels per steady step with 15.21 ms aggregate GPU kernel time,
versus about 2,132 kernels and 7.70 ms for the profiled PyTorch/Inductor step.
About 239 `conv2d_grouped_direct` launches consumed 61.2% of RXLA kernel time.
The name was misleading. YOLOX-tiny has `depthwise=false`; the launches came
from RXLA's old max-pool adjoint, which expanded each spatial window position
into a transpose-convolution scatter. The 5x5, 9x9, and 13x13 SPP windows alone
created 275 such operations per training graph.

Max-pool reverse mode now remains a first-class RXLA IR operation and lowers to
one StableHLO `select_and_scatter` per pool. It also follows PyTorch/JAX-style
single-winner tie semantics instead of dividing a cotangent among equal maxima.
On the same real-data BF16 run, steady training steps fell to 12.2–16.3 ms. A
100-step run processed 800 output images in 2.618 seconds (305.6 images/s end to
end); JPEG/Mosaic preparation and producer waits, rather than the device graph,
now set that throughput. An 11-step Nsight capture measured 4.07 ms aggregate
GPU kernel time
and about 402 kernel launches per step. Only one grouped-direct launch remained
per step, consuming less than 0.01 ms. The native lowering therefore removed
the actual bottleneck instead of tuning unrelated convolutions.

Host augmentation is now split into deterministic sequential sampling and an
ordered parallel transformation stage. Multiple batches may decode and augment
concurrently, while a bounded reorder buffer preserves the exact seeded sample
order. On the same 100-step workload, 16 augmentation workers with 16 prefetched
batches raised end-to-end throughput from 305.6 to 513.6 images/s. Final
measurements for 1, 2, 4, 8, 16, and 24 workers were respectively 339.7, about
417, 429.1, 480.4, 513.6, and 497.7 images/s. The example therefore defaults to
two thirds of the available host parallelism for both values (16 on this
24-thread host), while retaining explicit overrides. The prepared sample IDs
remained identical across every run.

BF16 XLA autotune logs also confirm that forward and weight-gradient
convolutions remain BF16, while input gradients are represented as
reversed/dilated forward convolutions and ultimately autotuned as
`__cudnn$convForward`; PyTorch uses dedicated dgrad kernels. Rewriting the
reversal to JAX's explicit `stablehlo.reverse` form was mathematically correct
but slowed steady training
to 32.6--37.1 ms, so it was rejected. Autotune inspection showed that 12 BF16
shapes then became `__cudnn$convBackwardInput`, but an Nsight capture exposed
the regression: nine profiled steps spent 38.85 ms in 954
`dgrad2d_grouped_direct` launches, plus about 33 ms in FFT and GEMV support
kernels. The retained reversed-window forward formulation is substantially
faster for this graph. Closing the gap therefore requires a better convolution
plan rather than merely making XLA classify the operation as dgrad; a
depthwise-only kernel would not help this model.

Two narrower canonicalization experiments were also rejected. Using the dgrad
form only for 13x13, 3x3, ungrouped input gradients measured 23.4--24.8 ms and
278.7 images/s, still below the retained end-to-end baseline. Explicitly
transposing checkpoint-native OIHW parameters to JAX-style HWIO before every
convolution measured 22.9--24.4 ms and 272.7 images/s. This indicates that XLA
already canonicalizes the OIHW dimension numbers adequately; neither a
model-shape heuristic nor extra layout operations justify permanent API or IR
complexity.

RXLA now independently preserves genuinely grouped input gradients as one
`feature_group_count` StableHLO convolution instead of expanding a Rust loop.
CPU scalar-gradient and higher-order tests verify that lowering, but it is an
IR/correctness improvement rather than the explanation for YOLOX-tiny's speed.

The earlier real-data experiment that appeared to make `DEFAULT` convolution
precision slower was confounded by the old max-pool adjoint and host pipeline.
Repeating it after those fixes showed the opposite: strict synthetic F32 steps
fell from `26.075 ms` at `HIGHEST` to `17.108 ms` at `DEFAULT`. Nsight explains
the difference: the strict graph spent about 19.4 ms of GPU kernel time per
profiled step in SIMT/FFT convolution algorithms, while PyTorch used TF32
tensor-core kernels. `DEFAULT` is now an explicit per-convolution policy and is
selected by the training example; the model API retains `HIGHEST` as its
accuracy-first default.

Two other XLA configuration alternatives remain rejected. Enabling
`--xla_gpu_experimental_enable_conv_fusion=true` slowed them to 36--37 ms
(196 images/s), and forcing NCHW with `--xla_gpu_force_conv_nchw=true` reached
28.4--29.2 ms (235 images/s). The plugin already defaults convolution autotuning
to level 4; the former cuDNN frontend switch is no longer present in this XLA
version. The retained NHWC path is therefore the best measured general XLA
lowering rather than an accidentally disabled autotuning setup.

Disabling CUDA command buffers remains workload-dependent. On the sparse-target
strict BF16 benchmark, 100 measured steps improved from `12.719 ms` (`629.0
images/s`) to `12.459 ms` (`642.1 images/s`). Nsight showed about 173
`cuGraphLaunch` calls per step, with graph launches and node-parameter updates
accounting for roughly 3.3 ms of host CUDA API time. The seeded real COCO run,
however, regressed from `502.1` to `489.9 images/s`. Command buffers therefore
remain the default. Both training and synthetic benchmark examples expose
`--direct-launches` for hardware- and workload-specific measurement; internally
it uses `Compiler::without_cuda_command_buffers()`.

For bounded Nsight captures, `--cuda-profile --cuda-runtime /path/to/libcudart.so`
starts collection after the first training step. Use it with
`nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop`.

The exported checkpoint was then loaded by the 416×416 inference example on
both CPU and CUDA. CUDA inference measured 9.4 ms after compilation and produced
the same 15 decoded detections as CPU for the validation image used in the test.
