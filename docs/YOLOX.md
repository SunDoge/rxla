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
topology, SimOTA matcher, losses, and reverse-mode head gradients against the
official-compatible PyTorch implementation without image resize or decoding
differences:

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
all three raw heads was `2.54e-7` (default tensor tolerance `5e-5`).

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
  --size 416 --batch-size 8 --steps 100 --learning-rate 0.00015625 \
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
  --size 416 --batch-size 8 --prefetch 2 --shuffle-buffer 1024 --seed 7 \
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
The final no-augmentation phase consumes one image, disables Mosaic/MixUp, and
enables the official raw-box L1 loss. Its weight is a runtime scalar, so the
phase transition reuses the same compiled forward/backward/SGD program. Decoded
RGB images are converted to the BGR channel order used by the official OpenCV
input pipeline before entering the model.

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

The convolution-only BF16 path reduced steady device training time from
`36.4–37.2 ms` to `23.0–24.0 ms` on the same RTX 5080. A 20-step real-data run
reached 214.9 images/s end to end; at that point the 37–51 ms CPU Mosaic producer
became the limiter and occasional consumer waits reappeared. For context, an
official-compatible PyTorch eager benchmark with 12 synthetic targets per image
measured 23.87 ms in F32 and 26.61 ms under BF16 autocast. These are useful
device baselines, not a strict data-pipeline comparison.

Compiling the same PyTorch model with Inductor's default `torch.compile` mode
measured 18.48 ms (433.0 images/s). The official-compatible Python loss has a
graph break at its per-image `Tensor.item()` SimOTA loop, but Inductor still
compiles the convolutional regions. RXLA currently remains slower than this
stronger device baseline.

The input boundary now preserves decoded images as U8 through host batching and
PCIe transfer, then performs the exact U8-to-F32 conversion in the compiled
graph. Mosaic also exposes its four independent JPEG decode/resize jobs to the
same Rayon pool as the batch, rather than serializing them inside each output
image. On this 24-core host a 100-step real-data run sustained 295.6 images/s
(2.707 seconds for 800 images), 37.9% above the U8-only result. Steady device
steps were 22.6–24.8 ms, observed Mosaic preparation was 22.6–28.3 ms, and
consumer wait remained 0.0 ms. Host preparation is therefore hidden behind the
unified CUDA training graph at this batch size.

Nsight profiling explains the remaining CUDA gap. RXLA launched about 1,154
kernels per steady step with 15.21 ms aggregate GPU kernel time, versus about
2,132 kernels and 7.70 ms for the profiled PyTorch/Inductor step. Despite the
lower launch count, 61.2% of RXLA kernel time was cuDNN grouped-direct work:
YOLOX-tiny's depthwise convolutions caused roughly 239 per-channel executions
per step on this backend. RXLA now preserves grouped input gradients as one
`feature_group_count` StableHLO convolution instead of expanding a Rust loop;
CPU scalar-gradient and higher-order tests verify the new lowering. On this
CUDA stack, XLA/cuDNN still internally chooses the same per-channel execution,
so this correctness/IR improvement does not yet reduce the 22–27 ms wall time.
An experimental nine-tap tensor decomposition was rejected: XLA recognized it
as slower F32 convolutions and doubled profiled GPU work. Closing this gap needs
a specialized depthwise CUDA kernel/lowering rather than more graph expansion.

Three XLA configuration alternatives were also measured against the same real
training path and rejected. Changing convolution precision from `HIGHEST` to
`DEFAULT` slowed steady steps to 27.7--29.0 ms (250 images/s). Enabling
`--xla_gpu_experimental_enable_conv_fusion=true` slowed them to 36--37 ms
(196 images/s), and forcing NCHW with `--xla_gpu_force_conv_nchw=true` reached
28.4--29.2 ms (235 images/s). The plugin already defaults convolution autotuning
to level 4; the former cuDNN frontend switch is no longer present in this XLA
version. The retained NHWC, `HIGHEST` path is therefore the best measured
general XLA lowering, rather than an accidentally disabled autotuning setup.

For bounded Nsight captures, `--cuda-profile --cuda-runtime /path/to/libcudart.so`
starts collection after the first training step. Use it with
`nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop`.

The exported checkpoint was then loaded by the 416×416 inference example on
both CPU and CUDA. CUDA inference measured 9.4 ms after compilation and produced
the same 15 decoded detections as CPU for the validation image used in the test.
