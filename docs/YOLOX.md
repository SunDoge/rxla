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
topology, SimOTA matcher, and losses against the official-compatible PyTorch
implementation without image resize or decoding differences:

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
the largest loss error was `4.30e-6` (default tensor tolerance `5e-5`).

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

The model is compiled once. Each step runs the current weights through the
inference graph, computes dynamic SimOTA assignments on the host, then executes
the loss, backward pass, BatchNorm updates, and SGD update as one stateful XLA
program. Switching between the inference and training programs clones shared
PJRT buffer handles rather than copying parameter payloads. The output excludes
optimizer moments but includes parameters and model BatchNorm state, so it can
be passed directly to `yolox_infer`.

The example accepts a static `--batch-size`; variable numbers of targets are
kept in host-side records and SimOTA emits dense assignment tensors for the
compiled batch. JPEG decode and letterboxing run across the batch with Rayon,
and SimOTA assigns each image in parallel. Dataset shuffling, augmentation, and
prefetch belong in the data pipeline rather than the YOLO model definition.

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

On an RTX 5080 with JAX's CUDA 13 PJRT plugin, a 64×64 two-step COCO smoke run
produced finite losses on both steps. After warmup, inference for assignment was
3.1 ms and the fused training update was 9.1 ms. The same run passed with the
ZML CPU PJRT plugin; its warm step measured 8.5 ms for inference and 24.6 ms for
training. These are smoke-test timings, not throughput benchmarks.

At 416×416 and batch size 8, a twenty-step run over the Fory TFRecord measured
about 7.5–8.5 ms for assignment inference and 36.4–36.5 ms for the fused
forward/backward/SGD program after warmup (roughly 180 images/s for device work).
Parallel JPEG decode plus letterboxing took 12–16 ms and host SimOTA typically
took 12–15 ms. The bounded producer keeps two prepared batches ahead, reducing
consumer wait to 0.0 ms after the first batch. The complete 160-image measured
region, including first-step execution warmup, took 1.377 seconds (116.2
images/s), compared with roughly 96–111 images/s when input preparation was
sequential. Moving matching onto the device remains the largest pipeline-level
opportunity. Timings vary with image contents, target counts, host CPU, and
PJRT/XLA versions.

With Mosaic, affine, MixUp, HSV, and flip all enabled, the corresponding
twenty-step run took 1.482 seconds for 160 output images (107.9 images/s),
including first-step device warmup and two final no-augmentation/L1 steps.
Augmentation prepared a typical batch in 63–66 ms, but a three-batch bounded
prefetch kept measured consumer wait at 0.0 ms throughout the run.

The exported checkpoint was then loaded by the 416×416 inference example on
both CPU and CUDA. CUDA inference measured 9.4 ms after compilation and produced
the same 15 decoded detections as CPU for the validation image used in the test.
