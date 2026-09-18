# YOLOX training and inference

RXLA implements the official YOLOX-tiny topology, checkpoint-compatible model
paths, image preprocessing, decode/NMS, SimOTA matching, reverse-mode training,
and resident Adam updates. CPU and CUDA use the same model and training code;
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
  /data/datasets/coco2017/annotations/instances_train2017.json \
  /data/datasets/coco2017/train2017 \
  9 --size 64 --steps 100 --learning-rate 1e-5 \
  --output yolox_tiny_trained.safetensors
```

The model is compiled once. Each step runs the current weights through the
inference graph, computes dynamic SimOTA assignments on the host, then executes
the loss, backward pass, BatchNorm updates, and Adam update as one stateful XLA
program. Switching between the inference and training programs clones shared
PJRT buffer handles rather than copying parameter payloads. The output excludes
optimizer moments but includes parameters and model BatchNorm state, so it can
be passed directly to `yolox_infer`.

The current example is deliberately a correctness-oriented batch-size-one
trainer. Dataset shuffling, batched variable-length targets, augmentation, and
prefetch belong in the data pipeline rather than the YOLO model definition.

## Verified baseline

On an RTX 5080 with JAX's CUDA 13 PJRT plugin, a 64×64 two-step COCO smoke run
produced finite losses on both steps. After warmup, inference for assignment was
3.1 ms and the fused training update was 9.1 ms. The same run passed with the
ZML CPU PJRT plugin; its warm step measured 8.5 ms for inference and 24.6 ms for
training. These are smoke-test timings, not throughput benchmarks.

The exported checkpoint was then loaded by the 416×416 inference example on
both CPU and CUDA. CUDA inference measured 9.4 ms after compilation and produced
the same 15 decoded detections as CPU for the validation image used in the test.
