# Imagenette ResNet training

The `imagenette_train` example trains an ImageNet-style ResNet-18 from random
initialization against an ImageFolder tree. It uses the standard 2-2-2-2
BasicBlock layout, projection shortcuts at stage boundaries, and stateful
BatchNorm. The 160px stem uses 7x7 stride-two convolution followed by 3x3
stride-two average pooling. This last detail differs from canonical ResNet-18's
max pool because RXLA does not yet implement a max-pool VJP.

The input path is a bounded three-stage pipeline. Rayon performs JPEG decode,
resize, random crop, and collation; a dedicated CPU PJRT runtime performs
horizontal flip and ImageNet normalization as an RXLA Tensor program; the main
thread performs pinned uploads and CUDA training. Owned host batches cross
capacity-two channels, providing backpressure without sharing PJRT handles
between threads. Forward, backward, BatchNorm state transitions, and SGD execute
as one compiled CUDA program.

```bash
cargo run -p rxla-train --release --example imagenette_train -- \
  --cpu-plugin "$PJRT_CPU_PLUGIN_PATH" \
  --gpu-plugin "$PJRT_CUDA_PLUGIN_PATH" \
  --dataset /data/users/me/datasets/imagenette2 \
  --batch-size 16 \
  --steps 600 \
  --learning-rate 0.03 \
  --profile
```

## RTX 5080 validation

Validated on 2026-09-16 with 9,469 training images, 3,925 validation images,
CUDA 13.0, cuDNN 9.24.0, seed 42, and plain SGD. Fixed shapes evaluate the
largest 3,920-image validation prefix:

```text
step    0: loss 2.460432, accuracy 18.8%
step  100: loss 1.727793, accuracy 37.5%
step  200: loss 1.733194, accuracy 50.0%
step  300: loss 1.201419, accuracy 62.5%
step  400: loss 1.559697, accuracy 37.5%
step  500: loss 2.390429, accuracy 25.0%
step  599: loss 0.642041, accuracy 81.2%
train: loss 2.460432 -> 0.642041, mean accuracy 45.2%, 433.9 images/s
validation: loss 1.520570, top-1 51.66% (2025/3920)
```

The current native StableHLO convolution-filter gradient reaches 433.9 images/s
at batch 16. This is 3.4% below the earlier slice-and-GEMM lowering's 449.3
images/s on the same run shape, so the cleaner lowering is not yet a CUDA speed
win. It does remove the nine-GEMM expansion of every 3x3 filter gradient and is
covered by scalar-reference tests for stride, dilation, and asymmetric padding.

The original serialized input loop and the bounded pipeline compare as follows.
End-to-end measurements include input work and per-step metrics; the short
batch-16 and batch-64 runs use 100 steps, and batch 128 uses 50 steps:

| Batch | Serialized | Pipelined | Speedup | Pipelined steady state |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 433.9 images/s | 1,043.4 images/s | 2.40x | 1,110.7 images/s |
| 64 | 707.3 images/s | 1,630.0 images/s | 2.30x | 1,710.9 images/s |
| 128 | 759.3 images/s | 1,989.6 images/s | 2.62x | 2,176.0 images/s |

At batch 16, steady-state input wait is 0.62 ms and upload is 0.49 ms,
compared with 10.91 ms in CUDA execution. At batch 128 the GPU accounts for
91.8% of the measured step, so the input pipeline is no longer the primary
bottleneck.

These are end-to-end figures rather than isolated model kernel benchmarks. The
remaining XLA `gemm_fusion` register spills leave substantial optimization work
in the generated GPU program.

The training throughput excludes compilation and initial directory scanning,
but includes JPEG input work, CPU augmentation, transfers, CUDA execution, and
per-step metric downloads. This short run is approximately one pass over the
training set. Momentum, weight decay, a learning-rate schedule, mixed precision,
checkpointing, and a full canonical training recipe are not yet included.
