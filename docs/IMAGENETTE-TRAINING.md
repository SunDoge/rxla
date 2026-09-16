# Imagenette ResNet training

The `imagenette_train` example trains an ImageNet-style ResNet-18 from random
initialization against an ImageFolder tree. It uses the standard 2-2-2-2
BasicBlock layout, projection shortcuts at stage boundaries, and stateful
BatchNorm. The 160px stem uses 7x7 stride-two convolution followed by 3x3
stride-two average pooling. This last detail differs from canonical ResNet-18's
max pool because RXLA does not yet implement a max-pool VJP.

JPEG decode, resize, random crop, and collation run in parallel on the CPU.
Horizontal flip and ImageNet normalization are an RXLA Tensor program on the
CPU PJRT backend. Pinned buffers upload to CUDA, where forward, backward,
BatchNorm state transitions, and SGD execute as one compiled program.

```bash
cargo run -p rxla-train --release --example imagenette_train -- \
  --cpu-plugin "$PJRT_CPU_PLUGIN_PATH" \
  --gpu-plugin "$PJRT_CUDA_PLUGIN_PATH" \
  --dataset /data/users/me/datasets/imagenette2 \
  --batch-size 16 \
  --steps 600 \
  --learning-rate 0.03
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

Larger batches expose considerably more of the RTX 5080's throughput:

| Batch | Training throughput |
| ---: | ---: |
| 16 | 433.9 images/s |
| 64 | 707.3 images/s |
| 128 | 759.3 images/s |

These are end-to-end figures rather than isolated model kernel benchmarks. The
flattening above batch 64, together with remaining XLA `gemm_fusion` register
spills, leaves substantial optimization work in both the input pipeline and
generated GPU program.

The training throughput excludes compilation and initial directory scanning,
but includes JPEG input work, CPU augmentation, transfers, CUDA execution, and
per-step metric downloads. This short run is approximately one pass over the
training set. Momentum, weight decay, a learning-rate schedule, mixed precision,
checkpointing, and a full canonical training recipe are not yet included.
