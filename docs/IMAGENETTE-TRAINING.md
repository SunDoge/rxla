# Imagenette ResNet training

The `imagenette_train` example trains an ImageNet-style ResNet-18 from random
initialization against an ImageFolder tree. It uses the standard 2-2-2-2
BasicBlock layout, projection shortcuts at stage boundaries, and stateful
BatchNorm. The 160px stem uses 7x7 stride-two convolution followed by 3x3
stride-two average pooling. This last detail differs from canonical ResNet-18's
max pool because RXLA does not yet implement a max-pool VJP.

The input path uses the reusable `rxla_train::BoundedPipeline`. Rayon performs
JPEG decode, resize, random crop, and collation into U8 images. The main thread
starts pinned uploads, while a named device RNG effect supplies reproducible
flip decisions. Horizontal flip, U8-to-F32 conversion, ImageNet normalization,
forward, backward, BatchNorm and RNG state transitions, and SGD all execute in
one compiled CUDA program. A bounded capacity-four channel preserves order and
backpressure without materializing normalized F32 images on the host.

Trainable parameters are resident session state. Each step therefore submits
only the image and label buffers; it does not rebuild a path-to-buffer map or
return replacement parameters through the executable ABI. After training, the
session is consumed into path-addressed device buffers and restored into a
separately traced inference session without downloading or uploading weights.

```bash
cargo run -p rxla-train --release --example imagenette_train -- \
  --gpu-plugin "$PJRT_CUDA_PLUGIN_PATH" \
  --dataset /data/users/me/datasets/imagenette2 \
  --batch-size 16 \
  --steps 600 \
  --learning-rate 0.03 \
  --profile
```

## RTX 5080 validation

The measurements below predate the resident-parameter migration described
above. They remain useful as historical lowering and input-pipeline baselines,
but do not claim the current training loop has identical throughput.

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

| Batch | Serialized | CPU-PJRT pipeline | Speedup | Pipelined steady state |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 433.9 images/s | 1,118.1 images/s | 2.58x | 1,199.9 images/s |
| 64 | 707.3 images/s | 1,630.0 images/s | 2.30x | 1,710.9 images/s |
| 128 | 759.3 images/s | 1,989.6 images/s | 2.62x | 2,176.0 images/s |

At batch 16, the reusable pipeline's steady-state input wait is 0.77 ms and
upload is 0.51 ms, compared with 10.95 ms in CUDA execution. Accuracy is reduced on-device and one
`[loss, correct_count]` buffer is downloaded instead of full logits and labels;
host metric work falls from 2.35 ms to 0.98 ms. At batch 128 the GPU accounts for
91.8% of the measured step, so the input pipeline is no longer the primary
bottleneck.

Moving tensor preprocessing into the training IR removes the separate CPU PJRT
execution and cuts the image upload from F32 to U8. A subsequent batch-16,
100-step run reached 1,197.3 images/s end to end and 1,243.2 images/s after the
ten-step warmup, versus 1,118.1 and 1,199.9 images/s for the CPU-PJRT path. Its
steady-state breakdown was 0.62 ms input wait, 0.33 ms upload, 10.96 ms CUDA
execution, and 0.93 ms metrics. This is a 7.1% end-to-end improvement while also
removing one executable and a host synchronization boundary. Short training
runs are numerically nondeterministic on the CUDA convolution path, so loss and
accuracy are correctness smoke signals rather than comparable benchmark metrics.

The next revision moved flip sampling from host-generated input values to the
transactional `augmentation` RNG effect. Two batch-16 runs measured 1,179.6 and
1,182.9 images/s end to end; the latter steady-state result was 1,232.6 images/s
with 0.77 ms input wait, 0.31 ms upload, and 10.94 ms CUDA execution. This is
within normal JPEG/input-wait variance of the host-flip run rather than a speed
claim. Its semantic benefit is that the seed and counter are resident named
state, advance atomically with an accepted training step, and need no extra
per-batch scripting-language or host input. Inference retains the same state
schema but samples probability zero without advancing the stream.

These are end-to-end figures rather than isolated model kernel benchmarks. The
remaining XLA `gemm_fusion` register spills leave substantial optimization work
in the generated GPU program.

The training throughput excludes compilation and initial directory scanning,
but includes JPEG input work, CPU augmentation, transfers, CUDA execution, and
per-step metric downloads. This short run is approximately one pass over the
training set. Momentum, weight decay, a learning-rate schedule, mixed precision,
checkpointing, and a full canonical training recipe are not yet included.
