# CIFAR-10 ResNet-20 training

RXLA's training example uses the CIFAR ResNet-20 layout: one stem convolution,
three stages with three two-convolution residual blocks each, and a linear
classifier. Stage transitions use stride-two option-A shortcuts. All 19
BatchNorm layers update resident mean and variance state in the same compiled
program as forward, backward, and SGD.

Prepare the dataset through the uv-managed torchvision dependency:

```bash
uv run python scripts/export_cifar10.py \
  --root .cache/torchvision \
  --output .cache/cifar-10-binary
```

Then run the heterogeneous CPU/CUDA trainer:

```bash
cargo run -p rxla-train --release --example cifar_train -- \
  --cpu-plugin "$PJRT_CPU_PLUGIN_PATH" \
  --gpu-plugin "$PJRT_CUDA_PLUGIN_PATH" \
  --dataset .cache/cifar-10-binary \
  --batch-size 128 \
  --steps 100 \
  --learning-rate 0.05
```

## RTX 5080 validation

Validated on 2026-09-16 with CUDA 13.0 and cuDNN 9.24.0. The input was the
torchvision CIFAR-10 training split, seed 42, batch size 128, and plain SGD:

```text
step    0: loss 2.409989, accuracy 17.2%
step   20: loss 2.071885, accuracy 24.2%
step   40: loss 1.811213, accuracy 24.2%
step   60: loss 1.760878, accuracy 33.6%
step   80: loss 1.679030, accuracy 39.1%
step   99: loss 1.631837, accuracy 34.4%
summary: loss 2.409989 -> 1.631837, mean accuracy 31.3%, 8981.7 images/s
```

The throughput covers the training loop, CPU Tensor augmentation, pinned H2D
uploads, device execution, and per-step loss/logit downloads. It excludes XLA
compilation and dataset loading. This run proves real-data convergence and the
complete stateful training path; it is not a full-epoch benchmark or a claim
about final test accuracy. Random crop, momentum/weight decay, learning-rate
scheduling, checkpointing, and inference-mode test evaluation remain future
training-example work.

An extended 2,000-step run (about 5.1 passes over 50,000 images) with the same
configuration remained stable and reached:

```text
step    0: loss 2.409989, accuracy 17.2%
step  400: loss 1.501513, accuracy 45.3%
step  800: loss 1.129642, accuracy 58.6%
step 1200: loss 0.849458, accuracy 70.3%
step 1600: loss 0.806189, accuracy 68.8%
step 1999: loss 0.729034, accuracy 77.3%
summary: loss 2.409989 -> 0.729034, mean accuracy 61.6%, 9330.1 images/s
```
