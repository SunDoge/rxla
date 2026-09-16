# Tensor image augmentation experiment

This experiment evaluates whether several image preprocessing operations become
useful when expressed as one lazy Tensor program. It is inspired by
[dm-pix](https://github.com/google-deepmind/dm_pix), whose image functions are
ordinary JAX array expressions that can be transformed and compiled together.

The result is scoped: graph-native preprocessing is substantially faster when
the input is already resident, especially for a batch, but a separate GPU
round-trip is slower than host preprocessing. The intended use is therefore to
join augmentation to the first model stage, not to dispatch it as an isolated
GPU service.

## Tensor API

The added deterministic NHWC operations are:

- arbitrary-size asymmetric nearest-neighbor resize;
- half-pixel bilinear resize with edge clamping;
- per-channel normalization;
- brightness and per-image/channel contrast adjustment;
- horizontal and vertical flips;
- center crop and centered crop-or-zero-pad;
- RGB-to-grayscale using dm-pix's Rec. 601, Rec. 709 or BT. 2001 coefficients;
- solarization.

They compose through the public Tensor API. `TensorFunction` gives the same
Tensor-only model a reusable static input ABI without exposing `Graph` or
`Tracer`:

```rust,ignore
let preprocess = TensorFunction::new([8, 320, 320, 3], DType::U8, |image| {
    image
        .cast(DType::F32)?
        .mul_scalar(1.0 / 255.0)?
        .resize_bilinear2d([286, 286])?
        .center_crop([256, 256])?
        .flip_left_right()?
        .adjust_brightness(0.05)?
        .adjust_contrast(1.1)?
        .normalize_nhwc(&[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225])
})?;

let output = preprocess.call(&mut runtime, &input)?;
```

Resize currently has no antialias filter. Random crop/flip, affine transforms,
blur, HSV/HSL adjustment and arbitrary channel axes remain future work. Static
input/output shapes mean each size bucket has its own executable.

## Fast frontend validation

A lazy Tensor can now be snapshotted into a verified `Program` and StableHLO
without loading PJRT:

```rust,ignore
let program = output.program()?;
let stablehlo = program.lowered_program();
```

The host-only image test checks the complete U8 cast, resize, crop, flip,
brightness, contrast and normalization chain, its typed ABI, output shape and
key StableHLO operations. On the development machine the already-built targeted
test itself completes in approximately 0.01 seconds. This should be the default
operator-development loop:

```sh
cargo test -p rxla-core image::tests::chained_augmentation_lowers_without_a_runtime --lib
```

Native numerical tests are a narrower second gate. Run GPU tests serially: each
parallel integration test otherwise creates an independent PJRT client whose
default BFC pool attempts to reserve most of the RTX 5080 memory.

```sh
PJRT_PLUGIN_PATH=/trusted/plugin.so \
  cargo test -p rxla-core --test image --test resize -- \
  --ignored --test-threads=1
```

Development commands should select the relevant library/test/example target.
`cargo test --tests --no-run` links every integration-test binary and is not an
appropriate inner loop. Use `cargo check` before a native run and rely on the
existing Cargo artifact cache; XLA compilation is reserved for the selected
numerical/performance gates.

## Benchmark

`image_augment_bench` runs this fixed chain:

```text
U8 NHWC
  -> F32 / 255
  -> half-pixel bilinear 320x320 to 286x286
  -> center crop 256x256
  -> horizontal flip
  -> brightness +0.05
  -> contrast x1.1 around each image/channel mean
  -> ImageNet channel normalization
  -> squared mean checksum
```

The checksum keeps the complete pipeline live while downloading one scalar.
`resident` excludes upload and compilation. `end-to-end` includes construction
of a copied host Tensor, synchronous upload, execution and scalar download. The
host reference is an independent scalar implementation with resize coordinates
precomputed outside the timed path and cache-friendly horizontal and vertical
interpolation passes. A separate direct four-neighbor implementation checks its
numerical result. It is still not a comparison with a tuned SIMD/threaded image
library. Warmup trials are excluded, and compilation is reported separately.

Measurements were taken on 2026-09-16 using an Intel Core Ultra 9 285K and
NVIDIA GeForce RTX 5080. PJRT was the pinned ZML build documented in
[CUDA validation](CUDA-validation.md). Values are milliseconds; medians are
from 30 trials for batch 1 and 20 trials for batch 8.

| Backend | Batch | XLA compile | Host median / p95 | Resident median / p95 | End-to-end median / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU reference only | 1 | n/a | 1.305 / 1.319 | n/a | n/a |
| XLA CPU | 1 | 48.065 | 1.305 / 1.319 | 1.370 / 1.901 | 3.415 / 3.904 |
| RTX 5080 CUDA | 1 | 191.064 | 1.308 / 1.321 | 0.888 / 1.198 | 2.859 / 3.337 |
| CPU reference only | 8 | n/a | 5.464 / 10.406 | n/a | n/a |
| XLA CPU | 8 | 45.035 | 5.464 / 10.406 | 3.513 / 4.552 | 13.673 / 14.685 |
| RTX 5080 CUDA | 8 | 184.629 | 5.490 / 9.111 | 1.673 / 2.473 | 18.927 / 21.359 |

For batch 8, resident XLA CPU is about 1.56x faster than the scalar host
reference and resident CUDA is about 3.28x faster. Batch-1 resident CUDA is
about 1.47x faster, while resident XLA CPU is slightly slower. Re-uploading
every batch loses in all measured cases: batch-8 end-to-end CUDA is about 3.45x
slower than the host reference.

The optimized-HLO diagnostics contain fusion computations for the chain, but a
string occurrence count is not a kernel count or proof that the entire program
is one kernel. Nsight/XProf profiling is required before making a kernel-level
fusion claim.

Reproduce with:

```sh
cargo run --release -p rxla-core --example image_augment_bench -- \
  --plugin /trusted/plugin.so --batch 8 --runs 20 --warmups 3
```

## Consequences

The profitable architecture is:

```text
decode/upload once
  -> device-resident augmentation
  -> VAE/text/image encoder or training forward
  -> loss/model output
```

An `eval()` between augmentation steps, or downloading the augmented image
before the model, defeats the main opportunity. Future benchmarks should append
a representative encoder/LoRA training step, compare against a tuned threaded
CPU library and JAX/dm-pix, and profile allocation, copies and generated kernels.
