# Pure-XLA Gaussian splatting experiment

RXLA includes a differentiable, pure StableHLO reference renderer in
`rxla_models::gaussian_splatting`. It projects full camera-space 3D covariance
matrices through a pinhole camera, performs stable front-to-back depth sorting,
and alpha-composites RGB plus a background. It does not use a custom call or a
CUDA-specific kernel.

This is deliberately a dense implementation. Its major intermediates have
shape `[height, width, gaussians]`, so time and memory are `O(HWG)`. It is useful
for validating the Tensor API, StableHLO lowering, autodiff, and small scenes;
it is not yet the tiled/sorted rasterizer required for production 3DGS.

## Run

```bash
PJRT_PLUGIN_PATH=/path/to/xla_cuda_plugin.so \
cargo run --release -p rxla-models --example gaussian_splatting -- \
  --width 256 --height 256 --gaussians 48 \
  --output gaussian-splatting.png
```

The example constructs a small colored spiral, compiles one XLA executable,
reuses it for timed iterations, downloads the final image, and writes PNG.

## Verified boundary

On 2026-09-18, the reference renderer was checked on the local RTX 5080 through
JAX's CUDA 13 PJRT plugin:

- 128x128, 24 Gaussians: 0.256 s compilation and 0.118 ms mean synchronized
  execution over 50 timed iterations.
- 256x256, 48 Gaussians: 1.236 s compilation and 0.176 ms mean synchronized
  execution over 50 timed iterations. `ptxas` reported register spilling in two
  input-reduction fusions, leaving a concrete optimization target.
- Native front-to-back compositing produced the expected RGB and alpha values.
- A loss over rendered RGB compiled and executed gradients for means, full
  covariance matrices, colors, and opacities; all outputs were finite and the
  color/opacity gradients were nonzero.
- The generated StableHLO contains ordinary sort and exponential operations and
  no custom call. The forward correctness test also passes with the CPU PJRT
  plugin.

The timing is only evidence for this synthetic dense case, not a comparison to
gsplat or another tiled CUDA implementation. Every timed submission waits on
its PJRT completion event; PNG download is outside the timed region.

## Semantics and limitations

- Inputs are camera-space means `[G, 3]`, symmetric positive covariances
  `[G, 3, 3]`, linear RGB `[G, 3]`, opacity `[G]`, a pixel-center grid
  `[H, W, 2]`, and background RGB `[3]`.
- Opacities are values rather than logits and are clamped to `[0, 0.99]`.
- Depth ordering and the near/far visibility decision are nondifferentiable.
  The selected continuous means, covariance, color, and opacity paths remain
  differentiable.
- Covariance validity is a caller invariant. A small screen-space diagonal and
  determinant floor protect inversion, but do not turn arbitrary matrices into
  valid covariance matrices.
- Tile binning, compact contributor lists, spherical harmonics, quaternion/scale
  covariance construction, densification/pruning, antialiasing, and distributed
  rendering are future work.

The intended next architectural step is to retain this implementation as the
portable reference and introduce tile/binning transformations or a specialized
CUDA custom call behind the same public renderer contract.
