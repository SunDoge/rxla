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
  Negligible Gaussian tails default to an alpha cutoff of `1/255`, configurable
  on `GaussianCamera`; this prevents many invisible tails from accumulating.
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

## Real-scene experiment

`gaussian_splatting_real` loads the common compact `.splat` record layout and a
Graphdeco-style `cameras.json`. It decodes world-space means, linear scales,
RGBA and quantized quaternions, constructs full covariances, applies the selected
camera transform, and renders a bounded subset through the same StableHLO path.

The experiment used the public Bonsai mini checkpoint from the
`dylanebert/3dgs` dataset: 272,956 real trained Gaussians, with roughly 80,000
centers inside camera 0. CUDA successfully rendered subsets from 256 through
100,000 Gaussians. A synchronized 100,000-Gaussian render at 24x16 took about
2.34 ms after a 2.61 s compile on the local RTX 5080.

The resulting full-scene image is not a quality result. Dense `[H,W,G]`
compositing cannot retain the scene at useful resolution and does not construct
the exact per-tile contributor stream used by production 3DGS. This experiment
therefore establishes real checkpoint parsing and XLA execution while also
rejecting the dense algorithm as the production path. The next renderer should
use the pure-JAX `jax-gs` structure: projected radii, static Gaussian/tile
interactions, packed tile/depth sorting, tile boundaries, and fixed-size scan
compositing. A CUDA custom-call implementation can later share that public
contract.

```bash
PJRT_PLUGIN_PATH=/path/to/xla_cuda_plugin.so \
cargo run --release -p rxla-models --example gaussian_splatting_real -- \
  --splat /path/to/scene.splat --cameras /path/to/cameras.json \
  --camera 0 --width 160 --gaussians 4096 \
  --output gaussian-splatting-real.png
```
