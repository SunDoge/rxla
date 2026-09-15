# Model benchmarks

The quantized Qwen3.5-0.8B text benchmark, converter and BF16 oracle are
documented in
[`docs/QWEN3.5-BENCHMARK.md`](../docs/QWEN3.5-BENCHMARK.md).

## Stable Diffusion

Measured results and reproduction notes are recorded in
[`docs/STABLE-DIFFUSION-BENCHMARK.md`](../docs/STABLE-DIFFUSION-BENCHMARK.md).

The first end-to-end model component is the official TAESD decoder for Stable
Diffusion 1.x/2.x. It consumes a `[1, 64, 64, 4]` latent and produces a
`[1, 512, 512, 3]` image. Both runners load the same Diffusers safetensors file
and generate the same deterministic input.

Download the approximately 9.8 MB checkpoint outside the repository:

```sh
mkdir -p "${XDG_CACHE_HOME:-$HOME/.cache}/rxla/models/taesd"
curl -L --fail \
  -o "${XDG_CACHE_HOME:-$HOME/.cache}/rxla/models/taesd/diffusion_pytorch_model.safetensors" \
  https://huggingface.co/madebyollin/taesd/resolve/main/diffusion_pytorch_model.safetensors
```

Run RXLA (the CUDA runtime library directories depend on the PJRT distribution):

```sh
LD_LIBRARY_PATH="$CUDA_RUNTIME_LIBS" cargo run -q -p rxla-checkpoint \
  --features model --example taesd_decode_bench -- \
  "$PJRT_CUDA_PLUGIN_PATH" "$TAESD_WEIGHTS" 64 5 20 rxla-taesd.f32
```

Run the PyTorch oracle through the project environment managed by `uv`:

```sh
uv run python benchmarks/taesd_torch.py "$TAESD_WEIGHTS" \
  --device cuda --latent 64 --warmup 5 --iterations 20 \
  --compare rxla-taesd.f32
```

September 15, 2026 snapshot on an RTX 5080, CUDA 13.4 and cuDNN 9.26:

| Runner | mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| RXLA/PJRT/XLA | 4.977 ms | 4.968 ms | 5.134 ms |
| PyTorch/cuDNN | 5.972 ms | 5.984 ms | 5.999 ms |

The maximum absolute output difference was `0.003212` and the mean absolute
difference was `0.0001712`. Timings cover synchronized device execution and
exclude the separately reported output download. They are a single-machine
snapshot, not a general performance claim. This benchmark covers the decoder,
not yet CLIP tokenization/text encoding, UNet denoising, scheduling, safety
checking, or image encoding.

## Conditional UNet

The second benchmark runs the real 304-tensor conditional UNet from
`bumblebee-testing/tiny-stable-diffusion`. RXLA uses NHWC internally while the
PyTorch oracle converts the identical deterministic sample to NCHW. Both receive
the same timestep (`501`) and `[1, 77, 32]` encoder hidden states.
The RXLA runner uses the IR-first `model` API: parameters are declared at their
use sites, loaded from safetensors by schema path, and bound to a Pliron-derived
StableHLO program without constructing legacy module parameter objects.

Download the model component outside the repository (the files can also be
obtained by any Hugging Face cache client):

```sh
mkdir -p "${XDG_CACHE_HOME:-$HOME/.cache}/rxla/models/tiny-stable-diffusion/unet"
curl -L --fail \
  -o "${XDG_CACHE_HOME:-$HOME/.cache}/rxla/models/tiny-stable-diffusion/unet/config.json" \
  https://huggingface.co/bumblebee-testing/tiny-stable-diffusion/resolve/main/unet/config.json
curl -L --fail \
  -o "${XDG_CACHE_HOME:-$HOME/.cache}/rxla/models/tiny-stable-diffusion/unet/diffusion_pytorch_model.safetensors" \
  https://huggingface.co/bumblebee-testing/tiny-stable-diffusion/resolve/main/unet/diffusion_pytorch_model.safetensors
```

Run both synchronized benchmarks and compare the final NHWC output:

```sh
LD_LIBRARY_PATH="$CUDA_RUNTIME_LIBS" cargo run -q -p rxla-checkpoint \
  --features model --example stable_diffusion_unet_bench -- \
  "$PJRT_CUDA_PLUGIN_PATH" "$TINY_SD_UNET_WEIGHTS" \
  64 5 20 rxla-unet.f32

uv run python benchmarks/stable_diffusion_unet_torch.py \
  "$TINY_SD_UNET_DIRECTORY" --spatial 64 --warmup 5 --iterations 20 \
  --compare rxla-unet.f32
```

September 15, 2026 snapshot on the same RTX 5080:

| Runner | mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| RXLA/PJRT/XLA | 2.567 ms | 2.543 ms | 2.783 ms |
| PyTorch | 3.877 ms | 3.849 ms | 4.019 ms |

The maximum absolute output difference was `0.002411` and the mean absolute
difference was `0.0001826`. Compilation (RXLA: 6.22 s), checkpoint loading and
the separately reported device-to-host copy are excluded.

A fresh 8×8 functional-API correctness run loaded all 304 tensors by schema
path and executed on the same RTX 5080. Against Diffusers through the project
`uv` environment, maximum/mean absolute differences were `0.000753164291` /
`0.000189408165`. RXLA compilation took `3.855 s`; the single un-warmed device
execution took `18.214 ms`, so this run is a correctness check rather than a
throughput comparison.

## Complete tiny Stable Diffusion pipeline

`stable_diffusion` exercises the complete inference path: CLIP BPE tokenization,
text encoding, classifier-free guidance, device-resident PNDM/PLMS history,
conditional UNet denoising, and VAE decoding. It uses the small
`bumblebee-testing/tiny-stable-diffusion` checkpoint, so its output is a
`128×128` image; it is an integration/performance target, not an SD 1.x quality
model. The initial latent is optionally written as little-endian f32 so the
PyTorch runner receives exactly the same noise.

The tensor module layer also exposes explicit `stable_diffusion_v1()` CLIP and
UNet presets and an `AutoencoderKlDecoderConfig::stable_diffusion()` VAE preset
for the standard SD 1.x architecture. The measured runner intentionally stays
on the small checkpoint until a full checkpoint benchmark is added.

Run RXLA and save that latent and the final NHWC image:

```sh
LD_LIBRARY_PATH="$CUDA_RUNTIME_LIBS" cargo run -q -p rxla-checkpoint \
  --features model --example stable_diffusion -- \
  "$PJRT_CUDA_PLUGIN_PATH" "$TINY_SD_DIRECTORY" \
  "a photo of an astronaut riding a horse" 20 7.5 2 5 \
  rxla-sd.ppm rxla-sd.f32 rxla-sd-latent.f32
```

Then use the same model, prompt, scheduler implementation and latent in the
PyTorch oracle:

```sh
uv run python benchmarks/stable_diffusion_torch.py \
  "$TINY_SD_DIRECTORY" rxla-sd-latent.f32 --steps 20 --guidance 7.5 \
  --warmup 2 --iterations 5 --compare rxla-sd.f32
```

September 15, 2026 snapshot on the RTX 5080 above, excluding RXLA's one-time
`13.65 s` compilation and both runners' model loading:

| Runner | mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| RXLA/PJRT/XLA | 88.463 ms | 88.947 ms | 89.651 ms |
| PyTorch | 90.032 ms | 90.075 ms | 91.161 ms |

After 20 denoising steps, the image maximum absolute difference was `0.07105`
and the mean absolute difference was `0.008708`. CUDA floating-point variation
from every denoising update compounds through the scheduler; the isolated CLIP,
UNet, and VAE component checks stay in the `10⁻⁶`–`10⁻³` range. These are
synchronized, single-machine measurements and exclude safety checking.

A fresh fixed-latent two-step CUDA comparison on the same RTX 5080 (one
iteration, no warmup) produced RXLA/PyTorch image maximum/mean absolute
differences of `0.00138485432` / `0.000176319154`. RXLA took `32.838 ms`
excluding its `13.621 s` compile; this is a short correctness gate, not a
throughput comparison because the un-warmed PyTorch invocation includes its
first-use overhead.

The same runner was also exercised against the ZML CPU PJRT plugin for two
steps. It completed in `818.6 ms` after `687.6 ms` compilation; using its fixed
seed, its final image differed from CUDA by maximum/mean absolute values of
`0.000879` / `0.000142`. This is a compatibility check, not a CPU performance
claim.
