"""PyTorch oracle and benchmark for RXLA's complete tiny Stable Diffusion path."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    model: Annotated[Path, Arg(help="local Diffusers Stable Diffusion model directory")]
    latent: Annotated[Path, Arg(help="little-endian f32 NHWC initial latent from RXLA")]
    prompt: Annotated[str, Arg("--prompt")] = "a photo of an astronaut riding a horse"
    steps: Annotated[int, Arg("--steps")] = 20
    guidance: Annotated[float, Arg("--guidance")] = 7.5
    warmup: Annotated[int, Arg("--warmup")] = 1
    iterations: Annotated[int, Arg("--iterations")] = 3
    output: Annotated[Path | None, Arg("--output")] = None
    compare: Annotated[Path | None, Arg("--compare")] = None
    precision: Annotated[str, Arg("--precision")] = "f32"


def percentile(samples: list[float], fraction: float) -> float:
    return sorted(samples)[round((len(samples) - 1) * fraction)]


def main() -> None:
    args = Args.parse_args()
    if (
        not 2 <= args.steps <= 1000
        or not np.isfinite(args.guidance)
        or args.iterations <= 0
        or args.precision not in {"f32", "f16", "bf16"}
    ):
        raise ValueError(
            "steps must be in 2..=1000, guidance finite, iterations positive, "
            "and precision f32, f16, or bf16"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.precision != "f32" and device.type != "cuda":
        raise ValueError("reduced-precision benchmark requires CUDA")
    dtype = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.precision]
    tokenizer = CLIPTokenizer.from_pretrained(args.model / "tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(args.model / "text_encoder", local_files_only=True).to(device=device, dtype=dtype).eval()
    unet = UNet2DConditionModel.from_pretrained(args.model / "unet", local_files_only=True).to(device=device, dtype=dtype).eval()
    vae = AutoencoderKL.from_pretrained(args.model / "vae", local_files_only=True).to(device=device, dtype=dtype).eval()
    latents = torch.from_numpy(np.fromfile(args.latent, dtype="<f4").reshape(1, 64, 64, 4))
    latents = latents.permute(0, 3, 1, 2).contiguous().to(device=device, dtype=dtype)
    ids = tokenizer(
        ["", args.prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    ).input_ids.to(device)
    scheduler = PNDMScheduler.from_pretrained(args.model / "scheduler", local_files_only=True)
    scheduler.set_timesteps(args.steps, device=device)

    def evaluate() -> tuple[torch.Tensor, float, float, float]:
        with torch.inference_mode():
            start = time.perf_counter()
            context = text_encoder(ids).last_hidden_state
            if device.type == "cuda":
                torch.cuda.synchronize()
            clip_ms = (time.perf_counter() - start) * 1e3
            current = latents.clone()
            start = time.perf_counter()
            for timestep in scheduler.timesteps:
                predicted = unet(
                    torch.cat((current, current)), timestep, encoder_hidden_states=context
                ).sample
                unconditional, conditional = predicted.chunk(2)
                guided = unconditional + args.guidance * (conditional - unconditional)
                current = scheduler.step(guided, timestep, current).prev_sample
            if device.type == "cuda":
                torch.cuda.synchronize()
            denoise_ms = (time.perf_counter() - start) * 1e3
            start = time.perf_counter()
            image = (vae.decode(current / vae.config.scaling_factor).sample / 2 + 0.5).clamp(0, 1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            return image, clip_ms, denoise_ms, (time.perf_counter() - start) * 1e3

    for _ in range(args.warmup):
        evaluate()
    totals: list[float] = []
    clip_times: list[float] = []
    denoise_times: list[float] = []
    vae_times: list[float] = []
    result = None
    for _ in range(args.iterations):
        start = time.perf_counter()
        result, clip_ms, denoise_ms, vae_ms = evaluate()
        totals.append((time.perf_counter() - start) * 1e3)
        clip_times.append(clip_ms)
        denoise_times.append(denoise_ms)
        vae_times.append(vae_ms)
    assert result is not None
    image = result.float().permute(0, 2, 3, 1).contiguous().cpu().numpy().astype("<f4", copy=False)
    if args.output is not None:
        image.tofile(args.output)
    comparison = ""
    if args.compare is not None:
        expected = np.fromfile(args.compare, dtype="<f4").reshape(image.shape)
        difference = np.abs(image - expected)
        comparison = f" max_abs_diff={difference.max():.9g} mean_abs_diff={difference.mean():.9g}"
    print(
        f"backend=pytorch device={device} precision={args.precision} sdpa=true "
        f"steps={args.steps} warmup={args.warmup} "
        f"iterations={args.iterations} mean_ms={np.mean(totals):.3f} "
        f"p50_ms={percentile(totals, .5):.3f} p95_ms={percentile(totals, .95):.3f} "
        f"clip_mean_ms={np.mean(clip_times):.3f} denoise_mean_ms={np.mean(denoise_times):.3f} "
        f"vae_mean_ms={np.mean(vae_times):.3f} checksum={image.astype(np.float64).sum():.9f}{comparison}"
    )


if __name__ == "__main__":
    main()
