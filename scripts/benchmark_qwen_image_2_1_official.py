"""Benchmark the official Diffusers Qwen-Image-2.1 generation core.

Uses cached prompt embeddings so this measures the same DiT + scheduler + VAE
boundary as RXLA's Rust example rather than the separate Qwen3-VL encoder.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Annotated, Literal

import torch
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from safetensors.torch import load_file
from transformers import Qwen3VLProcessor
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    model_root: Annotated[Path, Arg("--model-root")] = Path(
        ".cache/qwen-image-2.1"
    )
    conditioning: Annotated[Path, Arg("--conditioning")] = Path(
        ".cache/qwen-image-2.1/conditioning_chair.safetensors"
    )
    width: Annotated[int, Arg("--width")] = 512
    height: Annotated[int, Arg("--height")] = 512
    steps: Annotated[int, Arg("--steps")] = 4
    warmup: Annotated[int, Arg("--warmup")] = 1
    iterations: Annotated[int, Arg("--iterations")] = 3
    seed: Annotated[int, Arg("--seed")] = 42
    output: Annotated[Literal["latent", "image"], Arg("--output")] = "image"
    residency: Annotated[
        Literal["model-offload", "resident-transformer"], Arg("--residency")
    ] = "model-offload"
    kv_cache: Annotated[Literal["on", "off"], Arg("--kv-cache")] = "on"


def synchronize() -> None:
    torch.cuda.synchronize()


def main() -> None:
    args = Args.parse_args()
    if args.width <= 0 or args.height <= 0 or args.steps <= 0:
        raise ValueError("width, height and steps must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")

    load_start = time.perf_counter()
    transformer = QwenImage21Transformer2DModel.from_pretrained(
        args.model_root / "transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    vae = AutoencoderKLQwenImage21.from_pretrained(
        args.model_root / "vae",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_root / "scheduler"
    )
    processor = Qwen3VLProcessor.from_pretrained(args.model_root / "processor")
    pipe = QwenImage21Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=None,
        processor=processor,
        transformer=transformer,
    )
    if args.residency == "model-offload":
        # This is the memory-constrained path documented in the official README.
        pipe.enable_model_cpu_offload()
    else:
        if args.output != "latent":
            raise ValueError("resident-transformer supports latent output only")
        # Diagnostic compute-only path: the complete BF16 transformer narrowly
        # fits, while VAE remains on CPU and is skipped by latent output.
        pipe.transformer.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    load_seconds = time.perf_counter() - load_start

    prompt_embeds = load_file(args.conditioning)["prompt_embeds"].to(
        device="cuda", dtype=torch.bfloat16
    )
    prompt_embeds_mask = torch.ones(
        prompt_embeds.shape[:2], dtype=torch.long, device="cuda"
    )

    def generate() -> None:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        pipe(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            generator=generator,
            output_type="latent" if args.output == "latent" else "pt",
            use_kv_cache=args.kv_cache == "on",
        )

    for _ in range(args.warmup):
        generate()
        synchronize()

    samples: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.iterations):
        synchronize()
        start = time.perf_counter()
        generate()
        synchronize()
        samples.append(time.perf_counter() - start)

    report = {
        "backend": "official-diffusers",
        "device": torch.cuda.get_device_name(),
        "diffusers": __import__("diffusers").__version__,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "dtype": "bfloat16",
        "residency": args.residency,
        "cached_prompt_embeddings": True,
        "output": args.output,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "kv_cache": args.kv_cache,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "load_seconds": load_seconds,
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.fmean(samples),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
