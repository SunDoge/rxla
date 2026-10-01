"""Benchmark the official Diffusers Qwen-Image-2.1 generation core.

Uses cached prompt embeddings so this measures the same DiT + scheduler + VAE
boundary as RXLA's Rust example rather than the separate Qwen3-VL encoder.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Annotated, Literal

import torch
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from diffusers.models.autoencoders.vae import DecoderOutput
from safetensors import safe_open
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
    batch_size: Annotated[int, Arg("--batch-size")] = 1
    vae_batch_size: Annotated[int, Arg("--vae-batch-size")] = 4
    output: Annotated[Literal["latent", "image"], Arg("--output")] = "image"
    residency: Annotated[
        Literal["model-offload", "resident-transformer", "resident-pipeline"],
        Arg("--residency"),
    ] = "model-offload"
    kv_cache: Annotated[Literal["on", "off"], Arg("--kv-cache")] = "on"
    transformer_weights: Annotated[
        Literal["bf16", "rxla-w8"], Arg("--transformer-weights")
    ] = "bf16"
    save_image: Annotated[Path | None, Arg("--save-image")] = None


class GroupedW8Linear(torch.nn.Module):
    """Reference eager dequantization of RXLA's U8-offset grouped INT8 weights.

    This is a weight-matched diagnostic, not an optimized quantized GEMM kernel.
    """

    def __init__(self, out_features: int, in_features: int, group_size: int) -> None:
        super().__init__()
        self.group_size = group_size
        self.register_buffer(
            "weight", torch.empty(out_features, in_features, dtype=torch.uint8, device="meta")
        )
        self.register_buffer(
            "scale", torch.empty(out_features, in_features // group_size, device="meta")
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rows, columns = self.weight.shape
        weight = (self.weight.float() - 128).reshape(rows, -1, self.group_size)
        weight = (weight * self.scale.unsqueeze(-1)).reshape(rows, columns)
        return torch.nn.functional.linear(inputs, weight.to(inputs.dtype))


def load_w8_transformer(root: Path) -> QwenImage21Transformer2DModel:
    config = QwenImage21Transformer2DModel.load_config(root / "transformer")
    # Nonpersistent buffers (for example timestep frequencies) must be
    # initialized normally; they have no checkpoint entry to replace meta data.
    with init_empty_weights():
        model = QwenImage21Transformer2DModel.from_config(config)
        with safe_open(root / "transformer_w8.safetensors", framework="pt") as checkpoint:
            metadata = checkpoint.metadata() or {}
            if metadata.get("format") != "rxla-qwen-image-2.1-w8-v1":
                raise ValueError("unexpected grouped-W8 checkpoint format")
            group = int(metadata["group_size"])
            for key in checkpoint.keys():
                if not key.endswith(".scale"):
                    continue
                path = key.removesuffix(".scale")
                original = model.get_submodule(path)
                if not isinstance(original, torch.nn.Linear) or original.bias is not None:
                    raise ValueError(f"expected bias-free Linear at {path}")
                parent_path, _, name = path.rpartition(".")
                parent = model.get_submodule(parent_path) if parent_path else model
                setattr(
                    parent, name,
                    GroupedW8Linear(original.out_features, original.in_features, group),
                )
    with safe_open(root / "transformer_w8.safetensors", framework="pt") as checkpoint:
        state = {
            key: checkpoint.get_tensor(key) if key.endswith(".scale")
            or checkpoint.get_slice(key).get_dtype() == "U8"
            else checkpoint.get_tensor(key).to(torch.bfloat16)
            for key in checkpoint.keys()
        }
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


def synchronize() -> None:
    torch.cuda.synchronize()


def microbatch_vae(vae: AutoencoderKLQwenImage21, batch_size: int) -> None:
    decode = vae.decode

    def decode_batches(z: torch.Tensor, return_dict: bool = True):
        if z.shape[0] <= batch_size:
            return decode(z, return_dict=return_dict)
        decoded = torch.cat(
            [decode(chunk, return_dict=False)[0] for chunk in z.split(batch_size)],
            dim=0,
        )
        return DecoderOutput(sample=decoded) if return_dict else (decoded,)

    vae.decode = decode_batches


def main() -> None:
    args = Args.parse_args()
    if any(value <= 0 for value in (
        args.width, args.height, args.steps, args.batch_size, args.vae_batch_size,
    )):
        raise ValueError("width, height, steps and batch sizes must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")
    if args.save_image is not None and args.output != "image":
        raise ValueError("--save-image requires image output")

    load_start = time.perf_counter()
    if args.transformer_weights == "rxla-w8":
        transformer = load_w8_transformer(args.model_root)
    else:
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
    if args.output == "image" and args.batch_size > args.vae_batch_size:
        microbatch_vae(vae, args.vae_batch_size)
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
    elif args.residency == "resident-transformer":
        if args.output != "latent":
            raise ValueError("resident-transformer supports latent output only")
        # Diagnostic compute-only path: the complete BF16 transformer narrowly
        # fits, while VAE remains on CPU and is skipped by latent output.
        pipe.transformer.to("cuda")
    else:
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    load_seconds = time.perf_counter() - load_start

    prompt_embeds = load_file(args.conditioning)["prompt_embeds"].to(
        device="cuda", dtype=torch.bfloat16
    )
    prompt_embeds = prompt_embeds.repeat(args.batch_size, 1, 1)
    prompt_embeds_mask = torch.ones(
        prompt_embeds.shape[:2], dtype=torch.long, device="cuda"
    )

    def generate():
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        return pipe(
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
        print("warmup complete", file=sys.stderr, flush=True)

    samples: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.iterations):
        synchronize()
        start = time.perf_counter()
        result = generate()
        synchronize()
        samples.append(time.perf_counter() - start)
        print(f"sample={len(samples) - 1} seconds={samples[-1]:.3f}", file=sys.stderr, flush=True)

    report = {
        "backend": "official-diffusers",
        "device": torch.cuda.get_device_name(),
        "diffusers": __import__("diffusers").__version__,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "dtype": "bfloat16",
        "transformer_weights": args.transformer_weights,
        "quantized_linear": "eager-f32-dequant-bf16-gemm" if args.transformer_weights == "rxla-w8" else None,
        "residency": args.residency,
        "cached_prompt_embeddings": True,
        "output": args.output,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "vae_batch_size": args.vae_batch_size if args.output == "image" else None,
        "kv_cache": args.kv_cache,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "load_seconds": load_seconds,
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.fmean(samples),
        "median_images_per_second": args.batch_size / statistics.median(samples),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "allocator_config": os.environ.get(
            "PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        ),
        "timing_boundary": (
            "denoising only; GPU latent output"
            if args.output == "latent"
            else "denoising plus VAE; GPU output, excludes output download and PNG encoding"
        ),
    }
    print(json.dumps(report, indent=2))
    if args.save_image is not None:
        from PIL import Image
        pixels = result.images[0].detach().float().cpu().permute(1, 2, 0)
        pixels = (pixels.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        args.save_image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(args.save_image)


if __name__ == "__main__":
    main()
