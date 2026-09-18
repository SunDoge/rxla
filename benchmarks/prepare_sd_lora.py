import json
from pathlib import Path
from typing import Annotated

import torch
from diffusers import AutoencoderKL, DDPMScheduler
from PIL import Image
from safetensors.torch import save_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import center_crop, normalize, resize, to_tensor
from transformers import CLIPTextModel, CLIPTokenizer
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    model: Annotated[Path, Arg(metavar="MODEL")]
    image: Annotated[Path, Arg(metavar="IMAGE")]
    output: Annotated[Path, Arg(metavar="OUTPUT")]
    prompt: Annotated[str, Arg("--prompt")]
    image_size: Annotated[int, Arg("--image-size")] = 128
    timestep: Annotated[int, Arg("--timestep")] = 500
    seed: Annotated[int, Arg("--seed")] = 42
    device: Annotated[str, Arg("--device")] = "cuda"


def timestep_embedding(timestep: int, width: int, device: torch.device) -> torch.Tensor:
    half = width // 2
    indices = torch.arange(half, dtype=torch.float32, device=device)
    angles = torch.exp(
        -torch.log(torch.tensor(10_000.0, device=device)) * indices / half
    )
    angles = angles * timestep
    return torch.cat((angles.cos(), angles.sin())).unsqueeze(0)


def main() -> None:
    args = Args.parse_args()
    if args.image_size <= 0 or args.image_size % 2:
        raise ValueError("image size must be positive and even")
    if not 0 <= args.timestep < 1_000:
        raise ValueError("timestep must be in [0, 1000)")

    device = torch.device(args.device)
    tokenizer = CLIPTokenizer.from_pretrained(args.model / "tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model / "text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.model / "vae").to(device)
    scheduler = DDPMScheduler.from_pretrained(args.model / "scheduler")
    text_encoder.eval()
    vae.eval()

    image = Image.open(args.image).convert("RGB")
    image = resize(image, args.image_size, interpolation=InterpolationMode.BICUBIC)
    image = center_crop(image, [args.image_size, args.image_size])
    pixels = normalize(to_tensor(image), [0.5], [0.5]).unsqueeze(0).to(device)
    tokens = tokenizer(
        args.prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad():
        context = text_encoder(tokens).last_hidden_state.float()
        latents = vae.encode(pixels).latent_dist.sample(generator=generator)
        latents = latents.float() * vae.config.scaling_factor
        noise = torch.randn(latents.shape, generator=generator, device=device)
        timesteps = torch.tensor([args.timestep], device=device, dtype=torch.long)
        sample = scheduler.add_noise(latents, noise, timesteps)
        unet_config = json.loads((args.model / "unet" / "config.json").read_text())
        embedding = timestep_embedding(
            args.timestep, int(unet_config["block_out_channels"][0]), device
        )

    tensors = {
        "sample": sample.permute(0, 2, 3, 1).cpu().contiguous(),
        "target": noise.permute(0, 2, 3, 1).cpu().contiguous(),
        "timestep_embedding": embedding.cpu().contiguous(),
        "context": context.cpu().contiguous(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        args.output,
        metadata={
            "format": "rxla-sd-lora-batch-v1",
            "model": str(args.model),
            "image": str(args.image),
            "prompt": args.prompt,
            "timestep": str(args.timestep),
            "seed": str(args.seed),
        },
    )
    print(
        f"saved={args.output} sample={tuple(tensors['sample'].shape)} "
        f"context={tuple(tensors['context'].shape)}"
    )


if __name__ == "__main__":
    main()
