"""Prepare official Qwen3 FLUX.2 context and deterministic image noise."""

from pathlib import Path
from typing import Annotated

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    prompt: Annotated[str, Arg(help="text prompt")]
    output: Annotated[Path, Arg(help="output safetensors file")]
    height: Annotated[int, Arg("--height")] = 512
    width: Annotated[int, Arg("--width")] = 512
    seed: Annotated[int, Arg("--seed")] = 42


def position_ids(length: int, image_height: int | None = None, image_width: int | None = None) -> torch.Tensor:
    if image_height is None or image_width is None:
        return torch.stack(
            (
                torch.zeros(length),
                torch.zeros(length),
                torch.zeros(length),
                torch.arange(length),
            ),
            dim=1,
        ).unsqueeze(0)
    h = torch.arange(image_height).repeat_interleave(image_width)
    w = torch.arange(image_width).repeat(image_height)
    return torch.stack((torch.zeros(length), h, w, torch.zeros(length)), dim=1).unsqueeze(0)


def main() -> None:
    args = Args.parse_args()
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        raise ValueError("height and width must be positive multiples of 16")
    model_id = "Qwen/Qwen3-4B-FP8"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=None,
        device_map="cuda",
    ).eval()
    with torch.inference_mode():
        output = model(
            input_ids=encoded["input_ids"].cuda(),
            attention_mask=encoded["attention_mask"].cuda(),
            output_hidden_states=True,
            use_cache=False,
        )
        context = torch.cat([output.hidden_states[index] for index in (9, 18, 27)], dim=2)

    latent_height = args.height // 16
    latent_width = args.width // 16
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    noise = torch.randn(
        (1, 128, latent_height, latent_width),
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    ).permute(0, 2, 3, 1).reshape(1, latent_height * latent_width, 128)
    context = context.float().cpu().contiguous()
    noise = noise.float().cpu().contiguous()
    tensors = {
        "context": context,
        "context_ids": position_ids(context.shape[1]),
        "image": noise,
        "image_ids": position_ids(
            latent_height * latent_width, latent_height, latent_width
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        args.output,
        metadata={
            "model": model_id,
            "prompt": args.prompt,
            "height": str(args.height),
            "width": str(args.width),
            "seed": str(args.seed),
        },
    )
    print(
        f"context={tuple(context.shape)} image={tuple(noise.shape)} "
        f"context_checksum={context.sum().item():.9f} noise_checksum={noise.sum().item():.9f}"
    )


if __name__ == "__main__":
    main()
