"""Generate an oracle from Diffusers' official Qwen-Image-2.1 VAE source."""

import importlib.util
import sys
from pathlib import Path
from typing import Annotated

import diffusers.models.autoencoders.vae as vae_module
import torch
from safetensors.torch import load_file, save_file
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    source: Annotated[Path, Arg(help="official autoencoder_kl_qwenimage21.py")]
    weights: Annotated[Path, Arg(help="official VAE safetensors")]
    output: Annotated[Path, Arg(help="oracle safetensors output")]


def load_official_class(source: Path):
    # Diffusers 0.35 lacks the method-only mixin added on main. The Qwen class
    # does not need those convenience methods for direct construction/decode.
    if not hasattr(vae_module, "AutoencoderMixin"):
        vae_module.AutoencoderMixin = type("AutoencoderMixin", (), {})
    module_name = "diffusers.models.autoencoders.autoencoder_kl_qwenimage21_oracle"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official source {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.AutoencoderKLQwenImage21


def main() -> None:
    args = Args.parse_args()
    cls = load_official_class(args.source)
    model = cls(
        base_dim=96,
        decoder_base_dim=144,
        dim_mult=[1, 2, 4, 8, 8],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True, True],
        dropout=0.0,
        z_dim=64,
        is_residual=True,
        in_channels=4,
        out_channels=4,
        patch_size=None,
        scale_factor_temporal=8,
        scale_factor_spatial=16,
    ).eval()
    missing, unexpected = model.load_state_dict(load_file(args.weights), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    generator = torch.Generator().manual_seed(20260929)
    latent = torch.randn(1, 64, 1, 1, 1, generator=generator)
    with torch.inference_mode():
        output = model.decode(latent).sample.squeeze(2).permute(0, 2, 3, 1).contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "latent": latent.squeeze(2).permute(0, 2, 3, 1).reshape(1, 1, 64).contiguous(),
            "output": output,
        },
        args.output,
    )
    print(
        f"shape={tuple(output.shape)} checksum={output.sum().item():.9f} "
        f"minimum={output.min().item():.9f} maximum={output.max().item():.9f}"
    )


if __name__ == "__main__":
    main()
