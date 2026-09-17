"""Generate deterministic official-FLUX.2 tiny weights, inputs and output."""

import sys
from pathlib import Path
from typing import Annotated

import torch
from safetensors.torch import save_file
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    flux2_source: Annotated[Path, Arg(help="black-forest-labs/flux2 checkout")]
    output_dir: Annotated[Path, Arg(help="directory for oracle safetensors")]


def main() -> None:
    args = Args.parse_args()
    sys.path.insert(0, str((args.flux2_source / "src").resolve()))
    from flux2.model import Flux2, Klein4BParams

    torch.manual_seed(20260918)
    params = Klein4BParams(
        in_channels=8,
        context_in_dim=12,
        hidden_size=32,
        num_heads=4,
        depth=1,
        depth_single_blocks=1,
        axes_dim=[2, 2, 2, 2],
        theta=2000,
        mlp_ratio=2.0,
    )
    model = Flux2(params).float().eval()
    image = torch.randn(1, 3, 8)
    image_ids = torch.tensor(
        [[[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]],
        dtype=torch.float32,
    )
    timestep = torch.tensor([0.75], dtype=torch.float32)
    context = torch.randn(1, 2, 12)
    context_ids = torch.tensor(
        [[[0, 0, 0, 0], [0, 0, 0, 1]]], dtype=torch.float32
    )
    with torch.inference_mode():
        output = model(image, image_ids, timestep, context, context_ids, None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.detach().contiguous() for name, value in model.state_dict().items()},
        args.output_dir / "weights.safetensors",
    )
    save_file(
        {
            "image": image,
            "image_ids": image_ids,
            "timestep": timestep,
            "context": context,
            "context_ids": context_ids,
            "output": output.contiguous(),
        },
        args.output_dir / "oracle.safetensors",
    )
    print(
        f"parameters={sum(value.numel() for value in model.parameters())} "
        f"checksum={output.sum().item():.9f}"
    )


if __name__ == "__main__":
    main()
