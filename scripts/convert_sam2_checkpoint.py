"""Convert an official SAM 2 PyTorch checkpoint to RXLA safetensors."""

from pathlib import Path
from typing import Annotated

import torch
from safetensors.torch import save_file
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    checkpoint: Annotated[Path, Arg(help="official SAM 2 .pt checkpoint")]
    output: Annotated[Path, Arg(help="output safetensors file")]


def main() -> None:
    args = Args.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint does not contain a nonempty model state_dict")
    tensors = {
        name: tensor.detach().contiguous()
        for name, tensor in state.items()
        if isinstance(name, str) and isinstance(tensor, torch.Tensor)
    }
    if len(tensors) != len(state):
        raise ValueError("checkpoint model contains non-tensor entries")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        args.output,
        metadata={
            "format": "rxla-sam2",
            "source": "facebookresearch/sam2",
        },
    )
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"wrote {len(tensors)} tensors ({size_mb:.1f} MiB) to {args.output}")


if __name__ == "__main__":
    main()
