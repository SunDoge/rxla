"""Convert an official YOLOX PyTorch checkpoint to RXLA safetensors."""

from pathlib import Path
from typing import Annotated

import torch
from safetensors.torch import save_file
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    checkpoint: Annotated[Path, Arg(help="official YOLOX .pth checkpoint")]
    output: Annotated[Path, Arg(help="RXLA safetensors output")]


def main() -> None:
    args = Args.parse_args()
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = loaded.get("model", loaded)
    tensors: dict[str, torch.Tensor] = {}
    for source_name, value in state.items():
        if source_name.endswith(".num_batches_tracked"):
            continue
        name = source_name.removeprefix("model.")
        if name.endswith(".running_var"):
            name = name.removesuffix(".running_var") + ".running_variance"
        if name in tensors:
            raise ValueError(f"duplicate converted tensor name: {name}")
        tensors[name] = value.detach().float().contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        args.output,
        metadata={
            "format": "rxla-yolox-v1",
            "source": "Megvii-BaseDetection/YOLOX",
        },
    )
    print(f"wrote {len(tensors)} tensors to {args.output}")


if __name__ == "__main__":
    main()
