"""Convert sharded official Qwen-Image-2.1 DiT weights to grouped W8."""

import json
from pathlib import Path
from typing import Annotated

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    index: Annotated[Path, Arg(help="official safetensors index JSON")]
    output: Annotated[Path, Arg(help="RXLA W8 safetensors output")]
    group_size: Annotated[int, Arg("--group-size")] = 64


def main() -> None:
    args = Args.parse_args()
    if args.group_size <= 0:
        raise ValueError("group size must be positive")
    index = json.loads(args.index.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        path = args.index.parent / shard
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                value = checkpoint.get_tensor(name)
                if value.ndim != 2:
                    tensors[name] = value.contiguous()
                    continue
                rows, columns = value.shape
                if columns % args.group_size:
                    raise ValueError(
                        f"{name}: width {columns} is not divisible by {args.group_size}"
                    )
                grouped = value.float().reshape(rows, -1, args.group_size)
                scale = grouped.abs().amax(2).clamp_min(1e-12) / 127
                tensors[name] = (
                    (grouped / scale.unsqueeze(2))
                    .round()
                    .clamp(-127, 127)
                    .add(128)
                    .to(torch.uint8)
                    .reshape(rows, columns)
                    .contiguous()
                )
                tensors[name.removesuffix(".weight") + ".scale"] = scale.contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        args.output,
        metadata={
            "format": "rxla-qwen-image-2.1-w8-v1",
            "group_size": str(args.group_size),
            "source": "Qwen/Qwen-Image-2.1",
        },
    )


if __name__ == "__main__":
    main()
