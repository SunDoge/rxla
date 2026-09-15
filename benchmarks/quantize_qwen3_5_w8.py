"""Convert official Qwen3.5-0.8B text weights to RXLA grouped symmetric W8."""
from pathlib import Path
from typing import Annotated

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    source: Annotated[Path, Arg(help="official model safetensors file")]
    output: Annotated[Path, Arg(help="RXLA W8 safetensors output")]
    group_size: Annotated[int, Arg("--group-size")] = 128


def main() -> None:
    args = Args.parse_args()
    if args.group_size <= 0:
        raise ValueError("group size must be positive")
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(args.source, framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            if not name.startswith("model.language_model."):
                continue
            value = checkpoint.get_tensor(name)
            if value.ndim == 2:
                rows, columns = value.shape
                if columns % args.group_size:
                    raise ValueError(
                        f"{name}: width {columns} is not divisible by {args.group_size}"
                    )
                grouped = value.float().reshape(rows, -1, args.group_size)
                scale = grouped.abs().amax(dim=2).clamp_min(1e-12) / 127.0
                encoded = (
                    (grouped / scale.unsqueeze(2))
                    .round()
                    .clamp(-127, 127)
                    .add(128)
                    .to(torch.uint8)
                    .reshape(rows, columns)
                )
                tensors[name] = encoded.contiguous()
                tensors[name.removesuffix(".weight") + ".scale"] = scale.contiguous()
            else:
                tensors[name] = value.float().contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        args.output,
        metadata={
            "format": "rxla-qwen3.5-w8-v1",
            "group_size": str(args.group_size),
            "source": "Qwen/Qwen3.5-0.8B",
        },
    )


if __name__ == "__main__":
    main()
