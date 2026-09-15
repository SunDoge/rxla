from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from diffusers import UNet2DConditionModel
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    model: Annotated[Path, Arg(help="directory containing UNet config and weights")]
    spatial: Annotated[int, Arg("--spatial")] = 64
    warmup: Annotated[int, Arg("--warmup")] = 2
    iterations: Annotated[int, Arg("--iterations")] = 10
    output: Annotated[Path | None, Arg("--output")] = None
    compare: Annotated[Path | None, Arg("--compare")] = None


def values(count: int, modulus: int, scale: float) -> torch.Tensor:
    return (torch.arange(count, dtype=torch.float32) % modulus - modulus // 2) / scale


def percentile(samples: list[float], fraction: float) -> float:
    return sorted(samples)[round((len(samples) - 1) * fraction)]


def main() -> None:
    args = Args.parse_args()
    if args.spatial <= 0 or args.spatial % 2 or args.iterations <= 0:
        raise ValueError("spatial must be positive/even and iterations must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2DConditionModel.from_pretrained(args.model, local_files_only=True).to(device).eval()
    sample = values(args.spatial * args.spatial * 4, 251, 64.0).reshape(
        1, args.spatial, args.spatial, 4
    ).permute(0, 3, 1, 2).contiguous().to(device)
    context = values(77 * 32, 241, 128.0).reshape(1, 77, 32).to(device)
    timestep = torch.tensor(501, device=device)

    def evaluate() -> torch.Tensor:
        with torch.inference_mode():
            result = model(sample, timestep, encoder_hidden_states=context).sample
        if device.type == "cuda":
            torch.cuda.synchronize()
        return result

    for _ in range(args.warmup):
        evaluate()
    samples: list[float] = []
    result = None
    for _ in range(args.iterations):
        start = time.perf_counter()
        result = evaluate()
        samples.append((time.perf_counter() - start) * 1e3)
    assert result is not None
    download_start = time.perf_counter()
    output = result.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    download_ms = (time.perf_counter() - download_start) * 1e3
    if args.output is not None:
        output.astype("<f4", copy=False).tofile(args.output)
    comparison = ""
    if args.compare is not None:
        expected = np.fromfile(args.compare, dtype="<f4").reshape(output.shape)
        difference = np.abs(output - expected)
        comparison = (
            f" max_abs_diff={float(difference.max()):.9g}"
            f" mean_abs_diff={float(difference.mean()):.9g}"
        )
    print(
        f"backend=pytorch device={device} spatial={args.spatial} "
        f"parameters={len(model.state_dict())} warmup={args.warmup} iterations={args.iterations} "
        f"mean_ms={sum(samples) / len(samples):.3f} p50_ms={percentile(samples, .5):.3f} "
        f"p95_ms={percentile(samples, .95):.3f} output_download_ms={download_ms:.3f} "
        f"checksum={float(output.astype(np.float64).sum()):.9f}{comparison}"
    )


if __name__ == "__main__":
    main()
