"""PyTorch BF16 baseline for RXLA's Qwen3.5 text-prefill benchmark."""
import time
from pathlib import Path
from typing import Annotated

import torch
import numpy as np
from transformers import AutoModelForImageTextToText
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    model: Annotated[Path, Arg(help="local official Qwen3.5 model directory")]
    sequence: Annotated[int, Arg("--sequence")] = 16
    warmup: Annotated[int, Arg("--warmup")] = 2
    iterations: Annotated[int, Arg("--iterations")] = 10
    compare: Annotated[Path | None, Arg("--compare")] = None


def percentile(samples: list[float], fraction: float) -> float:
    return sorted(samples)[round((len(samples) - 1) * fraction)]


def main() -> None:
    args = Args.parse_args()
    if args.sequence <= 0 or args.iterations <= 0 or args.warmup < 0:
        raise ValueError("sequence/iterations must be positive and warmup nonnegative")
    model = (
        AutoModelForImageTextToText.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16
        )
        .eval()
        .cuda()
    )
    tokens = (torch.arange(args.sequence, device="cuda") + 1000).reshape(1, -1)

    def evaluate() -> torch.Tensor:
        with torch.inference_mode():
            return model(input_ids=tokens, use_cache=False).logits

    for _ in range(args.warmup):
        evaluate()
    torch.cuda.synchronize()
    samples: list[float] = []
    output = None
    for _ in range(args.iterations):
        start = time.perf_counter()
        output = evaluate()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    assert output is not None
    mean = sum(samples) / len(samples)
    print(
        f"backend=torch-cuda precision=bf16 sequence={args.sequence} "
        f"warmup={args.warmup} iterations={args.iterations} mean_ms={mean:.3f} "
        f"p50_ms={percentile(samples, 0.50):.3f} "
        f"p95_ms={percentile(samples, 0.95):.3f} "
        f"tokens_per_second={args.sequence / (mean / 1e3):.3f} "
        f"checksum={output.float().sum().item():.9f}"
    )
    if args.compare is not None:
        expected = torch.from_numpy(
            np.fromfile(args.compare, dtype="<f4").reshape(output.shape)
        ).float()
        actual = output.detach().float().cpu()
        difference = (actual - expected).abs()
        top1_agreement = (actual.argmax(dim=-1) == expected.argmax(dim=-1)).float().mean()
        print(
            f"compare_max_abs={difference.max().item():.6f} "
            f"compare_mean_abs={difference.mean().item():.6f} "
            f"reference_mean_abs={actual.abs().mean().item():.6f} "
            f"top1_agreement={top1_agreement.item():.6f}"
        )


if __name__ == "__main__":
    main()
