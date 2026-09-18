"""Official PyTorch FLUX.2 Klein 4B benchmark on RXLA conditioning data."""

import sys
import time
from pathlib import Path
from typing import Annotated

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    flux2_source: Annotated[Path, Arg(help="black-forest-labs/flux2 checkout")]
    transformer: Annotated[Path, Arg(help="official transformer safetensors")]
    conditioning: Annotated[Path, Arg(help="flux2_prompt.py output")]
    warmup: Annotated[int, Arg("--warmup")] = 1
    iterations: Annotated[int, Arg("--iterations")] = 3


def percentile(samples: list[float], fraction: float) -> float:
    return sorted(samples)[round((len(samples) - 1) * fraction)]


def main() -> None:
    args = Args.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")
    sys.path.insert(0, str((args.flux2_source / "src").resolve()))
    from flux2.model import Flux2, Klein4BParams
    from flux2.sampling import denoise, get_schedule

    with torch.device("meta"):
        model = Flux2(Klein4BParams()).to(torch.bfloat16)
    state = load_file(args.transformer, device="cuda")
    model.load_state_dict(state, strict=True, assign=True)
    model = model.eval().cuda()
    del state

    values: dict[str, torch.Tensor] = {}
    with safe_open(args.conditioning, framework="pt", device="cpu") as checkpoint:
        for name in ("image", "image_ids", "context", "context_ids"):
            values[name] = checkpoint.get_tensor(name).cuda()
    image = values["image"].to(torch.bfloat16)
    image_ids = values["image_ids"]
    context = values["context"].to(torch.bfloat16)
    context_ids = values["context_ids"]
    schedule = get_schedule(4, image.shape[1])

    def evaluate() -> torch.Tensor:
        return denoise(
            model,
            image.clone(),
            image_ids,
            context,
            context_ids,
            schedule,
            guidance=1.0,
        )

    with torch.inference_mode():
        for _ in range(args.warmup):
            output = evaluate()
        torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            output = evaluate()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1e3)
    print(
        f"image_tokens={image.shape[1]} context_tokens={context.shape[1]} "
        f"warmup={args.warmup} iterations={args.iterations} "
        f"mean_ms={sum(samples) / len(samples):.3f} "
        f"p50_ms={percentile(samples, 0.5):.3f} "
        f"p95_ms={percentile(samples, 0.95):.3f} "
        f"checksum={output.float().sum().item():.9f} "
        f"peak_mib={torch.cuda.max_memory_allocated() / 2**20:.1f}"
    )


if __name__ == "__main__":
    main()
