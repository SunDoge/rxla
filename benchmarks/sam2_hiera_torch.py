"""PyTorch eager/compile baseline for the SAM 2.1 Hiera-small trunk."""

import sys
import time
from pathlib import Path
from typing import Annotated

import torch
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    checkpoint: Annotated[Path, Arg(help="official SAM 2.1 .pt checkpoint")]
    sam2_source: Annotated[Path, Arg(help="facebookresearch/sam2 checkout")]
    mode: Annotated[str, Arg("--mode")] = "compile"
    precision: Annotated[str, Arg("--precision")] = "f16"
    warmup: Annotated[int, Arg("--warmup")] = 2
    iterations: Annotated[int, Arg("--iterations")] = 10


def percentile(samples: list[float], fraction: float) -> float:
    return sorted(samples)[round((len(samples) - 1) * fraction)]


def main() -> None:
    args = Args.parse_args()
    if args.mode not in {"eager", "compile", "max-autotune"}:
        raise ValueError("mode must be eager, compile, or max-autotune")
    if args.precision not in {"f32", "bf16", "f16"}:
        raise ValueError("precision must be f32, bf16, or f16")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")
    sys.path.insert(0, str(args.sam2_source.resolve()))
    from sam2.modeling.backbones.hieradet import Hiera

    model = Hiera(
        embed_dim=96,
        num_heads=1,
        stages=(1, 2, 11, 2),
        global_att_blocks=(7, 10, 13),
        window_pos_embed_bkg_spatial_size=(7, 7),
    ).eval().cuda()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]
    prefix = "image_encoder.trunk."
    state = {name.removeprefix(prefix): value for name, value in checkpoint.items() if name.startswith(prefix)}
    model.load_state_dict(state, strict=True)
    evaluate = model.forward
    if args.mode == "compile":
        evaluate = torch.compile(evaluate, fullgraph=True, dynamic=False)
    elif args.mode == "max-autotune":
        evaluate = torch.compile(evaluate, mode="max-autotune", fullgraph=True, dynamic=False)

    image = torch.zeros((1, 3, 1024, 1024), device="cuda")
    autocast_dtype = {
        "f32": torch.float32,
        "bf16": torch.bfloat16,
        "f16": torch.float16,
    }[args.precision]
    torch.cuda.reset_peak_memory_stats()
    compile_started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=autocast_dtype, enabled=args.precision != "f32"
    ):
        outputs = evaluate(image)
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - compile_started) * 1e3 if args.mode != "eager" else 0.0
    for _ in range(args.warmup):
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=autocast_dtype, enabled=args.precision != "f32"
        ):
            outputs = evaluate(image)
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=autocast_dtype, enabled=args.precision != "f32"
        ):
            outputs = evaluate(image)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e3)
    checksum = sum(output.float().sum().item() for output in outputs)
    mean = sum(samples) / len(samples)
    print(
        f"backend=torch-cuda mode={args.mode} precision={args.precision} "
        f"model=sam2.1-hiera-small "
        f"batch=1 image=1024 compile_ms={compile_ms:.3f} warmup={args.warmup} "
        f"iterations={args.iterations} mean_ms={mean:.3f} "
        f"p50_ms={percentile(samples, 0.50):.3f} "
        f"p95_ms={percentile(samples, 0.95):.3f} "
        f"peak_memory_mb={torch.cuda.max_memory_allocated() / 1024**2:.1f} "
        f"checksum={checksum:.9f}"
    )


if __name__ == "__main__":
    main()
