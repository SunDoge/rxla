"""PyTorch eager/compile baseline for the SAM 2.1 Hiera-small trunk."""

import sys
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    checkpoint: Annotated[Path, Arg(help="official SAM 2.1 .pt checkpoint")]
    sam2_source: Annotated[Path, Arg(help="facebookresearch/sam2 checkout")]
    mode: Annotated[str, Arg("--mode")] = "compile"
    precision: Annotated[str, Arg("--precision")] = "f16"
    warmup: Annotated[int, Arg("--warmup")] = 2
    iterations: Annotated[int, Arg("--iterations")] = 10
    output_prefix: Annotated[Path | None, Arg("--output-prefix")] = None
    compare_prefix: Annotated[Path | None, Arg("--compare-prefix")] = None
    tap_blocks: Annotated[bool, Arg("--tap-blocks")] = False


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
    if args.tap_blocks and args.mode != "eager":
        raise ValueError("--tap-blocks requires --mode eager")
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
    block_outputs: list[torch.Tensor] = []
    if args.tap_blocks:
        for block in model.blocks:
            block.register_forward_hook(
                lambda _module, _inputs, output: block_outputs.append(output)
            )
    evaluate = model.forward
    if args.mode == "compile":
        evaluate = torch.compile(evaluate, fullgraph=True, dynamic=False)
    elif args.mode == "max-autotune":
        evaluate = torch.compile(evaluate, mode="max-autotune", fullgraph=True, dynamic=False)

    image = (
        (torch.arange(1024 * 1024 * 3, dtype=torch.float32) % 251 - 125.0)
        .div_(128.0)
        .reshape(1, 1024, 1024, 3)
        .permute(0, 3, 1, 2)
        .contiguous()
        .cuda()
    )
    autocast_dtype = {
        "f32": torch.float32,
        "bf16": torch.bfloat16,
        "f16": torch.float16,
    }[args.precision]
    torch.cuda.reset_peak_memory_stats()
    compile_started = time.perf_counter()
    block_outputs.clear()
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=autocast_dtype, enabled=args.precision != "f32"
    ):
        outputs = evaluate(image)
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - compile_started) * 1e3 if args.mode != "eager" else 0.0
    for _ in range(args.warmup):
        block_outputs.clear()
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=autocast_dtype, enabled=args.precision != "f32"
        ):
            outputs = evaluate(image)
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(args.iterations):
        block_outputs.clear()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=autocast_dtype, enabled=args.precision != "f32"
        ):
            outputs = evaluate(image)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e3)
    selected_outputs = block_outputs if args.tap_blocks else outputs
    labels = (
        [f"block{index}" for index in range(len(selected_outputs))]
        if args.tap_blocks
        else [f"stage{index}" for index in range(len(selected_outputs))]
    )
    checksum = sum(output.float().sum().item() for output in selected_outputs)
    host_outputs = [output.float().cpu().numpy() for output in selected_outputs]
    if args.output_prefix is not None:
        for label, output in zip(labels, host_outputs, strict=True):
            output.astype("<f4", copy=False).tofile(
                f"{args.output_prefix}.{label}.f32"
            )
    comparisons: list[str] = []
    if args.compare_prefix is not None:
        for label, actual in zip(labels, host_outputs, strict=True):
            expected = np.fromfile(
                f"{args.compare_prefix}.{label}.f32", dtype="<f4"
            ).reshape(actual.shape)
            difference = np.abs(actual - expected)
            denominator = np.maximum(np.abs(expected), 1e-6)
            cosine = float(
                np.dot(actual.ravel().astype(np.float64), expected.ravel())
                / (
                    np.linalg.norm(actual.ravel().astype(np.float64))
                    * np.linalg.norm(expected.ravel().astype(np.float64))
                )
            )
            comparisons.append(
                f"{label}:max_abs={float(difference.max()):.9g},"
                f"mean_abs={float(difference.mean()):.9g},"
                f"mean_rel={float((difference / denominator).mean()):.9g},"
                f"cosine={cosine:.12f}"
            )
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
        + (f" comparison={'|'.join(comparisons)}" if comparisons else "")
    )


if __name__ == "__main__":
    main()
