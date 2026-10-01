"""Measure persistent RXLA Qwen generation and sample total GPU memory usage."""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Annotated

from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    binary: Annotated[Path, Arg("--binary")] = Path(
        "target/release/examples/qwen_image_2_1_generate"
    )
    plugin: Annotated[Path, Arg("--plugin")] = Path(
        ".venv/lib/python3.14/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so"
    )
    model_root: Annotated[Path, Arg("--model-root")] = Path(".cache/qwen-image-2.1")
    output_dir: Annotated[Path, Arg("--output-dir")] = Path(
        ".cache/qwen-image-2.1/benchmark-40steps"
    )
    batches: Annotated[str, Arg("--batches")] = "1,8"
    steps: Annotated[int, Arg("--steps")] = 40
    warmup: Annotated[int, Arg("--warmup")] = 1
    iterations: Annotated[int, Arg("--iterations")] = 3
    width: Annotated[int, Arg("--width")] = 512
    height: Annotated[int, Arg("--height")] = 512


def gpu_memory_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.strip())


def seconds(value: str) -> float:
    match = re.fullmatch(r"([\d.]+)(ns|µs|ms|s)", value)
    if match is None:
        raise ValueError(f"unexpected Rust duration {value!r}")
    return float(match[1]) * {"ns": 1e-9, "µs": 1e-6, "ms": 1e-3, "s": 1}[match[2]]


def benchmark(args: Args, batch: int) -> dict:
    root = args.model_root
    command = [
        str(args.binary.resolve()), str(args.plugin),
        str(root / "transformer_w8.safetensors"),
        str(root / "vae/diffusion_pytorch_model.safetensors"),
        str(root / "vae_config.json"), str(root / "conditioning_chair.safetensors"),
        str(args.output_dir / f"batch{batch}.png"),
        "--width", str(args.width), "--height", str(args.height),
        "--steps", str(args.steps), "--batch-size", str(batch),
        "--vae-batch-size", "4", "--requests", str(args.warmup + args.iterations),
        "--memory-fraction", "0.9", "--preallocate", "false",
        "--cache-dir", str(root / "compile-cache"),
    ]
    baseline = gpu_memory_mib()
    samples = [baseline]
    sampler_errors: list[str] = []
    stop = threading.Event()

    def sample_memory() -> None:
        while not stop.is_set():
            try:
                samples.append(gpu_memory_mib())
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                sampler_errors.append(str(error))
            stop.wait(0.2)

    sampler = threading.Thread(target=sample_memory)
    start = time.perf_counter()
    with (args.output_dir / f"batch{batch}.log").open("w") as log:
        sampler.start()
        try:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        finally:
            stop.set()
            sampler.join()
    log = (args.output_dir / f"batch{batch}.log").read_text()
    requests = [
        {"request": int(index), "dit_seconds": seconds(dit),
         "vae_seconds": seconds(vae), "total_seconds": seconds(total)}
        for index, dit, vae, total in re.findall(
            r"request=(\d+) denoise=(\S+) vae=(\S+) total=(\S+)", log
        )
    ]
    steady = requests[args.warmup:]
    report = {
        "backend": "rxla", "weight_format": "grouped-w8", "compute_dtype": "bf16",
        "cached_prompt_embeddings": True, "width": args.width, "height": args.height,
        "steps": args.steps, "batch_size": batch, "warmup": args.warmup,
        "iterations": args.iterations, "command": command,
        "returncode": completed.returncode, "process_seconds": time.perf_counter() - start,
        "requests": requests, "gpu_baseline_mib": baseline,
        "sampled_gpu_peak_mib": max(samples),
        "sampled_gpu_peak_above_baseline_mib": max(samples) - baseline,
        "memory_sampling_errors": sampler_errors,
        "timing_boundary": "denoising plus VAE and output download; excludes PNG encoding",
    }
    if completed.returncode == 0 and len(steady) == args.iterations:
        median = statistics.median(item["total_seconds"] for item in steady)
        report.update(
            median_seconds=median, median_images_per_second=batch / median,
            median_dit_seconds=statistics.median(item["dit_seconds"] for item in steady),
            median_vae_seconds=statistics.median(item["vae_seconds"] for item in steady),
        )
    else:
        report["error"] = "generation failed or request timing samples are incomplete; see log"
    return report


def main() -> None:
    args = Args.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    if any(value <= 0 for value in [*batches, args.steps, args.iterations, args.width, args.height]):
        raise ValueError("batch sizes, steps, iterations and image dimensions must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for batch in batches:
        report = benchmark(args, batch)
        reports.append(report)
        (args.output_dir / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if "error" in report:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
