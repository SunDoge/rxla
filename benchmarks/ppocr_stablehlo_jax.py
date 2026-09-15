from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import jax
import numpy as np
from jax._src import xla_bridge
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    fixture: Annotated[Path, Arg(help="directory exported by ppocr_detector")]
    warmup: Annotated[int, Arg("--warmup")] = 1
    iterations: Annotated[int, Arg("--iterations")] = 20


def load_arguments(directory: Path) -> list[jax.Array]:
    shapes = []
    for line in (directory / "arguments.txt").read_text().splitlines():
        shapes.append(tuple(int(dim) for dim in line.split(",") if dim))
    device = jax.devices("cpu")[0]
    arguments = []
    for index, shape in enumerate(shapes):
        host = np.fromfile(directory / f"arg-{index:04}.f32", dtype="<f4").reshape(shape)
        arguments.append(jax.device_put(host, device))
    return arguments


def percentile(samples: list[float], fraction: float) -> float:
    return sorted(samples)[round((len(samples) - 1) * fraction)]


def main() -> None:
    args = Args.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    backend = xla_bridge.get_backend("cpu")
    module = (args.fixture / "module.mlir").read_text()
    started = time.perf_counter()
    executable = backend.compile_and_load(module, backend.devices())
    compile_ms = (time.perf_counter() - started) * 1e3
    arguments = load_arguments(args.fixture)

    output = None
    for _ in range(args.warmup):
        output = executable.execute(arguments)[0]
        output.block_until_ready()
    samples = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        output = executable.execute(arguments)[0]
        output.block_until_ready()
        samples.append((time.perf_counter() - started) * 1e3)
    assert output is not None
    actual = np.asarray(output).reshape(-1)
    expected = np.fromfile(args.fixture / "expected.f32", dtype="<f4")
    max_error = float(np.max(np.abs(actual - expected)))
    print(
        f"jax={jax.__version__} platform={backend.platform_version} "
        f"compile_ms={compile_ms:.3f} iterations={args.iterations} "
        f"mean_ms={sum(samples) / len(samples):.3f} "
        f"p50_ms={percentile(samples, .5):.3f} "
        f"p95_ms={percentile(samples, .95):.3f} "
        f"max_abs_error={max_error:.9g} arguments={len(arguments)}"
    )


if __name__ == "__main__":
    main()
