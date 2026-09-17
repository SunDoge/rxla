"""Measure isolated Hiera attention shapes with JAX or PyTorch."""

from typing import Annotated

import numpy as np
from typed_args import Arg, TypedArgs


SHAPES = (
    (1, 256, 2, 64),
    (25, 196, 4, 96),
    (1024, 64, 1, 96),
    (1024, 16, 2, 96),
    (25, 49, 8, 96),
)


class Args(TypedArgs):
    backend: Annotated[str, Arg(help="jax or torch")]
    iterations: Annotated[int, Arg("--iterations")] = 100


def run_jax(iterations: int) -> None:
    import jax
    import jax.numpy as jnp
    from jax.experimental.mosaic.gpu import profiler

    for shape in SHAPES:
        keys = jax.random.split(jax.random.key(sum(shape)), 3)
        q, k, v = (
            jax.random.normal(key, shape, dtype=jnp.float16) for key in keys
        )
        measurements = []
        for implementation in ("cudnn", "xla"):
            function = jax.jit(
                lambda q, k, v, implementation=implementation: (
                    jax.nn.dot_product_attention(
                        q, k, v, implementation=implementation
                    )
                )
            )
            output, samples = profiler.measure(function, iterations=iterations)(q, k, v)
            output.block_until_ready()
            measurements.append(f"{implementation}={np.median(samples):.6f}ms")
        print(f"shape={shape} {' '.join(measurements)}")


def run_torch(iterations: int) -> None:
    import torch
    from torch.nn.functional import scaled_dot_product_attention

    for batch, sequence, heads, dimension in SHAPES:
        shape = (batch, heads, sequence, dimension)
        q, k, v = (
            torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3)
        )
        samples = []
        with torch.inference_mode(), torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            for _ in range(10):
                scaled_dot_product_attention(q, k, v)
            for _ in range(iterations):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                scaled_dot_product_attention(q, k, v)
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end))
        original_shape = (batch, sequence, heads, dimension)
        print(f"shape={original_shape} flash={np.median(samples):.6f}ms")


def main() -> None:
    args = Args.parse_args()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    match args.backend:
        case "jax":
            run_jax(args.iterations)
        case "torch":
            run_torch(args.iterations)
        case _:
            raise ValueError("backend must be jax or torch")


if __name__ == "__main__":
    main()
