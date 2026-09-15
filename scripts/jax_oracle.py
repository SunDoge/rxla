"""Emit reproducible JAX values and StableHLO for rxla differential tests."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
from typed_args import TypedArgs

jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp  # noqa: E402


@dataclass(frozen=True)
class Case:
    inputs: tuple[Any, ...]
    function: Callable[..., Any]


def broadcast(input_: jax.Array) -> jax.Array:
    bias = jnp.array([1.0, 2.0], dtype=jnp.float32)
    return jnp.broadcast_to(jnp.reshape(input_ + bias, (1, 2)), (3, 2))


def elementwise(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    return (lhs + rhs) * rhs


def matmul_reduce(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    return jnp.sum(lhs @ rhs, axis=1)


def attention(query: jax.Array, key: jax.Array, value: jax.Array) -> jax.Array:
    scores = query @ jnp.swapaxes(key, -2, -1)
    weights = jax.nn.softmax(scores * jnp.float32(0.0), axis=-1)
    return weights @ value


def unet_residual(
    input_: jax.Array, first_kernel: jax.Array, second_kernel: jax.Array
) -> jax.Array:
    def convolution(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        return jax.lax.conv_general_dilated(
            lhs,
            rhs,
            window_strides=(1, 1),
            padding=((0, 0), (0, 0)),
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )

    hidden = jnp.transpose(convolution(input_, first_kernel), (0, 3, 1, 2))
    grouped = jnp.reshape(hidden, (1, 1, 2, 2, 2))
    axes = (2, 3, 4)
    centered = grouped - jnp.mean(grouped, axis=axes, keepdims=True)
    normalized = centered * jax.lax.rsqrt(
        jnp.mean(centered * centered, axis=axes, keepdims=True)
        + jnp.float32(1e-5)
    )
    hidden = jnp.transpose(jnp.reshape(normalized, hidden.shape), (0, 2, 3, 1))
    hidden = hidden * jax.nn.sigmoid(hidden)
    return convolution(hidden, second_kernel) + input_


def convolution(input_: jax.Array, kernel: jax.Array) -> jax.Array:
    return jax.lax.conv_general_dilated(
        input_,
        kernel,
        window_strides=(2, 1),
        padding=((1, 0), (2, 1)),
        rhs_dilation=(1, 2),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )


CASES = {
    "attention": Case(
        (
            jnp.array([[1.0, 2.0]], dtype=jnp.float32),
            jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
            jnp.array([[3.0], [6.0]], dtype=jnp.float32),
        ),
        attention,
    ),
    "broadcast": Case((jnp.array([3.0, 4.0], dtype=jnp.float32),), broadcast),
    "elementwise": Case(
        (
            jnp.array([1.0, 2.0], dtype=jnp.float32),
            jnp.array([3.0, 4.0], dtype=jnp.float32),
        ),
        elementwise,
    ),
    "matmul-reduce": Case(
        (
            jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32),
            jnp.array(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32
            ),
        ),
        matmul_reduce,
    ),
    "unet-residual": Case(
        (
            jnp.arange(1, 9, dtype=jnp.float32).reshape(1, 2, 2, 2),
            jnp.eye(2, dtype=jnp.float32).reshape(1, 1, 2, 2),
            jnp.eye(2, dtype=jnp.float32).reshape(1, 1, 2, 2),
        ),
        unet_residual,
    ),
    "convolution": Case(
        (
            jnp.arange(1 * 4 * 5 * 2, dtype=jnp.float32).reshape(1, 4, 5, 2),
            jnp.arange(2 * 3 * 2 * 3, dtype=jnp.float32).reshape(2, 3, 2, 3)
            / 10.0,
        ),
        convolution,
    ),
}


class Args(TypedArgs):
    """Generate a deterministic JAX oracle case."""

    case: Literal[
        "attention",
        "broadcast",
        "convolution",
        "elementwise",
        "matmul-reduce",
        "unet-residual",
    ]
    """Reference computation to evaluate."""

    emit: Literal["json", "stablehlo", "output"] = "json"
    """Representation written to standard output."""


def main() -> None:
    args = Args.parse_args()
    case = CASES[args.case]
    lowered = jax.jit(case.function).lower(*case.inputs)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    output = jax.device_get(case.function(*case.inputs)).tolist()
    if args.emit == "stablehlo":
        print(stablehlo)
    elif args.emit == "output":
        print(json.dumps(output, allow_nan=False, separators=(",", ":")))
    else:
        print(
            json.dumps(
                {
                    "case": args.case,
                    "inputs": [jax.device_get(value).tolist() for value in case.inputs],
                    "output": output,
                    "stablehlo": stablehlo,
                    "jax_version": jax.__version__,
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
