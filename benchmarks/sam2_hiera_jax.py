"""Pure-JAX SAM 2.1 Hiera-small trunk benchmark using RXLA weights."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path
import time
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    weights: Annotated[Path, Arg(help="RXLA SAM 2.1 Hiera-small safetensors")]
    attention: Annotated[str, Arg("--attention", help="xla, cudnn, or hybrid")] = (
        "hybrid"
    )
    warmup: Annotated[int, Arg("--warmup")] = 2
    iterations: Annotated[int, Arg("--iterations")] = 10
    output_prefix: Annotated[Path | None, Arg("--output-prefix")] = None
    compare_prefix: Annotated[Path | None, Arg("--compare-prefix")] = None


def percentile(samples: list[float], fraction: float) -> float:
    return sorted(samples)[round((len(samples) - 1) * fraction)]


def load_weights(path: Path) -> dict[str, jax.Array]:
    prefix = "image_encoder.trunk."
    weights = {}
    with safe_open(path, framework="np") as checkpoint:
        for name in checkpoint.keys():
            if name.startswith(prefix):
                weights[name.removeprefix(prefix)] = jax.device_put(
                    checkpoint.get_tensor(name).astype(np.float16)
                )
    return weights


def linear(weights: dict[str, jax.Array], path: str, x: jax.Array) -> jax.Array:
    return x @ weights[f"{path}.weight"].T + weights[f"{path}.bias"]


def layer_norm(
    weights: dict[str, jax.Array], path: str, x: jax.Array
) -> jax.Array:
    value = x.astype(jnp.float32)
    mean = jnp.mean(value, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(value - mean), axis=-1, keepdims=True)
    normalized = ((value - mean) * jax.lax.rsqrt(variance + 1e-6)).astype(x.dtype)
    return normalized * weights[f"{path}.weight"] + weights[f"{path}.bias"]


def max_pool_2d(x: jax.Array) -> jax.Array:
    return jax.lax.reduce_window(
        x,
        jnp.array(-jnp.inf, x.dtype),
        jax.lax.max,
        window_dimensions=(1, 2, 2, 1),
        window_strides=(1, 2, 2, 1),
        padding="VALID",
    )


def cubic_weight(distance: float) -> float:
    distance = abs(distance)
    coefficient = -0.75
    if distance <= 1:
        return (coefficient + 2) * distance**3 - (coefficient + 3) * distance**2 + 1
    if distance < 2:
        return (
            coefficient * distance**3
            - 5 * coefficient * distance**2
            + 8 * coefficient * distance
            - 4 * coefficient
        )
    return 0


def resize_bicubic_2d(x: jax.Array, height: int, width: int) -> jax.Array:
    def resize_axis(value: jax.Array, axis: int, output: int) -> jax.Array:
        input_size = value.shape[axis]
        scale = input_size / output
        indices = [[] for _ in range(4)]
        weights = [[] for _ in range(4)]
        for coordinate in range(output):
            source = (coordinate + 0.5) * scale - 0.5
            base = int(np.floor(source))
            for tap in range(4):
                index = base + tap - 1
                indices[tap].append(min(max(index, 0), input_size - 1))
                weights[tap].append(cubic_weight(source - index))
        result = jnp.zeros_like(jnp.take(value, jnp.array(indices[0]), axis=axis))
        for tap in range(4):
            shape = [1] * value.ndim
            shape[axis] = output
            weight = jnp.array(weights[tap], dtype=value.dtype).reshape(shape)
            result = result + jnp.take(value, jnp.array(indices[tap]), axis=axis) * weight
        return result

    return resize_axis(resize_axis(x, 1, height), 2, width)


def window_partition(x: jax.Array, window: int) -> tuple[jax.Array, tuple[int, int]]:
    batch, height, width, channels = x.shape
    pad_height = (-height) % window
    pad_width = (-width) % window
    x = jnp.pad(x, ((0, 0), (0, pad_height), (0, pad_width), (0, 0)))
    padded_height, padded_width = height + pad_height, width + pad_width
    x = x.reshape(
        batch,
        padded_height // window,
        window,
        padded_width // window,
        window,
        channels,
    )
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, window, window, channels), (padded_height, padded_width)


def window_unpartition(
    x: jax.Array,
    window: int,
    padded: tuple[int, int],
    original: tuple[int, int],
) -> jax.Array:
    padded_height, padded_width = padded
    windows_per_image = padded_height // window * (padded_width // window)
    batch = x.shape[0] // windows_per_image
    x = x.reshape(
        batch,
        padded_height // window,
        padded_width // window,
        window,
        window,
        x.shape[-1],
    )
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(batch, padded_height, padded_width, x.shape[-1])[
        :, : original[0], : original[1]
    ]


def attention_implementation(mode: str, sequence: int) -> str:
    if mode == "hybrid":
        return "cudnn" if sequence >= 512 else "xla"
    return mode


def attention(
    weights: dict[str, jax.Array],
    path: str,
    x: jax.Array,
    dim_out: int,
    heads: int,
    q_stride: bool,
    attention_mode: str,
) -> jax.Array:
    batch, height, width, _ = x.shape
    head_dim = dim_out // heads
    qkv = linear(weights, f"{path}.qkv", x).reshape(
        batch, height * width, 3, heads, head_dim
    )
    query, key, value = (qkv[:, :, index] for index in range(3))
    if q_stride:
        query = max_pool_2d(query.reshape(batch, height, width, dim_out)).reshape(
            batch, height // 2 * (width // 2), heads, head_dim
        )
        query_height, query_width = height // 2, width // 2
    else:
        query_height, query_width = height, width
    implementation = attention_implementation(attention_mode, key.shape[1])
    output = jax.nn.dot_product_attention(
        query, key, value, implementation=implementation
    )
    output = output.reshape(batch, query_height, query_width, dim_out)
    return linear(weights, f"{path}.proj", output)


def block(
    weights: dict[str, jax.Array],
    index: int,
    x: jax.Array,
    dim: int,
    dim_out: int,
    heads: int,
    window: int,
    q_stride: bool,
    attention_mode: str,
) -> jax.Array:
    path = f"blocks.{index}"
    normalized = layer_norm(weights, f"{path}.norm1", x)
    shortcut = x
    if dim != dim_out:
        shortcut = max_pool_2d(linear(weights, f"{path}.proj", normalized))
    height, width = normalized.shape[1:3]
    if window:
        partitioned, padded = window_partition(normalized, window)
    else:
        partitioned, padded = normalized, (height, width)
    hidden = attention(
        weights,
        f"{path}.attn",
        partitioned,
        dim_out,
        heads,
        q_stride,
        attention_mode,
    )
    if window:
        output_window = window // 2 if q_stride else window
        if q_stride:
            output_padded = tuple(
                size + (-size) % output_window for size in shortcut.shape[1:3]
            )
        else:
            output_padded = padded
        hidden = window_unpartition(
            hidden, output_window, output_padded, shortcut.shape[1:3]
        )
    hidden = shortcut + hidden
    mlp = layer_norm(weights, f"{path}.norm2", hidden)
    mlp = linear(weights, f"{path}.mlp.layers.0", mlp)
    mlp = jax.nn.gelu(mlp, approximate=False)
    mlp = linear(weights, f"{path}.mlp.layers.1", mlp)
    return hidden + mlp


def model(
    weights: dict[str, jax.Array], image: jax.Array, attention_mode: str
) -> tuple[jax.Array, ...]:
    kernel = weights["patch_embed.proj.weight"].transpose(2, 3, 1, 0)
    hidden = jax.lax.conv_general_dilated(
        image,
        kernel,
        window_strides=(4, 4),
        padding=((3, 3), (3, 3)),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    ) + weights["patch_embed.proj.bias"]
    position = resize_bicubic_2d(
        weights["pos_embed"].transpose(0, 2, 3, 1),
        hidden.shape[1],
        hidden.shape[2],
    )
    window = weights["pos_embed_window"].transpose(0, 2, 3, 1)
    window = jnp.tile(window, (1, hidden.shape[1] // 8, hidden.shape[2] // 8, 1))
    hidden = hidden + position + window

    stage_ends = (0, 2, 13, 15)
    q_pool_blocks = (1, 3, 14)
    window_spec = (8, 4, 14, 7)
    global_blocks = (7, 10, 13)
    outputs = []
    dim, heads, stage = 96, 1, 0
    for index in range(16):
        window_stage = stage
        transition = index > 0 and index - 1 in stage_ends
        dim_out = dim * 2 if transition else dim
        if transition:
            heads *= 2
            stage += 1
        window_size = 0 if index in global_blocks else window_spec[window_stage]
        hidden = block(
            weights,
            index,
            hidden,
            dim,
            dim_out,
            heads,
            window_size,
            index in q_pool_blocks,
            attention_mode,
        )
        dim = dim_out
        if index in stage_ends:
            outputs.append(hidden.transpose(0, 3, 1, 2))
    return tuple(outputs)


def main() -> None:
    args = Args.parse_args()
    if args.attention not in {"xla", "cudnn", "hybrid"}:
        raise ValueError("attention must be xla, cudnn, or hybrid")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")
    weights = load_weights(args.weights)
    image = (
        ((jnp.arange(1024 * 1024 * 3) % 251 - 125) / 128)
        .reshape(1, 1024, 1024, 3)
        .astype(jnp.float16)
    )
    evaluate = jax.jit(lambda image: model(weights, image, args.attention))
    started = time.perf_counter()
    outputs = evaluate(image)
    jax.block_until_ready(outputs)
    compile_ms = (time.perf_counter() - started) * 1e3
    for _ in range(args.warmup):
        jax.block_until_ready(evaluate(image))
    samples = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        outputs = evaluate(image)
        jax.block_until_ready(outputs)
        samples.append((time.perf_counter() - started) * 1e3)
    host_outputs = [np.asarray(output, dtype=np.float32) for output in outputs]
    if args.output_prefix is not None:
        for index, output in enumerate(host_outputs):
            output.astype("<f4", copy=False).tofile(
                f"{args.output_prefix}.stage{index}.f32"
            )
    comparisons = []
    if args.compare_prefix is not None:
        for index, actual in enumerate(host_outputs):
            expected = np.fromfile(
                f"{args.compare_prefix}.stage{index}.f32", dtype="<f4"
            ).reshape(actual.shape)
            difference = np.abs(actual - expected)
            cosine = np.dot(actual.ravel().astype(np.float64), expected.ravel()) / (
                np.linalg.norm(actual.ravel().astype(np.float64))
                * np.linalg.norm(expected.ravel().astype(np.float64))
            )
            comparisons.append(
                f"stage{index}:max_abs={difference.max():.9g},"
                f"mean_abs={difference.mean():.9g},cosine={cosine:.12f}"
            )
    print(
        f"backend=jax-cuda attention={args.attention} precision=f16 "
        f"model=sam2.1-hiera-small compile_ms={compile_ms:.3f} "
        f"mean_ms={np.mean(samples):.3f} p50_ms={percentile(samples, 0.5):.3f} "
        f"p95_ms={percentile(samples, 0.95):.3f} "
        f"checksum={sum(output.astype(np.float64).sum() for output in host_outputs):.9f}"
        + (f" comparison={'|'.join(comparisons)}" if comparisons else "")
    )


if __name__ == "__main__":
    main()
