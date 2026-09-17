"""Generate and validate a minimal offline Mosaic GPU StableHLO artifact."""

from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
import jax.experimental.pallas.mosaic_gpu as plgpu
from typed_args import Arg, TypedArgs


class Args(TypedArgs):
    output: Annotated[Path, Arg(help="output StableHLO text path")]


@jax.jit
def add(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    def kernel(lhs_ref, rhs_ref, output_ref):
        output_ref[...] = plgpu.load(lhs_ref, optimized=False) + plgpu.load(
            rhs_ref, optimized=False
        )

    return plgpu.kernel(kernel, out_type=jax.typeof(lhs))(lhs, rhs)


def main() -> None:
    args = Args.parse_args()
    lhs = jnp.arange(512, dtype=jnp.float32).reshape(64, 8)
    rhs = lhs + 1
    actual = add(lhs, rhs)
    if not bool(jnp.array_equal(actual, lhs + rhs)):
        raise RuntimeError("Mosaic GPU result mismatch")
    module = str(add.lower(lhs, rhs).compiler_ir(dialect="stablehlo"))
    if module.count("mosaic_gpu_v2") != 1:
        raise RuntimeError("expected exactly one Mosaic GPU custom call")
    args.output.write_text(module)
    print(f"wrote {len(module)} bytes to {args.output}")


if __name__ == "__main__":
    main()
