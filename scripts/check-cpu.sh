#!/usr/bin/env sh
set -eu

offline=
if test "${1:-}" = "--offline"; then
  offline=--offline
elif test "$#" -ne 0; then
  echo 'usage: scripts/check-cpu.sh [--offline]' >&2
  exit 2
fi

if test -z "${PJRT_CPU_PLUGIN_PATH:-}" || ! test -f "$PJRT_CPU_PLUGIN_PATH"; then
  echo 'PJRT_CPU_PLUGIN_PATH must name a trusted compatible CPU plugin file' >&2
  exit 2
fi

PJRT_CPU_PLUGIN_PATH=$(CDPATH= cd -- "$(dirname -- "$PJRT_CPU_PLUGIN_PATH")" && pwd)/$(basename -- "$PJRT_CPU_PLUGIN_PATH")
export PJRT_CPU_PLUGIN_PATH
export PJRT_PLUGIN_PATH=$PJRT_CPU_PLUGIN_PATH

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

# Exercise the public lazy Tensor facade plus low-level transfers and representative
# StableHLO compilation/execution. Larger model and performance tests stay outside
# the ordinary pull-request gate.
cargo $offline run --locked -p rxla --example basic -- --plugin "$PJRT_CPU_PLUGIN_PATH"
cargo $offline test --locked -p rxla-pjrt \
  --test bf16 --test download --test host_transfer \
  -- --include-ignored --test-threads=1
cargo $offline test --locked -p rxla-core \
  --test convolution --test matmul --test unified_tensor \
  -- --include-ignored --test-threads=1
cargo $offline test --locked -p rxla-train --lib \
  -- --include-ignored --test-threads=1

