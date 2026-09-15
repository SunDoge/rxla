#!/usr/bin/env sh
set -eu

offline=
if test "${1:-}" = "--offline"; then
  offline=--offline
elif test "$#" -ne 0; then
  echo 'usage: scripts/check-cuda.sh [--offline]' >&2
  exit 2
fi

if test -z "${PJRT_PLUGIN_PATH:-}" || ! test -f "$PJRT_PLUGIN_PATH"; then
  echo 'PJRT_PLUGIN_PATH must name a trusted compatible CUDA plugin file' >&2
  exit 2
fi

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"
cargo $offline run --locked -p rxla-core --example cuda_clients
cargo $offline test --locked -p rxla-pjrt --test bf16 --test memory_stats --test download \
  --test host_transfer -- --include-ignored --test-threads=1
cargo $offline test --locked -p rxla-core --test submit --test unified_tensor \
  --test matmul --test convolution --test kv_cache_checked \
  -- --include-ignored --test-threads=1
