#!/usr/bin/env sh
set -eu

offline=false
case "${1:-}" in
  "") ;;
  --offline) offline=true ;;
  *) echo 'usage: sh scripts/check-xla-heterogeneous.sh [--offline]' >&2; exit 2 ;;
esac

for variable in PJRT_CPU_PLUGIN_PATH PJRT_CUDA_PLUGIN_PATH; do
  eval "path=\${$variable:-}"
  if test -z "$path" || ! test -f "$path"; then
    echo "$variable must point to a trusted compatible plugin file" >&2
    exit 2
  fi
  path=$(CDPATH= cd -- "$(dirname -- "$path")" && pwd)/$(basename -- "$path")
  eval "export $variable=\$path"
done

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"
run_cargo() {
  if "$offline"; then
    command cargo --offline "$@"
  else
    command cargo "$@"
  fi
}

run_cargo run --locked -p rxla-core --example heterogeneous
run_cargo run --locked -p rxla-core --example heterogeneous_state

for plugin in "$PJRT_CPU_PLUGIN_PATH" "$PJRT_CUDA_PLUGIN_PATH"; do
  PJRT_PLUGIN_PATH=$plugin run_cargo test --locked -p rxla-pjrt \
    --test host_transfer -- --ignored --exact real_host_staged_copy_preserves_scalars_empty_shapes_and_payloads
done

echo 'CPU/CUDA heterogeneous execution, state continuation, and host transfers passed.'
