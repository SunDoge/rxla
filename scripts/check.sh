#!/usr/bin/env sh
set -eu

offline=
if test "${1:-}" = "--offline"; then
  offline=--offline
elif test "$#" -ne 0; then
  echo 'usage: scripts/check.sh [--offline]' >&2
  exit 2
fi

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"
cargo $offline fmt --all -- --check
cargo $offline check --locked --workspace --all-targets
cargo $offline check --locked -p rxla-safetensors --features model,disk-cache --examples
cargo $offline test --locked --workspace --lib
cargo $offline test --locked -p rxla-nn --test model_attribute --test model_ui
cargo $offline test --locked -p rxla --test model_attribute
cargo $offline clippy --locked --workspace --all-targets -- -D warnings
