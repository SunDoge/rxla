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
cargo $offline test --locked --workspace --lib
cargo $offline clippy --locked --workspace --all-targets -- -D warnings
