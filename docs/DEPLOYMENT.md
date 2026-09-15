# Precompiled OCR: same-environment deployment smoke

`ocr_graph --export-artifact DIRECTORY` writes a raw PJRT executable and an
experimental version-4 JSON manifest with weight order/shapes, input/output shapes and
diagnostic target metadata, plus `weights.safetensors` containing only the named
model weights. The directory must not exist. On Linux, export writes all payloads
and the manifest into a private temporary sibling directory, then publishes it
with `renameat2(RENAME_NOREPLACE)`. Existing targets (including empty directories
and dangling symlinks) are never replaced, even if created after preflight.
Compilation still
happens in this exporter; export is separate from numerical acceptance and may
produce an artifact even when the final strict ORT comparison fails.

`ocr_precompiled` is a separate process with **no tensor graph or compiler
imports**. It imports only the PJRT client and checkpoint reader, checks the
manifest's platform/version/API/device kind, deserializes the executable, uploads
the packaged weights and a caller-selected tensor from a separate input file,
and executes it. It neither reconstructs
the graph nor calls a compiler nor falls back to compilation. Backend-internal
work performed during native loading is not asserted absent.

Before loading the plugin or deserializing native code, the loader verifies
BLAKE3 checksums for both `executable.bin` and `weights.safetensors`, against the
manifest. Before loading the plugin, preflight also validates declared input,
output and weight F32 shapes, rejects negative dimensions and arithmetic
overflow, and limits each tensor to 256 MiB. Scalar and empty tensors are valid.
This is not an aggregate allocation quota or verification of the native ABI.
The manifest is bounded to 1 MiB; each payload is bounded to 256 MiB in
this OCR prototype. Reads stop after the limit plus one byte, even if a file grows.
The checked bytes are retained for native deserialization and checkpoint reads,
so later path replacement cannot substitute unchecked payloads. These checks
detect accidental corruption/mismatched files, **not authenticity**: an attacker
who can replace the manifest can replace the checksums too. The plugin and package
must still be trusted. The input checkpoint is separate and outside these payload
limits; this is not a comprehensive untrusted-file sandbox or total memory bound.

The loader then validates
checkpoint headers: manifest weight names must be nonempty and unique, the
weight file must contain exactly that set, and each weight and the selected
input must have the declared shape and F32 dtype. No implicit F16/BF16 conversion
is accepted in this deployment format. This preflight reads no tensor payloads;
it does not authenticate files or establish the executable's actual ABI.
Six host tests exercise valid headers, bounded reads, checksums and rejection paths in the independent
runtime project's standard gate. A wrong-shaped input was also rejected with a
nonexistent plugin path, verifying rejection precedes plugin loading. Valid CPU
and CUDA packages still reproduce all export pixels exactly after preflight.

This is a trusted same-environment prototype, **not a portable distribution
format**. The required `--trusted-same-environment` switch acknowledges that:

- Artifact and plugin are trusted native code, not untrusted model data.
- Diagnostic metadata is not a complete compatibility fingerprint. The caller
  must ensure the producing plugin build, CPU/GPU target, driver/native libraries
  and other required execution-environment details are compatible.
- Payload digests are checked, but no signature is verified. The manifest and code must remain
  consistent and unmodified. Files/directories and their ancestors must be protected.
- Publication has atomic visibility on supported Linux filesystems, not crash
  durability: there is no complete fsync protocol. Ordinary write/rename failures
  clean up the private staging directory when possible; process termination or
  cleanup failure may leave a hidden `.ocr-artifact-*` sibling. Such directories
  are not published packages. Never distribute them. The parent directory must
  be trusted; this is not protection against hostile ancestor replacement.

Atomic publication currently requires Linux and a filesystem supporting
`RENAME_NOREPLACE`; other platforms or unsupported filesystems fail export
instead of falling back to an overwriting/non-atomic rename. The `rustix` API is
an exporter-only Linux dev dependency, already present transitively through
tempfile; it is not added to the independent runtime's direct dependencies.
Host tests inject writer failure, late file/empty-directory/symlink destinations,
and two concurrent publishers (exactly one wins without mixing files).
This publication protocol does not change manifest version 4 or turn a
numerically rejected model into an accepted one: the exporter still publishes
before its separate strict ORT comparison.

The atomic publisher was also exercised by a fresh real CPU export to
`/tmp/xla-ocr-atomic-cpu`, then the independent runtime restored all 266240
pixels exactly (`-export.json` and `-load.json` sibling reports). The exporter
retained the known 10-pixel strict ORT failure. The full CPU development gate,
including exporter host tests, Clippy and independent runtime checks, passed.
This publication change was not a new CUDA/full-model performance run.

Before public distribution, this needs a versioned deployment API, trusted
signature verification, complete tested compatibility policy, bounded
loading and target/shape variants.
Version 4 rejects the earlier experimental version-1/version-2/version-3 manifests;
regenerate artifacts rather than implicitly trusting unchecked payloads.

After the tensor-size preflight change (`3c38611f46`), the independently rebuilt
runtime restored the existing v4 CPU and CUDA packages again in separate
processes on 2026-09-13. Every one of the 266240 output values and the
`[1,1,416,640]` shape matched its corresponding export report exactly.
Local reports are `/tmp/xla-deployment-recheck.wR1Ijo/{cpu,cuda}.json`.
An attempted v1 load was rejected for missing required manifest fields and
created no report. Both original v4 exporters still have `passed=false` for
strict ORT numerical agreement: successful restore is not an accuracy fix.
These runs used existing native artifacts, not graph reconstruction or export,
and do not establish cross-environment compatibility or startup performance.

Version 4 additionally requires `plugin_blake3`, computed from the producing
PJRT plugin file before loading it. The independent loader canonicalizes the
configured plugin path and verifies that file's digest before calling Client::load
or deserializing native code. Files must be regular and at most 1 GiB; hashing
uses a 64 KiB buffer and reads at most the limit plus one byte. Ordinary
non-export OCR evaluation does not hash the plugin. Hashing time is outside the
reported deserialize/load interval, and adds startup I/O even for valid packages.

This is an exact-file compatibility check, not a full build/environment identity
or authentication mechanism. It intentionally rejects even otherwise-compatible
plugin files with different bytes. Transitive shared libraries, drivers, compiler
flags and CPU instruction features remain outside the digest; matching files
do not prove a safe execution environment. Unlike retained artifact payloads,
the plugin is reopened by the native loader: its file and ancestor directories
must remain protected and immutable through hashing/loading/export. No defense
against malicious file replacement or loader dependency injection is claimed.
Keep `--trusted-same-environment`; the digest is not a reason to remove it.

Version-4 validation on 2026-09-13 exported new CPU and CUDA packages under
`/tmp/xla-ocr-v4-{cpu,cuda}` and loaded them in fresh independent runtime
processes. Each reproduced all 266240 export pixels and its output shape exactly;
manifest and runtime-report plugin digests matched. Sibling `-export.json`,
`-load.json` and `.log` files record the runs. Export numerical acceptance still
failed (10 strict ORT mismatches on CPU, 8 on CUDA in these runs). This is not an
accuracy fix or a cross-hardware compatibility test. A CPU package with either
the CUDA plugin file or an ordinary non-library file was rejected specifically
with `artifact PJRT plugin fingerprint mismatch`, before native loading and
without creating a report. The full CPU development gate also passed.
The earlier restoration measurements below describe version 1, not package size
or startup timing for version 2.

Version-2 validation on 2026-09-13 exported and restored the book fixture on both
CPU and CUDA through the independently built runtime. All 266240 pixels and
output shapes again matched the corresponding export exactly. Each package's
`weights.safetensors` contains exactly the 169 manifest weight names (1726512
bytes), with no `input/` or `expected/` entries. Reports and packages are under
`/tmp/xla-ocr-package.0KIz5z`. The loader rejected a version-1 package before plugin
loading and rejected an input with the wrong shape; neither wrote a report.
The full CPU default/cache/native/downstream/Clippy gate passed. Strict ORT
numerical acceptance remains failed; package restoration does not override it.

Version-3 CPU and CUDA packages were exported and restored through the independent
runtime on 2026-09-13. Both reproduced all 266240 output pixels exactly. Artifacts
are `/tmp/xla-ocr-v3-{cpu,cuda}`, with sibling `-export.json` and `-load.json`
reports. Exporters still report the known strict ORT mismatch (10 pixels for this
fixture on each backend); checksum validation and restoration do not fix it.
Separate executable-digest and weight-digest mismatch trials both failed with
`artifact payload checksum mismatch` while pointing to a nonexistent PJRT plugin,
confirming rejection precedes plugin loading; neither produced a report. The full
CPU default/cache/native/downstream/Clippy gate passed for this change. CUDA
validation here is the real export/restore comparison, not a new full GPU suite.

## Verified native execution

On 2026-09-13, the existing book fixture `[1,3,416,640]` was exported separately
through the pinned CPU and RTX 5080 CUDA plugins, then restored in fresh processes:

| Target | Native executable bytes | Deserialize/load interval | Output agreement |
| --- | ---: | ---: | --- |
| CPU | 613974 | 31.637 ms | All 266240 pixels exactly equal |
| CUDA | 148369 | 18.385 ms | All 266240 pixels exactly equal |

Sizes exclude weights, plugin and native dependencies. Load times exclude
process/plugin initialization, artifact file reading, weight/input upload and
first execution; they are single-run observations, not startup benchmarks.
Both export reports retain `passed=false` for the previously documented strict
ORT agreement failures. The loader checks shape/finiteness, while the separate
report comparison establishes exact output equality; successful restoration
does **not** establish OCR accuracy. Loading the CUDA manifest with the CPU
plugin is rejected before native executable deserialization. These limited
checks do not validate arbitrary hardware/library changes.

Artifacts and full reports are under `/tmp/xla-ocr-aot.oIKTk6` (`cpu/`, `cuda/`,
`cpu-export-report.json`, `cpu-load-report.json`, and CUDA counterparts).

## Reproduction

From repository root, set the trusted `PJRT_PLUGIN_PATH` and, for CUDA, the
environment in [CUDA validation](CUDA-validation.md). Use new artifact/report
paths; the original fixture must already exist.

```sh
cargo --config fast-load.toml run --offline \
  --manifest-path Cargo.toml \
  -p rxla-safetensors --example ocr_graph -- \
  target/ocr-xla-image-fixture \
  /tmp/new-export-report.json --include-output --export-artifact /tmp/new-ocr-artifact

cargo --config fast-load.toml run --offline \
  --manifest-path Cargo.toml \
  -p rxla-safetensors --example ocr_precompiled -- \
  --trusted-same-environment /tmp/new-ocr-artifact \
  target/ocr-xla-image-fixture/tensors.safetensors \
  input/0 /tmp/new-load-report.json

jq -s '.[0].outputs[0] == .[1].output and .[0].output_shape == .[1].shape' \
  /tmp/new-export-report.json /tmp/new-load-report.json
```

The exporter currently exits 1 after writing the book report because numerical
agreement fails. Inspect that report before running the loader; do not suppress
arbitrary export failures. The final comparison must print `true`. Runtime shape
and dtype are currently fixed F32 for this model.
