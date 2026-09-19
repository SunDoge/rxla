# RXLA model artifact format

## Status

This document proposes the portable model boundary. It is intentionally
separate from the existing OCR native-executable prototype, compilation cache,
and resumable training snapshots. Those formats solve different problems and
must not silently become a public model format.

## Decision

Use a small **Protobuf manifest** that references immutable external blobs:

```text
model.rxla/
├── manifest.pb
├── programs/
│   ├── inference.stablehlo.mlirbc
│   └── train-lora.stablehlo.mlirbc
├── tensors/
│   ├── model-00000.safetensors
│   └── model-00001.safetensors
├── assets/
└── native/                         # optional, disposable variants
```

The directory is the canonical runtime representation in version 1. A tarball,
object-store prefix, OCI artifact, or package-manager archive is only a transport
encoding. Runtime readers must not require extracting an opaque compressed
single file before they can inspect the manifest.

The manifest describes identity, graph entry points, ABI values, parameter and
state trees, tensor shards, required capabilities, custom calls, and blob
digests. It does not contain large tensor payloads or native executables inline.

## Why Protobuf rather than FlatBuffers

The manifest is normally kilobytes, while weights are megabytes or gigabytes.
Zero-copy access to manifest fields cannot materially change model load time.
Protobuf provides the more useful properties here:

- stable numeric field tags and well-understood additive evolution;
- mature Rust, Python, TypeScript, C++, Java, and Go tooling;
- convenient generated APIs for build tools and model inspection;
- bounded decoding can happen before any tensor or executable payload is read;
- `prost` is already used in the RXLA workspace.

FlatBuffers remains reasonable for a very large table queried directly from a
mapped file. It is a poor default for this manifest: its generated Python API is
more cumbersome, unions and defaults require care during evolution, and the
manifest still needs independent bounds, digest, and semantic validation.
FlatBuffers does not make an incompatible schema compatible by itself.

Schema-less JSON, MessagePack, Fory, CBOR, `bincode`, and `rkyv` are not the
canonical manifest encoding. JSON may be emitted as a diagnostic view. The
others are useful local encodings but do not offer a better cross-language,
long-lived schema contract than Protobuf for this small control-plane object.

## Payload choices

| Payload | Canonical encoding | Reason |
| --- | --- | --- |
| Manifest | Protobuf | Cross-language schema evolution |
| Portable program | StableHLO MLIR bytecode | Typed portable compiler boundary |
| Model tensors | SafeTensors shards | Existing ecosystem, bounded header, mmap/range-friendly payloads |
| Small opaque assets | Raw blob with declared media type | Tokenizers, labels, configs, lookup tables |
| Native executable | Backend-owned bytes, optional | Startup cache only; never the sole model representation |
| Human inspection | Derived JSON/text | Not authoritative and not hashed as the manifest |

StableHLO text may be emitted for debugging, but bytecode is the distribution
form. The manifest records its StableHLO compatibility/version requirements and
the exact calling convention used by RXLA.

## Logical schema sketch

The final `.proto` should remain small and use explicit messages instead of
maps where deterministic ordering matters. A representative shape is:

```proto
syntax = "proto3";

message ModelArtifact {
  uint32 format_major = 1;
  uint32 format_minor = 2;
  string model_id = 3;
  string producer = 4;
  repeated Blob blobs = 5;
  repeated Tensor tensors = 6;
  repeated Program programs = 7;
  repeated Capability required_capabilities = 8;
  repeated Asset assets = 9;
}

message Blob {
  uint64 id = 1;
  string path = 2;
  uint64 byte_length = 3;
  Digest digest = 4;
  string media_type = 5;
}

message Tensor {
  uint64 id = 1;
  repeated PathSegment path = 2;
  DType dtype = 3;
  repeated uint64 shape = 4;
  uint64 blob_id = 5;
  string safetensors_key = 6;
  TensorRole role = 7;
  optional uint64 alias_of = 8;
}

message Program {
  string id = 1;
  ProgramKind kind = 2;
  uint64 stablehlo_blob_id = 3;
  repeated Value inputs = 4;
  repeated Value outputs = 5;
  repeated StateBinding states = 6;
  repeated TensorBinding tensors = 7;
  repeated Capability required_capabilities = 8;
  repeated CustomCall custom_calls = 9;
  TargetConstraint target = 10;
}
```

The sketch is not permission to serialize Rust structs mechanically. Every
field needs a language-independent semantic definition and loader limits.

## Paths, trees, and aliasing

Parameter identity is an explicit sequence of path segments, not a dotted
string. A segment is either UTF-8 text or an unsigned integer. Thus
`layer1 / 0 / conv1 / weight` remains unambiguous and can round-trip through
Rust, Python, and TypeScript without parsing naming conventions.

Tensor IDs are artifact-local stable integers. Multiple bindings may refer to
one tensor ID for tied weights. `alias_of` is only for representational aliases;
readers must reject cycles, incompatible dtype/shape, and ambiguous ownership.
Tensor order is never semantic.

Roles distinguish at least:

- immutable parameter;
- trainable parameter;
- persistent model state;
- optimizer/transform state;
- RNG state;
- constant table.

A deployment model normally excludes optimizer and data-pipeline state. A
resumable checkpoint includes them in a separate checkpoint manifest.

## Programs and variants

One artifact may contain multiple named entry points, for example:

- `inference`;
- `train_lora`;
- `merge_lora`;
- `vae_decode`;
- `preprocess_rgb_416`.

Each program owns a complete ordered ABI. Inputs and outputs carry dtype,
shape, dynamic bounds, semantic name, and role. Tensor/state bindings map model
tree identities to ABI positions explicitly; loaders never infer ordering from
file names or dictionary iteration.

Variants are separate `Program` entries with target constraints and required
capabilities. Selection is a compatibility query, not “take the first program.”
Portable StableHLO is preferred. A native executable variant may accelerate a
known backend, but failure to validate or load it falls back to a compatible
portable program, not to a different model.

Custom calls declare:

- stable symbolic target name;
- ABI version;
- platform and architecture constraints;
- required shared-library/kernel blob and digest;
- portable fallback program when one exists.

An artifact containing an unknown required custom call is rejected before
compilation. Optional capabilities may be ignored only when the selected
program does not reference them.

## Model artifacts versus checkpoints

An immutable model artifact and a mutable training checkpoint are different
types:

```text
ModelArtifact
  portable programs + model tensors + assets

TrainingCheckpoint
  model_artifact_digest
  step/epoch and schedule identity
  current parameters and model state
  optimizer/EMA/RNG state
  optional data cursor
```

A checkpoint may reference a base artifact by digest and store only changed
tensors. It must still declare the exact model schema and training-program
identity it expects. Restoring weights into fresh optimizer state is not a
resume operation.

LoRA is an immutable overlay artifact:

```text
OverlayArtifact
  base_model_digest
  adapter tensors
  target tensor IDs/paths
  scale/rank semantics
  optional merge/apply StableHLO entry point
```

This permits fast adapter loading without copying the base model and makes an
incorrect base model a deterministic compatibility error.

## Compatibility rules

Compatibility is capability-based and fail-closed:

1. `format_major` changes only for incompatible manifest semantics.
2. A reader accepts its supported major and any minor whose required fields and
   capabilities it understands.
3. New optional fields use new Protobuf tags and safe defaults.
4. Field tags are never reused; removed fields are reserved forever.
5. Enum value zero always means unspecified/invalid, never a useful default.
6. Unknown required capabilities reject the artifact.
7. Unknown optional metadata may be ignored, but tooling must not promise
   lossless read-modify-write of unknown fields.
8. Program calling-convention versions are checked independently of manifest
   versions.
9. StableHLO, custom-call, native-backend, and tensor-format compatibility are
   separate constraints rather than one misleading global version.
10. Shape/dtype/path/tree mismatches reject before allocating device buffers or
    loading executable code.

The format version describes syntax and semantics, not a release number. Model
architecture identity belongs in hashes and explicit program/schema IDs.

## Integrity and trust

Every referenced blob has a byte length and digest. SHA-256 is the required
distribution digest because it is universally available in registries and
standard libraries. Implementations may additionally record BLAKE3 for fast
local verification, but cache keys and interoperability cannot require it.

Digests detect corruption and mismatched files; they do not authenticate the
publisher. A future signature envelope signs the canonical manifest bytes and
root blob digest set. Native executables, plugins, and custom-call libraries are
trusted code even when correctly signed.

Readers enforce limits before allocation:

- manifest bytes and recursion depth;
- blob count and aggregate declared size;
- tensor rank, dimensions, and checked element counts;
- UTF-8 path length and segment count;
- relative paths only, with no traversal or symlink escape;
- unique IDs, paths, program names, and ABI positions;
- digest verification before parsing executable/native payloads.

## Loading sequence

1. Read the bounded `manifest.pb` only.
2. Validate format version, IDs, uniqueness, paths, sizes, and required
   capabilities.
3. Select a program variant for the requested entry point and target.
4. Validate its complete ABI, tensor bindings, state schema, and custom calls.
5. Verify only the blobs required by that selected program.
6. Map/read tensor shards and create host/device buffers under explicit limits.
7. Compile portable StableHLO or validate and load an exact native variant.
8. Bind tensors/state by declared IDs and ABI positions.
9. Publish the initialized session only after every check succeeds.

No loader guesses missing metadata from a StableHLO module or SafeTensors file.

## Remote and single-file storage

The v1 directory maps naturally to local files, S3/R2 object prefixes, HTTP
range-addressable blobs, and OCI layers. Tensor shards are sized independently
so a serving system can fetch only the selected precision or component.

A future single-file form may use a small fixed header, a bounded Protobuf
manifest near the front, aligned immutable blob ranges, and a checksummed footer.
It must preserve the same logical manifest and blob IDs. Do not use inline
Protobuf `bytes` for multi-gigabyte weights, and do not invent this container
until mmap, range-read, atomic publication, and corruption tests justify it.

## Publication

Local directory publication follows the existing no-clobber staging pattern:
write a private sibling directory, finalize and verify all digests, write the
manifest last inside staging, then atomically publish without replacing an
existing target. Remote publication uploads content-addressed blobs first and
publishes the root manifest/reference last.

## Implementation phases

1. Add `rxla-artifact` containing generated Protobuf types plus semantic
   validation independent of PJRT.
2. Export one YOLOX inference artifact with StableHLO, EMA SafeTensors, ABI, and
   COCO class metadata; load it through RXLA and JAX and compare raw heads.
3. Add sharding, LoRA overlays, and training-checkpoint manifests.
4. Add target/capability selection and optional native executable variants.
5. Only then evaluate a single-file blob arena and signature envelope.

The first implementation must keep the current SafeTensors APIs usable. The
artifact layer composes them; it does not turn `rxla-safetensors` into a general
model/runtime crate.
