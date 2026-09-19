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
│   ├── model-00000.rxt
│   └── model-00001.rxt
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

Schema-less JSON, MessagePack, CBOR, `bincode`, and `rkyv` are not the canonical
manifest encoding. JSON may be emitted as a diagnostic view. The others are
useful local encodings but do not offer a better cross-language, long-lived
schema contract than Protobuf for this small control-plane object.

Fory is a serious option, but solves a different problem. Its xlang object
format is useful when peers reconstruct object graphs, and Standard Row is
useful for schema-driven random access to many structured records. An RXLA
manifest is one small control-plane value, while weights are large typed byte
ranges. Making either depend on Fory would add a runtime/schema dependency
without improving the hot weight-loading path. Fory remains a candidate for
datasets, event streams, and structured opaque assets; it is not part of the
version 1 model ABI.

## Payload choices

| Payload | Canonical encoding | Reason |
| --- | --- | --- |
| Manifest | Protobuf | Cross-language schema evolution |
| Portable program | StableHLO MLIR bytecode | Typed portable compiler boundary |
| Model tensors | RXLA tensor arena shards | Aligned raw bytes, mmap/range reads, aliases, and explicit integrity |
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
  bytes artifact_uuid = 10; // exactly 16 bytes
  repeated Metadata metadata = 11;
}

message Blob {
  uint64 id = 1;
  string path = 2;
  uint64 byte_length = 3;
  Digest digest = 4;
  string media_type = 5;
  bytes shard_uuid = 6; // exactly 16 bytes for tensor arenas
}

message Tensor {
  uint64 id = 1;
  repeated PathSegment path = 2;
  DType dtype = 3;
  repeated uint64 shape = 4;
  TensorRole role = 5;
  TensorEncoding encoding = 6;
  repeated TensorSlice slices = 7;
  optional uint64 storage_alias_of = 8;
  optional Sharding sharding = 9;
  repeated Metadata metadata = 10;
}

message TensorSlice {
  uint64 blob_id = 1;
  uint64 byte_offset = 2;
  uint64 byte_length = 3;
  repeated SliceDimension dimensions = 4;
  optional Digest digest = 5;
  uint32 alignment = 6;
  optional uint64 replica_group = 7;
  repeated MeshCoordinate mesh_coordinates = 8;
}

message SliceDimension {
  uint64 start = 1;
  uint64 length = 2;
}

message MeshCoordinate {
  string axis = 1;
  uint64 index = 2;
}

message Sharding {
  repeated string mesh_axes = 1;
  repeated uint64 preferred_mesh_shape = 2;
  repeated DimensionSharding dimensions = 3;
  bool reshardable = 4;
}

message Metadata {
  string key = 1;
  MetadataValue value = 2;
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

## Typed metadata

GGUF demonstrates that a model container benefits from self-describing typed
key/value metadata rather than one opaque JSON object. RXLA follows that idea
for information which is useful to tools but is not part of the execution ABI.
`MetadataValue` is a closed tagged union containing booleans, signed/unsigned
integers, `f32`/`f64`, UTF-8 strings, bytes, and homogeneous arrays of those
scalar types. Nested arbitrary maps and dynamically typed arrays are excluded.

Keys are namespaced lowercase identifiers such as `general.architecture`,
`general.license`, `rxla.quantization_version`, or `coco.class_names`. The
`rxla.*` namespace is reserved. Architecture owners should use their own
namespace, and readers must preserve unknown metadata when copying an artifact.
Metadata cannot override typed manifest fields, program ABI, tensor shape,
encoding, integrity information, or required capabilities. Anything required
for safe execution gets a real schema field instead of a magic key.

RXLA deliberately departs from GGUF where its scope is broader. GGUF places a
tensor name, dimensions, element/quantization type, and aligned offset beside a
single tensor data region. RXLA retains those explicit descriptors but uses
structured parameter paths, multiple external arenas, logical tensor slices,
replication/sharding semantics, StableHLO programs, and typed calling
conventions. This keeps inference, fine-tuning, distributed checkpoints, and
LoRA overlays in one model without turning naming conventions into semantics.

## Tensor arena

RXLA owns the tensor payload format. It borrows the useful idea from
SafeTensors--a bounded metadata layer describes slices of contiguous raw
bytes--without adopting JSON metadata or making an ecosystem interchange
format the model's storage ABI.

In the directory representation, an `.rxt` shard is an immutable byte arena.
Tensor descriptors live in `manifest.pb`; the shard does not repeat names,
shapes, or arbitrary string metadata. Each descriptor gives a blob ID, checked
64-bit offset and length, dtype, shape, encoding, and logical slice. Version 1
tensor payloads are:

- dense, row-major, and little-endian;
- stored without compression in the normal serving path;
- naturally aligned to at least 64 bytes, with writers optionally using page
  alignment for independently mapped or fetched components;
- addressable through HTTP/S3 range reads as well as local mmap;
- immutable after publication.

The blob digest protects the whole shard. A tensor may additionally carry a
digest when independent range verification is required. A small fixed shard
header records magic, format version, artifact UUID, and shard UUID so that
files from different artifacts cannot be mixed accidentally. The manifest
binds those UUIDs to the shard digest; the header cannot contain the manifest
digest because that would create a digest cycle. Offsets must begin after this
header, satisfy the declared alignment, stay within the declared blob length,
and never overlap unless the tensors explicitly share one storage identity.

Tensor tying does not duplicate bytes. Multiple bindings refer to the same
tensor ID; representational views refer to a storage tensor plus a checked byte
range and view metadata. Version 1 does not persist arbitrary strides. A writer
materializes unsupported layouts, keeping runtime loading predictable and
allowing one mapped range to become a host buffer directly.

A logical tensor may have multiple `TensorSlice` entries. Every slice records
its half-open range in every logical dimension, so its meaning is independent
of file order, process rank, host name, and device ordinal. Partition slices
must be non-overlapping and cover the complete logical tensor unless an
explicitly declared sparse encoding says otherwise. Equivalent replicas may
cover the same range when they share a replica group; their encoding, logical
range, byte length, and decoded content digest must agree. A reader may select
any verified replica. Version 1 does not allow one physical slice payload to
cross blob boundaries.

`Sharding` records semantic mesh axes and how logical dimensions are partitioned
or replicated. Slice mesh coordinates describe the saved semantic layout. The
layout may contain a preferred mesh shape, but never persistent assignments
such as `cuda:0` or worker addresses. At load time the runtime maps the semantic
mesh onto available PJRT devices. When the saved and requested layouts differ
and `reshardable` is true, it constructs a checked reshard plan; otherwise
loading fails before device allocation.

Published inference artifacts should prefer broadly loadable canonical layouts.
Training checkpoints may preserve the current physical slicing so every worker
can write and restore its own shards concurrently without gathering the full
tensor on one host. Both forms describe the same global logical tensor.

Compression is a blob transport concern, not a tensor encoding default. A
compressed archive must be expanded before mmap and partial fetching are
possible. Future encodings such as quantized blocks must have a named,
versioned binary contract and may not be smuggled through opaque metadata.

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
- slice bounds, alignment, non-overlap, and complete logical coverage;
- mesh-axis uniqueness and valid partition/replication specifications;
- typed metadata count, key length, array length, and namespace rules;
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

No loader guesses missing metadata from a StableHLO module or tensor shard.

## Remote and single-file storage

The v1 directory maps naturally to local files, S3/R2 object prefixes, HTTP
range-addressable blobs, and OCI layers. Tensor shards are sized independently
so a serving system can fetch only the selected precision or component.

The directory already uses tensor arenas, but the package itself remains a
directory. A future single-file package may use a small fixed header, a bounded
Protobuf manifest near the front, aligned immutable blob ranges, and a
checksummed footer. It must preserve the same logical manifest and blob IDs. Do
not use inline Protobuf `bytes` for multi-gigabyte weights, and do not define
the package container until mmap, range-read, atomic publication, and corruption
tests justify it.

## Publication

Local directory publication follows the existing no-clobber staging pattern:
write a private sibling directory, finalize and verify all digests, write the
manifest last inside staging, then atomically publish without replacing an
existing target. Remote publication uploads content-addressed blobs first and
publishes the root manifest/reference last.

## Implementation phases

1. Add `rxla-artifact` containing generated Protobuf types plus semantic
   validation independent of PJRT.
2. Implement the minimal `.rxt` arena writer/reader with checked offsets,
   alignment, mmap, range reads, and corruption tests.
3. Export one YOLOX inference artifact with StableHLO, EMA weights, ABI, and
   COCO class metadata; load it through RXLA and JAX and compare raw heads.
4. Add sharding, LoRA overlays, and training-checkpoint manifests.
5. Add target/capability selection and optional native executable variants.
6. Only then evaluate a single-file package and signature envelope.

The current SafeTensors APIs remain an import/export compatibility bridge for
existing checkpoints. The artifact format does not depend on them, and
`rxla-safetensors` does not become a general model/runtime crate.
