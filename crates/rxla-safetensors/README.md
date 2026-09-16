# rxla-safetensors

The native `device_rng_resume` integration test (`training` feature) checkpoints
four resident I32 Threefry key/counter slots and an F32 accumulation slot using
`save_state_new`. A fresh subprocess rebuilds the graph, validates the explicit
sampling-policy metadata, and restores all five slots. Six subsequent draws
match uninterrupted execution exactly, including raw I32 random bits, F32 state
bits, and a low-word counter carry. Each process compiles once. This verifies
completed-step device RNG restoration on the tested CPU plugin, not automatic
RNG discovery, cross-backend reproducibility, or a complete training protocol.

Optional safetensors checkpoint I/O crate, independent of the tensor graph and
its protobuf schema. Depends on the small PJRT wrapper for uploads. Adding a
checkpoint reader does not add model I/O dependencies to `rxla-core` builds.

```rust,ignore
let mut weights = rxla_safetensors::SafeTensors::open("model.safetensors")?;
let weight = weights.upload_f32(&client, "model.norm.weight")?;
// Bind weight as a runtime parameter, not a graph constant.
```

The reader loads and validates a bounded header (100 MB maximum) using upstream
safetensors metadata validation, checks the exact file length, then seeks to each
requested tensor. It does not mmap or read the whole checkpoint. `read_f32`
allocates the selected raw payload plus its F32 conversion; `upload_f32` drops the
temporary host data after the synchronous upload. This is not zero-copy or a
bounded-chunk streaming upload. Keep the file unchanged for a consistent snapshot.

Supported source formats: little-endian F32, F16, BF16. Shapes and axis order are
retained. Scalars, zero-element tensors and IEEE non-finite values are supported;
missing names, other dtypes and malformed metadata/payload lengths are errors.
This is conversion to the current F32 backend, not native mixed-precision execution.

`SafeTensors::new` also accepts a `Read + Seek` source, useful for tests and other
storage implementations. `info` and `names` inspect metadata without payload I/O.

```sh
cargo run -p rxla-safetensors --example inspect -- model.safetensors model.norm.weight
PJRT_PLUGIN_PATH=/trusted/libzml_cpu.so cargo test -p rxla-safetensors -- --include-ignored
```

Tests cover source dtypes, malformed files, exact requested-payload read counts,
and a real PJRT linear layer using BF16/F16 checkpoint weights. The local TinyLlama
checkpoint's header and BF16 final norm weight were also read successfully.
Sharded index files, quantized formats, architecture/config parsing, tokenizer
loading and a complete model importer are not implemented here.

## Module parameters

With feature `training`, explicit module-path → checkpoint-key mappings support
parameter deduplication, including tied aliases. Every alias must map to the same
key as its primary path. Names, supported dtypes and exact shapes are checked
before payload reads/uploads; failures during I/O may retain completed work in
statistics but never modify a session.

For fixed/input-only modules, `load_module` returns parameter bindings as before.
For modules containing `trainable_parameter` handles, use `load_module_parts`:

```rust,ignore
let loaded = checkpoint.load_module_parts(&client, &model, &mapping)?;
let mut initial = loaded.states;
initial.extend(other_initial_state); // KV cache, optimizer moments, etc.
let mut session = program.session(initial)?;
session.bind_parameters(loaded.inputs)?;
```

`ModuleBuffers::states` owns trainable parameter buffers with state identities;
`inputs` holds fixed/dynamic input parameter buffers in Rc. Classification does
not copy or download tensor data. Only parameter state is initialized: no implicit
optimizer moments, counters, unrelated model state, or session creation.
A native mixed BF16/F16 test loads a tied resident weight and fixed bias once
each, verifies preflight rejection without payload I/O, then trains through
Linear while the bias remains unchanged.

## Parameter-effect models

New IR-first models use `ParamSchema` rather than module-owned parameter
handles. `load_parameter_schema` validates every schema path, shape and storage
dtype before any payload is read or uploaded. Its default checkpoint mapping is
path-to-identical-path; use `load_parameter_schema_with_mapping` for a renamed
checkpoint.

```rust,ignore
let (schema, _) = rxla_nn::init(model)?;
let applied = rxla_nn::apply(&schema, model)?;
let weights = checkpoint.load_parameter_schema(&client, &schema)?;
let compiled = applied.compile(&mut compiler)?;
let model = compiled.bind_parameters(weights.bindings())?;
let output: Buffer = model.run(&input)?;
```

Inputs retain the explicit model-function order. Parameters are selected by
their stable lexical paths, then `CompiledModel::bind_parameters` freezes their
PJRT ABI order for repeated execution. This prevents a checkpoint from becoming
coupled to incidental trace or compiler parameter numbering.

## Export current module weights

With `modules`, `rxla_safetensors::serialize_module(&session, &model)` returns F32
safetensors bytes for current fixed-bound and resident parameters. It checks all
parameter bindings before downloading payloads. Dynamic unbound parameters and
foreign schemas are errors. No file is opened or overwritten; the caller owns
publication and durability policy for the returned bytes.

Each unique parameter is saved under its first collector path. Tied aliases are
not duplicated: when reloading, explicitly map all alias paths to that primary
key. Names are module paths, not automatically the original checkpoint keys.
Metadata and payloads are accepted by the upstream safetensors reader; a native
test exports after training, verifies tied-weight deduplication, continues
training without changing the exported bytes, and reloads into a fresh inference
module with matching predictions. Empty parameter sets are also supported.

The byte-returning API retains the whole serialized result, so large models may
require substantial host memory. It is not a complete training-resume checkpoint: optimizer
moments, counters, RNG, other session state and executable code are not included.
All values are saved as F32, not the source checkpoint's F16/BF16 storage format.

`write_module(&mut writer, &session, &model)` streams to a caller-owned `Write`
implementation. It constructs and validates metadata with upstream safetensors,
checks all bindings and rejects reserved `__metadata__` primary names before
writing anything. F32 parameters are sorted by primary name, downloaded one at a
time, and encoded in chunks no larger than 64 KiB. Additional host memory includes
the full largest parameter, header, encoding buffer, and the writer's buffering;
native partial-tensor downloads are not supported. `serialize_module` now wraps
this writer path with a Vec destination rather than retaining all input payloads.

No file is opened, overwritten, flushed, synchronized or atomically published by
the library. A write/download error can leave a partial output: discard it and
apply your own temporary-file/publication policy. Session state is not changed.
Native tests verify byte-for-byte compatibility with upstream serialization for
scalars, empty tensors and multi-chunk data, deduplicated aliases, preflight
failure with zero writes, and an injected partial-write failure. This is not a
large-model memory benchmark or a durability guarantee.
