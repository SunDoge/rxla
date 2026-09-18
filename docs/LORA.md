# LoRA training and export

RXLA models can declare a base affine projection and its low-rank adapter at the
same use site:

```rust,ignore
let projection = cx.named_layer(
    "to_q",
    LoraLinear::new(hidden_size, rank).alpha(alpha).bias(false),
)?;
let q = input.apply(&projection)?;
```

The stable parameter tree is:

```text
to_q.weight
to_q.lora_A.weight  [rank, input_width]
to_q.lora_B.weight  [output_width, rank]
```

`lora_B.weight` uses a zero initializer, so adding the adapter does not change
the initial base-model output. The forward expression is
`base(x) + (alpha / rank) * B(A(x))`.

## Train only the adapter

Trainability is a transformation choice rather than a permanent property of the
layer. Discover the adapter paths from an ordinary trace, retrace only that
selection as resident state, and apply Adam:

```rust,ignore
let definition = Model::new(loss_model);
let schema_trace = definition.trace()?;
let adapter = rxla_train::select_lora(schema_trace.schema());
let mut training = definition.trace_resident(&adapter)?;
let loss = training.outputs()[0].clone();
rxla_train::apply_model_adam(
    &mut training,
    &adapter,
    &loss,
    AdamOptions::default(),
)?;

let compiled = training.compile_stateful(&mut compiler)?;
let mut session = compiled.initialize_session(seed)?;
```

Base weights remain ordinary immutable parameter bindings. Adapter parameters,
Adam moments and the step counter remain device-resident and are updated in one
compiled forward/backward/optimizer program.

## Export and restore

Export only the selected adapter. Base parameters, Adam state, RNG and unrelated
model state are deliberately excluded:

```rust,ignore
save_parameter_selection_new(
    "adapter.safetensors",
    &training,
    session.raw(),
    &adapter,
    Some(&metadata),
)?;
```

The caller should record at least the format/version, rank, alpha and base-model
identity in safetensors metadata. Loading validates every selected path, shape
and dtype before the first upload:

```rust,ignore
let mut checkpoint = SafeTensors::open("adapter.safetensors")?;
let buffers = checkpoint.load_parameter_selection(&client, &adapter)?;
let restored = buffers.initialize(compiled.session())?.build()?;
```

The native integration test trains a rank-one adapter for 250 Adam steps, saves
exactly `lora_A.weight` and `lora_B.weight`, rebuilds a fresh inference session,
and verifies the restored result on both the ZML CPU PJRT plugin and JAX CUDA
PJRT. Current scope is F32 affine LoRA. Convolution adapters, adapter merging,
quantized-base training and external Diffusers/PEFT key conversion remain
separate extensions.
