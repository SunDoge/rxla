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

## Stable Diffusion attention LoRA

`UnetConfig::attention_lora(rank, alpha)` inserts adapters into every
cross-attention query, key, value and output projection without renaming the
base Diffusers parameter paths. The same UNet definition is therefore used for
ordinary inference and adapter training.

Prepare one real image/prompt noise-prediction batch with the Diffusers VAE,
CLIP text encoder and scheduler:

```console
uv run python benchmarks/prepare_sd_lora.py \
  /path/to/stable-diffusion-v1-5 \
  image.jpg \
  /tmp/sd-lora-batch.safetensors \
  --prompt "a photo of a church" \
  --image-size 128
```

Then compile the RXLA UNet forward pass, autodiff and Adam update into one
stateful GPU program. Only the LoRA tensors are resident and trainable; the
original checkpoint is loaded as immutable parameter bindings:

```console
cargo run -p rxla-safetensors --features model,export \
  --example stable_diffusion_lora_train --offline -- \
  .venv/lib/python3.14/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so \
  /path/to/stable-diffusion-v1-5 \
  /tmp/sd-lora-batch.safetensors \
  /tmp/sd-lora.safetensors \
  --preset sd-v1 --rank 1 --alpha 1 \
  --learning-rate 0.0001 --steps 2
```

The September 18, 2026 smoke test on an RTX 5080 trained all 256 SD 1.5
attention adapter tensors and exported an 814 KiB SafeTensors file. Loss on the
fixed real-image batch changed from `0.25562716` to `0.25503471` in two steps.
The full F32 compile was close to the 16 GiB device limit: XLA estimated
13.41 GiB after memory scheduling. A 100-step tiny Stable Diffusion run changed
loss from `1.09543443` to `1.06830454` and exported 96 adapter tensors in a
36 KiB file.

This is deliberately an end-to-end correctness and architecture test, not a
claim about useful image adaptation from one sample. Dataset loading,
multi-example batching, BF16 compute, rematerialization, gradient accumulation,
validation image generation and PEFT/Diffusers key conversion are the next
pieces needed for a production trainer.

### Memory-fitting controls

LoRA adapters may use F32 storage independently of a BF16 base checkpoint.
This is important for FLUX: freezing BF16 base parameters avoids base gradients
and optimizer slots, while the much smaller adapter and Adam state retain F32
precision. CUDA compilation can additionally request XLA's native memory
fitting schedule:

```rust,ignore
let compiler = Compiler::new(client, CacheLimits::default())
    .with_memory_fitting_level(MemoryFittingLevel::O2);
```

On the same RTX 5080 SD 1.5 smoke test, the earlier default compile reported an
estimated 13.41 GiB schedule against XLA's 11.67 GiB BFC pool. The explicit O2
compile completed without that over-budget warning and preserved the two-step
loss (`0.25562727` to `0.25503474`). The allocator's reserved pool is unchanged,
so this is evidence of a fitting schedule rather than a claim that 11.67 GiB
was physically consumed.

Whole-program BF16 compute now lowers slice-gradient padding correctly and the
same program compiles and executes. It is still experimental for a graph that
also contains Adam: the smoke test produced a finite first loss and a NaN on
the second update because the current lowering also demotes optimizer math.
Production mixed-precision training therefore remains F32 by default until the
IR can retain an explicit F32 optimizer island.

## FLUX.2 LoRA

`Flux2Config::attention_lora(rank, alpha)` adds F32 adapters to the fused
projections in both transformer block families:

- double-stream `img_attn` and `txt_attn` QKV/output projections;
- single-stream fused `linear1` and `linear2` projections.

The base paths and BF16 storage stay compatible with the official FLUX.2 Klein
checkpoint. A topology test verifies all adapter paths, selection and storage
dtypes. For Klein 4B at rank one this selects 120 adapter tensors containing
about 1.1 million F32 values; base gradients and base Adam states are absent.
The single-stream projections fuse attention and MLP, so adapting them is wider
than an attention-only projection in SD. A real Klein checkpoint training run
is still required before treating 16 GiB support as established.
