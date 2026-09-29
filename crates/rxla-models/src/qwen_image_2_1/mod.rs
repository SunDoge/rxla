//! Qwen-Image-2.1 block-causal diffusion transformer.
//!
//! This module initially exposes the text-to-image path.  Parameter names and
//! math intentionally follow Diffusers' `QwenImage21Transformer2DModel`; image
//! editing uses the same blocks but additionally needs condition-image token
//! interleaving.

use super::{Error, ModelDefinitionError, Result};
use rxla_core::{DType, RotaryLayout, Tensor};
use rxla_nn::{Cx, Linear, QuantizedLinear, TensorApply};

mod scheduler;
pub use scheduler::{QwenImage21FlowScheduler, QwenImage21FlowStep};
mod vae;
pub use vae::qwen_image_2_1_vae_decoder;

/// Static topology and latent grid for a Qwen-Image-2.1 transformer trace.
#[derive(Clone, Debug)]
pub struct QwenImage21Config {
    in_channels: i64,
    out_channels: i64,
    hidden_size: i64,
    context_in_dim: i64,
    num_heads: i64,
    num_layers: usize,
    mlp_ratio: i64,
    axes_dims_rope: [i64; 3],
    rope_theta: f32,
    epsilon: f32,
    latent_height: i64,
    latent_width: i64,
    parameter_dtype: DType,
    compute_dtype: DType,
    quant_group_size: Option<i64>,
}

impl QwenImage21Config {
    /// Official 7B Qwen-Image-2.1 transformer at a selected latent grid.
    pub fn official(latent_height: i64, latent_width: i64) -> Self {
        Self {
            in_channels: 64,
            out_channels: 64,
            hidden_size: 4096,
            context_in_dim: 4096,
            num_heads: 32,
            num_layers: 32,
            mlp_ratio: 3,
            axes_dims_rope: [16, 56, 56],
            rope_theta: 10_000.0,
            epsilon: 1e-6,
            latent_height,
            latent_width,
            parameter_dtype: DType::BF16,
            compute_dtype: DType::BF16,
            quant_group_size: None,
        }
    }

    /// Official topology using RXLA symmetric grouped W8 matrix storage.
    /// Group size 64 is required by the transformer's 64-channel image input.
    pub fn official_w8(latent_height: i64, latent_width: i64) -> Self {
        Self {
            parameter_dtype: DType::F32,
            quant_group_size: Some(64),
            ..Self::official(latent_height, latent_width)
        }
    }

    /// Complete small topology for compiler and numerical oracle tests.
    pub fn tiny(latent_height: i64, latent_width: i64) -> Self {
        Self {
            in_channels: 8,
            out_channels: 8,
            hidden_size: 32,
            context_in_dim: 24,
            num_heads: 4,
            num_layers: 2,
            mlp_ratio: 2,
            axes_dims_rope: [2, 2, 4],
            rope_theta: 10_000.0,
            epsilon: 1e-6,
            latent_height,
            latent_width,
            parameter_dtype: DType::F32,
            compute_dtype: DType::F32,
            quant_group_size: None,
        }
    }

    pub fn tiny_w8(latent_height: i64, latent_width: i64) -> Self {
        Self {
            quant_group_size: Some(8),
            ..Self::tiny(latent_height, latent_width)
        }
    }

    /// Select the arithmetic used by affine projections while retaining F32
    /// tensor boundaries. Official inference defaults to BF16; F32 is useful
    /// for numerical diagnostics.
    pub fn compute_dtype(mut self, dtype: DType) -> Self {
        self.compute_dtype = dtype;
        self
    }

    fn head_dim(&self) -> i64 {
        self.hidden_size / self.num_heads
    }

    fn image_tokens(&self) -> i64 {
        self.latent_height * self.latent_width
    }

    fn validate(&self) -> Result<()> {
        let rope_width: i64 = self.axes_dims_rope.iter().sum();
        let valid = self.in_channels > 0
            && self.out_channels > 0
            && self.hidden_size > 0
            && self.context_in_dim > 0
            && self.num_heads > 0
            && self.hidden_size % self.num_heads == 0
            && self.num_layers > 0
            && self.mlp_ratio > 0
            && self.latent_height > 0
            && self.latent_width > 0
            && self
                .axes_dims_rope
                .iter()
                .all(|dim| *dim > 0 && *dim % 2 == 0)
            && rope_width == self.head_dim()
            && self.rope_theta.is_finite()
            && self.rope_theta > 0.0
            && self.epsilon.is_finite()
            && self.epsilon > 0.0
            && matches!(self.parameter_dtype, DType::BF16 | DType::F32)
            && matches!(self.compute_dtype, DType::BF16 | DType::F32);
        let valid = valid
            && self.quant_group_size.is_none_or(|group| {
                group > 0
                    && self.in_channels % group == 0
                    && self.context_in_dim % group == 0
                    && self.hidden_size % group == 0
                    && 256 % group == 0
            });
        if !valid {
            return Err(Error::InvalidModel {
                kind: ModelDefinitionError::InvalidQwenImage21Configuration,
            });
        }
        Ok(())
    }
}

fn linear(
    cx: Cx,
    input: &Tensor,
    width: i64,
    bias: bool,
    config: &QwenImage21Config,
) -> Result<Tensor> {
    match config.quant_group_size {
        Some(group) => {
            debug_assert!(!bias, "Qwen-Image-2.1 W8 projections are bias-free");
            Ok(input.through(cx, QuantizedLinear::new(width, group))?)
        }
        None => Ok(input.through(cx, Linear::new(width).bias(bias))?),
    }
}

fn timestep_embedding(
    cx: Cx,
    timestep: &Tensor,
    hidden: i64,
    config: &QwenImage21Config,
) -> Result<Tensor> {
    let half = 128;
    let frequencies = (0..half)
        .map(|index| (-10_000.0_f32.ln() * index as f32 / half as f32).exp())
        .collect::<Vec<_>>();
    let frequency = cx.constant([1, half], frequencies)?;
    let phase = timestep
        .mul_scalar(1000.0)?
        .unsqueeze(1)?
        .broadcast_to([timestep.shape()[0], half])?
        .mul(&frequency.broadcast_to([timestep.shape()[0], half])?)?;
    let embedded = Tensor::concatenate(&[phase.cos()?, phase.sin()?], 1)?;
    let scope = cx.scope("timestep_embedder")?;
    let embedded = linear(scope.scope("linear_1")?, &embedded, hidden, false, config)?.silu()?;
    linear(scope.scope("linear_2")?, &embedded, hidden, false, config)
}

fn zero_center_rms_norm(cx: Cx, input: &Tensor, epsilon: f32) -> Result<Tensor> {
    let weight = cx.param("weight", [input.shape()[input.shape().len() - 1]])?;
    input
        .rms_norm(&weight.add_scalar(1.0)?, epsilon)
        .map_err(Into::into)
}

fn text_projection(cx: Cx, input: &Tensor, config: &QwenImage21Config) -> Result<Tensor> {
    let hidden = zero_center_rms_norm(cx.scope("text_norm")?, input, config.epsilon)?;
    let hidden = linear(
        cx.scope("in_layer")?,
        &hidden,
        config.hidden_size,
        false,
        config,
    )?
    .gelu_tanh()?;
    linear(
        cx.scope("out_layer")?,
        &hidden,
        config.hidden_size,
        false,
        config,
    )
}

fn rotary_angles(cx: &Cx, text_tokens: i64, config: &QwenImage21Config) -> Result<Tensor> {
    let image_tokens = config.image_tokens();
    let tokens = text_tokens + image_tokens;
    let mut phases = Vec::with_capacity((tokens * config.head_dim() / 2) as usize);
    for token in 0..tokens {
        let (frame, height, width) = if token < text_tokens {
            (token, token, token)
        } else {
            let image_index = token - text_tokens;
            (
                text_tokens,
                image_index / config.latent_width
                    - (config.latent_height - config.latent_height / 2),
                image_index % config.latent_width - (config.latent_width - config.latent_width / 2),
            )
        };
        for (position, axis_width) in [frame, height, width]
            .into_iter()
            .zip(config.axes_dims_rope)
        {
            for pair in 0..axis_width / 2 {
                phases.push(
                    position as f32
                        * config
                            .rope_theta
                            .powf(-2.0 * pair as f32 / axis_width as f32),
                );
            }
        }
    }
    Ok(cx.constant([1, 1, tokens, config.head_dim() / 2], phases)?)
}

fn text_causal_bias(cx: &Cx, text_tokens: i64) -> Result<Tensor> {
    let mut bias = Vec::with_capacity((text_tokens * text_tokens) as usize);
    for query in 0..text_tokens {
        for key in 0..text_tokens {
            bias.push(if key <= query { 0.0 } else { f32::NEG_INFINITY });
        }
    }
    Ok(cx.constant([1, 1, text_tokens, text_tokens], bias)?)
}

fn token_modulation(value: &Tensor, text_tokens: i64, image_tokens: i64) -> Result<Tensor> {
    let batch = value.shape()[0] - 1;
    let width = value.shape()[1];
    let real =
        value
            .narrow(0, 0, batch)?
            .unsqueeze(1)?
            .broadcast_to([batch, image_tokens, width])?;
    let zero = value
        .narrow(0, batch, 1)?
        .broadcast_to([batch, width])?
        .unsqueeze(1)?
        .broadcast_to([batch, text_tokens, width])?;
    Ok(Tensor::concatenate(&[zero, real], 1)?)
}

fn attention(
    cx: Cx,
    input: &Tensor,
    angles: &Tensor,
    text_bias: &Tensor,
    text_tokens: i64,
    config: &QwenImage21Config,
) -> Result<Tensor> {
    let [batch, tokens, _] = input.shape() else {
        unreachable!("Qwen-Image blocks receive rank-three tensors")
    };
    let project = |name: &str| -> Result<Tensor> {
        Ok(
            linear(cx.scope(name)?, input, config.hidden_size, false, config)?.reshape([
                *batch,
                *tokens,
                config.num_heads,
                config.head_dim(),
            ])?,
        )
    };
    let mut query = project("to_q")?;
    let mut key = project("to_k")?;
    let value = project("to_v")?;
    let query_scale = cx.scope("norm_q")?.param("weight", [config.head_dim()])?;
    let key_scale = cx.scope("norm_k")?.param("weight", [config.head_dim()])?;
    query = query.rms_norm(&query_scale, config.epsilon)?;
    key = key.rms_norm(&key_scale, config.epsilon)?;
    query = query
        .transpose([0, 2, 1, 3])?
        .rotary_embedding_angles(angles, RotaryLayout::Interleaved)?;
    key = key
        .transpose([0, 2, 1, 3])?
        .rotary_embedding_angles(angles, RotaryLayout::Interleaved)?;
    let value = value.transpose([0, 2, 1, 3])?;
    // Exact non-Flex block-causal specialization for text-to-image: text is a
    // causal prefix and the target image is one internally bidirectional block
    // that can attend to the complete prefix plus itself. Splitting the two
    // query regions avoids materializing a quadratic full-sequence mask and
    // leaves the large image attention mask-free for cuDNN/XLA fusion.
    let text_attended = query
        .narrow(2, 0, text_tokens)?
        .scaled_dot_product_attention(
            &key.narrow(2, 0, text_tokens)?,
            &value.narrow(2, 0, text_tokens)?,
            Some(text_bias),
            None,
        )?;
    let image_tokens = *tokens - text_tokens;
    let image_attended = query
        .narrow(2, text_tokens, image_tokens)?
        .scaled_dot_product_attention(&key, &value, None, None)?;
    let attended = Tensor::concatenate(&[text_attended, image_attended], 2)?;
    let attended =
        attended
            .transpose([0, 2, 1, 3])?
            .reshape([*batch, *tokens, config.hidden_size])?;
    linear(
        cx.scope("to_out")?.scope("0")?,
        &attended,
        config.hidden_size,
        false,
        config,
    )
}

fn transformer_block(
    cx: Cx,
    input: &Tensor,
    modulation: &Tensor,
    angles: &Tensor,
    text_bias: &Tensor,
    text_tokens: i64,
    config: &QwenImage21Config,
) -> Result<Tensor> {
    let parts = modulation.split(2, &[config.hidden_size * 2, config.hidden_size * 2])?;
    let apply_modulation = |input: &Tensor, value: &Tensor| -> Result<(Tensor, Tensor)> {
        let value = value.split(2, &[config.hidden_size, config.hidden_size])?;
        let scale = value[0].add_scalar(1.0)?;
        let normalized = input.layer_norm(&[config.hidden_size], None, None, config.epsilon)?;
        Ok((normalized.mul(&scale)?, value[1].tanh()?))
    };
    let (normalized, gate) = apply_modulation(input, &parts[0])?;
    let hidden = input.add(
        &attention(
            cx.scope("attn")?,
            &normalized,
            angles,
            text_bias,
            text_tokens,
            config,
        )?
        .mul(&gate)?,
    )?;
    let (normalized, gate) = apply_modulation(&hidden, &parts[1])?;
    let mlp = cx.scope("img_mlp")?;
    let projected = linear(
        mlp.scope("proj")?,
        &normalized,
        config.hidden_size * config.mlp_ratio,
        false,
        config,
    )?;
    let gated = linear(
        mlp.scope("gate_layer")?,
        &normalized,
        config.hidden_size * config.mlp_ratio,
        false,
        config,
    )?
    .silu()?
    .mul(&projected)?;
    let delta = linear(mlp.scope("out")?, &gated, config.hidden_size, false, config)?;
    Ok(hidden.add(&delta.mul(&gate)?)?)
}

/// Apply the official text-to-image denoising transformer.
///
/// `image` contains the target latent grid flattened in row-major order;
/// `context` contains unpadded Qwen3-VL embeddings.  Image-editing condition
/// blocks will be added as a separate packing API rather than weakening this
/// function's static invariants.
pub fn qwen_image_2_1_transformer(
    cx: Cx,
    image: &Tensor,
    timestep: &Tensor,
    context: &Tensor,
    config: &QwenImage21Config,
) -> Result<Tensor> {
    config.validate()?;
    let cx = cx
        .with_parameter_storage_dtype(config.parameter_dtype)?
        .with_layer_compute_dtype(config.compute_dtype)?;
    let ([batch, image_tokens, image_width], [context_batch, text_tokens, context_width]) =
        (image.shape(), context.shape())
    else {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidQwenImage21Input,
        });
    };
    let valid = image.dtype() == DType::F32
        && context.dtype() == DType::F32
        && timestep.dtype() == DType::F32
        && batch == context_batch
        && *image_tokens == config.image_tokens()
        && *image_width == config.in_channels
        && *context_width == config.context_in_dim
        && *text_tokens > 0
        && timestep.shape() == [*batch];
    if !valid {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidQwenImage21Input,
        });
    }

    let image = linear(
        cx.scope("img_in")?,
        image,
        config.hidden_size,
        false,
        config,
    )?;
    let context = text_projection(cx.scope("txt_in")?, context, config)?;
    let mut hidden = Tensor::concatenate(&[context, image], 1)?;
    let angles = rotary_angles(&cx, *text_tokens, config)?;
    let text_bias = text_causal_bias(&cx, *text_tokens)?;

    let zero_timestep = cx.constant([1], [0.0])?;
    let all_timesteps = Tensor::concatenate(&[timestep.clone(), zero_timestep], 0)?;
    let time = timestep_embedding(
        cx.scope("time_text_embed")?,
        &all_timesteps,
        config.hidden_size,
        config,
    )?;
    let modulation = linear(
        cx.scope("modulation")?.scope("1")?,
        &time.silu()?,
        config.hidden_size * 4,
        false,
        config,
    )?;
    let modulation = token_modulation(&modulation, *text_tokens, *image_tokens)?;

    let blocks = cx.scope("transformer_blocks")?;
    for index in 0..config.num_layers {
        hidden = transformer_block(
            blocks.scope(index.to_string())?,
            &hidden,
            &modulation,
            &angles,
            &text_bias,
            *text_tokens,
            config,
        )?;
    }

    let output = hidden.narrow(1, *text_tokens, *image_tokens)?;
    let scale = linear(
        cx.scope("norm_out")?.scope("linear")?,
        &time.narrow(0, 0, *batch)?.silu()?,
        config.hidden_size,
        false,
        config,
    )?
    .unsqueeze(1)?
    .broadcast_to(output.shape())?
    .add_scalar(1.0)?;
    let output = output
        .layer_norm(&[config.hidden_size], None, None, config.epsilon)?
        .mul(&scale)?;
    linear(
        cx.scope("proj_out")?,
        &output,
        config.out_channels,
        false,
        config,
    )
}

/// One reusable device-resident Qwen-Image-2.1 FlowMatch Euler step.
/// `step_size` is `next_sigma - sigma` with shape `[batch]`.
pub fn qwen_image_2_1_euler_step(
    cx: Cx,
    sample: &Tensor,
    timestep: &Tensor,
    step_size: &Tensor,
    context: &Tensor,
    config: &QwenImage21Config,
) -> Result<Tensor> {
    let prediction = qwen_image_2_1_transformer(cx, sample, timestep, context, config)?;
    if step_size.dtype() != DType::F32 || step_size.shape() != [sample.shape()[0]] {
        return Err(Error::InvalidModel {
            kind: ModelDefinitionError::InvalidQwenImage21Input,
        });
    }
    let coefficient = step_size
        .reshape([sample.shape()[0], 1, 1])?
        .broadcast_to(sample.shape())?;
    Ok(sample.add(&prediction.mul(&coefficient)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_core::{CacheLimits, Client, Compiler};
    use rxla_nn::{Model, ModelInput};

    #[test]
    fn tiny_transformer_traces_with_official_parameter_paths() {
        let config = QwenImage21Config::tiny(2, 3);
        let model = Model::new(|cx: Cx| {
            let image = cx.input([1, 6, 8])?;
            let timestep = cx.input([1])?;
            let context = cx.input([1, 5, 24])?;
            qwen_image_2_1_transformer(cx, &image, &timestep, &context, &config)
        })
        .trace()
        .unwrap();
        assert_eq!(model.outputs()[0].shape(), [1, 6, 8]);
        for path in [
            "img_in.weight",
            "txt_in.text_norm.weight",
            "txt_in.in_layer.weight",
            "time_text_embed.timestep_embedder.linear_1.weight",
            "modulation.1.weight",
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.norm_q.weight",
            "transformer_blocks.0.img_mlp.gate_layer.weight",
            "norm_out.linear.weight",
            "proj_out.weight",
        ] {
            assert!(model.schema().get(path).is_some(), "missing {path}");
        }
        model.prepare().unwrap();
    }

    #[test]
    fn official_topology_is_valid() {
        let config = QwenImage21Config::official(128, 128);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.in_channels, 64);
        config.validate().unwrap();
    }

    #[test]
    fn official_schema_matches_checkpoint_index() {
        let config = QwenImage21Config::official(1, 1);
        let model = Model::new(|cx: Cx| {
            let image = cx.input([1, 1, 64])?;
            let timestep = cx.input([1])?;
            let context = cx.input([1, 1, 4096])?;
            qwen_image_2_1_transformer(cx, &image, &timestep, &context, &config)
        })
        .trace()
        .unwrap();
        let parameters = model.schema().parameters();
        assert_eq!(parameters.len(), 297);
        assert!(
            parameters
                .iter()
                .all(|parameter| parameter.storage_dtype() == DType::BF16)
        );
        assert_eq!(
            parameters
                .iter()
                .map(|parameter| parameter.shape().iter().product::<i64>())
                .sum::<i64>(),
            7_115_124_736
        );
        assert!(
            model
                .schema()
                .get("time_text_embed.timestep_embedder.linear_1.bias")
                .is_none()
        );
    }

    #[test]
    fn official_w8_schema_fits_a_sixteen_gib_device() {
        let config = QwenImage21Config::official_w8(1, 1);
        let model = Model::new(|cx: Cx| {
            let image = cx.input([1, 1, 64])?;
            let timestep = cx.input([1])?;
            let context = cx.input([1, 1, 4096])?;
            qwen_image_2_1_transformer(cx, &image, &timestep, &context, &config)
        })
        .trace()
        .unwrap();
        assert_eq!(model.schema().parameters().len(), 529);
        assert_eq!(
            model.schema().get("img_in.weight").unwrap().storage_dtype(),
            DType::U8
        );
        assert_eq!(
            model.schema().get("img_in.scale").unwrap().shape(),
            [4096, 1]
        );
        assert_eq!(
            model
                .schema()
                .get("transformer_blocks.0.attn.norm_q.weight")
                .unwrap()
                .storage_dtype(),
            DType::F32
        );
        let storage_bytes = model
            .schema()
            .parameters()
            .iter()
            .map(|parameter| {
                let elements = parameter.shape().iter().product::<i64>();
                elements
                    * match parameter.storage_dtype() {
                        DType::U8 => 1,
                        DType::F32 => 4,
                        dtype => panic!("unexpected W8 schema dtype {dtype:?}"),
                    }
            })
            .sum::<i64>();
        assert!(storage_bytes < 7_700_000_000, "{storage_bytes}");
    }

    #[test]
    fn reusable_flow_step_has_one_parameter_tree() {
        let config = QwenImage21Config::tiny_w8(2, 3);
        let model = Model::new(|cx: Cx| {
            let image = cx.input([1, 6, 8])?;
            let timestep = cx.input([1])?;
            let step_size = cx.input([1])?;
            let context = cx.input([1, 5, 24])?;
            qwen_image_2_1_euler_step(cx, &image, &timestep, &step_size, &context, &config)
        })
        .trace()
        .unwrap();
        assert_eq!(model.outputs()[0].shape(), [1, 6, 8]);
        assert_eq!(model.schema().parameters().len(), 49);
        model.prepare().unwrap();
    }

    #[test]
    #[ignore = "requires a trusted PJRT_CUDA_PLUGIN_PATH and compiles the complete 32-layer transformer"]
    fn official_w8_step_compiles_on_cuda() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let plugin = std::env::var("PJRT_CUDA_PLUGIN_PATH")?;
        let client = unsafe { Client::load(plugin) }?;
        let config = QwenImage21Config::official_w8(32, 32);
        let model = Model::new(|cx: Cx, sample, timestep, step_size, context| {
            qwen_image_2_1_euler_step(cx, &sample, &timestep, &step_size, &context, &config)
        })
        .inputs((
            ModelInput::new([1, 1024, 64]),
            ModelInput::new([1]),
            ModelInput::new([1]),
            ModelInput::new([1, 32, 4096]),
        ))
        .trace()?;
        model.compile(&mut Compiler::new(client, CacheLimits::default()))?;
        Ok(())
    }
}
