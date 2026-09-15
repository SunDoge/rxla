//! Quantized text inference for the hybrid Qwen3.5 architecture.

mod layers;

use super::*;
use layers::{full_attention, gated_delta_attention, mlp};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayerType {
    LinearAttention,
    FullAttention,
}

/// Static text-tower architecture and RXLA weight-only quantization policy.
#[derive(Clone, Debug)]
pub struct Qwen3_5Config {
    vocabulary: i64,
    hidden_size: i64,
    intermediate_size: i64,
    layers: Vec<LayerType>,
    attention_heads: i64,
    key_value_heads: i64,
    head_dim: i64,
    linear_heads: i64,
    linear_head_dim: i64,
    conv_kernel: i64,
    max_positions: i64,
    rope_theta: f32,
    rotary_dim: i64,
    epsilon: f32,
    quant_group_size: i64,
}

impl Qwen3_5Config {
    /// Small hybrid topology for correctness and compiler tests.
    pub fn tiny() -> Self {
        Self {
            vocabulary: 128,
            hidden_size: 32,
            intermediate_size: 64,
            layers: vec![LayerType::LinearAttention, LayerType::FullAttention],
            attention_heads: 4,
            key_value_heads: 2,
            head_dim: 8,
            linear_heads: 4,
            linear_head_dim: 8,
            conv_kernel: 4,
            max_positions: 128,
            rope_theta: 10_000_000.0,
            rotary_dim: 4,
            epsilon: 1e-6,
            quant_group_size: 8,
        }
    }

    /// Official Qwen3.5-0.8B text topology. Two-dimensional weights use the
    /// RXLA symmetric W8 group format; conversion is intentionally explicit.
    pub fn qwen3_5_0_8b_w8() -> Self {
        Self {
            vocabulary: 248_320,
            hidden_size: 1_024,
            intermediate_size: 3_584,
            layers: (0..24)
                .map(|index| {
                    if (index + 1) % 4 == 0 {
                        LayerType::FullAttention
                    } else {
                        LayerType::LinearAttention
                    }
                })
                .collect(),
            attention_heads: 8,
            key_value_heads: 2,
            head_dim: 256,
            linear_heads: 16,
            linear_head_dim: 128,
            conv_kernel: 4,
            max_positions: 262_144,
            rope_theta: 10_000_000.0,
            rotary_dim: 64,
            epsilon: 1e-6,
            quant_group_size: 128,
        }
    }

    pub fn quant_group_size(mut self, value: i64) -> Self {
        self.quant_group_size = value;
        self
    }

    fn validate(&self) -> Result<()> {
        let valid = self.vocabulary > 0
            && self.hidden_size > 0
            && self.intermediate_size > 0
            && !self.layers.is_empty()
            && self.attention_heads > 0
            && self.key_value_heads > 0
            && self.attention_heads % self.key_value_heads == 0
            && self.head_dim > 0
            && self.linear_heads > 0
            && self.linear_head_dim > 0
            && self.conv_kernel > 0
            && self.max_positions > 0
            && self.rotary_dim > 0
            && self.rotary_dim <= self.head_dim
            && self.rotary_dim % 2 == 0
            && self.quant_group_size > 0
            && self.hidden_size % self.quant_group_size == 0
            && self.rope_theta.is_finite()
            && self.rope_theta > 0.0
            && self.epsilon.is_finite()
            && self.epsilon > 0.0;
        if !valid {
            return Err(Error::InvalidDefinition {
                message: "invalid Qwen3.5 configuration".into(),
            });
        }
        Ok(())
    }
}

fn quantized_embedding(cx: &mut Cx, ids: &Tensor, config: &Qwen3_5Config) -> Result<Tensor> {
    let groups = config.hidden_size / config.quant_group_size;
    let weight = cx
        .param_dtype(
            "weight",
            &[config.vocabulary, config.hidden_size],
            DType::U8,
        )?
        .take(ids, 0)?
        .cast(DType::F32)?
        .add_scalar(-128.0)?
        .reshape(&[
            ids.shape()[0],
            ids.shape()[1],
            groups,
            config.quant_group_size,
        ])?;
    let scale = cx
        .param("scale", &[config.vocabulary, groups])?
        .take(ids, 0)?
        .unsqueeze(3)?
        .broadcast_to(weight.shape())?;
    Ok(weight
        .mul(&scale)?
        .reshape(&[ids.shape()[0], ids.shape()[1], config.hidden_size])?)
}

fn tied_lm_head(cx: &mut Cx, hidden: &Tensor, config: &Qwen3_5Config) -> Result<Tensor> {
    let groups = config.hidden_size / config.quant_group_size;
    let weight = cx
        .param_dtype(
            "weight",
            &[config.vocabulary, config.hidden_size],
            DType::U8,
        )?
        .cast(DType::F32)?
        .add_scalar(-128.0)?
        .reshape(&[config.vocabulary, groups, config.quant_group_size])?;
    let scale = cx
        .param("scale", &[config.vocabulary, groups])?
        .unsqueeze(2)?
        .broadcast_to(weight.shape())?;
    Ok(hidden.linear(
        &weight
            .mul(&scale)?
            .reshape(&[config.vocabulary, config.hidden_size])?,
        None,
    )?)
}

/// Full-sequence text-only inference for Qwen3.5 with weight-only W8 storage.
/// Token IDs are `[batch, sequence]`; logits are `[batch, sequence, vocabulary]`.
pub fn qwen3_5(cx: &mut Cx, token_ids: &Tensor, config: &Qwen3_5Config) -> Result<Tensor> {
    config.validate()?;
    let [_, sequence] = token_ids.shape() else {
        return Err(Error::InvalidDefinition {
            message: "Qwen3.5 token IDs must have shape [batch, sequence]".into(),
        });
    };
    if token_ids.dtype() != DType::I32 || *sequence <= 0 || *sequence > config.max_positions {
        return Err(Error::InvalidDefinition {
            message: "Qwen3.5 requires nonempty in-range I32 token IDs".into(),
        });
    }
    let positions = cx.iota_i32(&[*sequence], 0)?;
    let half_rotary = config.rotary_dim / 2;
    let frequencies = (0..half_rotary)
        .map(|index| {
            config
                .rope_theta
                .powf(-2.0 * index as f32 / config.rotary_dim as f32)
        })
        .collect::<Vec<_>>();
    let angle_shape = [1, 1, *sequence, half_rotary];
    let angles = positions
        .to_f32()?
        .reshape(&[1, 1, *sequence, 1])?
        .broadcast_to(&angle_shape)?
        .mul(
            &cx.constant(&[1, 1, 1, half_rotary], &frequencies)?
                .broadcast_to(&angle_shape)?,
        )?;
    let causal_mask = positions.causal_attention_mask(&positions)?;

    let mut model = cx.named("model")?;
    let mut language_model = model.named("language_model")?;
    let mut embedding = language_model.named("embed_tokens")?;
    let mut hidden = quantized_embedding(&mut embedding, token_ids, config)?;
    drop(embedding);
    for (index, layer_type) in config.layers.iter().enumerate() {
        let mut layers = language_model.named("layers")?;
        let mut layer = layers.named(&index.to_string())?;
        let normalized = layer
            .named("input_layernorm")?
            .rms_norm()
            .epsilon(config.epsilon)
            .zero_centered(true)
            .apply(&hidden)?;
        let mixed = match layer_type {
            LayerType::LinearAttention => {
                let mut mixer = layer.named("linear_attn")?;
                gated_delta_attention(&mut mixer, &normalized, config)?
            }
            LayerType::FullAttention => {
                let mut mixer = layer.named("self_attn")?;
                full_attention(&mut mixer, &normalized, &angles, &causal_mask, config)?
            }
        };
        hidden = hidden.add(&mixed)?;
        let normalized = layer
            .named("post_attention_layernorm")?
            .rms_norm()
            .epsilon(config.epsilon)
            .zero_centered(true)
            .apply(&hidden)?;
        let mut feed_forward = layer.named("mlp")?;
        hidden = hidden.add(&mlp(&mut feed_forward, &normalized, config)?)?;
    }
    hidden = language_model
        .named("norm")?
        .rms_norm()
        .epsilon(config.epsilon)
        .zero_centered(true)
        .apply(&hidden)?;
    let mut embedding = language_model.named("embed_tokens")?;
    tied_lm_head(&mut embedding, &hidden, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rxla_core::{CacheLimits, Client, Compiler};
    use rxla_nn::Model;

    #[test]
    fn tiny_quantized_qwen_builds_both_mixer_types_and_lowers() {
        let config = Qwen3_5Config::tiny();
        let model = Model::new(|cx: &mut Cx| {
            let ids = cx.input_dtype(&[1, 3], DType::I32)?;
            qwen3_5(cx, &ids, &config)
        });
        let (schema, applied) = model.trace().unwrap();
        assert_eq!(applied.outputs()[0].shape(), [1, 3, 128]);
        assert!(
            schema
                .get("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
                .is_some()
        );
        assert!(
            schema
                .get("model.language_model.layers.1.self_attn.q_norm.weight")
                .is_some()
        );
        assert_eq!(
            schema
                .get("model.language_model.embed_tokens.weight")
                .unwrap()
                .dtype(),
            DType::U8
        );
        applied.prepare().unwrap();
    }

    #[test]
    #[ignore = "requires trusted PJRT_CPU_PLUGIN_PATH"]
    fn tiny_quantized_qwen_compiles_with_pjrt() {
        let plugin = std::env::var("PJRT_CPU_PLUGIN_PATH").unwrap();
        let client = unsafe { Client::load(plugin) }.unwrap();
        let config = Qwen3_5Config::tiny();
        let model = Model::new(|cx: &mut Cx| {
            let ids = cx.input_dtype(&[1, 3], DType::I32)?;
            qwen3_5(cx, &ids, &config)
        });
        let (_, applied) = model.trace().unwrap();
        applied
            .compile(&mut Compiler::new(client, CacheLimits::default()))
            .unwrap();
    }
}
